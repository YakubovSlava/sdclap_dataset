#!/usr/bin/env python3
"""
Финальный манифест датасета (шаг 1.6) — одна таблица, объединяющая всё: аудио-фрагмент (путь +
таймкоды), текст (speech/description), метаданные книги/чтеца, мягкие метаданные манеры и явные
индексы во все посчитанные наборы эмбеддингов (2 текстовых + 7 аудио, wavlm_large добавлен
2026-08-02 по мотивам ParaSpeechCLAP).

Строки — только фрагменты, прошедшие жёсткие фильтры `07_derive_scores.py`
(~junk_any_hard & ~junk_badmatch & ~junk_few_words & ~junk_few_asr_words, см.
docs/filtering_pipeline.md) — та же база, что и текущие 89015 фрагментов / 146.4ч.

`split` (train/val/test, reader-disjoint) сюда сознательно НЕ включён — отдельное решение о
пропорциях, отдельный последующий скрипт.

Эмбеддинги сами по себе НЕ копируются в манифест (это раздуло бы таблицу на порядки) — только
индекс строки в соответствующем .npy:
    text:  description_embedding_row_bge, description_embedding_row_qwen3
           (ключ джойна — description, т.к. эмбеддинг считан на уникальные описания, дедуп)
    audio: audio_embedding_row_<model> для всех 7 моделей
           (ключ джойна — fragment_id; row != fragment_id, это позиция после сортировки при
           сохранении в 13_embed_audio.py — джойним явно, не полагаемся на совпадение)

Запуск (из корня репозитория, venv активирован):
    python scripts/14_build_manifest.py
"""
import logging

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_manifest")

BASE = "data/processed/audio_fragments"
AUDIO_MODELS = [
    "panns_cnn14", "wav2vec2_base", "wavlm_base_plus", "wavlm_large",
    "emotion2vec_plus", "wav2vec2_large_robust", "laion_clap",
    # промежуточные слои (16_embed_audio_layers.py, 2026-08-02) — см. dataset.py AUDIO_EMBEDDINGS
    "wav2vec2_base_l6", "wav2vec2_base_l9",
    "wavlm_base_plus_l6", "wavlm_base_plus_l9",
    "wav2vec2_large_robust_l6", "wav2vec2_large_robust_l9",
    "wavlm_large_l12", "wavlm_large_l18",
]

MANNER_COLS = [
    "loudness", "pitch", "tempo", "tension",
    "loudness_anom", "loudness_agree", "pitch_anom", "pitch_agree",
    "tempo_anom", "tempo_agree", "tension_anom", "tension_agree",
    "alignment_score", "manner_bucket", "anomaly_score", "underdelivery",
]


def main() -> None:
    log.info("чтение scores.csv...")
    scores_cols = [
        "fragment_id", "duration_s", "reader", "book_name", "speech", "description",
        "match_distance_norm", "n_words_matched",
        "junk_any_hard", "junk_badmatch", "junk_few_words", "junk_few_asr_words",
        *MANNER_COLS,
    ]
    scores = pd.read_csv(f"{BASE}/scores.csv", usecols=scores_cols)
    hard_ok = (~scores["junk_any_hard"] & ~scores["junk_badmatch"]
               & ~scores["junk_few_words"] & ~scores["junk_few_asr_words"])
    scores = scores[hard_ok].drop(columns=["junk_any_hard", "junk_badmatch", "junk_few_words", "junk_few_asr_words"])
    log.info("после жёстких фильтров: %d фрагментов", len(scores))

    log.info("чтение dataset.csv...")
    dataset = pd.read_csv(
        f"{BASE}/dataset.csv",
        usecols=["fragment_id", "path_book", "book_author", "character",
                  "audio_fragment_path", "match_start_ts", "match_end_ts", "cut_status"],
    )
    df = scores.merge(dataset, on="fragment_id", how="left")
    df = df[df["cut_status"] == "skipped_exists"].drop(columns=["cut_status"])
    log.info("после проверки, что FLAC реально нарезан: %d фрагментов", len(df))

    log.info("джойн book_genres...")
    registry = pd.read_csv("data/books_full_corpus_ready.csv", usecols=["path_book", "book_genres"])
    registry = registry.drop_duplicates("path_book")
    df = df.merge(registry, on="path_book", how="left")

    log.info("джойн reader_gender...")
    gender = pd.read_csv("data/reader_gender.csv")
    df = df.merge(gender, on="reader", how="left")
    n_missing_gender = df["gender"].isna().sum()
    if n_missing_gender:
        log.warning("нет пола для %d фрагментов (новый чтец вне data/reader_gender.csv?)", n_missing_gender)
    df = df.rename(columns={"gender": "reader_gender"})

    log.info("джойн текстовых эмбеддингов (bge-m3, qwen3)...")
    bge_idx = pd.read_csv(f"{BASE}/description_embeddings_index.csv").rename(
        columns={"row": "description_embedding_row_bge"})
    df = df.merge(bge_idx, on="description", how="left")
    qwen_idx_path = f"{BASE}/description_embeddings_qwen3_index.csv"
    qwen_idx = pd.read_csv(qwen_idx_path).rename(columns={"row": "description_embedding_row_qwen3"})
    df = df.merge(qwen_idx, on="description", how="left")

    log.info("джойн аудио-эмбеддингов (%d моделей)...", len(AUDIO_MODELS))
    for model in AUDIO_MODELS:
        idx_path = f"{BASE}/audio_embeddings_{model}_index.csv"
        idx = pd.read_csv(idx_path).rename(columns={"row": f"audio_embedding_row_{model}"})
        df = df.merge(idx, on="fragment_id", how="left")
        n_missing = df[f"audio_embedding_row_{model}"].isna().sum()
        if n_missing:
            log.warning("%s: нет эмбеддинга для %d фрагментов", model, n_missing)

    column_order = [
        "fragment_id",
        "path_book", "book_name", "book_author", "book_genres",
        "reader", "reader_gender", "character",
        "audio_fragment_path", "match_start_ts", "match_end_ts", "duration_s",
        "speech", "description",
        "match_distance_norm", "n_words_matched",
        "description_embedding_row_bge", "description_embedding_row_qwen3",
        *[f"audio_embedding_row_{m}" for m in AUDIO_MODELS],
        *MANNER_COLS,
    ]
    df = df[column_order]

    out_path = f"{BASE}/manifest.csv"
    df.to_csv(out_path, index=False)
    log.info("сохранено: %s (%d строк, %d колонок)", out_path, len(df), df.shape[1])


if __name__ == "__main__":
    main()
