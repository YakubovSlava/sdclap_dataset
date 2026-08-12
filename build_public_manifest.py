#!/usr/bin/env python3
"""
Публикуемый манифест датасета (сопроводительные материалы к статье, воспроизводимость) — берёт
канонический `data/processed/audio_fragments/manifest.csv` (см. `for_paper/scripts/14_build_
manifest.py`/`15_assign_reader_split.py`) и строит `manifest_public.csv`: без локальных путей и
служебных индексов эмбеддингов (бессмысленны вне репозитория), но с достаточной информацией, чтобы
исследователь с легальным доступом к соответствующим аудиозаписям мог сам нарезать те же
фрагменты и сверить их по контрольной сумме.

Колонки `manifest_public.csv`:
    fragment_id, book_name, book_author, book_genres, reader, reader_gender, character,
    speech, description, match_start_ts, match_end_ts, duration_s,
    match_distance_norm, n_words_matched,
    loudness, pitch, tempo, tension, loudness_anom, loudness_agree, pitch_anom, pitch_agree,
    tempo_anom, tempo_agree, tension_anom, tension_agree, alignment_score, manner_bucket,
    anomaly_score, underdelivery,
    split,
    knigavuhe_link, source_audio_sha256

Явно ИСКЛЮЧЕНЫ: path_book, audio_fragment_path (локальные пути), все
description_embedding_row_*/audio_embedding_row_* (индексы в локальные .npy, бессмысленны без
файлов эмбеддингов), ссылка на источник текста.

Запуск (из корня основного репозитория, venv активирован, после 15_assign_reader_split.py):
    python for_paper/build_public_manifest.py
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_public_manifest")

MANIFEST = "data/processed/audio_fragments/manifest.csv"
REGISTRY = "data/books_full_corpus_ready.csv"
CHECKSUM_CACHE = "for_paper/source_audio_checksums.csv"
OUT = "for_paper/manifest_public.csv"

MANNER_COLS = [
    "loudness", "pitch", "tempo", "tension",
    "loudness_anom", "loudness_agree", "pitch_anom", "pitch_agree",
    "tempo_anom", "tempo_agree", "tension_anom", "tension_agree",
    "alignment_score", "manner_bucket", "anomaly_score", "underdelivery",
]

KEEP_FROM_MANIFEST = [
    "fragment_id", "path_book", "book_name", "book_author", "book_genres",
    "reader", "reader_gender", "character", "speech", "description",
    "match_start_ts", "match_end_ts", "duration_s",
    "match_distance_norm", "n_words_matched",
    *MANNER_COLS,
    "split",
]


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def find_source_mp3(path_audio: str) -> Path | None:
    matches = list(Path(path_audio, "raw").glob("*.mp3"))
    if len(matches) != 1:
        return None
    return matches[0]


def main() -> None:
    log.info("чтение %s...", MANIFEST)
    df = pd.read_csv(MANIFEST, usecols=KEEP_FROM_MANIFEST)
    log.info("строк: %d", len(df))

    log.info("джойн knigavuhe_link/path_audio из %s...", REGISTRY)
    registry = pd.read_csv(REGISTRY, usecols=["path_book", "knigavuhe_link", "path_audio"])
    registry = registry.drop_duplicates("path_book")
    df = df.merge(registry, on="path_book", how="left")
    n_missing_link = df["knigavuhe_link"].isna().sum()
    if n_missing_link:
        log.warning("нет knigavuhe_link для %d строк (не нашлось path_book в реестре)", n_missing_link)

    log.info("контрольные суммы исходных mp3 по уникальным (reader, book_name)...")
    if Path(CHECKSUM_CACHE).exists():
        cache = pd.read_csv(CHECKSUM_CACHE)
        log.info("загружен кэш: %d записей", len(cache))
    else:
        cache = pd.DataFrame(columns=["reader", "book_name", "path_audio", "source_audio_sha256"])

    known = set(zip(cache["reader"], cache["book_name"]))
    pairs = df[["reader", "book_name", "path_audio"]].drop_duplicates(["reader", "book_name"])
    new_rows = []
    n_notfound = 0
    for i, row in enumerate(pairs.itertuples(index=False), 1):
        key = (row.reader, row.book_name)
        if key in known or pd.isna(row.path_audio):
            continue
        mp3 = find_source_mp3(row.path_audio)
        if mp3 is None:
            n_notfound += 1
            log.warning("не найден (или неоднозначен) исходный mp3 для %s / %s (%s)",
                        row.reader, row.book_name, row.path_audio)
            continue
        checksum = sha256_file(mp3)
        new_rows.append({"reader": row.reader, "book_name": row.book_name,
                          "path_audio": row.path_audio, "source_audio_sha256": checksum})
        if i % 100 == 0:
            log.info("  захешировано %d/%d уникальных пар...", i, len(pairs))

    if new_rows:
        cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
        cache.to_csv(CHECKSUM_CACHE, index=False)
        log.info("обновлён кэш контрольных сумм: +%d, всего %d", len(new_rows), len(cache))
    if n_notfound:
        log.warning("не найден источник mp3 для %d уникальных (reader, book) пар", n_notfound)

    df = df.merge(cache[["reader", "book_name", "source_audio_sha256"]],
                  on=["reader", "book_name"], how="left")
    n_missing_sha = df["source_audio_sha256"].isna().sum()
    if n_missing_sha:
        log.warning("нет source_audio_sha256 для %d строк", n_missing_sha)

    df = df.drop(columns=["path_book", "path_audio"])

    n_books = df["book_name"].nunique()
    n_readers = df["reader"].nunique()
    log.info("итог: %d строк, %d уникальных книг, %d уникальных чтецов", len(df), n_books, n_readers)

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    log.info("сохранено -> %s", OUT)


if __name__ == "__main__":
    main()
