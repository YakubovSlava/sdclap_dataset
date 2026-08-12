#!/usr/bin/env python3
"""
Уровень 2 фильтрации (шаг 1.4b): производные скоры поверх сырых фич. Строки НЕ удаляются —
всё пишется колонками, отбор датасета — на сборке финального манифеста.

Вход:
- аудио-фичи — выход `scripts/04_extract_audio_features.py --egemaps` (`--audio`);
- текстовые оси — выход `scripts/05_extract_text_axes.py` (`--axes`, по `fragment_id` или `description`);
- нарраторская нейтральная база — выход `scripts/10_narration_baseline.py` (`--baseline`);
- манифест `dataset.csv` — `reader`/`book_name`/`speech`/`match_distance_norm`/`n_words_matched`.

Что считает:

1. **Аномалия относительно нейтрали диктора** `<axis>_anom` — на сколько робастных σ фрагмент
   отклонился от НАРРАТОРСКОЙ базы диктора (по чтецу+книге, фолбэк чтец→глобально). Нарраторская
   база (а не среднее по диалогам) — потому что среднее по диалогам смещено эмоцией; валидация:
   разделение крик/шёпот +1.13σ против +0.66σ у диалоговой (history.md 2026-07-23).
2. **Согласие с текстом** `<axis>_agree = знак(high=+1/low=−1) × anom`. `alignment_score` = loudness_agree
   (единственная валидированная ось; pitch/tempo/tension — метадата, доверять осторожно).
3. **`manner_bucket`** — стратегия отбора:
   - `neutral` — текст нейтрален по всем осям (берём как есть; `anomaly_score` — фича для хвоста);
   - `confirmed` — текст отклоняется И аудио подтверждает по громкости (`loudness_agree >= --confirm-sigma`);
   - `unconfirmed` — текст отклоняется, аудио не подтвердило (резерв, НЕ мусор).
   `underdelivery` — грубый недоигрыш: `loudness=high`, а аудио сильно ниже нейтрали (`< -under-sigma`).
4. **`junk_*`** — флаги мусора (метаданные):
   - этап 0: `junk_zerodur`, `junk_toolong` (`>= --max-dur`);
   - корректность текста: `junk_badmatch` (`match_distance_norm >= --max-dist`), `junk_few_words`
     (`speech <= 3` слов), `junk_few_asr_words` (`n_words_matched <= 3`);
   - акустика (жёсткое): `junk_tooshort`;
   - акустика (справочные, в `junk_any_hard` НЕ входят — ловят немного и ненадёжны/не провалидированы):
     `junk_nonspeech`, `junk_silence`, `junk_truncation` (edge — 82% ложных), `junk_flatness_soft`
     (ловит тихую речь);
   - `junk_any_hard` = этап0 ∪ tooshort (надёжный акустический мусор).

Жёсткие фильтры для датасета (применяются на сборке): `~junk_any_hard & ~junk_badmatch &
~junk_few_words & ~junk_few_asr_words`; далее корзины `confirmed + neutral`.

Пример:
    python scripts/07_derive_scores.py \
        --audio data/processed/audio_fragments/audio_features.csv \
        --axes  data/processed/audio_fragments/text_axes.A.csv text_axes.B.csv \
        --baseline data/processed/audio_fragments/reader_book_baseline.csv \
        --out   data/processed/audio_fragments/scores.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

AXES = ["loudness", "pitch", "tempo", "tension"]
EXPECT = {"high": 1.0, "low": -1.0}  # neutral → не проверяем

# ось -> измеримая eGeMAPS-фича (все есть и в scores.csv, и в нарраторской базе как egm_*_med/_scale).
# Направление: high по оси => фича ВЫШЕ. loudness валидирована; остальные — метадата.
AXIS_FEATURE = {
    "loudness": "egm_loudness_sma3_amean",
    "pitch":    "egm_F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
    "tempo":    "egm_VoicedSegmentsPerSec",
    "tension":  "egm_jitterLocal_sma3nz_amean",
}


def resolve_baseline(df: pd.DataFrame, baseline: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """На каждую строку df вернуть med/scale по фичам `feats` из нарраторской базы с фолбэком
    (чтец, книга) → (чтец) → глобально. Векторизовано."""
    cols = [f"{f}_med" for f in feats] + [f"{f}_scale" for f in feats]
    cols = [c for c in cols if c in baseline.columns]
    ok = baseline[baseline["base_status"] == "ok"] if "base_status" in baseline.columns else baseline
    book = ok.groupby(["reader", "book_name"])[cols].median()
    reader = ok.groupby("reader")[cols].median()
    glob = ok[cols].median()

    m = df[["reader", "book_name"]].merge(book, on=["reader", "book_name"], how="left")
    rd = df[["reader"]].merge(reader, on="reader", how="left")[cols]
    for c in cols:
        m[c] = m[c].fillna(rd[c]).fillna(glob[c])
    m.index = df.index
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    base = "data/processed/audio_fragments"
    ap.add_argument("--audio", default=f"{base}/audio_features.csv")
    ap.add_argument("--axes", nargs="*", default=[f"{base}/text_axes.A.csv"],
                    help="Файлы осей (05); несколько — склеиваются. Ключ — fragment_id или description.")
    ap.add_argument("--manifest", default=f"{base}/dataset.csv")
    ap.add_argument("--baseline", default=f"{base}/reader_book_baseline.csv",
                    help="Нарраторская база (10_narration_baseline.py).")
    ap.add_argument("--out", default=f"{base}/scores.csv")
    ap.add_argument("--confirm-sigma", type=float, default=0.5,
                    help="loudness_agree >= порога → аудио подтверждает текст (bucket confirmed).")
    ap.add_argument("--under-sigma", type=float, default=1.0,
                    help="loudness=high и anom < -порога → underdelivery (грубый недоигрыш).")
    ap.add_argument("--max-dur", type=float, default=60.0, help="Этап 0: длиннее — over-capture.")
    ap.add_argument("--max-dist", type=float, default=0.4, help="junk_badmatch: match_distance_norm >= порога.")
    args = ap.parse_args()

    print(f"чтение аудио-фич ({args.audio})...", flush=True)
    audio = pd.read_csv(args.audio)
    man_cols = ["fragment_id", "reader", "book_name", "speech", "description",
                "match_distance_norm", "n_words_matched"]
    manifest = pd.read_csv(args.manifest)
    manifest = manifest[[c for c in man_cols if c in manifest.columns]]
    df = audio.merge(manifest, on="fragment_id", how="left")
    print(f"загружено {len(df)} строк", flush=True)

    axes_files = [p for p in args.axes if Path(p).exists()]
    if axes_files:
        ax = pd.concat([pd.read_csv(p) for p in axes_files], ignore_index=True)
        key = "fragment_id" if "fragment_id" in ax.columns else "description"
        ax = ax.drop_duplicates(subset=[key])
        df = df.merge(ax[[key, *[a for a in AXES if a in ax.columns]]], on=key, how="left")
        print(f"оси: {len(axes_files)} файл(ов), джойн по '{key}'", flush=True)
    for a in AXES:
        if a not in df.columns:
            df[a] = "neutral"
        df[a] = df[a].fillna("neutral")

    # --- аномалия от нарраторской базы + согласие по осям ---
    if Path(args.baseline).exists():
        print("чтение нарраторской базы...", flush=True)
        baseline = pd.read_csv(args.baseline)
        feats = [AXIS_FEATURE[a] for a in AXES if AXIS_FEATURE[a] in df.columns]
        bl = resolve_baseline(df, baseline, feats)
        for axis in AXES:
            f = AXIS_FEATURE[axis]
            if f not in df.columns or f"{f}_scale" not in bl.columns:
                continue
            scale = bl[f"{f}_scale"].where(bl[f"{f}_scale"] > 0)
            anom = (df[f] - bl[f"{f}_med"]) / scale
            agree = df[axis].map(EXPECT) * anom
            df[f"{axis}_anom"] = anom.round(3)
            df[f"{axis}_agree"] = agree.round(3)
        df["alignment_score"] = df.get("loudness_agree")
    else:
        print(f"ВНИМАНИЕ: база {args.baseline} не найдена — аномалии/корзины не считаются", flush=True)

    def col(name, default):
        return df[name] if name in df.columns else pd.Series(default, index=df.index)

    # --- корзины манеры (только по валидированной оси loudness; pitch/tempo/tension — метадата,
    #     tension вообще переразмечена LLM ~46%, в корзину её брать нельзя) ---
    if "loudness_agree" in df.columns:
        loud_dev = df["loudness"].isin(EXPECT)                    # loudness ненейтрален
        confirmed = loud_dev & (df["loudness_agree"] >= args.confirm_sigma)
        bucket = np.where(~loud_dev, "neutral",
                          np.where(confirmed, "confirmed", "unconfirmed"))
        df["manner_bucket"] = bucket
        df["underdelivery"] = (df["loudness"] == "high") & (df["loudness_anom"] < -args.under_sigma)
        df["anomaly_score"] = df["loudness_anom"]  # «степень несоответствия» для всех, вкл. нейтральные

    # --- флаги мусора ---
    n_speech = col("speech", "").fillna("").astype(str).str.split().str.len()
    df["junk_zerodur"] = col("duration_s", 1.0) <= 0
    df["junk_toolong"] = col("duration_s", 0.0) >= args.max_dur
    df["junk_nonspeech"] = col("voiced_ratio", 1.0) < 0.30
    df["junk_silence"] = col("silence_ratio", 0.0) > 0.55
    df["junk_tooshort"] = col("duration_s", 9.0) < 0.40
    df["junk_few_words"] = n_speech <= 3
    df["junk_few_asr_words"] = col("n_words_matched", 99) <= 3
    df["junk_truncation"] = (col("edge_start_ratio", 0.0) > 3) | (col("edge_end_ratio", 0.0) > 3)  # 82% ложных
    df["junk_flatness_soft"] = col("spec_flatness_mean", 0.0) > 0.15  # ловит тихую речь
    df["junk_badmatch"] = col("match_distance_norm", 0.0) >= args.max_dist
    # junk_nonspeech/junk_silence сознательно НЕ входят: ловят мало (~1000 фрагментов сверх
    # остальных фильтров на полном корпусе) и не провалидированы вручную — оставлены справочными
    # колонками, как junk_truncation/junk_flatness_soft.
    df["junk_any_hard"] = df[["junk_zerodur", "junk_toolong", "junk_tooshort"]].any(axis=1)

    print(f"запись {args.out} ({df.shape[1]} колонок)...", flush=True)
    df.to_csv(args.out, index=False)

    # --- сводка ---
    n = len(df)
    print(f"{n} фрагментов -> {args.out}")
    print("мусор:")
    for c in ["junk_zerodur", "junk_toolong", "junk_nonspeech", "junk_silence", "junk_tooshort",
              "junk_few_words", "junk_few_asr_words", "junk_any_hard", "junk_badmatch",
              "junk_truncation", "junk_flatness_soft"]:
        print(f"  {c:20s} {int(df[c].sum()):6d}  ({100*df[c].mean():4.1f}%)")
    if "manner_bucket" in df.columns:
        print("корзины манеры:", {k: int(v) for k, v in df["manner_bucket"].value_counts().items()})
        print(f"underdelivery (loudness=high, аудио вяло): {int(df['underdelivery'].sum())}")
        # чистый датасет = жёсткие фильтры + confirmed/neutral
        hard = df[["junk_any_hard", "junk_badmatch", "junk_few_words", "junk_few_asr_words"]].any(axis=1)
        keep = ~hard & df["manner_bucket"].isin(["confirmed", "neutral"])
        h = df.loc[keep, "duration_s"].sum() / 3600
        print(f"датасет (жёсткие фильтры + confirmed/neutral): {int(keep.sum())} фрагм, {h:.1f} ч")


if __name__ == "__main__":
    main()
