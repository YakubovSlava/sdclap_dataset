#!/usr/bin/env python3
"""
Сборка одной таблицы для разведки/сверки: манифест фрагментов + аудио-фичи
(`scripts/04_extract_audio_features.py`) + текстовые оси (`scripts/05_extract_text_axes.py`).

Джойн: аудио-фичи по `fragment_id`, оси по `fragment_id` (из `--axes`, файла `--join-output`
скрипта 05). Строки без аудио-фич отбрасываются (inner join по аудио), оси подтягиваются слева
(могут быть пустыми, если описание не размечено).

Пример:
    python scripts/06_join_features.py
    python scripts/06_join_features.py --out data/processed/audio_fragments/explore_joined.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

BASE = "data/processed/audio_fragments"
# колонки для человекочитаемого разведочного среза (в таком порядке); недостающие молча пропускаются
VIEW_COLS = [
    "reader", "speech", "description",
    "loudness", "pitch", "tempo", "tension",
    "loudness_dbfs", "dyn_range_db", "f0_std_semitones", "onset_rate",
    "silence_ratio", "voiced_ratio", "edge_start_ratio", "edge_end_ratio",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=f"{BASE}/dataset.csv")
    ap.add_argument("--audio", default=f"{BASE}/explore_audio.csv", help="выход скрипта 04")
    ap.add_argument("--axes", default=f"{BASE}/explore_axes_by_fragment.csv",
                    help="выход скрипта 05 (--join-output); необязателен")
    ap.add_argument("--out", default=f"{BASE}/explore_joined.csv")
    ap.add_argument("--full", action="store_true",
                    help="сохранить все колонки, а не только человекочитаемый срез VIEW_COLS")
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    audio = pd.read_csv(args.audio)
    meta_cols = [c for c in ["fragment_id", "reader", "book_name", "speech", "description"]
                 if c in manifest.columns]
    df = audio.merge(manifest[meta_cols], on="fragment_id", how="inner")

    axis_cols = ["loudness", "pitch", "tempo", "tension"]
    if Path(args.axes).exists():
        axes = pd.read_csv(args.axes)
        keep = ["fragment_id"] + [c for c in axis_cols if c in axes.columns]
        df = df.merge(axes[keep], on="fragment_id", how="left")
    else:
        print(f"оси не найдены ({args.axes}) — таблица без осей")

    if not args.full:
        cols = [c for c in VIEW_COLS if c in df.columns]
        df = df[["fragment_id", *cols]]

    df.to_csv(args.out, index=False)
    print(f"{len(df)} строк -> {args.out}")


if __name__ == "__main__":
    main()
