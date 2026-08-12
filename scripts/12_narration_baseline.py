#!/usr/bin/env python3
"""
Нейтральная база диктора по (чтец, книга) — для нормировки аудио-фич фрагментов (шаг 1.4b).

Идея: «нейтраль» диктора — это его немаркированная речь. Среднее по извлечённым ДИАЛОГОВЫМ
фрагментам для этого не годится — оно смещено эмоцией (в нём же крики/шёпоты) и шумит на книгах с
малым числом реплик. Поэтому база строится по **случайной выборке ASR-сегментов исходной записи**
(по 200 на книгу): большинство сегментов — нарраторская нейтральная речь, медиана тянется к ней.
Разделять нарратив/диалог не пытаемся — над медианой 200 сегментов диалоги в меньшинстве.

Для каждой книги: `--n-segments` случайных ASR-сегментов (длительность >= `--min-dur`) читаются
срезами из исходного mp3 (soundfile, один хендл на книгу, seek+read), ресемплятся в 24 кГц (как
FLAC-фрагменты), считаются eGeMAPS (openSMILE), из них берутся **медиана** (центр) и **робастный
масштаб** (1.4826·MAD) по каждой фиче.

Вход — `dataset.csv` (уникальные пары чтец/книга + `path_audio`) и ASR-json/`mp3` в
`<path_audio>/raw/`. Выход — CSV на (чтец, книга): `egm_<feat>_med`, `egm_<feat>_scale`,
`n_segments`, `base_status`. Фолбэк (чтец → глобально) для разреженных строится позже, в `07`.

Пример:
    python scripts/10_narration_baseline.py --workers 8
    python scripts/10_narration_baseline.py --limit 5   # тест на 5 книгах
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

DEFAULT_MANIFEST = "data/processed/audio_fragments/dataset.csv"
DEFAULT_OUTPUT = "data/processed/audio_fragments/reader_book_baseline.csv"
SR = 24000  # как у FLAC-фрагментов (04), чтобы база и фрагменты были сопоставимы

_SMILE = None


def _get_smile():
    global _SMILE
    if _SMILE is None:
        import opensmile
        _SMILE = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
    return _SMILE


def egemaps_columns() -> list[str]:
    import opensmile
    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                            feature_level=opensmile.FeatureLevel.Functionals)
    return list(smile.feature_names)


def _process_book(args: tuple) -> dict:
    """Воркер: одна книга -> строка базы (медианы/масштабы eGeMAPS по 200 сегментам)."""
    import soundfile as sf
    import soxr

    reader, book_name, path_audio, n_segments, min_dur, seed = args
    row: dict = {"reader": reader, "book_name": book_name}
    try:
        js = glob.glob(f"{path_audio}/raw/*.json")
        mp3 = glob.glob(f"{path_audio}/raw/*.mp3")
        if not js or not mp3:
            row["n_segments"] = 0
            row["base_status"] = "missing_json_or_mp3"
            return row
        segs = [s for s in json.load(open(js[0]))
                if s.get("end", 0) - s.get("start", 0) >= min_dur and s.get("words")]
        if not segs:
            row["n_segments"] = 0
            row["base_status"] = "no_segments"
            return row
        rng = random.Random(seed)
        samp = rng.sample(segs, min(n_segments, len(segs)))
        samp.sort(key=lambda x: x["start"])  # последовательные seek дешевле

        smile = _get_smile()
        feats = []
        with sf.SoundFile(mp3[0]) as f:
            sr = f.samplerate
            for s in samp:
                f.seek(int(s["start"] * sr))
                y = f.read(int((s["end"] - s["start"]) * sr), dtype="float32")
                if y.ndim > 1:
                    y = y.mean(axis=1)
                if len(y) < sr * 0.3:
                    continue
                if sr != SR:
                    y = soxr.resample(y, sr, SR)
                feats.append(smile.process_signal(y, SR).iloc[0])
        if not feats:
            row["n_segments"] = 0
            row["base_status"] = "no_features"
            return row
        F = pd.DataFrame(feats).astype(float)
        med = F.median()
        mad = (F - med).abs().median() * 1.4826  # робастный масштаб
        for c in F.columns:
            row[f"egm_{c}_med"] = float(med[c])
            row[f"egm_{c}_scale"] = float(mad[c])
        row["n_segments"] = len(feats)
        row["base_status"] = "ok"
    except Exception as e:  # noqa: BLE001
        row["n_segments"] = 0
        row["base_status"] = f"error:{type(e).__name__}"
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--n-segments", type=int, default=200)
    ap.add_argument("--min-dur", type=float, default=1.0, help="мин. длительность сегмента, с")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--readers", default=None, help="только эти чтецы (через запятую)")
    ap.add_argument("--limit", type=int, default=None, help="обработать только N книг (тест)")
    ap.add_argument("--resume", action="store_true", help="пропустить уже посчитанные (чтец, книга)")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest, usecols=["reader", "book_name", "path_audio"])
    if args.readers:
        wanted = {r.strip() for r in args.readers.split(",")}
        df = df[df["reader"].isin(wanted)]
    books = df.dropna(subset=["path_audio"]).drop_duplicates(["reader", "book_name"]).reset_index(drop=True)
    if args.resume and Path(args.output).exists():
        done = pd.read_csv(args.output, usecols=["reader", "book_name"])
        key = set(map(tuple, done[["reader", "book_name"]].values))
        before = len(books)
        books = books[~books.apply(lambda r: (r.reader, r.book_name) in key, axis=1)].reset_index(drop=True)
        print(f"resume: пропущено {before - len(books)} уже посчитанных", file=sys.stderr)
    if args.limit:
        books = books.head(args.limit)

    tasks = [(r.reader, r.book_name, r.path_audio, args.n_segments, args.min_dur, args.seed)
             for r in books.itertuples(index=False)]
    print(f"Книг к обработке: {len(tasks)}, воркеров: {args.workers}", file=sys.stderr)
    if not tasks:
        return

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_process_book, t) for t in tasks]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="baseline", unit="book"):
            rows.append(fut.result())

    cols = ["reader", "book_name"]
    for c in egemaps_columns():
        cols += [f"egm_{c}_med", f"egm_{c}_scale"]
    cols += ["n_segments", "base_status"]
    out = pd.DataFrame(rows).reindex(columns=cols)

    n_err = int((out["base_status"] != "ok").sum())
    if n_err:
        from collections import Counter
        print(f"Проблемных книг: {n_err}, {dict(Counter(s for s in out.base_status if s != 'ok'))}",
              file=sys.stderr)
    out_path = Path(args.output)
    if args.resume and out_path.exists():
        out.to_csv(out_path, mode="a", header=False, index=False)
    else:
        out.to_csv(out_path, index=False)
    print(f"Готово: {len(out)} книг -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
