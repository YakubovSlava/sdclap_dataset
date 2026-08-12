#!/usr/bin/env python3
"""
Диагностика: для каждой пары (path_book, path_audio) из реестра проверяет, действительно ли
текст epub соответствует содержимому аудио — независимо от каких-либо результатов LLM-экстракции
или скрипта матчинга (02_match_fragments_to_audio.py).

Метод: берётся один представительный абзац из epub (примерно из середины книги, чтобы не попасть
на титул/оглавление/предисловие), и его расстояние редактирования ищется по ВСЕМУ аудио-транскрипту
целиком (edlib, mode="HW", позиционно-независимый поиск). Если реального совпадения нет нигде в
книге — это не баг матчинга, а несоответствие текста и аудио в самом реестре (ошибка сопоставления
на шаге 1.1, вероятно — сопоставление источников текста и аудио по строке названия без проверки автора).

Ничего не удаляет и не изменяет исходные данные — только пишет отчёт.

Запуск:
    python scripts/check_text_audio_mismatch.py --output data/processed/text_audio_mismatch_check.csv
"""
import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import edlib
import pandas as pd
from tqdm import tqdm

from text_utils import epub_paragraphs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("check_text_audio_mismatch")

MISMATCH_RATIO_THRESHOLD = 0.35


def pick_sample_paragraph(paragraphs: list[str]) -> str | None:
    mid = paragraphs[len(paragraphs) // 3: 2 * len(paragraphs) // 3] or paragraphs
    candidates = [p for p in mid if 120 <= len(p) <= 350]
    if not candidates:
        candidates = [p for p in mid if len(p) >= 80]
    if not candidates:
        return None
    return max(candidates, key=len)[:300]


def check_pair(row: dict) -> dict:
    path_book = Path(row["path_book"])
    path_audio = Path(row["path_audio"])
    title = path_audio.name
    json_path = path_audio / "raw" / f"{title}.json"

    out = {
        "path_book": str(path_book), "path_audio": str(path_audio),
        "book_name": row.get("book_names"), "book_author": row.get("book_authors"),
        "reader": row.get("book_reader_names"),
        "sample_text": None, "ratio": None, "status": None,
    }

    if not json_path.exists():
        out["status"] = "no_audio_json"
        return out

    try:
        paragraphs = epub_paragraphs(path_book)
    except Exception as e:
        out["status"] = f"epub_parse_error: {e}"
        return out

    sample = pick_sample_paragraph(paragraphs)
    if sample is None:
        out["status"] = "no_sample_paragraph"
        return out
    out["sample_text"] = sample

    try:
        with json_path.open(encoding="utf-8") as f:
            transcribed = json.load(f)
    except Exception as e:
        out["status"] = f"audio_json_error: {e}"
        return out

    words = [w["word"] for seg in transcribed for w in seg["words"]]
    # регистр приводим к нижнему — капслочные реплики (крики) в epub иначе дают огромное
    # расстояние из-за одного только регистра, а не реального несовпадения (см. 02_
    # match_fragments_to_audio.py, тот же баг был найден и исправлен в матчинге фрагментов)
    full_text = " ".join(words).lower()
    sample = sample.lower()
    if not full_text:
        out["status"] = "empty_audio"
        return out

    k = max(int(len(sample) * (MISMATCH_RATIO_THRESHOLD + 0.15)), 10)
    res = edlib.align(sample, full_text, mode="HW", task="distance", k=k)
    dist = res["editDistance"]
    if dist < 0:
        out["ratio"] = None
        out["status"] = "mismatch"
    else:
        ratio = dist / len(sample)
        out["ratio"] = round(ratio, 3)
        out["status"] = "ok" if ratio < MISMATCH_RATIO_THRESHOLD else "mismatch"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--registry-csv", default="data/books_postdownload_filtered.csv")
    parser.add_argument("--output", default="data/processed/text_audio_mismatch_check.csv")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.registry_csv).dropna(subset=["path_book", "path_audio"])
    df = df.drop_duplicates(subset=["path_book", "path_audio"])
    if args.limit:
        df = df.head(args.limit)
    rows = df.to_dict("records")
    log.info("checking %d (path_book, path_audio) pairs...", len(rows))

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(check_pair, r) for r in rows]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="checking pairs"):
            results.append(fut.result())

    out_df = pd.DataFrame(results)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    counts = out_df["status"].value_counts()
    log.info("status breakdown:\n%s", counts.to_string())
    n_checked = (out_df["status"].isin(["ok", "mismatch"])).sum()
    n_mismatch = (out_df["status"] == "mismatch").sum()
    if n_checked:
        log.info("mismatch rate among checked: %d/%d (%.1f%%)", n_mismatch, n_checked, 100 * n_mismatch / n_checked)
    log.info("written: %s", out_path)


if __name__ == "__main__":
    main()
