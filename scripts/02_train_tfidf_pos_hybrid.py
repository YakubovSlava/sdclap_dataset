#!/usr/bin/env python3
"""
Обучение гибридного фильтра (TF-IDF + POS-признаки + контекст соседних абзацев, LogReg) на
уже накопленных результатах LLM-экстракции (data/processed/speech_fragments/*.jsonl) и
сохранение весов в scripts/models/.

Запускать из корня репозитория, venv/ активирован. Постановка эксперимента, метод и
результаты — для статьи/tfidf_pos_hybrid_experiment.md.

Два режима:

  holdout — обучиться на одном наборе книг, проверить recall/экономию на другом (честная
            проверка на данных, которых модель не видела). Пример — обучение на первых 50
            книгах, проверка на следующих 50:

    python scripts/train_tfidf_pos_hybrid.py holdout \
        --train-csv data/books_50_original.csv \
        --test-csv data/books_50_new.csv \
        --output scripts/models/tfidf_pos_hybrid_holdout.pkl

  cv — k-fold кросс-валидация по книгам на одном наборе (честная оценка качества без
       отдельного hold-out), затем полный рефит на всех книгах набора и сохранение весов.
       Пример — CV на первых 50 книгах:

    python scripts/train_tfidf_pos_hybrid.py cv \
        --csv data/books_50_original.csv \
        --folds 5 \
        --output scripts/models/tfidf_pos_hybrid_cv50.pkl

--train-csv/--test-csv/--csv — любой CSV с колонкой path_book (например
data/books_postdownload_filtered.csv или подмножество вроде books_50_new.csv из
для статьи/tfidf_pos_hybrid_experiment.md) — используются только строки, для которых уже
есть обработанный jsonl в --fragments-dir. Дедупликация текстов по md5 файла — всегда
включена (см. поправку 2 в для статьи/tfidf_classifier_experiment.md про утечку данных
из-за книг-дублей в реестре).
"""
import argparse
import glob
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import text_utils as m
from tfidf_pos_hybrid_filter import HybridModel, fit_hybrid_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train_tfidf_pos_hybrid")

THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 19)]  # 0.05 .. 0.90, шаг 0.05


def load_book_data(csv_path: str, fragments_dir: str) -> list[dict]:
    """Возвращает список {path_book, paragraphs (после диалогового фильтра), valid (список
    валидных фрагментов из jsonl)} для книг из csv_path, у которых есть обработанный jsonl."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["path_book"]).drop_duplicates("path_book")
    wanted_paths = set(df["path_book"])

    books = []
    seen_md5: set[str] = set()
    for f in sorted(glob.glob(f"{fragments_dir}/*.jsonl")):
        lines = [json.loads(l) for l in open(f) if l.strip()]
        if not lines:
            continue
        path_book = lines[0]["path_book"]
        if path_book not in wanted_paths:
            continue
        p = Path(path_book)
        if not p.exists():
            log.warning("epub missing on disk, skip: %s", path_book)
            continue
        h = hashlib.md5(p.read_bytes()).hexdigest()
        if h in seen_md5:
            log.warning("duplicate text (md5), skip: %s", path_book)
            continue
        seen_md5.add(h)

        valid = [l for l in lines if not m.is_junk_description(l["description"], l["speech"])]
        paragraphs = m.epub_paragraphs(p)
        filtered = m.filter_dialogue_relevant_paragraphs(paragraphs, context=1)
        books.append({"path_book": path_book, "paragraphs": filtered, "valid": valid})

    log.info("loaded %d unique books from %s (matched against %s)", len(books), fragments_dir, csv_path)
    return books


def labels_for(paragraphs: list[str], valid: list[dict]) -> list[int]:
    valid_speeches = [v["speech"] for v in valid]
    return [1 if any(s in p for s in valid_speeches) else 0 for p in paragraphs]


def eval_on_books(model: HybridModel, books: list[dict]) -> dict:
    total_after_dialogue = 0
    valid_total = 0
    kept_counts = {t: 0 for t in THRESHOLDS}
    surv_counts = {t: 0 for t in THRESHOLDS}
    for b in books:
        paragraphs, valid = b["paragraphs"], b["valid"]
        total_after_dialogue += len(m.chunk_paragraphs(paragraphs, 6000))
        valid_total += len(valid)
        if not paragraphs:
            continue
        probs = model.score(paragraphs)
        for t in THRESHOLDS:
            kept = [p for p, pr in zip(paragraphs, probs) if pr >= t]
            kept_counts[t] += len(m.chunk_paragraphs(kept, 6000))
            full = "\n\n".join(kept)
            surv_counts[t] += sum(1 for v in valid if v["speech"] in full)
    return {
        "total_after_dialogue": total_after_dialogue,
        "valid_total": valid_total,
        "kept_counts": kept_counts,
        "surv_counts": surv_counts,
    }


def print_table(stats: dict, title: str) -> None:
    print(f"\n=== {title} ===")
    print(f"chunks (after dialogue filter): {stats['total_after_dialogue']}, valid fragments: {stats['valid_total']}")
    print(f"{'порог':>6} {'kept':>8} {'economy':>9} {'recall':>8}")
    for t in THRESHOLDS:
        kept = stats["kept_counts"][t]
        economy = 1 - kept / stats["total_after_dialogue"] if stats["total_after_dialogue"] else float("nan")
        recall = stats["surv_counts"][t] / stats["valid_total"] if stats["valid_total"] else float("nan")
        print(f"{t:>6} {kept:>8} {economy:>8.1%} {recall:>7.1%}")


def run_holdout(args: argparse.Namespace) -> None:
    train_books = load_book_data(args.train_csv, args.fragments_dir)
    test_books = load_book_data(args.test_csv, args.fragments_dir)

    train_paths = {b["path_book"] for b in train_books}
    test_paths = {b["path_book"] for b in test_books}
    overlap = train_paths & test_paths
    if overlap:
        raise SystemExit(f"train/test пересекаются по path_book, это утечка: {overlap}")

    labels = [labels_for(b["paragraphs"], b["valid"]) for b in train_books]
    model = fit_hybrid_model(
        [b["paragraphs"] for b in train_books],
        labels,
        meta={"mode": "holdout", "train_csv": args.train_csv, "test_csv": args.test_csv, "n_train_books": len(train_books)},
    )

    stats = eval_on_books(model, test_books)
    print_table(stats, f"HOLD-OUT: train={len(train_books)} книг, test={len(test_books)} книг (не пересекаются)")

    model.save(Path(args.output))
    log.info("model saved to %s", args.output)


def run_cv(args: argparse.Namespace) -> None:
    books = load_book_data(args.csv, args.fragments_dir)
    idx = np.arange(len(books))
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    pooled = {
        "total_after_dialogue": 0,
        "valid_total": 0,
        "kept_counts": {t: 0 for t in THRESHOLDS},
        "surv_counts": {t: 0 for t in THRESHOLDS},
    }
    for fold, (train_i, test_i) in enumerate(kf.split(idx)):
        train_subset = [books[i] for i in train_i]
        test_subset = [books[i] for i in test_i]
        labels = [labels_for(b["paragraphs"], b["valid"]) for b in train_subset]
        fold_model = fit_hybrid_model(
            [b["paragraphs"] for b in train_subset], labels, meta={"mode": "cv-fold", "fold": fold}
        )
        stats = eval_on_books(fold_model, test_subset)
        pooled["total_after_dialogue"] += stats["total_after_dialogue"]
        pooled["valid_total"] += stats["valid_total"]
        for t in THRESHOLDS:
            pooled["kept_counts"][t] += stats["kept_counts"][t]
            pooled["surv_counts"][t] += stats["surv_counts"][t]
        log.info("fold %d done: train %d books, test %d books", fold, len(train_i), len(test_i))

    print_table(pooled, f"{args.folds}-FOLD CV: {len(books)} книг ({args.csv})")

    log.info("refitting final model on all %d books for deployment...", len(books))
    labels = [labels_for(b["paragraphs"], b["valid"]) for b in books]
    final_model = fit_hybrid_model(
        [b["paragraphs"] for b in books], labels, meta={"mode": "cv-full-refit", "csv": args.csv, "folds": args.folds}
    )
    final_model.save(Path(args.output))
    log.info("final model (trained on all %d books) saved to %s", len(books), args.output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fragments-dir", default="data/processed/speech_fragments")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_holdout = sub.add_parser("holdout", help="train на --train-csv, eval на --test-csv")
    p_holdout.add_argument("--train-csv", required=True)
    p_holdout.add_argument("--test-csv", required=True)
    p_holdout.add_argument("--output", default="scripts/models/tfidf_pos_hybrid_holdout.pkl")

    p_cv = sub.add_parser("cv", help="k-fold CV на --csv, затем полный рефит и сохранение")
    p_cv.add_argument("--csv", required=True)
    p_cv.add_argument("--folds", type=int, default=5)
    p_cv.add_argument("--seed", type=int, default=42)
    p_cv.add_argument("--output", default="scripts/models/tfidf_pos_hybrid_cv.pkl")

    args = parser.parse_args()
    if args.mode == "holdout":
        run_holdout(args)
    elif args.mode == "cv":
        run_cv(args)


if __name__ == "__main__":
    main()
