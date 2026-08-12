#!/usr/bin/env python3
"""
Формирует случайные непересекающиеся выборки книг из реестра для обучения/проверки
TF-IDF+POS-гибридного фильтра (train+CV набор и отдельный hold-out набор) — шаг перед
`01_extract_speech_fragments.py` и `train_tfidf_pos_hybrid.py`.

Раньше выборка бралась по порядку строк реестра (`--limit`/`--offset`, без рандомизации,
детерминированно первые N книг) — теперь по явному запросу использует случайную выборку с
зафиксированным сидом (воспроизводимо, но не привязано к порядку строк реестра).

Запускать из корня репозитория, venv/ активирован.

Пример:
    python scripts/00_sample_books_for_tfidf.py \
        --registry-csv data/books_full_corpus_ready.csv \
        --n-train-cv 50 --n-holdout 50 --seed 42 \
        --output-dir data/tfidf_samples

Результат — три файла в --output-dir:
    train_cv_books.csv   — для train+CV (передать как --csv в 01_extract / train_tfidf... cv)
    holdout_books.csv    — для hold-out проверки (train_tfidf... holdout --test-csv)
    books_list.md         — человекочитаемый список обеих выборок (для документации/для статьи)
"""
import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("sample_books_for_tfidf")


def load_registry(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["path_book"])
    return df.drop_duplicates(subset=["path_book"]).reset_index(drop=True)


def sample_disjoint(
    df: pd.DataFrame, n_train_cv: int, n_holdout: int, seed: int, exclude_paths: set[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pool = df[~df["path_book"].isin(exclude_paths)]
    total_needed = n_train_cv + n_holdout
    if len(pool) < total_needed:
        raise ValueError(
            f"в реестре после исключений только {len(pool)} книг, нужно {total_needed} "
            f"({n_train_cv} train+CV + {n_holdout} holdout)"
        )
    shuffled = pool.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    train_cv = shuffled.iloc[:n_train_cv].reset_index(drop=True)
    holdout = shuffled.iloc[n_train_cv : n_train_cv + n_holdout].reset_index(drop=True)
    return train_cv, holdout


def write_books_list_md(train_cv: pd.DataFrame, holdout: pd.DataFrame, seed: int, out_path: Path) -> None:
    lines = [
        "# Случайная выборка книг для обучения/проверки TF-IDF+POS-гибридного фильтра",
        "",
        f"Сгенерировано `scripts/00_sample_books_for_tfidf.py`, `--seed {seed}` "
        "(воспроизводимо при том же реестре и сиде).",
        "",
        f"## Train + CV ({len(train_cv)} книг)",
        "",
        "| # | Книга | Автор | Чтец |",
        "|---|---|---|---|",
    ]
    for i, row in train_cv.iterrows():
        lines.append(
            f"| {i + 1} | {row.get('book_names', row.get('title', ''))} | "
            f"{row.get('book_authors', '')} | {row.get('book_reader_names', '')} |"
        )
    lines += ["", f"## Hold-out ({len(holdout)} книг)", "", "| # | Книга | Автор | Чтец |", "|---|---|---|---|"]
    for i, row in holdout.iterrows():
        lines.append(
            f"| {i + 1} | {row.get('book_names', row.get('title', ''))} | "
            f"{row.get('book_authors', '')} | {row.get('book_reader_names', '')} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--registry-csv", default="data/books_full_corpus_ready.csv")
    parser.add_argument("--output-dir", default="data/tfidf_samples")
    parser.add_argument("--n-train-cv", type=int, default=50)
    parser.add_argument("--n-holdout", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exclude-csv", action="append", default=[],
        help="CSV с колонкой path_book — книги оттуда исключаются из пула перед выборкой "
             "(можно указать несколько раз, например чтобы не пересекаться с прошлыми экспериментами)",
    )
    args = parser.parse_args()

    registry = load_registry(Path(args.registry_csv))
    log.info("загружен реестр: %d уникальных книг из %s", len(registry), args.registry_csv)

    exclude_paths: set[str] = set()
    for exc_path in args.exclude_csv:
        exc_df = pd.read_csv(exc_path)
        exclude_paths |= set(exc_df["path_book"].dropna())
        log.info("исключено %d книг из %s", exc_df["path_book"].notna().sum(), exc_path)

    train_cv, holdout = sample_disjoint(registry, args.n_train_cv, args.n_holdout, args.seed, exclude_paths)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_cv_path = output_dir / "train_cv_books.csv"
    holdout_path = output_dir / "holdout_books.csv"
    list_md_path = output_dir / "books_list.md"

    train_cv.to_csv(train_cv_path, index=False)
    holdout.to_csv(holdout_path, index=False)
    write_books_list_md(train_cv, holdout, args.seed, list_md_path)

    log.info("train+CV: %d книг -> %s", len(train_cv), train_cv_path)
    log.info("holdout: %d книг -> %s", len(holdout), holdout_path)
    log.info("список для документации -> %s", list_md_path)

    overlap = set(train_cv["path_book"]) & set(holdout["path_book"])
    assert not overlap, f"пересечение train_cv/holdout не должно быть возможно, но найдено: {overlap}"
    log.info("пересечений между train+CV и holdout: 0 (проверено)")


if __name__ == "__main__":
    main()
