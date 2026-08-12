"""
Гибридный фильтр абзацев: TF-IDF (текущий абзац + контекст соседних) + POS-флаги
(scripts/pos_filter.py) поверх LogisticRegression.

Статус: НЕ подключён к продовскому пайплайну по умолчанию — включается флагом
--hybrid-filter в scripts/01_extract_speech_fragments.py. Обучение и сохранение весов —
scripts/train_tfidf_pos_hybrid.py. Постановка эксперимента, метод, результаты (CV +
честный hold-out на новых 50 книгах) — для статьи/tfidf_pos_hybrid_experiment.md.

Признаки на абзац: TF-IDF(текущий) ⊕ TF-IDF(prev+next) ⊕ [pos_v1, pos_v2, prev_pos_v1,
prev_pos_v2, next_pos_v1, next_pos_v2] (6 бинарных POS-флагов).
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from pos_filter import pos_pass


def pos_flags(paragraphs: list[str]) -> tuple[np.ndarray, np.ndarray]:
    v1 = np.array([1.0 if pos_pass(p, extended=False) else 0.0 for p in paragraphs])
    v2 = np.array([1.0 if pos_pass(p, extended=True) else 0.0 for p in paragraphs])
    return v1, v2


def context_texts(paragraphs: list[str]) -> list[str]:
    prev_text = [""] + paragraphs[:-1]
    next_text = paragraphs[1:] + [""]
    return [prev_text[j] + " " + next_text[j] for j in range(len(paragraphs))]


def pos_feature_matrix(paragraphs: list[str]) -> csr_matrix:
    v1, v2 = pos_flags(paragraphs)
    prev_v1 = np.concatenate([[0.0], v1[:-1]])
    prev_v2 = np.concatenate([[0.0], v2[:-1]])
    next_v1 = np.concatenate([v1[1:], [0.0]])
    next_v2 = np.concatenate([v2[1:], [0.0]])
    return csr_matrix(np.column_stack([v1, v2, prev_v1, prev_v2, next_v1, next_v2]))


def new_vectorizers() -> tuple[TfidfVectorizer, TfidfVectorizer]:
    vec_curr = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=50000)
    vec_ctx = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=50000)
    return vec_curr, vec_ctx


@dataclass
class HybridModel:
    vec_curr: TfidfVectorizer
    vec_ctx: TfidfVectorizer
    clf: LogisticRegression
    meta: dict

    def transform(self, paragraphs: list[str]) -> csr_matrix:
        X_curr = self.vec_curr.transform(paragraphs)
        X_ctx = self.vec_ctx.transform(context_texts(paragraphs))
        X_pos = pos_feature_matrix(paragraphs)
        return hstack([X_curr, X_ctx, X_pos]).tocsr()

    def score(self, paragraphs: list[str]) -> np.ndarray:
        if not paragraphs:
            return np.array([])
        return self.clf.predict_proba(self.transform(paragraphs))[:, 1]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "HybridModel":
        with open(path, "rb") as f:
            return pickle.load(f)


def fit_hybrid_model(paragraphs_by_book: list[list[str]], labels_by_book: list[list[int]], meta: dict) -> HybridModel:
    """paragraphs_by_book/labels_by_book — списки списков (по одному на книгу), уже
    отфильтрованных через filter_dialogue_relevant_paragraphs. labels: 1, если абзац
    содержит speech хотя бы одного валидного фрагмента, иначе 0."""
    all_curr: list[str] = []
    all_ctx: list[str] = []
    all_labels: list[int] = []
    pos_blocks = []
    for paragraphs, labels in zip(paragraphs_by_book, labels_by_book):
        all_curr.extend(paragraphs)
        all_ctx.extend(context_texts(paragraphs))
        all_labels.extend(labels)
        pos_blocks.append(pos_feature_matrix(paragraphs))

    vec_curr, vec_ctx = new_vectorizers()
    X_curr = vec_curr.fit_transform(all_curr)
    X_ctx = vec_ctx.fit_transform(all_ctx)
    X_pos = pos_blocks[0] if len(pos_blocks) == 1 else vstack(pos_blocks)
    X = hstack([X_curr, X_ctx, X_pos]).tocsr()

    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    clf.fit(X, all_labels)
    return HybridModel(vec_curr=vec_curr, vec_ctx=vec_ctx, clf=clf, meta=meta)


def hybrid_pass(model: HybridModel, paragraphs: list[str], threshold: float) -> list[str]:
    if not paragraphs:
        return []
    probs = model.score(paragraphs)
    return [p for p, pr in zip(paragraphs, probs) if pr >= threshold]
