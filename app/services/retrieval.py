"""Semantic retrieval over the in-memory knowledge index."""

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from app.schemas import KnowledgeKind
from app.services.dataset import KnowledgeIndex
from app.services.embeddings import EmbeddingProvider, normalize_vector

logger = logging.getLogger(__name__)

# Documents whose embeddings are this similar are treated as the same content.
NEAR_DUPLICATE_THRESHOLD = 0.98

_NON_WORD = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True, slots=True)
class RetrievedDocument:
    id: str
    kind: KnowledgeKind
    text: str
    borough: str | None
    category: str | None
    score: float


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two 1-D vectors.

    Returns 0.0 for a zero vector and raises ``ValueError`` on dimension
    mismatch or non-finite input, so callers never see NaN.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("cosine_similarity expects 1-D vectors")
    if a.shape != b.shape:
        raise ValueError(f"dimension mismatch: {a.shape[0]} != {b.shape[0]}")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("vectors must contain only finite values")
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.clip(np.dot(a, b) / (norm_a * norm_b), -1.0, 1.0))


def score_matrix(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Cosine scores of every normalized matrix row against a query.

    Rows are assumed L2-normalized (the loader guarantees this). Invalid
    scores are mapped to ``-inf`` so they always rank last.
    """
    if matrix.ndim != 2 or query.ndim != 1:
        raise ValueError("score_matrix expects a 2-D matrix and a 1-D query")
    if matrix.shape[0] == 0:
        return np.empty(0, dtype=np.float32)
    if matrix.shape[1] != query.shape[0]:
        raise ValueError(f"dimension mismatch: {matrix.shape[1]} != {query.shape[0]}")
    scores = matrix @ normalize_vector(query)
    return np.where(np.isfinite(scores), scores, -np.inf).astype(np.float32, copy=False)


def _text_key(text: str) -> str:
    return _NON_WORD.sub(" ", text.lower()).strip()


class RetrievalService:
    """Ranks knowledge documents against a question embedding."""

    def __init__(
        self,
        index: KnowledgeIndex,
        embedder: EmbeddingProvider,
        top_k: int,
        min_score: float,
    ) -> None:
        self._index = index
        self._embedder = embedder
        self._top_k = top_k
        self._min_score = min_score

    def retrieve(
        self,
        question: str,
        boroughs: Sequence[str] = (),
        categories: Sequence[str] = (),
    ) -> list[RetrievedDocument]:
        """Up to ``top_k`` distinct documents scoring at least ``min_score``.

        When filters are given, documents tagged with a different borough or
        category are skipped so evidence matches the question's scope.
        """
        if self._index.document_count == 0:
            return []
        query = self._embedder.embed_query(question)
        scores = score_matrix(self._index.matrix, query)
        order = np.argsort(-scores, kind="stable")
        borough_set = set(boroughs)
        category_set = set(categories)

        selected: list[RetrievedDocument] = []
        selected_indices: list[int] = []
        seen_text: set[str] = set()
        for index in order:
            score = float(scores[index])
            if score < self._min_score:
                break
            document = self._index.documents[index]
            if borough_set and document.borough is not None and document.borough not in borough_set:
                continue
            if category_set and document.category is not None and document.category not in category_set:
                continue
            if self._is_duplicate(int(index), document.text, selected_indices, seen_text):
                continue
            selected.append(
                RetrievedDocument(
                    document.id, document.kind, document.text, document.borough, document.category, score
                )
            )
            selected_indices.append(int(index))
            seen_text.add(_text_key(document.text))
            if len(selected) >= self._top_k:
                break

        logger.debug(
            "retrieval complete",
            extra={"candidates": int((scores >= self._min_score).sum()), "selected": len(selected)},
        )
        return selected

    def _is_duplicate(self, index: int, text: str, selected_indices: list[int], seen_text: set[str]) -> bool:
        if _text_key(text) in seen_text:
            return True
        if not selected_indices:
            return False
        row = self._index.matrix[index]
        similarities = self._index.matrix[selected_indices] @ row
        return bool(np.any(similarities >= NEAR_DUPLICATE_THRESHOLD))
