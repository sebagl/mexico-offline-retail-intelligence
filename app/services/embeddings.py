"""Local text embeddings.

The application depends on the small ``EmbeddingProvider`` protocol rather
than on FastEmbed directly, so the model backend can be swapped (or faked in
tests) without touching retrieval or ingestion code.
"""

import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)

EMBEDDING_PROVIDER_NAME = "fastembed"


class EmbeddingProvider(Protocol):
    @property
    def provider_name(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, dim) float32 matrix of L2-normalized vectors."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Return a single L2-normalized float32 vector."""
        ...


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize each row of a 2-D array; zero rows stay zero."""
    if matrix.ndim != 2:
        raise ValueError("expected a 2-D matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe = np.where(norms == 0.0, 1.0, norms)
    return (matrix / safe).astype(np.float32, copy=False)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """L2-normalize a 1-D vector; a zero vector is returned unchanged."""
    if vector.ndim != 1:
        raise ValueError("expected a 1-D vector")
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector.astype(np.float32, copy=False)
    return (vector / norm).astype(np.float32, copy=False)


def resolve_cache_dir(configured: Path | None) -> Path | None:
    """Pick a writable model cache directory.

    Inside the container ``FASTEMBED_CACHE_PATH`` points at ``/app/.cache``.
    When the same ``.env`` is used on a developer machine that path may not be
    creatable, so fall back to a per-user cache instead of failing.
    """
    candidates: list[Path] = []
    if configured is not None:
        candidates.append(configured)
    candidates.append(Path.home() / ".cache" / "mexico-offline-retail-intelligence" / "fastembed")

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            if os.access(candidate, os.W_OK):
                if configured is not None and candidate != configured:
                    logger.warning(
                        "fastembed cache path not writable; using fallback",
                        extra={"fallback": str(candidate)},
                    )
                return candidate
        except OSError:
            continue
    logger.warning("no writable fastembed cache directory found; using library default")
    return None


class FastEmbedProvider:
    """CPU-only ONNX embeddings via FastEmbed. The model is loaded once, lazily."""

    def __init__(self, model_name: str, cache_dir: Path | None = None) -> None:
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._model = None
        self._dimension: int | None = None

    @property
    def provider_name(self) -> str:
        return EMBEDDING_PROVIDER_NAME

    @property
    def model_name(self) -> str:
        return self._model_name

    def _get_model(self):
        if self._model is None:
            from fastembed import TextEmbedding

            logger.info("loading embedding model", extra={"model": self._model_name})
            self._model = TextEmbedding(
                model_name=self._model_name,
                cache_dir=str(self._cache_dir) if self._cache_dir else None,
            )
        return self._model

    def _record_dimension(self, vectors: np.ndarray) -> None:
        dim = int(vectors.shape[-1])
        if self._dimension is None:
            self._dimension = dim
        elif self._dimension != dim:
            raise ValueError(f"embedding dimension changed from {self._dimension} to {dim}")

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dimension or 0), dtype=np.float32)
        model = self._get_model()
        rows = [np.asarray(vector, dtype=np.float32) for vector in model.embed(list(texts))]
        matrix = np.vstack(rows)
        self._record_dimension(matrix)
        return normalize_rows(matrix)

    def embed_query(self, text: str) -> np.ndarray:
        model = self._get_model()
        vector = np.asarray(next(iter(model.query_embed(text))), dtype=np.float32)
        self._record_dimension(vector)
        return normalize_vector(vector)
