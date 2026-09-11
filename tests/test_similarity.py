"""Cosine similarity and matrix scoring edge cases."""

import numpy as np
import pytest

from app.services.embeddings import normalize_rows, normalize_vector
from app.services.retrieval import cosine_similarity, score_matrix


def test_identical_vectors() -> None:
    assert cosine_similarity(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])) == pytest.approx(1.0)


def test_orthogonal_vectors() -> None:
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)


def test_opposite_vectors() -> None:
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([-1.0, 0.0])) == pytest.approx(-1.0)


def test_zero_vector_returns_zero() -> None:
    assert cosine_similarity(np.zeros(3), np.array([1.0, 2.0, 3.0])) == 0.0


def test_dimension_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="dimension mismatch"):
        cosine_similarity(np.array([1.0, 0.0]), np.array([1.0, 0.0, 0.0]))


def test_nan_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        cosine_similarity(np.array([np.nan, 1.0]), np.array([1.0, 1.0]))


def test_infinity_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        cosine_similarity(np.array([np.inf, 1.0]), np.array([1.0, 1.0]))


def test_result_is_clipped_to_unit_range() -> None:
    value = cosine_similarity(
        np.array([0.1, 0.1, 0.1], dtype=np.float32), np.array([0.1, 0.1, 0.1], dtype=np.float32)
    )
    assert -1.0 <= value <= 1.0


def test_score_matrix_ranks_rows() -> None:
    matrix = normalize_rows(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    scores = score_matrix(matrix, np.array([1.0, 0.0]))
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.0)
    assert scores[2] == pytest.approx(0.7071, abs=1e-3)


def test_score_matrix_handles_empty_matrix() -> None:
    assert score_matrix(np.empty((0, 3), dtype=np.float32), np.array([1.0, 0.0, 0.0])).shape == (0,)


def test_score_matrix_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="dimension mismatch"):
        score_matrix(np.ones((2, 3), dtype=np.float32), np.ones(2))


def test_normalize_vector_zero_stays_zero() -> None:
    assert np.array_equal(normalize_vector(np.zeros(4)), np.zeros(4, dtype=np.float32))
