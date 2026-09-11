"""Semantic retrieval over the synthetic knowledge index."""

from pathlib import Path

import numpy as np

from app.services.dataset import DocumentRecord, KnowledgeIndex, load_dataset
from app.services.embeddings import normalize_rows
from app.services.retrieval import RetrievalService
from tests.conftest import FAKE_MODEL_NAME, FakeEmbedder


def _index(embedder: FakeEmbedder, texts: list[tuple[str, str, str | None, str | None]]) -> KnowledgeIndex:
    matrix = embedder.embed_documents([t[1] for t in texts])
    docs = tuple(DocumentRecord(i, "aggregate", t, b, c) for i, t, b, c in texts)
    return KnowledgeIndex(documents=docs, matrix=normalize_rows(matrix))


def test_ranking_prefers_matching_borough_and_category(data_dir: Path, fake_embedder: FakeEmbedder) -> None:
    dataset = load_dataset(data_dir, FAKE_MODEL_NAME)
    service = RetrievalService(dataset.knowledge, fake_embedder, top_k=5, min_score=0.35)
    results = service.retrieve("How many grocery stores are in Iztapalapa?")
    assert results
    assert results[0].borough == "Iztapalapa"
    assert results[0].category == "grocery"
    assert results[0].score >= results[-1].score


def test_top_k_limits_results(fake_embedder: FakeEmbedder) -> None:
    texts = [(f"d{i}", f"Grocery stores in Coyoacán number {i}", "Coyoacán", "grocery") for i in range(6)]
    service = RetrievalService(_index(fake_embedder, texts), fake_embedder, top_k=2, min_score=0.0)
    assert len(service.retrieve("grocery Coyoacán")) <= 2


def test_min_score_filters_unrelated(fake_embedder: FakeEmbedder) -> None:
    texts = [("d1", "Pharmacies in Cuauhtémoc", "Cuauhtémoc", "pharmacy")]
    service = RetrievalService(_index(fake_embedder, texts), fake_embedder, top_k=5, min_score=0.35)
    assert service.retrieve("How do I bake a chocolate cake?") == []


def test_duplicate_documents_are_removed(fake_embedder: FakeEmbedder) -> None:
    texts = [
        ("d1", "Bakeries in Coyoacán: 3 establishments", "Coyoacán", "bakery"),
        ("d2", "Bakeries in Coyoacán: 3 establishments", "Coyoacán", "bakery"),
        ("d3", "Bakeries   in Coyoacán:  3 establishments.", "Coyoacán", "bakery"),
    ]
    service = RetrievalService(_index(fake_embedder, texts), fake_embedder, top_k=5, min_score=0.0)
    assert len(service.retrieve("bakeries Coyoacán")) == 1


def test_filters_restrict_scope(fake_embedder: FakeEmbedder) -> None:
    texts = [
        ("d1", "Grocery stores in Iztapalapa", "Iztapalapa", "grocery"),
        ("d2", "Grocery stores in Coyoacán", "Coyoacán", "grocery"),
        ("d3", "Grocery stores overall", None, "grocery"),
    ]
    service = RetrievalService(_index(fake_embedder, texts), fake_embedder, top_k=5, min_score=0.0)
    results = service.retrieve("grocery stores", boroughs=("Coyoacán",))
    assert {r.id for r in results} == {"d2", "d3"}


def test_empty_index_returns_nothing(fake_embedder: FakeEmbedder) -> None:
    index = KnowledgeIndex(documents=(), matrix=np.empty((0, 16), dtype=np.float32))
    service = RetrievalService(index, fake_embedder, top_k=5, min_score=0.0)
    assert service.retrieve("anything") == []


def test_metadata_preserved(fake_embedder: FakeEmbedder) -> None:
    texts = [("agg", "Pharmacies in Miguel Hidalgo: 4", "Miguel Hidalgo", "pharmacy")]
    service = RetrievalService(_index(fake_embedder, texts), fake_embedder, top_k=5, min_score=0.0)
    result = service.retrieve("pharmacies Miguel Hidalgo")[0]
    assert (result.id, result.kind, result.borough, result.category) == (
        "agg",
        "aggregate",
        "Miguel Hidalgo",
        "pharmacy",
    )
