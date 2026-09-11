"""Shared test doubles and a SYNTHETIC dataset builder.

Everything here is fabricated for tests: establishment names are
``TEST ESTABLISHMENT ...`` and never represent real DENUE records. No test
touches the network, downloads a model, or calls Gemini.
"""

import re
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.catalog import BOROUGHS, CATEGORIES, EMPLOYMENT_STRATA, ENTITY_CODE, ENTITY_NAME
from app.config import Settings
from app.main import create_app
from app.schemas import (
    SOURCE_NAME,
    SOURCE_URL,
    TRANSFORMATION_NOTICE,
    BoroughScope,
    CategoryScope,
    CountCheck,
    EstablishmentRecord,
    EstablishmentsFile,
    Evidence,
    GeographicScope,
    KnowledgeDocument,
    KnowledgeFile,
    KnowledgeMetadata,
    Manifest,
    ScopeFailure,
    attribution_text,
)
from app.services.dataset import (
    AGGREGATES_FILE,
    ESTABLISHMENTS_FILE,
    KNOWLEDGE_FILE,
    MANIFEST_FILE,
    SCHEMA_VERSION,
    checksum_bytes,
    compute_aggregates,
    serialize_model,
    write_bytes_atomic,
)
from app.services.embeddings import normalize_rows, normalize_vector
from app.services.generation import GenerationError, GenerationFailure
from app.services.question_parser import fold

FAKE_MODEL_NAME = "fake-embedding-model"
SYNTHETIC_RETRIEVED_AT = "2026-01-15T12:00:00+00:00"

# Synthetic counts per borough and category. Tests assert against these.
SYNTHETIC_COUNTS: dict[str, dict[str, int]] = {
    "Benito Juárez": {"grocery": 5, "pharmacy": 3, "bakery": 2, "cafe": 1},
    "Cuauhtémoc": {"grocery": 4, "pharmacy": 2, "restaurant": 3},
    "Miguel Hidalgo": {"grocery": 2, "pharmacy": 4, "restaurant": 1},
    "Coyoacán": {"grocery": 3, "bakery": 3},
    "Iztapalapa": {"grocery": 6, "butcher": 2},
}
SYNTHETIC_TOTAL = sum(sum(v.values()) for v in SYNTHETIC_COUNTS.values())  # 41

_TOKEN = re.compile(r"[a-z0-9]+")
_BOROUGH_TERMS = [(b.name, tuple(fold(a) for a in (b.name, *b.aliases))) for b in BOROUGHS]
_CATEGORY_TERMS = [(c.key, tuple(fold(a) for a in (c.label_en, c.label_es, *c.aliases))) for c in CATEGORIES]
FAKE_DIMENSION = len(BOROUGHS) + len(CATEGORIES) + 1


class FakeEmbedder:
    """Deterministic embeddings: one dimension per borough and per category.

    Text mentioning no known borough or category maps to a final "unrelated"
    axis, so unrelated questions have zero similarity with every document.
    """

    provider_name = "fake"

    def __init__(self, model_name: str = FAKE_MODEL_NAME) -> None:
        self._model_name = model_name
        self.query_calls: list[str] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    def _vector(self, text: str) -> np.ndarray:
        folded = fold(text)
        values: list[float] = []
        for _, aliases in _BOROUGH_TERMS:
            values.append(
                float(any(re.search(rf"(?<![a-z0-9]){re.escape(a)}(?![a-z0-9])", folded) for a in aliases))
            )
        for _, aliases in _CATEGORY_TERMS:
            values.append(
                float(any(re.search(rf"(?<![a-z0-9]){re.escape(a)}(?![a-z0-9])", folded) for a in aliases))
            )
        values.append(0.0 if any(values) else 1.0)
        return np.array(values, dtype=np.float32)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, FAKE_DIMENSION), dtype=np.float32)
        return normalize_rows(np.vstack([self._vector(text) for text in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        self.query_calls.append(text)
        return normalize_vector(self._vector(text))


class FakeGenerator:
    """Stand-in for Gemini: returns a fixed explanation or raises a failure."""

    provider_name = "gemini"

    def __init__(
        self,
        explanation: str | None = None,
        failure: GenerationFailure | None = None,
        configured: bool = True,
    ) -> None:
        self.explanation = explanation
        self.failure = failure
        self.configured = configured
        self.calls: list[tuple[str, dict[str, Any], list[Evidence]]] = []

    @property
    def is_configured(self) -> bool:
        return self.configured

    async def explain(self, question: str, payload: dict[str, Any], evidence: list[Evidence]) -> str:
        self.calls.append((question, payload, evidence))
        if self.failure is not None:
            raise GenerationError(self.failure)
        if self.explanation is not None:
            return self.explanation
        # Echo the deterministic answer so every number is consistent.
        return "Explained: " + str(payload["deterministic_answer"])


# --------------------------------------------------------------------------- #
# Synthetic dataset
# --------------------------------------------------------------------------- #


def synthetic_records() -> list[EstablishmentRecord]:
    records: list[EstablishmentRecord] = []
    serial = 0
    for borough in BOROUGHS:
        for category in CATEGORIES:
            count = SYNTHETIC_COUNTS.get(borough.name, {}).get(category.key, 0)
            for index in range(count):
                serial += 1
                stratum = "1" if index % 3 != 2 else "2"
                scian_class = category.scian_classes[index % len(category.scian_classes)]
                records.append(
                    EstablishmentRecord(
                        id=f"T{serial:04d}",
                        name=f"TEST ESTABLISHMENT {borough.code}-{category.key.upper()}-{index + 1}",
                        category=category.key,
                        scian_class=scian_class,
                        activity_label=category.official_labels[index % len(category.scian_classes)],
                        stratum=stratum,
                        stratum_label=EMPLOYMENT_STRATA[stratum],
                        state=ENTITY_NAME,
                        borough_code=borough.code,
                        borough=borough.name,
                        locality="TEST LOCALITY",
                        latitude=19.4,
                        longitude=-99.1,
                        establishment_type="Fijo",
                        source_date="2025-11",
                    )
                )
    return records


def synthetic_documents(
    records: list[EstablishmentRecord], embedder: FakeEmbedder
) -> list[KnowledgeDocument]:
    aggregates = compute_aggregates(records)
    texts: list[tuple[str, str, str, str | None, str | None]] = []
    for borough, per_category in aggregates.by_borough_category.items():
        for key, count in per_category.items():
            label = next(c.label_en for c in CATEGORIES if c.key == key).lower()
            texts.append(
                (
                    f"aggregate:{borough}:{key}",
                    "aggregate",
                    f"In the synthetic dataset, {borough} contains {count} {label} establishments. "
                    "Source: synthetic test data. Calculation performed by this application.",
                    borough,
                    key,
                )
            )
    first = records[0]
    texts.append(
        (
            f"sample:{first.id}",
            "establishment_sample",
            f"{first.name} is classified as {first.activity_label} in {first.borough}. Grocery sample.",
            first.borough,
            first.category,
        )
    )
    matrix = embedder.embed_documents([t[2] for t in texts])
    return [
        KnowledgeDocument(
            id=i, kind=k, text=t, borough=b, category=c, embedding=[float(v) for v in matrix[n]]
        )
        for n, (i, k, t, b, c) in enumerate(texts)
    ]


def build_synthetic_dataset(
    data_dir: Path,
    complete: bool = True,
    records: list[EstablishmentRecord] | None = None,
    embedder: FakeEmbedder | None = None,
) -> Path:
    """Write a fully consistent synthetic dataset into *data_dir*."""
    embedder = embedder or FakeEmbedder()
    records = synthetic_records() if records is None else records
    aggregates = compute_aggregates(records)
    documents = synthetic_documents(records, embedder)
    establishments = EstablishmentsFile(schema_version=SCHEMA_VERSION, establishments=records)
    establishments_bytes = serialize_model(establishments, compact=True)
    knowledge = KnowledgeFile(
        metadata=KnowledgeMetadata(
            created_at=SYNTHETIC_RETRIEVED_AT,
            embedding_provider=embedder.provider_name,
            embedding_model=embedder.model_name,
            embedding_dimension=FAKE_DIMENSION,
            document_count=len(documents),
        ),
        documents=documents,
    )
    failed = [] if complete else [ScopeFailure(borough="Iztapalapa", scian_class="464111", reason="timeout")]
    manifest = Manifest(
        schema_version=SCHEMA_VERSION,
        source_name=SOURCE_NAME,
        source_url=SOURCE_URL,
        retrieved_at=SYNTHETIC_RETRIEVED_AT,
        dataset_update_date="2025-11",
        geographic_scope=GeographicScope(
            entity_code=ENTITY_CODE,
            entity_name=ENTITY_NAME,
            boroughs=[BoroughScope(code=b.code, name=b.name) for b in BOROUGHS],
        ),
        economic_scope=[
            CategoryScope(
                key=c.key, label_en=c.label_en, label_es=c.label_es, scian_classes=list(c.scian_classes)
            )
            for c in CATEGORIES
        ],
        record_count=len(records),
        category_count=len(aggregates.by_category),
        borough_count=len(aggregates.by_borough),
        knowledge_document_count=len(documents),
        embedding_provider=embedder.provider_name,
        embedding_model=embedder.model_name,
        embedding_dimension=FAKE_DIMENSION,
        data_checksum=checksum_bytes(establishments_bytes),
        complete=complete,
        completeness_note="Synthetic test dataset."
        if complete
        else "Partial dataset: 1 scope(s) failed to download.",
        failed_scopes=failed,
        count_checks=[CountCheck(borough="Benito Juárez", scian_class="461110", expected=5, retrieved=5)],
        transformation_notice=TRANSFORMATION_NOTICE,
        attribution=attribution_text(SYNTHETIC_RETRIEVED_AT[:10]),
    )
    write_bytes_atomic(data_dir / ESTABLISHMENTS_FILE, establishments_bytes)
    write_bytes_atomic(data_dir / AGGREGATES_FILE, serialize_model(aggregates))
    write_bytes_atomic(data_dir / KNOWLEDGE_FILE, serialize_model(knowledge))
    write_bytes_atomic(data_dir / MANIFEST_FILE, serialize_model(manifest))
    return data_dir


def make_settings(data_dir: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "data_dir": data_dir,
        "embedding_model": FAKE_MODEL_NAME,
        "inegi_api_token": None,
        "gemini_api_key": None,
        "gemini_model": "",
        "allowed_origins": "http://testserver",
        "rate_limit_requests": 1000,
        "rate_limit_window_seconds": 60,
        "retrieval_top_k": 5,
        "retrieval_min_score": 0.35,
        "log_level": "WARNING",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def data_dir(tmp_path: Path, fake_embedder: FakeEmbedder) -> Path:
    return build_synthetic_dataset(tmp_path / "data", embedder=fake_embedder)


@pytest.fixture
def make_client(fake_embedder: FakeEmbedder, data_dir: Path) -> Iterator[Any]:
    """Factory producing a started TestClient with injected fakes."""
    clients: list[TestClient] = []

    def factory(
        settings: Settings | None = None,
        generator: FakeGenerator | None = None,
    ) -> TestClient:
        app = create_app(
            settings=settings or make_settings(data_dir),
            embedder=fake_embedder,
            generator=generator or FakeGenerator(configured=False),
        )
        client = TestClient(app, raise_server_exceptions=False)
        client.__enter__()
        clients.append(client)
        return client

    yield factory
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def client(make_client: Any) -> TestClient:
    return make_client()


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
