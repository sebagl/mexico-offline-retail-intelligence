"""Load, validate and atomically write the generated DENUE data artifacts."""

import hashlib
import json
import logging
import os
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ValidationError

from app.catalog import BOROUGHS_BY_NAME, CATEGORIES_BY_KEY, EMPLOYMENT_STRATA
from app.exceptions import DatasetError
from app.schemas import (
    Aggregates,
    EstablishmentRecord,
    EstablishmentsFile,
    KnowledgeFile,
    KnowledgeKind,
    Manifest,
)
from app.services.embeddings import normalize_rows

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
MANIFEST_FILE = "manifest.json"
ESTABLISHMENTS_FILE = "denue_establishments.json"
AGGREGATES_FILE = "denue_aggregates.json"
KNOWLEDGE_FILE = "knowledge.json"

# Fields that must never appear in a stored record (data minimization).
FORBIDDEN_FIELDS = frozenset(
    {
        "telefono",
        "correo_e",
        "sitio_internet",
        "razon_social",
        "calle",
        "num_exterior",
        "num_interior",
        "tipo_vialidad",
        "cp",
        "ubicacion",
    }
)


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    """A knowledge document without its vector; vectors live in ``KnowledgeIndex.matrix``."""

    id: str
    kind: KnowledgeKind
    text: str
    borough: str | None
    category: str | None


@dataclass(frozen=True, slots=True)
class Establishment:
    """Runtime row for one minimized establishment (same fields as ``EstablishmentRecord``)."""

    id: str
    name: str
    category: str
    scian_class: str
    activity_label: str
    stratum: str
    stratum_label: str
    state: str
    borough_code: str
    borough: str
    locality: str
    latitude: float | None
    longitude: float | None
    establishment_type: str
    source_date: str | None


@dataclass(frozen=True, slots=True)
class KnowledgeIndex:
    """Semantic documents with their L2-normalized embedding matrix (one copy)."""

    documents: tuple[DocumentRecord, ...]
    matrix: np.ndarray

    @property
    def document_count(self) -> int:
        return len(self.documents)


@dataclass(frozen=True, slots=True)
class Dataset:
    manifest: Manifest
    establishments: tuple[Establishment, ...]
    aggregates: Aggregates
    knowledge: KnowledgeIndex

    @property
    def record_count(self) -> int:
        return len(self.establishments)


# --------------------------------------------------------------------------- #
# Aggregation (used by ingestion to build, and by the loader to verify)
# --------------------------------------------------------------------------- #


def compute_aggregates(records: Sequence[EstablishmentRecord | Establishment]) -> Aggregates:
    """Exact counts by borough, category, stratum and their pairwise combinations."""
    by_borough: Counter[str] = Counter()
    by_category: Counter[str] = Counter()
    by_stratum: Counter[str] = Counter()
    by_borough_category: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_borough_stratum: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_category_stratum: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        by_borough[record.borough] += 1
        by_category[record.category] += 1
        by_stratum[record.stratum] += 1
        by_borough_category[record.borough][record.category] += 1
        by_borough_stratum[record.borough][record.stratum] += 1
        by_category_stratum[record.category][record.stratum] += 1

    def sorted_counter(counter: Counter[str]) -> dict[str, int]:
        return dict(sorted(counter.items()))

    return Aggregates(
        schema_version=SCHEMA_VERSION,
        total=len(records),
        by_borough=sorted_counter(by_borough),
        by_category=sorted_counter(by_category),
        by_stratum=sorted_counter(by_stratum),
        by_borough_category={k: sorted_counter(v) for k, v in sorted(by_borough_category.items())},
        by_borough_stratum={k: sorted_counter(v) for k, v in sorted(by_borough_stratum.items())},
        by_category_stratum={k: sorted_counter(v) for k, v in sorted(by_category_stratum.items())},
    )


def checksum_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _read_bytes(path: Path) -> bytes:
    if not path.is_file():
        raise DatasetError(f"missing data artifact: {path.name}")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise DatasetError(f"{path.name} is not readable: {exc.__class__.__name__}") from exc


def _parse[T: BaseModel](model: type[T], payload: bytes, name: str) -> T:
    """Validate JSON bytes directly in pydantic-core.

    Parsing straight from bytes avoids materialising the whole file as Python
    dicts first, which for ~50k records keeps peak memory well below the
    512 MB available on the free hosting tier.
    """
    try:
        return model.model_validate_json(payload)
    except ValidationError as exc:
        first = exc.errors()[0]
        if first.get("type") == "json_invalid":
            raise DatasetError(f"{name} is not readable JSON") from exc
        location = ".".join(str(part) for part in first.get("loc", ()))
        raise DatasetError(f"{name} schema error at '{location}': {first.get('msg')}") from exc


def _to_rows(records: list[EstablishmentRecord]) -> tuple[Establishment, ...]:
    """Convert validated models to compact slotted rows.

    Categorical strings (borough, labels, strata, dates) repeat across tens of
    thousands of records; interning them keeps one object per distinct value.
    """
    intern = sys.intern
    return tuple(
        Establishment(
            r.id,
            r.name,
            intern(r.category),
            intern(r.scian_class),
            intern(r.activity_label),
            intern(r.stratum),
            intern(r.stratum_label),
            intern(r.state),
            intern(r.borough_code),
            intern(r.borough),
            intern(r.locality),
            r.latitude,
            r.longitude,
            intern(r.establishment_type),
            intern(r.source_date) if r.source_date is not None else None,
        )
        for r in records
    )


def _validate_records(records: tuple[Establishment, ...]) -> None:
    """Catalog-consistency checks. Disallowed contact fields are already
    rejected by ``EstablishmentRecord`` (``extra="forbid"``)."""
    seen: set[str] = set()
    for record in records:
        if record.id in seen:
            raise DatasetError(f"duplicate establishment id: {record.id}")
        seen.add(record.id)
        if record.borough not in BOROUGHS_BY_NAME:
            raise DatasetError(f"record '{record.id}' has an unconfigured borough: {record.borough}")
        if BOROUGHS_BY_NAME[record.borough].code != record.borough_code:
            raise DatasetError(f"record '{record.id}' borough code does not match its borough name")
        category = CATEGORIES_BY_KEY.get(record.category)
        if category is None:
            raise DatasetError(f"record '{record.id}' has an unconfigured category: {record.category}")
        if record.scian_class not in category.scian_classes:
            raise DatasetError(
                f"record '{record.id}' SCIAN class is not part of category '{record.category}'"
            )
        if EMPLOYMENT_STRATA.get(record.stratum) != record.stratum_label:
            raise DatasetError(f"record '{record.id}' stratum label does not match its stratum code")


def _build_knowledge(data: KnowledgeFile, expected_model: str) -> KnowledgeIndex:
    metadata = data.metadata
    if metadata.embedding_model != expected_model:
        raise DatasetError(
            f"knowledge.json was built with '{metadata.embedding_model}' "
            f"but runtime EMBEDDING_MODEL is '{expected_model}'"
        )
    if metadata.document_count != len(data.documents):
        raise DatasetError("knowledge metadata document_count does not match the documents")
    if not data.documents:
        raise DatasetError("knowledge.json contains no documents")
    dim = metadata.embedding_dimension
    ids: set[str] = set()
    for document in data.documents:
        if document.id in ids:
            raise DatasetError(f"duplicate knowledge document id: {document.id}")
        ids.add(document.id)
        if len(document.embedding) != dim:
            raise DatasetError(
                f"document '{document.id}' has dimension {len(document.embedding)}, expected {dim}"
            )
        if document.borough is not None and document.borough not in BOROUGHS_BY_NAME:
            raise DatasetError(f"document '{document.id}' references an unconfigured borough")
        if document.category is not None and document.category not in CATEGORIES_BY_KEY:
            raise DatasetError(f"document '{document.id}' references an unconfigured category")
    matrix = np.asarray([document.embedding for document in data.documents], dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise DatasetError("knowledge.json contains NaN or infinite embedding values")
    if np.any(np.linalg.norm(matrix, axis=1) == 0.0):
        raise DatasetError("knowledge.json contains a zero embedding vector")
    documents = tuple(DocumentRecord(d.id, d.kind, d.text, d.borough, d.category) for d in data.documents)
    return KnowledgeIndex(documents=documents, matrix=normalize_rows(matrix))


def load_dataset(data_dir: Path, expected_model: str) -> Dataset:
    """Load and cross-validate all four artifacts or raise ``DatasetError``."""
    manifest = _parse(Manifest, _read_bytes(data_dir / MANIFEST_FILE), MANIFEST_FILE)

    establishments_bytes = _read_bytes(data_dir / ESTABLISHMENTS_FILE)
    if checksum_bytes(establishments_bytes) != manifest.data_checksum:
        raise DatasetError("establishments checksum does not match the manifest")
    establishments_file = _parse(EstablishmentsFile, establishments_bytes, ESTABLISHMENTS_FILE)
    del establishments_bytes
    records = _to_rows(establishments_file.establishments)
    del establishments_file  # the validated models are no longer needed
    _validate_records(records)
    if manifest.record_count != len(records):
        raise DatasetError("manifest.record_count does not match the establishments file")

    aggregates = _parse(Aggregates, _read_bytes(data_dir / AGGREGATES_FILE), AGGREGATES_FILE)
    if aggregates != compute_aggregates(records):
        raise DatasetError("aggregates do not match a recomputation from the establishment records")

    knowledge = _build_knowledge(
        _parse(KnowledgeFile, _read_bytes(data_dir / KNOWLEDGE_FILE), KNOWLEDGE_FILE), expected_model
    )
    if manifest.embedding_model != expected_model:
        raise DatasetError("manifest embedding model does not match the runtime EMBEDDING_MODEL")
    if manifest.embedding_dimension != knowledge.matrix.shape[1]:
        raise DatasetError("manifest embedding dimension does not match knowledge.json")
    if manifest.knowledge_document_count != knowledge.document_count:
        raise DatasetError("manifest knowledge_document_count does not match knowledge.json")

    configured_boroughs = {scope.name for scope in manifest.geographic_scope.boroughs}
    if configured_boroughs - set(BOROUGHS_BY_NAME):
        raise DatasetError("manifest lists a borough that is not configured in the catalog")
    configured_categories = {scope.key for scope in manifest.economic_scope}
    if configured_categories - set(CATEGORIES_BY_KEY):
        raise DatasetError("manifest lists a category that is not configured in the catalog")

    logger.info(
        "dataset loaded",
        extra={
            "establishments": len(records),
            "boroughs": manifest.borough_count,
            "categories": manifest.category_count,
            "knowledge_documents": knowledge.document_count,
            "complete": manifest.complete,
            "retrieved_at": manifest.retrieved_at,
        },
    )
    return Dataset(
        manifest=manifest, establishments=tuple(records), aggregates=aggregates, knowledge=knowledge
    )


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #


def serialize_model(model: BaseModel, compact: bool = False) -> bytes:
    """Deterministic JSON bytes; ``compact`` keeps large record files small."""
    payload = model.model_dump(mode="json")
    if compact:
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    else:
        text = json.dumps(payload, ensure_ascii=False, indent=1)
    return (text + "\n").encode("utf-8")


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    """Write to a temp file in the same directory, then atomically replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_name, 0o644)  # mkstemp creates 0600; artifacts are public data
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
