"""Pydantic models for the HTTP API and the generated data artifacts."""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ResponseMode = Literal["deterministic", "generated", "extractive", "insufficient_data", "unsupported"]
HealthStatus = Literal["ok", "degraded", "unavailable"]
DatasetBasis = Literal["complete_configured_dataset", "filtered_subset", "partial_dataset"]
KnowledgeKind = Literal["aggregate", "establishment_sample"]

SOURCE_NAME = "INEGI, Directorio Estadístico Nacional de Unidades Económicas (DENUE)"
SOURCE_SHORT_NAME = "INEGI DENUE"
SOURCE_URL = "https://www.inegi.org.mx/servicios/api_denue.html"
TRANSFORMATION_NOTICE = (
    "This independent demonstration transforms and analyzes public INEGI data. "
    "The analysis was not produced, reviewed, sponsored, or endorsed by INEGI."
)

_WHITESPACE = re.compile(r"\s+")


def normalize_question(text: str) -> str:
    """Collapse internal whitespace and trim the ends."""
    return _WHITESPACE.sub(" ", text).strip()


def attribution_text(date: str) -> str:
    return f"Fuente: {SOURCE_NAME}, {date}."


# --------------------------------------------------------------------------- #
# Generated data artifacts (shared by ingestion and runtime loading)
# --------------------------------------------------------------------------- #


class EstablishmentRecord(BaseModel):
    """Minimized DENUE establishment. Contact and street-address fields are
    never present; ``extra="forbid"`` makes their presence a load error."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    category: str = Field(..., min_length=1)
    scian_class: str = Field(..., pattern=r"^\d{6}$")
    activity_label: str = Field(..., min_length=1)
    stratum: str = Field(..., pattern=r"^[1-7]$")
    stratum_label: str = Field(..., min_length=1)
    state: str = Field(..., min_length=1)
    borough_code: str = Field(..., pattern=r"^\d{3}$")
    borough: str = Field(..., min_length=1)
    locality: str
    establishment_type: str
    source_date: str | None = None


class EstablishmentsFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(..., ge=1)
    establishments: list[EstablishmentRecord]


class Aggregates(BaseModel):
    """Exact counts computed in Python during ingestion."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(..., ge=1)
    total: int = Field(..., ge=0)
    by_borough: dict[str, int]
    by_category: dict[str, int]
    by_stratum: dict[str, int]
    by_borough_category: dict[str, dict[str, int]]
    by_borough_stratum: dict[str, dict[str, int]]
    by_category_stratum: dict[str, dict[str, int]]


class KnowledgeDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)
    kind: KnowledgeKind
    text: str = Field(..., min_length=1)
    borough: str | None = None
    category: str | None = None
    embedding: list[float] = Field(..., min_length=1)


class KnowledgeMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    created_at: str
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int = Field(..., ge=1)
    document_count: int = Field(..., ge=0)


class KnowledgeFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metadata: KnowledgeMetadata
    documents: list[KnowledgeDocument]


class BoroughScope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    name: str


class CategoryScope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    label_en: str
    label_es: str
    scian_classes: list[str]


class GeographicScope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_code: str
    entity_name: str
    boroughs: list[BoroughScope]


class ScopeFailure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    borough: str
    scian_class: str
    reason: str


class CountCheck(BaseModel):
    """Expected count from DENUE ``Cuantificar`` versus records retrieved."""

    model_config = ConfigDict(extra="forbid")

    borough: str
    scian_class: str
    expected: int | None
    retrieved: int


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(..., ge=1)
    source_name: str
    source_url: str
    retrieved_at: str
    dataset_update_date: str | None = None
    geographic_scope: GeographicScope
    economic_scope: list[CategoryScope]
    record_count: int = Field(..., ge=0)
    category_count: int = Field(..., ge=0)
    borough_count: int = Field(..., ge=0)
    knowledge_document_count: int = Field(..., ge=0)
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int = Field(..., ge=1)
    data_checksum: str = Field(..., pattern=r"^sha256:[0-9a-f]{64}$")
    complete: bool
    completeness_note: str
    failed_scopes: list[ScopeFailure]
    count_checks: list[CountCheck]
    transformation_notice: str
    attribution: str


# --------------------------------------------------------------------------- #
# API request / response models
# --------------------------------------------------------------------------- #


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., description="Question about the configured DENUE retail dataset.")

    @field_validator("question")
    @classmethod
    def _must_be_meaningful(cls, value: str) -> str:
        if not normalize_question(value):
            raise ValueError("question must not be empty")
        return value


class AppliedFilters(BaseModel):
    boroughs: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    strata: list[str] = Field(default_factory=list)


class Evidence(BaseModel):
    kind: KnowledgeKind
    text: str
    score: float = Field(..., ge=-1.0, le=1.0)
    borough: str | None = None
    category: str | None = None


class SourceInfo(BaseModel):
    name: str = SOURCE_SHORT_NAME
    url: str = SOURCE_URL


class ScopeInfo(BaseModel):
    boroughs: list[str]
    categories: list[str]
    complete: bool
    basis: DatasetBasis
    retrieved_at: str


class QueryResponse(BaseModel):
    answer: str
    mode: ResponseMode
    intent: str
    metrics: dict[str, object]
    filters: AppliedFilters
    evidence: list[Evidence]
    source: SourceInfo
    scope: ScopeInfo
    methodology: str
    attribution: str


class HealthResponse(BaseModel):
    status: HealthStatus
    dataset_loaded: bool
    establishments: int
    boroughs: int
    categories: int
    dataset_complete: bool
    embedding_provider: str
    embedding_model: str
    generation_provider: str
    generation_configured: bool
    fallback_available: bool


class SourceResponse(BaseModel):
    source_name: str
    source_url: str
    attribution: str
    retrieved_at: str
    dataset_update_date: str | None
    geographic_scope: GeographicScope
    economic_scope: list[CategoryScope]
    complete: bool
    completeness_note: str
    failed_scopes: list[ScopeFailure]
    transformation_notice: str


class CategorySummary(BaseModel):
    key: str
    label_en: str
    label_es: str
    count: int


class BoroughSummary(BaseModel):
    name: str
    count: int


class SummaryResponse(BaseModel):
    total_establishments: int
    boroughs: list[BoroughSummary]
    categories: list[CategorySummary]
    strata: dict[str, int]
    retrieved_at: str
    complete: bool
    attribution: str
    transformation_notice: str


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody
