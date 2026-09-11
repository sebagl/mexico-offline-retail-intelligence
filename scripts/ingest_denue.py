"""Build the DENUE data artifacts for the configured Mexico City scope.

Usage::

    python -m scripts.ingest_denue [--data-dir data]

Requires ``INEGI_API_TOKEN`` (register at
https://www.inegi.org.mx/servicios/api_denue.html). Only the documented DENUE
API methods ``Cuantificar`` and ``BuscarAreaAct`` are called, one configured
borough and SCIAN class at a time, with paging until a page comes back short.
Every scope is checked against the ``Cuantificar`` total; any failure or
mismatch marks the manifest as partial. All four artifacts are generated and
re-validated in a staging directory before replacing the previous dataset.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import shutil
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import __version__  # noqa: E402
from app.catalog import (  # noqa: E402
    BOROUGHS,
    CATEGORIES,
    CATEGORY_BY_SCIAN_CLASS,
    EMPLOYMENT_STRATA,
    ENTITY_CODE,
    ENTITY_NAME,
    Borough,
)
from app.config import get_settings  # noqa: E402
from app.logging_config import configure_logging  # noqa: E402
from app.schemas import (  # noqa: E402
    SOURCE_NAME,
    SOURCE_URL,
    TRANSFORMATION_NOTICE,
    Aggregates,
    BoroughScope,
    CategoryScope,
    CountCheck,
    EstablishmentRecord,
    EstablishmentsFile,
    GeographicScope,
    KnowledgeDocument,
    KnowledgeFile,
    KnowledgeMetadata,
    Manifest,
    ScopeFailure,
    attribution_text,
)
from app.services import analytics  # noqa: E402
from app.services.dataset import (  # noqa: E402
    AGGREGATES_FILE,
    ESTABLISHMENTS_FILE,
    KNOWLEDGE_FILE,
    MANIFEST_FILE,
    SCHEMA_VERSION,
    checksum_bytes,
    compute_aggregates,
    load_dataset,
    serialize_model,
    write_bytes_atomic,
)
from app.services.embeddings import FastEmbedProvider, resolve_cache_dir  # noqa: E402

logger = logging.getLogger("ingest_denue")

API_BASE = "https://www.inegi.org.mx/app/api/denue/v1/consulta"
USER_AGENT = f"MexicoOfflineRetailIntelligence/{__version__} (independent demo; documented DENUE API only)"
CONNECT_TIMEOUT_SECONDS = 10.0
READ_TIMEOUT_SECONDS = 60.0
PAGE_SIZE = 1000
REQUEST_DELAY_SECONDS = 0.3
MAX_RETRIES = 3
SAMPLES_PER_SCOPE = 10  # establishment descriptions embedded per borough x category
# Loose bounding box for Ciudad de México; coordinates outside are dropped.
LAT_RANGE = (19.0, 19.7)
LON_RANGE = (-99.4, -98.9)


@dataclass
class ScopeResult:
    borough: Borough
    scian_class: str
    expected: int | None = None
    records: list[dict[str, Any]] = field(default_factory=list)
    failure: str | None = None
    label_mismatches: Counter[str] = field(default_factory=Counter)
    borough_mismatches: int = 0  # records whose Ubicacion does not name the requested borough


class SetupError(Exception):
    """Raised when the ingestion environment is not usable."""


# --------------------------------------------------------------------------- #
# DENUE API client (documented methods only)
# --------------------------------------------------------------------------- #


class DenueClient:
    def __init__(self, token: str) -> None:
        self._token = token
        timeout = httpx.Timeout(
            connect=CONNECT_TIMEOUT_SECONDS, read=READ_TIMEOUT_SECONDS, write=10.0, pool=10.0
        )
        self._client = httpx.AsyncClient(
            base_url=API_BASE,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            timeout=timeout,
            follow_redirects=False,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _get(self, path: str) -> Any:
        """GET ``{API_BASE}/{path}/{token}`` with retries. The token is never logged."""
        last_error = "unknown"
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await self._client.get(f"/{path}/{self._token}")
            except httpx.TimeoutException:
                last_error = "timeout"
            except httpx.HTTPError as exc:
                last_error = f"network_error:{exc.__class__.__name__}"
            else:
                if response.status_code == 200:
                    return _parse_body(response)
                if response.status_code in (401, 403):
                    raise SetupError(f"DENUE rejected the token (HTTP {response.status_code})")
                last_error = f"http_{response.status_code}"
                if 400 <= response.status_code < 500 and response.status_code != 429:
                    break
            await asyncio.sleep(min(2.0**attempt, 8.0))
        raise RuntimeError(last_error)

    async def cuantificar(self, scian_class: str, geo_code: str) -> int | None:
        body = await self._get(f"Cuantificar/{scian_class}/{geo_code}/0")
        for item in body if isinstance(body, list) else []:
            if isinstance(item, dict) and "Total" in item:
                try:
                    return int(str(item["Total"]).replace(",", ""))
                except ValueError:
                    return None
        return 0 if body == [] else None

    async def buscar_area_act_page(
        self, borough: Borough, scian_class: str, start: int, end: int
    ) -> list[dict[str, Any]]:
        path = f"BuscarAreaAct/{ENTITY_CODE}/{borough.code}/0/0/0/0/0/0/{scian_class}/0/{start}/{end}/0"
        body = await self._get(path)
        if not isinstance(body, list):
            return []
        return [item for item in body if isinstance(item, dict)]


NO_RESULTS_SENTINEL = "no hay resultados"


def _parse_body(response: httpx.Response) -> Any:
    """Return the JSON payload, mapping DENUE's empty-scope sentinel to ``[]``.

    Verified against the live API: a scope with no establishments answers
    HTTP 200 with the JSON string ``"No hay resultados. "`` instead of a list.
    """
    try:
        body = response.json()
    except ValueError:
        if NO_RESULTS_SENTINEL in response.text.lower():
            return []
        raise RuntimeError("non_json_response") from None
    if isinstance(body, str):
        if NO_RESULTS_SENTINEL in body.lower():
            return []
        raise RuntimeError("unexpected_string_response")
    return body


# --------------------------------------------------------------------------- #
# Fetching every configured scope
# --------------------------------------------------------------------------- #


async def fetch_scope(client: DenueClient, borough: Borough, scian_class: str) -> ScopeResult:
    result = ScopeResult(borough=borough, scian_class=scian_class)
    try:
        result.expected = await client.cuantificar(scian_class, borough.geo_code)
        await asyncio.sleep(REQUEST_DELAY_SECONDS)
        start = 1
        while True:
            page = await client.buscar_area_act_page(borough, scian_class, start, start + PAGE_SIZE - 1)
            result.records.extend(page)
            await asyncio.sleep(REQUEST_DELAY_SECONDS)
            if len(page) < PAGE_SIZE:
                break
            start += PAGE_SIZE
    except SetupError:
        raise
    except RuntimeError as exc:
        result.failure = str(exc)
    logger.info(
        "scope fetched",
        extra={
            "borough": borough.name,
            "scian_class": scian_class,
            "expected": result.expected,
            "retrieved": len(result.records),
            "failure": result.failure,
        },
    )
    return result


async def fetch_all(token: str) -> list[ScopeResult]:
    client = DenueClient(token)
    try:
        results: list[ScopeResult] = []
        for borough in BOROUGHS:
            for category in CATEGORIES:
                for scian_class in category.scian_classes:
                    results.append(await fetch_scope(client, borough, scian_class))
        return results
    finally:
        await client.aclose()


# --------------------------------------------------------------------------- #
# Normalization and minimization
# --------------------------------------------------------------------------- #

_STRATUM_BY_LABEL = {_label.lower(): code for code, _label in EMPLOYMENT_STRATA.items()}


def fold(text: str) -> str:
    stripped = "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", stripped.lower()).strip()


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _stratum_code(value: Any) -> str | None:
    text = _clean(value)
    if text in EMPLOYMENT_STRATA:
        return text
    folded = fold(text)
    for label, code in _STRATUM_BY_LABEL.items():
        if fold(label) == folded:
            return code
    return None


def _coordinate(value: Any, valid_range: tuple[float, float]) -> float | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if not (valid_range[0] <= number <= valid_range[1]):
        return None
    return round(number, 6)


def normalize_record(raw: dict[str, Any], borough: Borough, scian_class: str) -> EstablishmentRecord | None:
    """Keep only permitted fields; return None when a required value is unusable."""
    category = CATEGORY_BY_SCIAN_CLASS[scian_class]
    identifier = _clean(raw.get("Id"))
    name = _clean(raw.get("Nombre"))
    stratum = _stratum_code(raw.get("Estrato"))
    if not identifier or not name or stratum is None:
        return None
    activity_label = (
        _clean(raw.get("Clase_actividad"))
        or category.official_labels[category.scian_classes.index(scian_class)]
    )
    return EstablishmentRecord(
        id=identifier,
        name=name,
        category=category.key,
        scian_class=scian_class,
        activity_label=activity_label,
        stratum=stratum,
        stratum_label=EMPLOYMENT_STRATA[stratum],
        state=ENTITY_NAME,
        borough_code=borough.code,
        borough=borough.name,
        locality=_clean(raw.get("Colonia")),
        latitude=_coordinate(raw.get("Latitud"), LAT_RANGE),
        longitude=_coordinate(raw.get("Longitud"), LON_RANGE),
        establishment_type=_clean(raw.get("Tipo")),
        source_date=_clean(raw.get("Fecha_Alta")) or None,
    )


def normalize_all(results: list[ScopeResult]) -> tuple[list[EstablishmentRecord], int, int]:
    """Return (records, dropped_invalid, dropped_duplicates)."""
    records: dict[str, EstablishmentRecord] = {}
    dropped = 0
    duplicates = 0
    for result in results:
        expected_label = fold(
            CATEGORY_BY_SCIAN_CLASS[result.scian_class].official_labels[
                CATEGORY_BY_SCIAN_CLASS[result.scian_class].scian_classes.index(result.scian_class)
            ]
        )
        borough_name = fold(result.borough.name)
        for raw in result.records:
            class_id = _clean(raw.get("CLASE_ACTIVIDAD_ID"))
            if class_id and class_id != result.scian_class:
                dropped += 1
                continue
            # Ubicacion is checked in flight and then discarded (never stored).
            location = fold(_clean(raw.get("Ubicacion")))
            if location and borough_name not in location:
                result.borough_mismatches += 1
                dropped += 1
                continue
            record = normalize_record(raw, result.borough, result.scian_class)
            if record is None:
                dropped += 1
                continue
            if fold(record.activity_label) != expected_label:
                result.label_mismatches[record.activity_label] += 1
            if record.id in records:
                duplicates += 1
                continue
            records[record.id] = record
    ordered = sorted(records.values(), key=lambda r: (r.borough_code, r.scian_class, r.id))
    return ordered, dropped, duplicates


# --------------------------------------------------------------------------- #
# Semantic descriptions
# --------------------------------------------------------------------------- #


def establishment_description(record: EstablishmentRecord) -> str:
    category = CATEGORY_BY_SCIAN_CLASS[record.scian_class]
    where = f"{record.locality}, {record.borough}" if record.locality else record.borough
    return (
        f"{record.name} is classified by INEGI DENUE as {record.activity_label} "
        f"({category.label_en.lower()}). It is located in {where}, Mexico City. "
        f"Its reported employment-size range is {record.stratum_label}. "
        f"Establishment type: {record.establishment_type or 'not reported'}. "
        "This is one sample record from the configured dataset, not a complete listing."
    )


def aggregate_documents(
    records: list[EstablishmentRecord], aggregates: Aggregates, complete: bool, date: str
) -> list[tuple[str, str, str | None, str | None]]:
    """(id, text, borough, category) for every aggregate document."""
    docs: list[tuple[str, str, str | None, str | None]] = []
    tail = "Source: INEGI DENUE. Calculation performed by this application."
    scope_note = (
        "The dataset covers the configured boroughs and categories completely."
        if complete
        else "The dataset is a partial retrieval; some scopes failed or were incomplete."
    )
    docs.append(
        (
            "aggregate:overview",
            f"The configured DENUE dataset contains {aggregates.total} establishments across "
            f"{len(aggregates.by_category)} retail categories in {len(aggregates.by_borough)} Mexico City "
            f"boroughs ({', '.join(aggregates.by_borough)}), retrieved on {date}. {scope_note} {tail}",
            None,
            None,
        )
    )
    for borough, per_category in aggregates.by_borough_category.items():
        borough_total = aggregates.by_borough[borough]
        top = sorted(per_category.items(), key=lambda item: (-item[1], item[0]))[:1]
        top_text = f"{analytics.category_label(top[0][0]).lower()} ({top[0][1]})" if top else "none"
        docs.append(
            (
                f"aggregate:borough:{borough}",
                f"In the configured DENUE dataset, {borough} contains {borough_total} establishments "
                f"across the covered retail categories; the largest category is {top_text}. {tail}",
                borough,
                None,
            )
        )
        for category_key, count in per_category.items():
            strata = analytics.count_by_stratum(records, (borough,), (category_key,))
            common = max(strata.items(), key=lambda item: (item[1], -int(item[0])))
            docs.append(
                (
                    f"aggregate:borough_category:{borough}:{category_key}",
                    f"In the configured DENUE dataset, {borough} contains {count} establishments "
                    f"classified as {analytics.category_label(category_key).lower()} "
                    f"({analytics.percentage(count, borough_total)}% of the borough's covered "
                    f"establishments). The most common employment-size range is "
                    f"{analytics.stratum_label(common[0])} "
                    f"with {common[1]} establishments. {tail}",
                    borough,
                    category_key,
                )
            )
    for category_key, count in aggregates.by_category.items():
        ranking = analytics.rank_boroughs(aggregates, (category_key,))
        lead = ranking[0] if ranking else None
        lead_text = f"{lead.key} ({lead.count})" if lead else "none"
        docs.append(
            (
                f"aggregate:category:{category_key}",
                f"Across the covered boroughs, the configured DENUE dataset contains {count} "
                f"establishments classified as {analytics.category_label(category_key).lower()} "
                f"({analytics.percentage(count, aggregates.total)}% of the dataset); "
                f"the borough with the most is {lead_text}. {tail}",
                None,
                category_key,
            )
        )
    return docs


def sample_documents(records: list[EstablishmentRecord]) -> list[tuple[str, str, str | None, str | None]]:
    by_scope: defaultdict[tuple[str, str], list[EstablishmentRecord]] = defaultdict(list)
    for record in records:
        by_scope[(record.borough, record.category)].append(record)
    docs: list[tuple[str, str, str | None, str | None]] = []
    for (borough, category_key), items in sorted(by_scope.items()):
        items.sort(key=lambda r: (r.name.casefold(), r.id))
        for record in items[:SAMPLES_PER_SCOPE]:
            docs.append((f"sample:{record.id}", establishment_description(record), borough, category_key))
    return docs


# --------------------------------------------------------------------------- #
# Artifact assembly
# --------------------------------------------------------------------------- #


def build_artifacts(
    results: list[ScopeResult],
    records: list[EstablishmentRecord],
    embedder: FastEmbedProvider,
    retrieved_at: str,
) -> tuple[Manifest, EstablishmentsFile, Aggregates, KnowledgeFile]:
    aggregates = compute_aggregates(records)

    failed = [
        ScopeFailure(borough=r.borough.name, scian_class=r.scian_class, reason=r.failure)
        for r in results
        if r.failure
    ]
    checks = [
        CountCheck(
            borough=r.borough.name, scian_class=r.scian_class, expected=r.expected, retrieved=len(r.records)
        )
        for r in results
        if not r.failure
    ]
    mismatched = [c for c in checks if c.expected is not None and c.expected != c.retrieved]
    complete = not failed and not mismatched and all(c.expected is not None for c in checks)
    if complete:
        note = "All configured borough/activity scopes were retrieved and match the DENUE Cuantificar totals."
    else:
        parts = []
        if failed:
            parts.append(f"{len(failed)} scope(s) failed to download")
        if mismatched:
            parts.append(f"{len(mismatched)} scope(s) differ from the DENUE Cuantificar total")
        if any(c.expected is None for c in checks):
            parts.append("some Cuantificar totals were unavailable")
        note = "Partial dataset: " + "; ".join(parts) + ". Answers describe a partial dataset."

    date = retrieved_at[:10]
    docs = aggregate_documents(records, aggregates, complete, date) + sample_documents(records)
    matrix = embedder.embed_documents([text for _, text, _, _ in docs])
    if matrix.ndim != 2 or matrix.shape[0] != len(docs) or not np.isfinite(matrix).all():
        raise RuntimeError("embedding output is invalid")
    dimension = int(matrix.shape[1])
    documents = [
        KnowledgeDocument(
            id=doc_id,
            kind="aggregate" if doc_id.startswith("aggregate:") else "establishment_sample",
            text=text,
            borough=borough,
            category=category,
            embedding=[round(float(v), 6) for v in matrix[index]],
        )
        for index, (doc_id, text, borough, category) in enumerate(docs)
    ]
    knowledge = KnowledgeFile(
        metadata=KnowledgeMetadata(
            created_at=retrieved_at,
            embedding_provider=embedder.provider_name,
            embedding_model=embedder.model_name,
            embedding_dimension=dimension,
            document_count=len(documents),
        ),
        documents=documents,
    )
    establishments = EstablishmentsFile(schema_version=SCHEMA_VERSION, establishments=records)
    latest_date = max((r.source_date for r in records if r.source_date), default=None)
    manifest = Manifest(
        schema_version=SCHEMA_VERSION,
        source_name=SOURCE_NAME,
        source_url=SOURCE_URL,
        retrieved_at=retrieved_at,
        dataset_update_date=latest_date,
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
        embedding_dimension=dimension,
        data_checksum=checksum_bytes(serialize_model(establishments, compact=True)),
        complete=complete,
        completeness_note=note,
        failed_scopes=failed,
        count_checks=checks,
        transformation_notice=TRANSFORMATION_NOTICE,
        attribution=attribution_text(date),
    )
    return manifest, establishments, aggregates, knowledge


def write_artifacts(
    data_dir: Path,
    manifest: Manifest,
    establishments: EstablishmentsFile,
    aggregates: Aggregates,
    knowledge: KnowledgeFile,
) -> None:
    """Stage all files, re-validate them with the runtime loader, then replace."""
    data_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=data_dir))
    try:
        write_bytes_atomic(staging / ESTABLISHMENTS_FILE, serialize_model(establishments, compact=True))
        write_bytes_atomic(staging / AGGREGATES_FILE, serialize_model(aggregates))
        write_bytes_atomic(staging / KNOWLEDGE_FILE, serialize_model(knowledge, compact=True))
        write_bytes_atomic(staging / MANIFEST_FILE, serialize_model(manifest))
        load_dataset(staging, manifest.embedding_model)  # raises DatasetError if inconsistent
        for name in (ESTABLISHMENTS_FILE, AGGREGATES_FILE, KNOWLEDGE_FILE, MANIFEST_FILE):
            (staging / name).replace(data_dir / name)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def print_report(
    results: list[ScopeResult], manifest: Manifest, dropped: int, duplicates: int, data_dir: Path
) -> None:
    print("DENUE ingestion report")
    print(f"  Data directory:        {data_dir}")
    print(f"  Retrieved at:          {manifest.retrieved_at}")
    print(f"  Boroughs:              {manifest.borough_count}  Categories: {manifest.category_count}")
    print(
        f"  Establishments:        {manifest.record_count}  "
        f"(dropped invalid: {dropped}, duplicates: {duplicates})"
    )
    print(
        f"  Knowledge documents:   {manifest.knowledge_document_count} "
        f"({manifest.embedding_model}, dim {manifest.embedding_dimension})"
    )
    print(f"  Complete:              {manifest.complete}")
    print(f"  Note:                  {manifest.completeness_note}")
    print("  Scopes (borough / SCIAN class: expected -> retrieved):")
    for result in results:
        status = (
            "FAILED " + result.failure
            if result.failure
            else ("ok" if result.expected == len(result.records) else "MISMATCH")
        )
        print(
            f"    {result.borough.name:15s} {result.scian_class}: "
            f"{result.expected} -> {len(result.records)}  {status}"
        )
        for label, count in result.label_mismatches.items():
            print(f"      ! unexpected activity label seen {count}x: {label}")
        if result.borough_mismatches:
            print(f"      ! {result.borough_mismatches} record(s) dropped: Ubicacion named another borough")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=None, help="Output directory (default: DATA_DIR)")
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level)
    data_dir = args.data_dir or settings.data_dir

    token = settings.inegi_token_value
    if not token:
        print(
            "INEGI_API_TOKEN is not set. Register for a free DENUE API token at\n"
            "https://www.inegi.org.mx/servicios/api_denue.html and add it to .env, then re-run.\n"
            "The web application still starts with the existing generated dataset.",
            file=sys.stderr,
        )
        return 2

    try:
        results = asyncio.run(fetch_all(token))
    except SetupError as exc:
        print(f"Ingestion aborted: {exc}", file=sys.stderr)
        return 2

    records, dropped, duplicates = normalize_all(results)
    if not records:
        print("Ingestion failed: no establishment records were retrieved.", file=sys.stderr)
        return 1

    embedder = FastEmbedProvider(settings.embedding_model, resolve_cache_dir(settings.fastembed_cache_path))
    retrieved_at = datetime.now(UTC).isoformat(timespec="seconds")
    manifest, establishments, aggregates, knowledge = build_artifacts(
        results, records, embedder, retrieved_at
    )
    write_artifacts(data_dir, manifest, establishments, aggregates, knowledge)
    logger.info(
        "ingestion complete",
        extra={
            "records": manifest.record_count,
            "complete": manifest.complete,
            "failed_scopes": len(manifest.failed_scopes),
            "knowledge_documents": manifest.knowledge_document_count,
        },
    )
    print_report(results, manifest, dropped, duplicates, data_dir)
    return 0 if manifest.complete else 3


if __name__ == "__main__":
    sys.exit(main())
