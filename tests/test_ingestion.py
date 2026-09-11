"""DENUE response normalization, minimization, dedup and partial handling.

Raw records below imitate the documented DENUE field names but every value
is synthetic (TEST ESTABLISHMENT ...). The live API is never called.
"""

from pathlib import Path

import httpx
import pytest

from app.catalog import BOROUGHS_BY_CODE
from app.schemas import EstablishmentRecord
from app.services.dataset import FORBIDDEN_FIELDS, load_dataset
from scripts.ingest_denue import (
    ScopeResult,
    aggregate_documents,
    build_artifacts,
    establishment_description,
    normalize_all,
    normalize_record,
    sample_documents,
    write_artifacts,
)
from tests.conftest import FakeEmbedder, synthetic_records

BJ = BOROUGHS_BY_CODE["014"]


def raw_record(identifier: str = "1001", **overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "CLEE": "09014461110000000000000",
        "Id": identifier,
        "Nombre": "TEST ESTABLISHMENT A",
        "Razon_social": "TEST LEGAL ENTITY SA DE CV",
        "Clase_actividad": "Comercio al por menor en tiendas de abarrotes, ultramarinos y misceláneas",
        "Estrato": "0 a 5 personas",
        "Tipo_vialidad": "CALLE",
        "Calle": "TEST STREET",
        "Num_Exterior": "1",
        "Num_Interior": "",
        "Colonia": "TEST LOCALITY",
        "CP": "03100",
        "Ubicacion": "TEST STREET 1, TEST LOCALITY, Benito Juárez, Ciudad de México",
        "Telefono": "5555555555",
        "Correo_e": "test@example.invalid",
        "Sitio_internet": "www.example.invalid",
        "Tipo": "Fijo",
        "Longitud": "-99.16",
        "Latitud": "19.38",
        "CLASE_ACTIVIDAD_ID": "461110",
        "Fecha_Alta": "2025-11",
    }
    record.update(overrides)
    return record


def test_parse_body_handles_denue_empty_sentinel() -> None:
    """Verified live: an empty scope is HTTP 200 with the JSON string "No hay resultados. "."""
    from scripts.ingest_denue import _parse_body

    def response(text: str) -> httpx.Response:
        return httpx.Response(200, text=text)

    assert _parse_body(response('"No hay resultados. "')) == []
    assert _parse_body(response("No hay resultados.")) == []  # plain text variant
    assert _parse_body(response('[{"AE":"461110","AG":"09014","Total":"697"}]')) == [
        {"AE": "461110", "AG": "09014", "Total": "697"}
    ]
    with pytest.raises(RuntimeError, match="unexpected_string_response"):
        _parse_body(response('"something else"'))
    with pytest.raises(RuntimeError, match="non_json_response"):
        _parse_body(response("<html>error</html>"))


def test_normalize_keeps_only_permitted_fields() -> None:
    record = normalize_record(raw_record(), BJ, "461110")
    assert record is not None
    stored = set(record.model_dump())
    assert not (stored & FORBIDDEN_FIELDS)
    for key in (
        "telefono",
        "correo_e",
        "sitio_internet",
        "razon_social",
        "calle",
        "cp",
        "ubicacion",
        "latitud",
    ):
        assert key not in stored
    assert record.name == "TEST ESTABLISHMENT A"
    assert record.category == "grocery"
    assert record.stratum == "1"
    assert record.stratum_label == "0 a 5 personas"
    assert record.borough == "Benito Juárez"
    assert record.borough_code == "014"
    assert record.locality == "TEST LOCALITY"
    assert not hasattr(record, "latitude")  # coordinates are not retained
    assert record.source_date == "2025-11"


def test_normalize_accepts_numeric_stratum_and_drops_unknown() -> None:
    assert normalize_record(raw_record(Estrato="3"), BJ, "461110").stratum == "3"
    assert normalize_record(raw_record(Estrato="unknown"), BJ, "461110") is None
    assert normalize_record(raw_record(Nombre="   "), BJ, "461110") is None
    assert normalize_record(raw_record(Id=""), BJ, "461110") is None


def test_dedup_and_wrong_class_rejection() -> None:
    scope = ScopeResult(borough=BJ, scian_class="461110", expected=2)
    scope.records = [
        raw_record("1"),
        raw_record("1"),
        raw_record("2"),
        raw_record("3", CLASE_ACTIVIDAD_ID="999999"),
    ]
    records, dropped, duplicates = normalize_all([scope])
    assert [r.id for r in records] == ["1", "2"]
    assert duplicates == 1
    assert dropped == 1


def test_label_mismatch_is_recorded() -> None:
    scope = ScopeResult(borough=BJ, scian_class="461110")
    scope.records = [raw_record("1", Clase_actividad="Some unexpected label")]
    normalize_all([scope])
    assert scope.label_mismatches == {"Some unexpected label": 1}


def test_descriptions_exclude_contact_data() -> None:
    record = normalize_record(raw_record(), BJ, "461110")
    text = establishment_description(record)
    assert "TEST ESTABLISHMENT A" in text
    assert "INEGI DENUE" in text
    assert "sample record" in text
    for leak in ("5555555555", "example.invalid", "TEST STREET", "03100", "SA DE CV"):
        assert leak not in text


def test_aggregate_documents_use_exact_counts() -> None:
    records = synthetic_records()
    from app.services.dataset import compute_aggregates

    docs = aggregate_documents(records, compute_aggregates(records), complete=True, date="2026-01-15")
    by_id = {d[0]: d[1] for d in docs}
    assert (
        "Iztapalapa contains 6 establishments classified as grocery stores"
        in by_id["aggregate:borough_category:Iztapalapa:grocery"]
    )
    assert "contains 41 establishments" in by_id["aggregate:overview"]
    assert "completely" in by_id["aggregate:overview"]
    samples = sample_documents(records)
    assert all(text.startswith("TEST ESTABLISHMENT") for _, text, _, _ in samples)


def test_partial_results_mark_manifest_incomplete(tmp_path: Path) -> None:
    records = synthetic_records()
    ok = ScopeResult(borough=BJ, scian_class="461110", expected=5)
    ok.records = [raw_record(str(i)) for i in range(5)]
    failed = ScopeResult(borough=BJ, scian_class="464111", failure="timeout")
    mismatch = ScopeResult(borough=BJ, scian_class="311812", expected=10)
    mismatch.records = [raw_record(str(i), CLASE_ACTIVIDAD_ID="311812") for i in range(2)]
    embedder = FakeEmbedder()
    manifest, establishments, aggregates, knowledge = build_artifacts(
        [ok, failed, mismatch], records, embedder, "2026-01-15T12:00:00+00:00"
    )
    assert manifest.complete is False
    assert manifest.failed_scopes[0].scian_class == "464111"
    assert "Partial dataset" in manifest.completeness_note
    assert any(c.expected != c.retrieved for c in manifest.count_checks)
    assert manifest.record_count == len(records)
    assert manifest.attribution.startswith("Fuente: INEGI")

    write_artifacts(tmp_path / "data", manifest, establishments, aggregates, knowledge)
    dataset = load_dataset(tmp_path / "data", embedder.model_name)
    assert dataset.manifest.complete is False
    assert dataset.record_count == len(records)


def test_complete_results_mark_manifest_complete() -> None:
    records: list[EstablishmentRecord] = synthetic_records()
    scopes = []
    for record in records[:3]:
        scope = ScopeResult(
            borough=BOROUGHS_BY_CODE[record.borough_code], scian_class=record.scian_class, expected=1
        )
        scope.records = [raw_record(record.id)]
        scopes.append(scope)
    manifest, *_ = build_artifacts(scopes, records, FakeEmbedder(), "2026-01-15T12:00:00+00:00")
    assert manifest.complete is True
    assert manifest.failed_scopes == []
