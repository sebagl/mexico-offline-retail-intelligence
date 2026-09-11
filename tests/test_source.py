"""Source, summary and attribution endpoints."""

from pathlib import Path

from app.schemas import SOURCE_URL, TRANSFORMATION_NOTICE
from tests.conftest import FakeEmbedder, build_synthetic_dataset, make_settings


def test_source_endpoint_carries_attribution_and_scope(client) -> None:
    body = client.get("/api/source").json()
    assert body["source_url"] == SOURCE_URL
    assert body["attribution"].startswith(
        "Fuente: INEGI, Directorio Estadístico Nacional de Unidades Económicas (DENUE), 2026-01-15"
    )
    assert body["transformation_notice"] == TRANSFORMATION_NOTICE
    assert body["complete"] is True
    assert body["failed_scopes"] == []
    names = {b["name"] for b in body["geographic_scope"]["boroughs"]}
    assert names == {"Benito Juárez", "Cuauhtémoc", "Miguel Hidalgo", "Coyoacán", "Iztapalapa"}
    assert {c["key"] for c in body["economic_scope"]} >= {"grocery", "pharmacy", "bakery"}


def test_source_reports_partial_dataset(make_client, tmp_path: Path, fake_embedder: FakeEmbedder) -> None:
    partial_dir = build_synthetic_dataset(tmp_path / "partial", complete=False, embedder=fake_embedder)
    client = make_client(settings=make_settings(partial_dir))
    body = client.get("/api/source").json()
    assert body["complete"] is False
    assert body["failed_scopes"][0]["borough"] == "Iztapalapa"
    assert client.get("/health").json()["dataset_complete"] is False
    answer = client.post("/api/query", json={"question": "How many grocery stores are there?"}).json()
    assert answer["scope"]["basis"] == "partial_dataset"
    assert "partial" in answer["methodology"]


def test_summary_matches_synthetic_counts(client) -> None:
    body = client.get("/api/summary").json()
    assert body["total_establishments"] == 41
    by_borough = {b["name"]: b["count"] for b in body["boroughs"]}
    assert by_borough == {
        "Benito Juárez": 11,
        "Cuauhtémoc": 9,
        "Iztapalapa": 8,
        "Miguel Hidalgo": 7,
        "Coyoacán": 6,
    }
    by_category = {c["key"]: c["count"] for c in body["categories"]}
    assert by_category["grocery"] == 20
    assert body["attribution"].startswith("Fuente: INEGI")
    assert body["transformation_notice"] == TRANSFORMATION_NOTICE


def test_source_endpoints_return_503_without_dataset(make_client, tmp_path: Path) -> None:
    client = make_client(settings=make_settings(tmp_path / "missing"))
    for path in ("/api/source", "/api/summary"):
        response = client.get(path)
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "dataset_unavailable"
