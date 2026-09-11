"""Health endpoint: status transitions and no secret exposure."""

from pathlib import Path

from tests.conftest import FakeGenerator, make_settings


def test_health_ok_when_dataset_and_generator_available(make_client) -> None:
    client = make_client(generator=FakeGenerator(configured=True))
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["version"]
    assert body["dataset_loaded"] is True
    assert body["establishments"] == 41
    assert body["boroughs"] == 5
    assert body["categories"] == 6
    assert body["dataset_complete"] is True
    assert body["generation_configured"] is True
    assert body["generation_provider"] == "gemini"
    assert body["fallback_available"] is True


def test_health_ok_without_generator(client) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"  # Gemini is optional; its absence is the documented default
    assert body["generation_configured"] is False
    assert body["generation_provider"] == "none"
    assert body["fallback_available"] is True


def test_health_degraded_after_recent_generation_failure(make_client) -> None:
    client = make_client(generator=FakeGenerator(failure="rate_limited"))
    assert client.get("/health").json()["status"] == "ok"
    response = client.post("/api/query", json={"question": "How many establishments are in the dataset?"})
    assert response.json()["mode"] == "deterministic"
    health = client.get("/health").json()
    assert health["status"] == "degraded"
    assert health["generation_last_failure"] == "rate_limited"  # category only, never provider detail


def test_health_unavailable_when_dataset_missing(make_client, tmp_path: Path) -> None:
    client = make_client(settings=make_settings(tmp_path / "missing"))
    response = client.get("/health")
    assert response.status_code == 503  # keeps a broken deploy out of rotation
    body = response.json()
    assert body["status"] == "unavailable"
    assert body["dataset_loaded"] is False
    assert body["establishments"] == 0
    assert body["fallback_available"] is False


def test_health_never_exposes_secrets_or_paths(make_client, data_dir: Path) -> None:
    settings = make_settings(
        data_dir,
        gemini_api_key="super-secret-key-value",
        gemini_model="some-model",
        inegi_api_token="tok-123",
    )
    client = make_client(settings=settings, generator=FakeGenerator(configured=True))
    text = client.get("/health").text
    assert "super-secret-key-value" not in text
    assert "tok-123" not in text
    assert str(data_dir) not in text
    assert "some-model" not in text
