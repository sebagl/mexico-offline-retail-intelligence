"""Request validation for POST /api/query."""

from pathlib import Path

from tests.conftest import make_settings


def test_valid_question(client) -> None:
    response = client.post("/api/query", json={"question": "How many establishments are in the dataset?"})
    assert response.status_code == 200
    assert response.json()["mode"] == "deterministic"


def test_empty_question_is_422(client) -> None:
    response = client.post("/api/query", json={"question": ""})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_whitespace_question_is_422(client) -> None:
    assert client.post("/api/query", json={"question": "   \n\t "}).status_code == 422


def test_null_question_is_422(client) -> None:
    assert client.post("/api/query", json={"question": None}).status_code == 422


def test_missing_question_is_422(client) -> None:
    assert client.post("/api/query", json={}).status_code == 422


def test_oversized_question_is_400(client) -> None:
    response = client.post("/api/query", json={"question": "grocery " * 100})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_question"


def test_custom_maximum_length(make_client, data_dir: Path) -> None:
    client = make_client(settings=make_settings(data_dir, max_question_length=20))
    assert client.post("/api/query", json={"question": "a" * 21}).status_code == 400
    assert client.post("/api/query", json={"question": "how many stores"}).status_code == 200


def test_unexpected_fields_are_rejected(client) -> None:
    response = client.post("/api/query", json={"question": "hi", "debug": True})
    assert response.status_code == 422
    assert "debug" in response.json()["error"]["message"]


def test_malformed_json_is_422_with_error_format(client) -> None:
    response = client.post("/api/query", content=b"{not json", headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert set(response.json()["error"]) == {"code", "message"}


def test_repetitive_input_is_handled_safely(client) -> None:
    response = client.post("/api/query", json={"question": "grocery grocery grocery " * 15})
    assert response.status_code in (200, 400)
    if response.status_code == 200:
        assert response.json()["mode"] in {"unsupported", "extractive", "deterministic"}


def test_oversized_body_is_413(make_client, data_dir: Path) -> None:
    client = make_client(settings=make_settings(data_dir, max_request_body_bytes=1024))
    response = client.post("/api/query", json={"question": "x" * 5000})
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"
