"""API surface: index page, headers, rate limiting, controlled errors."""

from pathlib import Path

from tests.conftest import make_settings


def test_index_serves_html_with_attribution_and_disclaimer(client) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Mexico Offline Retail Intelligence" in response.text
    assert "Fuente: INEGI, Directorio Estadístico Nacional de Unidades Económicas (DENUE)" in response.text
    assert "not produced, reviewed, sponsored, or endorsed by INEGI" in response.text
    assert "innerHTML" not in (client.get("/static/app.js").text)


def test_request_id_and_security_headers(client) -> None:
    response = client.get("/health")
    assert len(response.headers["x-request-id"]) == 16
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]


def test_client_supplied_request_id_is_ignored(client) -> None:
    response = client.get("/health", headers={"X-Request-ID": "evil\ninjected"})
    assert response.headers["x-request-id"] != "evil\ninjected"


def test_rate_limit_returns_429(make_client, data_dir: Path) -> None:
    client = make_client(
        settings=make_settings(data_dir, rate_limit_requests=2, rate_limit_window_seconds=60)
    )
    payload = {"question": "How many establishments are in the dataset?"}
    assert client.post("/api/query", json=payload).status_code == 200
    assert client.post("/api/query", json=payload).status_code == 200
    response = client.post("/api/query", json=payload)
    assert response.status_code == 429
    assert response.json()["error"]["code"] == "rate_limited"
    assert client.get("/health").status_code == 200  # health is not rate limited


def test_query_without_dataset_is_controlled_503(make_client, tmp_path: Path) -> None:
    client = make_client(settings=make_settings(tmp_path / "missing"))
    response = client.post("/api/query", json={"question": "How many stores?"})
    assert response.status_code == 503
    assert response.json() == {
        "error": {"code": "dataset_unavailable", "message": "The DENUE dataset is temporarily unavailable."}
    }


def test_unexpected_error_is_generic(client, monkeypatch) -> None:
    from app.services.query import QueryService

    async def boom(self, question: str):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(QueryService, "answer", boom)
    response = client.post("/api/query", json={"question": "How many stores?"})
    assert response.status_code == 500
    assert response.json()["error"] == {"code": "internal_error", "message": "An unexpected error occurred."}
    assert "secret internal detail" not in response.text
    assert "Traceback" not in response.text
    assert response.headers["x-content-type-options"] == "nosniff"
    assert len(response.headers["x-request-id"]) == 16


def test_not_found_uses_error_format(client) -> None:
    response = client.get("/api/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_docs_are_disabled(client) -> None:
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404


def test_forwarded_ip_is_validated(make_client, data_dir: Path) -> None:
    from app.middleware import resolve_client_ip

    scope = {"headers": [(b"x-forwarded-for", b"1.2.3.4, 10.0.0.9")], "client": ("127.0.0.1", 1)}
    assert resolve_client_ip(scope, trust_proxy_headers=True) == "10.0.0.9"
    assert resolve_client_ip(scope, trust_proxy_headers=False) == "127.0.0.1"
    bad = {"headers": [(b"x-forwarded-for", b"not-an-ip")], "client": ("127.0.0.1", 1)}
    assert resolve_client_ip(bad, trust_proxy_headers=True) == "127.0.0.1"


def test_cors_allows_configured_origin_only(client) -> None:
    allowed = client.get("/health", headers={"Origin": "http://testserver"})
    assert allowed.headers.get("access-control-allow-origin") == "http://testserver"
    denied = client.get("/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in denied.headers


def test_no_secret_reaches_browser_assets(client, data_dir: Path) -> None:
    for path in ("/", "/static/app.js", "/static/styles.css"):
        text = client.get(path).text
        assert "GEMINI_API_KEY" not in text
        assert "INEGI_API_TOKEN" not in text
        assert str(data_dir) not in text
