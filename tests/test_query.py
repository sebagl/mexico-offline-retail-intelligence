"""Query orchestration: modes, Gemini mocking, deterministic fallback."""

from app.services.query import numbers_consistent
from tests.conftest import FakeGenerator


def test_deterministic_mode_without_gemini(client) -> None:
    body = client.post("/api/query", json={"question": "How many grocery stores are in Iztapalapa?"}).json()
    assert body["mode"] == "deterministic"
    assert body["intent"] == "count"
    assert body["metrics"]["count"] == 6
    assert "No language model" in body["methodology"]


def test_generated_mode_uses_explanation_but_keeps_metrics(make_client) -> None:
    generator = FakeGenerator(explanation="Iztapalapa has 6 grocery stores in the dataset (41 total).")
    client = make_client(generator=generator)
    body = client.post("/api/query", json={"question": "How many grocery stores are in Iztapalapa?"}).json()
    assert body["mode"] == "generated"
    assert body["answer"].startswith("Iztapalapa has 6")
    assert body["metrics"]["count"] == 6
    assert "only rephrased" in body["methodology"]
    _, payload, _ = generator.calls[0]
    assert payload["metrics"]["count"] == 6
    assert payload["filters"]["boroughs"] == ["Iztapalapa"]
    assert payload["scope"]["complete"] is True


def test_gemini_is_not_called_without_evidence_or_for_unsupported(make_client) -> None:
    generator = FakeGenerator()
    client = make_client(generator=generator)
    body = client.post("/api/query", json={"question": "How do I make a chocolate cake?"}).json()
    assert body["mode"] == "unsupported"
    assert generator.calls == []


def test_explanation_with_foreign_numbers_is_discarded(make_client) -> None:
    generator = FakeGenerator(explanation="Iztapalapa has 9 grocery stores.")
    client = make_client(generator=generator)
    body = client.post("/api/query", json={"question": "How many grocery stores are in Iztapalapa?"}).json()
    assert body["mode"] == "deterministic"
    assert "6" in body["answer"]
    assert client.get("/health").json()["status"] == "degraded"


def test_fallback_after_each_generation_failure(make_client) -> None:
    for failure in (
        "timeout",
        "rate_limited",
        "invalid_credentials",
        "model_unavailable",
        "empty_response",
        "provider_unavailable",
    ):
        client = make_client(generator=FakeGenerator(failure=failure))
        body = client.post(
            "/api/query", json={"question": "What percentage of establishments are pharmacies?"}
        ).json()
        assert body["mode"] == "deterministic", failure
        assert body["metrics"]["percentage"] == 22.0


def test_unsupported_question_without_filters(client) -> None:
    body = client.post("/api/query", json={"question": "Who is the president of France?"}).json()
    assert body["mode"] == "unsupported"
    assert body["intent"] == "unknown"
    assert body["metrics"] == {}
    assert body["evidence"] == []


def test_extractive_mode_for_open_question_with_evidence(client) -> None:
    body = client.post("/api/query", json={"question": "Tell me about grocery stores in Iztapalapa"}).json()
    assert body["mode"] == "extractive"
    assert body["intent"] == "unknown"
    assert body["evidence"]
    assert body["evidence"][0]["borough"] == "Iztapalapa"
    assert "most relevant facts" in body["answer"]


def test_insufficient_data_for_comparison_without_two_boroughs(client) -> None:
    body = client.post("/api/query", json={"question": "Compare the retail composition of Coyoacán"}).json()
    assert body["mode"] == "insufficient_data"
    assert body["intent"] == "comparison"


def test_response_always_has_source_scope_and_attribution(client) -> None:
    body = client.post("/api/query", json={"question": "Show examples of bakeries in Coyoacán"}).json()
    assert body["source"] == {
        "name": "INEGI DENUE",
        "url": "https://www.inegi.org.mx/servicios/api_denue.html",
    }
    assert body["scope"]["boroughs"] == ["Coyoacán"]
    assert body["scope"]["categories"] == ["Bakeries"]
    assert body["scope"]["basis"] == "filtered_subset"
    assert body["attribution"].startswith("Fuente: INEGI")
    assert all(e["name"].startswith("TEST ESTABLISHMENT") for e in body["metrics"]["examples"])


def test_numbers_consistent_helper() -> None:
    payload = {"metrics": {"count": 1234, "percentage": 45.5}, "scope": {"retrieved_at": "2026-01-15"}}
    assert numbers_consistent("There are 1,234 establishments (45.5%) as of 2026-01-15.", payload)
    assert numbers_consistent("1234 establishments", payload)
    assert not numbers_consistent("There are 1,235 establishments.", payload)
    assert not numbers_consistent("Roughly 46% of them.", payload)
