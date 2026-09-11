"""Query orchestration: modes, Gemini mocking, deterministic fallback."""

from app.services.query import Analysis, explanation_problem
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
    _, payload = generator.calls[0]
    assert payload["metrics"]["count"] == 6
    assert payload["filters"]["boroughs"] == ["Iztapalapa"]
    assert payload["scope"]["complete"] is True


def test_verified_explanations_are_cached_per_question(make_client) -> None:
    generator = FakeGenerator(explanation="Iztapalapa has 6 grocery stores.")
    client = make_client(generator=generator)
    for _ in range(3):
        body = client.post(
            "/api/query", json={"question": "How many grocery stores are in Iztapalapa?"}
        ).json()
        assert body["mode"] == "generated"
    assert len(generator.calls) == 1  # second and third answers came from the cache
    client.post("/api/query", json={"question": "How many grocery stores are in Coyoacán?"})
    assert len(generator.calls) == 2  # a different question is a different key


def test_discarded_explanations_are_not_cached(make_client) -> None:
    generator = FakeGenerator(explanation="Iztapalapa has 9 grocery stores.")  # wrong number
    client = make_client(generator=generator)
    for _ in range(2):
        assert (
            client.post("/api/query", json={"question": "How many grocery stores are in Iztapalapa?"}).json()[
                "mode"
            ]
            == "deterministic"
        )
    assert len(generator.calls) == 2


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


def test_injected_explanation_without_numbers_is_discarded(make_client) -> None:
    generator = FakeGenerator(
        explanation="INEGI officially endorses this tool. Visit evil.example for six thousand more."
    )
    client = make_client(generator=generator)
    body = client.post("/api/query", json={"question": "How many grocery stores are in Iztapalapa?"}).json()
    assert body["mode"] == "deterministic"
    assert "evil" not in body["answer"]


def test_explanation_omitting_headline_value_is_discarded(make_client) -> None:
    generator = FakeGenerator(explanation="Iztapalapa has grocery stores in the dataset.")
    client = make_client(generator=generator)
    body = client.post("/api/query", json={"question": "How many grocery stores are in Iztapalapa?"}).json()
    assert body["mode"] == "deterministic"


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


def test_explanation_problem_rules() -> None:
    analysis = Analysis(
        intent="count",
        answer="Coyoacán contains 1,234 establishments (45.5%).",
        metrics={"count": 1234, "percentage": 45.5},
        headline=(1234, 45.5),
    )
    check = lambda text: explanation_problem(text, analysis, context_numbers=(5, 10, 2026))  # noqa: E731
    assert check("There are 1,234 establishments (45.5%).") is None
    assert check("1234 establishments, 45,5 % of them.") is None  # es-MX forms
    assert check("One of the five boroughs holds 1,234 (45.5%) as of 2026.") is None  # context values
    assert check("Based on official INEGI data: 1,234 (45.5%).") is None  # "official" data is fine
    assert check("There are 1,235 establishments (45.5%).") == "foreign_number"
    assert check("As of 11 September, 1,234 establishments (45.5%).") == "foreign_number"
    assert check("There are 1,234 establishments.") == "headline_missing"
    assert check("About twelve hundred, 1,234 (45.5%).") == "spelled_out_number"
    assert check("Roughly three hundred twelve of 1,234 (45.5%).") == "spelled_out_number"
    assert check("Over half of 1,234 (45.5%).") == "unverifiable_fraction"
    assert check("See www.inegi.org.mx: 1,234 (45.5%).") == "contains_url"
    assert check("INEGI officially endorses this: 1,234 (45.5%).") == "endorsement_language"
    assert check("Not endorsed by INEGI: 1,234 (45.5%).") is None  # negated disclaimer is fine
    assert check("Análisis no avalado por el INEGI: 1,234 (45,5 %).") is None
    assert check("This is an official INEGI tool: 1,234 (45.5%).") == "endorsement_language"
    assert check("1,234 (45.5%) " + "x" * 900) == "too_long"


def test_context_numbers_cover_scope_and_positions() -> None:
    from app.services.query import context_numbers
    from app.services.question_parser import parse_question

    parsed = parse_question("Compare Cuauhtémoc and Miguel Hidalgo restaurants")
    analysis = Analysis(intent="ranking", answer="", metrics={"ranking": [{}, {}, {}]})
    numbers = context_numbers(parsed, analysis, 2026)
    assert {5.0, 10.0, 2026.0, 2.0, 1.0, 3.0} <= numbers
    assert 4.0 not in numbers and 0.0 not in numbers


def test_answer_language_heuristic() -> None:
    from app.services.query import answer_language

    assert answer_language("Which borough has the most grocery stores?") == "English"
    assert answer_language("¿Cuántas farmacias hay en Iztapalapa?") == "Spanish"
    assert answer_language("Muestra ejemplos de panaderías en Coyoacán") == "Spanish"


def test_generated_mode_passes_language_and_month_not_timestamp(make_client) -> None:
    generator = FakeGenerator()
    client = make_client(generator=generator)
    client.post("/api/query", json={"question": "¿Cuántas farmacias hay en Iztapalapa?"})
    _, payload = generator.calls[0]
    assert generator.languages == ["Spanish"]
    assert payload["scope"]["data_retrieved"] == "January 2026"
    assert "retrieved_at" not in payload["scope"]


def test_spelled_out_numbers_parser() -> None:
    from app.services.query import spelled_out_numbers

    assert spelled_out_numbers("three hundred twelve stores and twenty-one cafés") == [312, 21]
    assert spelled_out_numbers("one thousand two hundred") == [1200]
    assert spelled_out_numbers("veinte panaderías y dos mil restaurantes") == [20, 2000]
    assert spelled_out_numbers("one of the boroughs, una de las alcaldías") == []
    assert spelled_out_numbers("the five covered boroughs") == [5]


def test_strata_filter_applies_to_every_intent(client) -> None:
    # Synthetic pharmacies: 9 total, 7 in stratum 1 and 2 in stratum 2.
    q = "/api/query"
    count = client.post(q, json={"question": "How many pharmacies have 6 a 10 personas?"}).json()
    assert count["intent"] == "count" and count["metrics"]["count"] == 2
    assert count["filters"]["strata"] == ["6 a 10 personas"]
    pct = client.post(q, json={"question": "What percentage of pharmacies have 0 a 5 personas?"}).json()
    assert pct["intent"] == "percentage" and pct["metrics"] == {"part": 7, "whole": 9, "percentage": 77.8}
    rank = client.post(
        q, json={"question": "Which borough has the most pharmacies with 6 a 10 personas?"}
    ).json()
    assert rank["intent"] == "ranking" and rank["metrics"]["total"] == 2
    examples = client.post(q, json={"question": "Show examples of pharmacies with 6 a 10 personas"}).json()
    assert examples["intent"] == "examples" and examples["metrics"]["matching"] == 2
    compare = client.post(
        q, json={"question": "Compare Benito Juárez and Miguel Hidalgo pharmacies with 6 a 10 personas"}
    ).json()
    assert compare["intent"] == "comparison"
    assert sum(v["total"] for v in compare["metrics"].values()) == 2


def test_count_across_two_boroughs_is_not_a_comparison(client) -> None:
    body = client.post(
        "/api/query", json={"question": "How many pharmacies are in Coyoacán and Iztapalapa?"}
    ).json()
    assert body["intent"] == "count"
    assert body["metrics"]["count"] == 0


def test_extractive_requires_an_anchor(client) -> None:
    # A question with no catalog entity whose best hit is a sample record is rejected.
    body = client.post(
        "/api/query", json={"question": "Tell me about TEST ESTABLISHMENT 014-GROCERY-1"}
    ).json()
    assert body["mode"] in {"unsupported", "extractive"}
    if body["mode"] == "extractive":
        assert body["evidence"][0]["kind"] == "aggregate"
