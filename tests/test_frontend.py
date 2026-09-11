"""The static frontend: suggested questions are supported, safety and key copy are present."""

import re
from pathlib import Path

from app.services.question_parser import parse_question

ROOT = Path(__file__).resolve().parent.parent
INDEX = (ROOT / "app" / "static" / "index.html").read_text(encoding="utf-8")
SCRIPT = (ROOT / "app" / "static" / "app.js").read_text(encoding="utf-8")


def _suggested_questions() -> list[str]:
    return re.findall(r'data-question="([^"]+)"', INDEX)


def test_every_suggested_question_maps_to_a_supported_intent() -> None:
    questions = _suggested_questions()
    assert len(questions) >= 6
    for question in questions:
        parsed = parse_question(question)
        assert parsed.intent != "unknown", question


def test_primary_interaction_and_copy_are_present() -> None:
    assert "Explore retail patterns across five Mexico City boroughs using public INEGI data." in INDEX
    assert 'id="question"' in INDEX and ">Analyze<" in INDEX
    assert 'href="https://github.com/sebagl/mexico-offline-retail-intelligence"' in INDEX
    assert 'rel="noopener noreferrer"' in INDEX
    assert INDEX.count('class="chip"') >= 6  # three primary examples plus the expandable set


def test_loading_and_error_states_are_honest_and_accessible() -> None:
    assert "Analyzing the selected dataset" in INDEX
    assert 'role="status"' in INDEX and 'role="alert"' in INDEX
    assert 'id="retry"' in INDEX
    assert "AbortController" in SCRIPT  # requests time out instead of hanging
    assert "progress" not in SCRIPT.lower()  # no invented progress percentages or stages


def test_frontend_never_hardcodes_dataset_totals() -> None:
    for literal in ("49,080", "49080"):
        assert literal not in INDEX and literal not in SCRIPT


def test_no_map_or_coordinates_in_the_frontend() -> None:
    lowered = (INDEX + SCRIPT).lower()
    assert "latitude" not in lowered and "longitude" not in lowered and "leaflet" not in lowered
