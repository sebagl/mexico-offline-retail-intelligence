"""Deterministic calculations over the synthetic records."""

import pytest

from app.services import analytics
from app.services.dataset import compute_aggregates
from tests.conftest import SYNTHETIC_TOTAL, synthetic_records

RECORDS = synthetic_records()
AGG = compute_aggregates(RECORDS)


def test_total_and_marginals() -> None:
    assert AGG.total == SYNTHETIC_TOTAL == 41
    assert AGG.by_borough["Benito Juárez"] == 11
    assert AGG.by_category["grocery"] == 20
    assert sum(AGG.by_stratum.values()) == 41


def test_count_combinations() -> None:
    assert analytics.count_establishments(AGG) == 41
    assert analytics.count_establishments(AGG, boroughs=("Coyoacán",)) == 6
    assert analytics.count_establishments(AGG, categories=("pharmacy",)) == 9
    assert analytics.count_establishments(AGG, boroughs=("Iztapalapa",), categories=("grocery",)) == 6
    assert (
        analytics.count_establishments(
            AGG, boroughs=("Iztapalapa", "Coyoacán"), categories=("grocery", "bakery")
        )
        == 12
    )
    assert analytics.count_establishments(AGG, boroughs=("Iztapalapa",), categories=("pharmacy",)) == 0


def test_percentages() -> None:
    assert analytics.percentage(20, 41) == 48.8
    assert analytics.percentage(0, 41) == 0.0
    assert analytics.percentage(5, 0) == 0.0
    assert analytics.percentage(1, 3) == 33.3


def test_rank_boroughs_by_category() -> None:
    ranking = analytics.rank_boroughs(AGG, categories=("grocery",))
    assert [item.key for item in ranking] == [
        "Iztapalapa",
        "Benito Juárez",
        "Cuauhtémoc",
        "Coyoacán",
        "Miguel Hidalgo",
    ]
    assert ranking[0].count == 6
    assert ranking[0].share == 30.0
    lowest = analytics.rank_boroughs(AGG, categories=("grocery",), ascending=True)
    assert lowest[0].key == "Miguel Hidalgo"


def test_rank_categories_in_borough_and_overall() -> None:
    ranking = analytics.rank_categories(AGG, boroughs=("Benito Juárez",))
    assert [item.key for item in ranking][:2] == ["grocery", "pharmacy"]
    assert ranking[-1].count == 0  # categories absent from the borough are listed with zero
    overall = analytics.rank_categories(AGG, ascending=True)
    assert overall[0].key == "cafe"
    assert overall[0].count == 1


def test_ties_are_broken_alphabetically() -> None:
    ranking = analytics.rank_categories(AGG, boroughs=("Coyoacán",))
    assert ranking[0].key == "bakery" and ranking[1].key == "grocery"  # both 3


def test_compare_boroughs() -> None:
    comparison = analytics.compare_boroughs(AGG, ("Cuauhtémoc", "Miguel Hidalgo"), ("grocery", "pharmacy"))
    assert comparison == {
        "Cuauhtémoc": {"grocery": 4, "pharmacy": 2},
        "Miguel Hidalgo": {"grocery": 2, "pharmacy": 4},
    }


def test_count_matching_with_strata_scans_records() -> None:
    assert analytics.count_matching(AGG, RECORDS, categories=("pharmacy",), strata=("2",)) == 2
    assert analytics.count_matching(AGG, RECORDS, boroughs=("Coyoacán",), strata=("1", "2")) == 6
    assert analytics.count_matching(AGG, RECORDS, boroughs=("Coyoacán",)) == 6  # aggregate path
    assert analytics.count_by_axis(RECORDS, "borough", categories=("pharmacy",), strata=("2",)) == {
        "Benito Juárez": 1,
        "Miguel Hidalgo": 1,
    }


def test_stratum_distribution_aggregate_fast_path_matches_scan() -> None:
    for kwargs in ({"boroughs": ("Coyoacán",)}, {"categories": ("pharmacy",)}, {}):
        assert analytics.count_by_stratum(RECORDS, aggregates=AGG, **kwargs) == analytics.count_by_stratum(
            RECORDS, **kwargs
        )


def test_stratum_distribution_matches_aggregates() -> None:
    counts = analytics.count_by_stratum(RECORDS, categories=("pharmacy",))
    assert sum(counts.values()) == 9
    assert counts == {"1": 7, "2": 2, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0}
    assert counts["1"] == AGG.by_category_stratum["pharmacy"]["1"]
    assert analytics.rank_strata(counts)[0].key == "1"


def test_examples_are_deterministic_and_bounded() -> None:
    first = analytics.example_establishments(RECORDS, ("Iztapalapa",), ("grocery",), limit=4)
    second = analytics.example_establishments(RECORDS, ("Iztapalapa",), ("grocery",), limit=4)
    assert [r.id for r in first] == [r.id for r in second]
    assert len(first) == 4
    assert all(r.borough == "Iztapalapa" and r.category == "grocery" for r in first)
    assert analytics.example_establishments(RECORDS, ("Iztapalapa",), ("cafe",)) == []


@pytest.mark.parametrize(("key", "label"), [("grocery", "Grocery stores"), ("unknown", "unknown")])
def test_category_label(key: str, label: str) -> None:
    assert analytics.category_label(key) == label
