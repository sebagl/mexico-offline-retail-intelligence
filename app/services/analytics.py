"""Deterministic retail analytics over the DENUE dataset.

Every quantitative result the application reports is produced here with plain
Python arithmetic over the validated records and precomputed aggregates. The
language model never calculates anything.
"""

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from app.catalog import CATEGORIES_BY_KEY, EMPLOYMENT_STRATA
from app.schemas import Aggregates, EstablishmentRecord
from app.services.dataset import Establishment

MAX_EXAMPLES = 8
# Ingestion works with validated pydantic records; the runtime uses compact rows.
EstablishmentLike = EstablishmentRecord | Establishment


@dataclass(frozen=True, slots=True)
class RankedItem:
    key: str
    count: int
    share: float  # percentage of the ranking's total, rounded to one decimal


def percentage(part: int, whole: int) -> float:
    """Percentage rounded to one decimal; 0.0 when the whole is zero."""
    if whole <= 0:
        return 0.0
    return round(part * 100.0 / whole, 1)


def _rank(counts: dict[str, int], ascending: bool) -> list[RankedItem]:
    total = sum(counts.values())
    ordered = sorted(counts.items(), key=lambda item: (item[1] if ascending else -item[1], item[0]))
    return [RankedItem(key, count, percentage(count, total)) for key, count in ordered]


def count_establishments(
    aggregates: Aggregates,
    boroughs: Sequence[str] = (),
    categories: Sequence[str] = (),
) -> int:
    """Exact count for any combination of boroughs and categories."""
    if not boroughs and not categories:
        return aggregates.total
    if boroughs and not categories:
        return sum(aggregates.by_borough.get(b, 0) for b in boroughs)
    if categories and not boroughs:
        return sum(aggregates.by_category.get(c, 0) for c in categories)
    return sum(aggregates.by_borough_category.get(b, {}).get(c, 0) for b in boroughs for c in categories)


def count_by_stratum(
    records: Iterable[EstablishmentLike],
    boroughs: Sequence[str] = (),
    categories: Sequence[str] = (),
) -> dict[str, int]:
    """Employment-size distribution for an arbitrary filter, scanning records."""
    borough_set = set(boroughs)
    category_set = set(categories)
    counts: Counter[str] = Counter()
    for record in records:
        if borough_set and record.borough not in borough_set:
            continue
        if category_set and record.category not in category_set:
            continue
        counts[record.stratum] += 1
    return {code: counts.get(code, 0) for code in EMPLOYMENT_STRATA}


def rank_boroughs(
    aggregates: Aggregates, categories: Sequence[str] = (), ascending: bool = False
) -> list[RankedItem]:
    """Boroughs ordered by establishment count (optionally within categories)."""
    if categories:
        counts = {
            borough: sum(per_category.get(c, 0) for c in categories)
            for borough, per_category in aggregates.by_borough_category.items()
        }
    else:
        counts = dict(aggregates.by_borough)
    return _rank(counts, ascending)


def rank_categories(
    aggregates: Aggregates, boroughs: Sequence[str] = (), ascending: bool = False
) -> list[RankedItem]:
    """Categories ordered by establishment count (optionally within boroughs)."""
    if boroughs:
        counts: Counter[str] = Counter()
        for borough in boroughs:
            counts.update(aggregates.by_borough_category.get(borough, {}))
        counts_dict = {key: counts.get(key, 0) for key in aggregates.by_category}
    else:
        counts_dict = dict(aggregates.by_category)
    return _rank(counts_dict, ascending)


def rank_strata(counts: dict[str, int], ascending: bool = False) -> list[RankedItem]:
    return _rank(counts, ascending)


def compare_boroughs(
    aggregates: Aggregates, boroughs: Sequence[str], categories: Sequence[str] = ()
) -> dict[str, dict[str, int]]:
    """Per-borough category counts for a side-by-side comparison."""
    keys = list(categories) if categories else list(aggregates.by_category)
    return {
        borough: {key: aggregates.by_borough_category.get(borough, {}).get(key, 0) for key in keys}
        for borough in boroughs
    }


def example_establishments(
    records: Iterable[EstablishmentLike],
    boroughs: Sequence[str] = (),
    categories: Sequence[str] = (),
    limit: int = MAX_EXAMPLES,
) -> list[EstablishmentLike]:
    """A deterministic sample: the first ``limit`` matches by name then id."""
    borough_set = set(boroughs)
    category_set = set(categories)
    matches = [
        record
        for record in records
        if (not borough_set or record.borough in borough_set)
        and (not category_set or record.category in category_set)
    ]
    matches.sort(key=lambda record: (record.name.casefold(), record.id))
    return matches[:limit]


def category_label(key: str) -> str:
    category = CATEGORIES_BY_KEY.get(key)
    return category.label_en if category else key


def stratum_label(code: str) -> str:
    return EMPLOYMENT_STRATA.get(code, code)
