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


def matches(
    record: EstablishmentLike,
    boroughs: Sequence[str] = (),
    categories: Sequence[str] = (),
    strata: Sequence[str] = (),
) -> bool:
    return (
        (not boroughs or record.borough in boroughs)
        and (not categories or record.category in categories)
        and (not strata or record.stratum in strata)
    )


def count_matching(
    aggregates: Aggregates,
    records: Iterable[EstablishmentLike],
    boroughs: Sequence[str] = (),
    categories: Sequence[str] = (),
    strata: Sequence[str] = (),
) -> int:
    """Exact count for any filter; uses precomputed aggregates unless strata are involved."""
    if not strata:
        return count_establishments(aggregates, boroughs, categories)
    return sum(1 for record in records if matches(record, boroughs, categories, strata))


def count_by_stratum(
    records: Iterable[EstablishmentLike],
    boroughs: Sequence[str] = (),
    categories: Sequence[str] = (),
    aggregates: Aggregates | None = None,
) -> dict[str, int]:
    """Employment-size distribution for a filter.

    Single-dimension filters are served from the precomputed pairwise
    aggregates; only borough x category combinations scan the records.
    """
    counts: Counter[str] = Counter()
    if aggregates is not None and not (boroughs and categories):
        if boroughs:
            for borough in boroughs:
                counts.update(aggregates.by_borough_stratum.get(borough, {}))
        elif categories:
            for category in categories:
                counts.update(aggregates.by_category_stratum.get(category, {}))
        else:
            counts.update(aggregates.by_stratum)
    else:
        for record in records:
            if matches(record, boroughs, categories):
                counts[record.stratum] += 1
    return {code: counts.get(code, 0) for code in EMPLOYMENT_STRATA}


def count_by_axis(
    records: Iterable[EstablishmentLike],
    axis: str,
    boroughs: Sequence[str] = (),
    categories: Sequence[str] = (),
    strata: Sequence[str] = (),
) -> dict[str, int]:
    """Counts grouped by ``borough`` or ``category`` under an arbitrary filter (record scan)."""
    counts: Counter[str] = Counter()
    for record in records:
        if matches(record, boroughs, categories, strata):
            counts[getattr(record, axis)] += 1
    return dict(counts)


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
    strata: Sequence[str] = (),
    limit: int = MAX_EXAMPLES,
) -> list[EstablishmentLike]:
    """A deterministic sample: the first ``limit`` matches by name then id."""
    selected = [record for record in records if matches(record, boroughs, categories, strata)]
    selected.sort(key=lambda record: (record.name.casefold(), record.id))
    return selected[:limit]


def category_label(key: str) -> str:
    category = CATEGORIES_BY_KEY.get(key)
    return category.label_en if category else key


def stratum_label(code: str) -> str:
    return EMPLOYMENT_STRATA.get(code, code)
