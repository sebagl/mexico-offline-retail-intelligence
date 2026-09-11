"""Rule-based parsing of natural-language questions into filters and intent.

Matching is accent- and case-insensitive and works for English and Spanish
phrasing. The parser is deliberately conservative: when no quantitative
intent is recognised the question is handed to semantic retrieval instead of
guessing a calculation.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Literal

from app.catalog import BOROUGHS, CATEGORIES, EMPLOYMENT_STRATA

Intent = Literal[
    "count",
    "percentage",
    "ranking",
    "comparison",
    "distribution",
    "examples",
    "unknown",
]


@dataclass(frozen=True, slots=True)
class ParsedQuestion:
    normalized: str
    intent: Intent
    boroughs: tuple[str, ...] = ()  # canonical borough names
    categories: tuple[str, ...] = ()  # category keys
    strata: tuple[str, ...] = ()  # stratum codes
    ascending: bool = False  # ranking direction: lowest/fewest first
    rank_axis: Literal["borough", "category", "stratum", "auto"] = field(default="auto")


def fold(text: str) -> str:
    """Lower-case, strip accents and collapse non-alphanumerics to spaces."""
    stripped = "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9%]+", " ", stripped.lower()).strip()


def _contains_phrase(text: str, phrase: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None


_PERCENT = ("%", "percent", "percentage", "porcentaje", "proportion", "proporcion", "share of")
_COMPARE = (
    "compare",
    "comparison",
    "versus",
    "vs",
    "compara",
    "comparar",
    "comparacion",
    "difference between",
)
_EXAMPLES = (
    "example",
    "examples",
    "ejemplo",
    "ejemplos",
    "show me",
    "list some",
    "name some",
    "muestra",
    "muestrame",
    "lista",
)
_EMPLOYMENT = (
    "employment",
    "employee",
    "employees",
    "size segment",
    "size range",
    "employment size",
    "stratum",
    "strata",
    "empleados",
    "tamano",
    "personal ocupado",
    "estrato",
)
# Category composition of an area ("mix", "breakdown") is a ranking of categories,
# not an employment distribution.
_COMPOSITION = ("distribution", "breakdown", "composition", "distribucion", "composicion", "mix")
_CONJUNCTION = ("and", "or", "y", "o", "vs")
_RANK_HIGH = (
    "most",
    "highest",
    "largest",
    "top",
    "biggest",
    "greatest",
    "mas",
    "mayor",
    "principal",
    "common",
    "comunes",
    "leading",
)
_RANK_LOW = (
    "lowest",
    "least",
    "fewest",
    "smallest",
    "rarest",
    "menos",
    "menor",
    "less common",
    "least common",
    "underrepresented",
    "lowest representation",
)
_COUNT = (
    "how many",
    "number of",
    "count",
    "total",
    "cuantos",
    "cuantas",
    "numero de",
    "cantidad",
    "included in",
    "represented in",
)
_BOROUGH_WORDS = (
    "borough",
    "boroughs",
    "alcaldia",
    "alcaldias",
    "municipio",
    "municipios",
    "delegacion",
    "area",
    "areas",
    "zone",
    "zones",
)
_CATEGORY_WORDS = (
    "category",
    "categories",
    "categoria",
    "categorias",
    "type",
    "types",
    "tipo",
    "tipos",
    "kind",
    "kinds",
    "giro",
    "giros",
    "retail categories",
    "sector",
    "sectors",
)


def _detect_boroughs(text: str) -> tuple[str, ...]:
    found: list[tuple[int, str]] = []
    for borough in BOROUGHS:
        for alias in (borough.name, *borough.aliases):
            match = re.search(rf"(?<![a-z0-9]){re.escape(fold(alias))}(?![a-z0-9])", text)
            if match:
                found.append((match.start(), borough.name))
                break
    ordered = [name for _, name in sorted(found)]
    return tuple(dict.fromkeys(ordered))


def _detect_categories(text: str) -> tuple[str, ...]:
    found: list[tuple[int, str]] = []
    for category in CATEGORIES:
        for alias in (category.label_en, category.label_es, *category.aliases):
            match = re.search(rf"(?<![a-z0-9]){re.escape(fold(alias))}(?![a-z0-9])", text)
            if match:
                found.append((match.start(), category.key))
                break
    ordered = [key for _, key in sorted(found)]
    return tuple(dict.fromkeys(ordered))


def _detect_strata(text: str) -> tuple[str, ...]:
    found: list[str] = []
    for code, label in EMPLOYMENT_STRATA.items():
        folded = fold(label)  # e.g. "0 a 5 personas"
        compact = folded.replace(" personas", "")
        variants = {folded, compact, compact.replace(" a ", " to "), compact.replace(" a ", " ")}
        if code == "7":
            variants |= {"251 y mas", "251 or more", "251 and more", "more than 250", "251"}
        if any(_contains_phrase(text, variant) for variant in variants):
            found.append(code)
    return tuple(found)


def _has_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(_contains_phrase(text, fold(phrase)) if phrase != "%" else "%" in text for phrase in phrases)


def _detect_intent(
    text: str, boroughs: tuple[str, ...], categories: tuple[str, ...], strata: tuple[str, ...]
) -> tuple[Intent, bool]:
    """Return (intent, ascending).

    Order encodes precedence: explicit cues first, implicit structure (two
    boroughs joined by a conjunction) last, so "how many X in A and B" is a
    count rather than a comparison.
    """
    ascending = _has_any(text, _RANK_LOW)
    if _has_any(text, _PERCENT):
        return "percentage", ascending
    if _has_any(text, _COMPARE):
        return "comparison", ascending
    if _has_any(text, _EXAMPLES):
        return "examples", ascending
    if _has_any(text, _RANK_HIGH) or ascending:
        return "ranking", ascending
    if _has_any(text, _COUNT):
        return "count", ascending
    if strata:
        # "restaurants with 6 to 10 employees in Coyoacán" is a filtered count.
        return "count", ascending
    if _has_any(text, _EMPLOYMENT):
        return "distribution", ascending
    if _has_any(text, _COMPOSITION):
        return "ranking", ascending
    if len(boroughs) >= 2 and _has_any(text, _CONJUNCTION):
        return "comparison", ascending
    if boroughs and _has_any(text, _CATEGORY_WORDS):
        return "ranking", ascending  # "what types of businesses are in Coyoacán?"
    return "unknown", ascending


def _detect_rank_axis(
    text: str, boroughs: tuple[str, ...], categories: tuple[str, ...]
) -> Literal["borough", "category", "stratum", "auto"]:
    if _has_any(text, _EMPLOYMENT):
        return "stratum"
    if _has_any(text, _BOROUGH_WORDS):
        return "borough"
    if _has_any(text, _CATEGORY_WORDS):
        return "category"
    if categories and not boroughs:
        return "borough"
    if boroughs and not categories:
        return "category"
    return "auto"


def parse_question(question: str) -> ParsedQuestion:
    text = fold(question)
    boroughs = _detect_boroughs(text)
    categories = _detect_categories(text)
    strata = _detect_strata(text)
    intent, ascending = _detect_intent(text, boroughs, categories, strata)
    axis = _detect_rank_axis(text, boroughs, categories)
    if intent == "ranking" and axis == "stratum" and strata:
        # Ranking size ranges while filtering to one range is meaningless; count instead.
        intent = "count"
    return ParsedQuestion(
        normalized=text,
        intent=intent,
        boroughs=boroughs,
        categories=categories,
        strata=strata,
        ascending=ascending,
        rank_axis=axis,
    )
