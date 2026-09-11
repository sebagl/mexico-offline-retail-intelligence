"""Hybrid query orchestration: parse, calculate, retrieve, explain, cite.

Quantitative answers are computed by Python from the validated dataset
(``analytics``). Semantic retrieval supplies supporting evidence and handles
open questions. Gemini, when configured, only rewrites the deterministic
result as a short explanation; its output is discarded if it introduces any
number that the calculated payload does not contain.
"""

import asyncio
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.catalog import BOROUGHS, CATEGORIES, CATEGORIES_BY_KEY
from app.exceptions import InvalidQuestionError
from app.schemas import (
    AppliedFilters,
    DatasetBasis,
    Evidence,
    QueryResponse,
    ResponseMode,
    ScopeInfo,
    SourceInfo,
    attribution_text,
    normalize_question,
)
from app.services import analytics
from app.services.dataset import Dataset
from app.services.generation import AnswerGenerator, GenerationError
from app.services.question_parser import ParsedQuestion, parse_question
from app.services.retrieval import RetrievalService
from app.services.status import ProviderStatus

logger = logging.getLogger(__name__)

UNSUPPORTED_ANSWER = (
    "I can only answer questions about the configured INEGI DENUE dataset: establishment counts, "
    "percentages, rankings, comparisons, employment-size distributions and examples for the covered "
    "Mexico City boroughs and retail categories. Try one of the suggested questions."
)
INSUFFICIENT_ANSWER = "The configured dataset does not contain enough information to answer that reliably."
EXTRACTIVE_PREFACE = (
    "That question does not map to an exact calculation, so here are the most relevant facts "
    "from the configured dataset:"
)
MAX_EVIDENCE = 3
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


@dataclass(frozen=True, slots=True)
class Analysis:
    """Deterministic result of a parsed question before any explanation."""

    intent: str
    answer: str
    metrics: dict[str, Any] = field(default_factory=dict)
    headline: tuple[float, ...] = ()  # values an explanation must reproduce
    supported: bool = True
    sufficient: bool = True


def fmt(number: int) -> str:
    return f"{number:,}"


def _borough_phrase(boroughs: tuple[str, ...]) -> str:
    if not boroughs:
        return "the five covered boroughs"
    if len(boroughs) == 1:
        return boroughs[0]
    return ", ".join(boroughs[:-1]) + " and " + boroughs[-1]


def _category_phrase(categories: tuple[str, ...]) -> str:
    if not categories:
        return "all covered retail categories"
    labels = [analytics.category_label(key).lower() for key in categories]
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + " and " + labels[-1]


def _strata_phrase(strata: tuple[str, ...]) -> str:
    if not strata:
        return ""
    return " with " + " / ".join(analytics.stratum_label(code) for code in strata) + " employed"


def _ranked_metrics(
    items: list[analytics.RankedItem], label_fn: Callable[[str], str]
) -> list[dict[str, Any]]:
    return [
        {"key": item.key, "label": label_fn(item.key), "count": item.count, "share_pct": item.share}
        for item in items
    ]


def plural(count: int, noun: str = "establishment") -> str:
    return f"{fmt(count)} {noun}{'' if count == 1 else 's'}"


def _describe_ranking(
    items: list[analytics.RankedItem], label_fn: Callable[[str], str], ascending: bool
) -> str:
    if not items:
        return ""
    lead = items[0]
    direction = "has the fewest" if ascending else "has the most"
    text = f"{label_fn(lead.key)} {direction} with {plural(lead.count)} ({lead.share}%)"
    rest = [item for item in items[1:4] if item.count > 0 or ascending]
    if rest:
        text += ", followed by " + ", ".join(
            f"{label_fn(item.key)} ({fmt(item.count)}, {item.share}%)" for item in rest
        )
    return text + "."


class QueryService:
    def __init__(
        self,
        dataset: Dataset,
        retrieval: RetrievalService,
        generator: AnswerGenerator,
        generation_status: ProviderStatus,
        max_question_length: int,
    ) -> None:
        self._dataset = dataset
        self._retrieval = retrieval
        self._generator = generator
        self._generation_status = generation_status
        self._max_question_length = max_question_length

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #

    async def answer(self, raw_question: str) -> QueryResponse:
        question = normalize_question(raw_question)
        if not question:
            raise InvalidQuestionError()
        if len(question) > self._max_question_length:
            raise InvalidQuestionError(
                f"The question exceeds the maximum length of {self._max_question_length} characters."
            )

        started = time.perf_counter()
        parsed = parse_question(question)
        analysis = self._analyze(parsed)
        evidence = await asyncio.to_thread(self._retrieve_evidence, question, parsed)
        retrieval_ms = _elapsed_ms(started)

        mode: ResponseMode
        answer = analysis.answer
        if not analysis.supported:
            if evidence and _grounded_in_dataset(parsed, evidence):
                mode, answer = "extractive", self._extractive_answer(evidence)
            else:
                mode, answer, evidence = "unsupported", UNSUPPORTED_ANSWER, []
        elif not analysis.sufficient:
            mode = "insufficient_data"
        else:
            mode = "deterministic"
            if self._generator.is_configured:
                explanation = await self._explain(question, parsed, analysis, evidence)
                if explanation is not None:
                    mode, answer = "generated", explanation

        response = self._build_response(parsed, analysis, mode, answer, evidence)
        logger.info(
            "query answered",
            extra={
                "mode": mode,
                "intent": analysis.intent,
                "question_length": len(question),
                "boroughs": len(parsed.boroughs),
                "categories": len(parsed.categories),
                "evidence": len(evidence),
                "retrieval_ms": retrieval_ms,
                "total_ms": _elapsed_ms(started),
            },
        )
        return response

    # ------------------------------------------------------------------ #
    # Deterministic analysis
    # ------------------------------------------------------------------ #

    def _analyze(self, parsed: ParsedQuestion) -> Analysis:
        handlers = {
            "count": self._count,
            "percentage": self._percentage,
            "ranking": self._ranking,
            "comparison": self._comparison,
            "distribution": self._distribution,
            "examples": self._examples,
        }
        handler = handlers.get(parsed.intent)
        if handler is None:
            return Analysis(intent="unknown", answer="", supported=False)
        return handler(parsed)

    def _count(self, parsed: ParsedQuestion) -> Analysis:
        aggregates = self._dataset.aggregates
        total = analytics.count_matching(
            aggregates, self._dataset.establishments, parsed.boroughs, parsed.categories, parsed.strata
        )
        answer = (
            f"Within the configured dataset, {_borough_phrase(parsed.boroughs)} "
            f"contain{'s' if len(parsed.boroughs) == 1 else ''} {plural(total)} "
            f"classified as {_category_phrase(parsed.categories)}{_strata_phrase(parsed.strata)}."
        )
        return Analysis(
            intent="count",
            answer=answer,
            metrics={"count": total, "dataset_total": aggregates.total},
            headline=(total,),
        )

    def _percentage(self, parsed: ParsedQuestion) -> Analysis:
        aggregates = self._dataset.aggregates
        records = self._dataset.establishments
        if not parsed.boroughs and not parsed.categories and not parsed.strata:
            return Analysis(
                intent="percentage",
                answer=(
                    "To compute a percentage, name a covered borough, a retail category "
                    "or an employment range."
                ),
                sufficient=False,
            )
        part = analytics.count_matching(
            aggregates, records, parsed.boroughs, parsed.categories, parsed.strata
        )
        if parsed.strata:
            # Share of the named employment range within the borough/category selection.
            whole = analytics.count_matching(aggregates, records, parsed.boroughs, parsed.categories)
            whole_phrase = f"{_category_phrase(parsed.categories)} in {_borough_phrase(parsed.boroughs)}"
            subject = f"establishments{_strata_phrase(parsed.strata)}"
        elif parsed.boroughs and parsed.categories:
            whole = analytics.count_establishments(aggregates, parsed.boroughs)
            whole_phrase = f"all covered establishments in {_borough_phrase(parsed.boroughs)}"
            subject = _category_phrase(parsed.categories)
        elif parsed.categories:
            whole = aggregates.total
            whole_phrase = "all establishments in the configured dataset"
            subject = _category_phrase(parsed.categories)
        else:
            whole = aggregates.total
            whole_phrase = "all establishments in the configured dataset"
            subject = f"establishments located in {_borough_phrase(parsed.boroughs)}"
        share = analytics.percentage(part, whole)
        answer = (
            f"{subject[0].upper() + subject[1:]} account for {share}% of {whole_phrase}: "
            f"{fmt(part)} of {fmt(whole)} establishments."
        )
        return Analysis(
            intent="percentage",
            answer=answer,
            metrics={"part": part, "whole": whole, "percentage": share},
            headline=(share,),
        )

    def _ranking(self, parsed: ParsedQuestion) -> Analysis:
        aggregates = self._dataset.aggregates
        records = self._dataset.establishments
        axis = parsed.rank_axis
        if axis == "auto":
            axis = "borough" if parsed.categories and not parsed.boroughs else "category"

        label_fn: Callable[[str], str]
        if axis == "stratum":
            counts = analytics.count_by_stratum(records, parsed.boroughs, parsed.categories, aggregates)
            items = analytics.rank_strata(counts, parsed.ascending)
            label_fn = analytics.stratum_label
            subject = "Employment-size ranges"
            scope = f"among {_category_phrase(parsed.categories)} in {_borough_phrase(parsed.boroughs)}"
        elif axis == "borough":
            if parsed.strata:
                counts = analytics.count_by_axis(records, "borough", (), parsed.categories, parsed.strata)
                items = analytics.rank_strata(
                    {b.name: counts.get(b.name, 0) for b in BOROUGHS}, parsed.ascending
                )
            else:
                items = analytics.rank_boroughs(aggregates, parsed.categories, parsed.ascending)
            label_fn = str
            subject = "Covered boroughs"
            scope = (
                f"by number of establishments classified as {_category_phrase(parsed.categories)}"
                f"{_strata_phrase(parsed.strata)}"
            )
        else:
            if parsed.strata:
                counts = analytics.count_by_axis(records, "category", parsed.boroughs, (), parsed.strata)
                items = analytics.rank_strata(
                    {c.key: counts.get(c.key, 0) for c in CATEGORIES}, parsed.ascending
                )
            else:
                items = analytics.rank_categories(aggregates, parsed.boroughs, parsed.ascending)
            label_fn = analytics.category_label
            subject = "Retail categories"
            scope = f"in {_borough_phrase(parsed.boroughs)}{_strata_phrase(parsed.strata)}"

        total = sum(item.count for item in items)
        if total == 0:
            return Analysis(intent="ranking", answer=INSUFFICIENT_ANSWER, sufficient=False)
        order = "lowest to highest" if parsed.ascending else "highest to lowest"
        answer = (
            f"{subject} ranked {scope}, {order}, within the configured dataset "
            f"({fmt(total)} establishments): " + _describe_ranking(items, label_fn, parsed.ascending)
        )
        return Analysis(
            intent="ranking",
            answer=answer,
            metrics={
                "axis": axis,
                "order": order,
                "total": total,
                "ranking": _ranked_metrics(items, label_fn),
            },
            headline=(items[0].count,),
        )

    def _comparison(self, parsed: ParsedQuestion) -> Analysis:
        aggregates = self._dataset.aggregates
        records = self._dataset.establishments
        if len(parsed.boroughs) < 2:
            if len(parsed.categories) >= 2:
                counts = {
                    key: analytics.count_matching(aggregates, records, parsed.boroughs, (key,), parsed.strata)
                    for key in parsed.categories
                }
                total = sum(counts.values())
                parts = ", ".join(
                    f"{analytics.category_label(key).lower()}: {fmt(value)} "
                    f"({analytics.percentage(value, total)}%)"
                    for key, value in counts.items()
                )
                answer = (
                    f"In {_borough_phrase(parsed.boroughs)}{_strata_phrase(parsed.strata)}, the "
                    f"configured dataset contains {parts} (shares are relative to the {fmt(total)} "
                    "establishments in these categories)."
                )
                return Analysis(
                    intent="comparison",
                    answer=answer,
                    metrics={"by_category": counts, "total": total},
                    headline=tuple(counts.values()),
                )
            return Analysis(
                intent="comparison",
                answer=(
                    "To compare, name two covered boroughs (for example Cuauhtémoc and Miguel Hidalgo) "
                    "or two retail categories."
                ),
                sufficient=False,
            )

        keys = list(parsed.categories) or list(aggregates.by_category)
        sentences = []
        metrics: dict[str, Any] = {}
        headline: list[float] = []
        for borough in parsed.boroughs:
            if parsed.strata:
                by_category = analytics.count_by_axis(records, "category", (borough,), keys, parsed.strata)
                counts = {key: by_category.get(key, 0) for key in keys}
            else:
                counts = analytics.compare_boroughs(aggregates, (borough,), parsed.categories)[borough]
            total = sum(counts.values())
            top = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:3]
            top_text = ", ".join(
                f"{analytics.category_label(key).lower()} {fmt(value)} "
                f"({analytics.percentage(value, total)}%)"
                for key, value in top
            )
            sentences.append(f"{borough} has {plural(total)}; largest categories: {top_text}")
            metrics[borough] = {
                "total": total,
                "by_category": counts,
                "share_pct": {key: analytics.percentage(value, total) for key, value in counts.items()},
            }
            headline.append(total)
        answer = (
            f"Comparison within the configured dataset{_strata_phrase(parsed.strata)}. "
            + ". ".join(sentences)
            + "."
        )
        return Analysis(intent="comparison", answer=answer, metrics=metrics, headline=tuple(headline))

    def _distribution(self, parsed: ParsedQuestion) -> Analysis:
        counts = analytics.count_by_stratum(
            self._dataset.establishments, parsed.boroughs, parsed.categories, self._dataset.aggregates
        )
        total = sum(counts.values())
        if total == 0:
            return Analysis(intent="distribution", answer=INSUFFICIENT_ANSWER, sufficient=False)
        items = analytics.rank_strata(counts)
        listing = "; ".join(
            f"{analytics.stratum_label(item.key)}: {fmt(item.count)} ({item.share}%)"
            for item in items
            if item.count
        )
        answer = (
            f"Employment-size distribution of {_category_phrase(parsed.categories)} in "
            f"{_borough_phrase(parsed.boroughs)} ({fmt(total)} establishments): {listing}. "
            f"The most common range is {analytics.stratum_label(items[0].key)}."
        )
        return Analysis(
            intent="distribution",
            answer=answer,
            metrics={"total": total, "by_stratum": _ranked_metrics(items, analytics.stratum_label)},
            headline=(total, items[0].count),
        )

    def _examples(self, parsed: ParsedQuestion) -> Analysis:
        records = analytics.example_establishments(
            self._dataset.establishments, parsed.boroughs, parsed.categories, parsed.strata
        )
        matching = analytics.count_matching(
            self._dataset.aggregates,
            self._dataset.establishments,
            parsed.boroughs,
            parsed.categories,
            parsed.strata,
        )
        if not records:
            return Analysis(intent="examples", answer=INSUFFICIENT_ANSWER, sufficient=False)
        lines = [
            f"• {record.name} — {record.activity_label}; {record.locality or 'locality not reported'}, "
            f"{record.borough}; {record.stratum_label}"
            for record in records
        ]
        answer = (
            f"Examples of {_category_phrase(parsed.categories)} in {_borough_phrase(parsed.boroughs)}"
            f"{_strata_phrase(parsed.strata)} (a deterministic sample of {len(records)} of "
            f"{fmt(matching)} matching establishments, sorted by name):\n" + "\n".join(lines)
        )
        return Analysis(
            intent="examples",
            answer=answer,
            headline=(matching,),
            metrics={
                "matching": matching,
                "sample_size": len(records),
                "examples": [
                    {
                        "id": r.id,
                        "name": r.name,
                        "activity_label": r.activity_label,
                        "locality": r.locality,
                        "borough": r.borough,
                        "stratum_label": r.stratum_label,
                    }
                    for r in records
                ],
            },
        )

    # ------------------------------------------------------------------ #
    # Evidence, explanation and response assembly
    # ------------------------------------------------------------------ #

    def _retrieve_evidence(self, question: str, parsed: ParsedQuestion) -> list[Evidence]:
        # Knowledge documents are English; appending the canonical English names of the
        # detected boroughs/categories lets Spanish questions reach the same documents.
        hints = [*parsed.boroughs, *(analytics.category_label(k) for k in parsed.categories)]
        query = question if not hints else f"{question} ({', '.join(hints)})"
        documents = self._retrieval.retrieve(query, parsed.boroughs, parsed.categories)
        return [
            Evidence(
                kind=d.kind, text=d.text, score=round(d.score, 3), borough=d.borough, category=d.category
            )
            for d in documents[:MAX_EVIDENCE]
        ]

    @staticmethod
    def _extractive_answer(evidence: list[Evidence]) -> str:
        return EXTRACTIVE_PREFACE + "\n" + "\n".join(f"• {item.text}" for item in evidence)

    def _payload(self, parsed: ParsedQuestion, analysis: Analysis) -> dict[str, Any]:
        manifest = self._dataset.manifest
        return {
            "intent": analysis.intent,
            "filters": {
                "boroughs": list(parsed.boroughs),
                "categories": [analytics.category_label(k) for k in parsed.categories],
                "employment_ranges": [analytics.stratum_label(s) for s in parsed.strata],
            },
            "metrics": analysis.metrics,
            "deterministic_answer": analysis.answer,
            "scope": {
                "complete": manifest.complete,
                "retrieved_at": manifest.retrieved_at,
                "boroughs_covered": [b.name for b in BOROUGHS],
                "categories_covered": [c.label_en for c in CATEGORIES],
            },
        }

    async def _explain(
        self, question: str, parsed: ParsedQuestion, analysis: Analysis, evidence: list[Evidence]
    ) -> str | None:
        payload = self._payload(parsed, analysis)
        started = time.perf_counter()
        try:
            text = await self._generator.explain(question, payload)
        except GenerationError as exc:
            self._generation_status.record_failure(exc.category)
            logger.warning(
                "explanation failed; using deterministic answer",
                extra={"provider": self._generator.provider_name, "category": exc.category},
            )
            return None
        problem = explanation_problem(text, analysis)
        if problem is not None:
            self._generation_status.record_failure(f"explanation_{problem}")
            logger.warning("explanation discarded", extra={"reason": problem})
            return None
        self._generation_status.record_success()
        logger.info("explanation generated", extra={"generation_ms": _elapsed_ms(started)})
        return text

    def _build_response(
        self,
        parsed: ParsedQuestion,
        analysis: Analysis,
        mode: ResponseMode,
        answer: str,
        evidence: list[Evidence],
    ) -> QueryResponse:
        manifest = self._dataset.manifest
        filtered = bool(parsed.boroughs or parsed.categories or parsed.strata)
        basis: DatasetBasis
        if not manifest.complete:
            basis = "partial_dataset"
        elif filtered:
            basis = "filtered_subset"
        else:
            basis = "complete_configured_dataset"
        boroughs = list(parsed.boroughs) or [b.name for b in BOROUGHS]
        categories = [analytics.category_label(k) for k in parsed.categories] or [
            c.label_en for c in CATEGORIES
        ]
        methodology = (
            "Quantitative values were computed by this application in Python from "
            f"{fmt(manifest.record_count)} validated DENUE establishment records retrieved on "
            f"{manifest.retrieved_at[:10]}. "
            + (
                "The dataset is a partial retrieval; see /api/source for failed scopes. "
                if not manifest.complete
                else ""
            )
            + (
                "The language model only rephrased the calculated result; it did not compute any value."
                if mode == "generated"
                else "No language model was involved in this answer."
            )
        )
        return QueryResponse(
            answer=answer,
            mode=mode,
            intent=analysis.intent,
            metrics=analysis.metrics if mode in {"deterministic", "generated"} else {},
            filters=AppliedFilters(
                boroughs=list(parsed.boroughs),
                categories=[CATEGORIES_BY_KEY[k].label_en for k in parsed.categories],
                strata=[analytics.stratum_label(s) for s in parsed.strata],
            ),
            evidence=evidence,
            source=SourceInfo(),
            scope=ScopeInfo(
                boroughs=boroughs,
                categories=categories,
                complete=manifest.complete,
                basis=basis,
                retrieved_at=manifest.retrieved_at,
            ),
            methodology=methodology,
            attribution=attribution_text(manifest.retrieved_at[:10]),
        )


_URL_OR_DOMAIN = re.compile(r"https?://|www\.|\b[a-z0-9-]+\.(?:com|org|net|mx|io|gob|edu)\b", re.I)
_NUMBER_WORDS = re.compile(
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|"
    r"sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
    r"hundred|thousand|million|dozen|half|quarter|third|"
    r"uno|una|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce|veinte|treinta|cuarenta|"
    r"cincuenta|cien|ciento|mil|millon|millones|docena|mitad|tercio|cuarto)\b",
    re.I,
)
_ENDORSEMENT = re.compile(r"official|endorse|approved|certified|avalad|oficial", re.I)
MAX_EXPLANATION_CHARS = 900


def _number_forms(value: float) -> set[str]:
    """Every textual form a calculated value may legitimately take in prose."""
    if float(value).is_integer():
        integer = int(value)
        grouped = f"{integer:,}"
        return {str(integer), grouped, grouped.replace(",", "."), grouped.replace(",", " ")}
    text = f"{value:.1f}".rstrip("0").rstrip(".")
    return {text, text.replace(".", ",")}


def _digits(text: str) -> str:
    return re.sub(r"[^0-9]", "", text)


def explanation_problem(text: str, analysis: Analysis) -> str | None:
    """Return why a generated explanation must be discarded, or ``None`` if it is faithful.

    Only calculated values (``metrics`` and the deterministic answer) may appear as
    numbers; every headline value must appear; no spelled-out numbers, URLs or
    endorsement language; bounded length. Timestamps and catalog labels are
    deliberately not part of the allowed set.
    """
    if len(text) > MAX_EXPLANATION_CHARS:
        return "too_long"
    if _URL_OR_DOMAIN.search(text):
        return "contains_url"
    if _NUMBER_WORDS.search(text):
        return "spelled_out_number"
    if _ENDORSEMENT.search(text):
        return "endorsement_language"

    allowed: set[str] = set()
    for match in _NUMBER.findall(_flatten(analysis.metrics) + " " + analysis.answer):
        allowed |= {_digits(form) for form in _number_forms(float(match.replace(",", "")))}
    for match in _NUMBER.findall(text):
        if _digits(match) not in allowed:
            return "foreign_number"
    for value in analysis.headline:
        if not any(form in text for form in _number_forms(value)):
            return "headline_missing"
    return None


def _flatten(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(f"{k} {_flatten(v)}" for k, v in value.items())
    if isinstance(value, list | tuple):
        return " ".join(_flatten(v) for v in value)
    return str(value)


def _grounded_in_dataset(parsed: ParsedQuestion, evidence: list[Evidence]) -> bool:
    """Extractive answers need an anchor beyond similarity: a catalog entity in the
    question, or an aggregate document as the strongest hit. A small embedding
    model scores unrelated text too close to relevant text to trust the score alone."""
    if parsed.boroughs or parsed.categories:
        return True
    return evidence[0].kind == "aggregate"


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
