"""Application state construction and FastAPI dependency providers."""

import logging
from dataclasses import dataclass

from fastapi import Request

from app.config import Settings
from app.exceptions import DatasetError, DatasetUnavailableError, RateLimitExceededError
from app.middleware import RateLimiter, resolve_client_ip
from app.services.dataset import Dataset, load_dataset
from app.services.embeddings import EmbeddingProvider, FastEmbedProvider, resolve_cache_dir
from app.services.generation import AnswerGenerator, GeminiGenerator
from app.services.query import QueryService
from app.services.retrieval import RetrievalService
from app.services.status import ProviderStatus

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AppState:
    """Everything built once at startup and shared read-only by requests."""

    settings: Settings
    embedder: EmbeddingProvider
    generator: AnswerGenerator
    generation_status: ProviderStatus
    rate_limiter: RateLimiter
    dataset: Dataset | None
    query_service: QueryService | None

    @property
    def ready(self) -> bool:
        return self.query_service is not None


def default_embedder(settings: Settings) -> EmbeddingProvider:
    return FastEmbedProvider(settings.embedding_model, resolve_cache_dir(settings.fastembed_cache_path))


def default_generator(settings: Settings) -> AnswerGenerator:
    key = settings.gemini_api_key.get_secret_value() if settings.gemini_api_key else ""
    return GeminiGenerator(key, settings.gemini_model, settings.request_timeout_seconds)


def _load_and_verify(settings: Settings, embedder: EmbeddingProvider) -> Dataset:
    dataset = load_dataset(settings.data_dir, embedder.model_name)
    # Loads the model once and proves the runtime model matches the stored vectors.
    probe = embedder.embed_query("startup probe")
    if probe.shape[0] != dataset.manifest.embedding_dimension:
        raise DatasetError(
            f"runtime embedding dimension {probe.shape[0]} does not match "
            f"stored dimension {dataset.manifest.embedding_dimension}"
        )
    return dataset


def build_app_state(
    settings: Settings,
    embedder: EmbeddingProvider | None = None,
    generator: AnswerGenerator | None = None,
) -> AppState:
    embedder = embedder or default_embedder(settings)
    generator = generator or default_generator(settings)
    generation_status = ProviderStatus()
    rate_limiter = RateLimiter(settings.rate_limit_requests, settings.rate_limit_window_seconds)

    dataset: Dataset | None = None
    query_service: QueryService | None = None
    try:
        dataset = _load_and_verify(settings, embedder)
    except DatasetError as exc:
        logger.error("dataset unavailable", extra={"reason": str(exc)})
    except Exception:
        # A failing embedding backend (e.g. no cached model) must not take
        # /health down; the query endpoint returns a controlled 503 instead.
        logger.exception("embedding model failed to load")
    else:
        retrieval = RetrievalService(
            dataset.knowledge, embedder, settings.retrieval_top_k, settings.retrieval_min_score
        )
        query_service = QueryService(
            dataset, retrieval, generator, generation_status, settings.max_question_length
        )

    return AppState(
        settings=settings,
        embedder=embedder,
        generator=generator,
        generation_status=generation_status,
        rate_limiter=rate_limiter,
        dataset=dataset,
        query_service=query_service,
    )


def get_state(request: Request) -> AppState:
    return request.app.state.container


def get_dataset(request: Request) -> Dataset:
    state = get_state(request)
    if state.dataset is None:
        raise DatasetUnavailableError()
    return state.dataset


def get_query_service(request: Request) -> QueryService:
    state = get_state(request)
    if state.query_service is None:
        raise DatasetUnavailableError()
    return state.query_service


def enforce_rate_limit(request: Request) -> None:
    state = get_state(request)
    client_ip = resolve_client_ip(request.scope, state.settings.trust_proxy_headers)
    if not state.rate_limiter.allow(client_ip):
        logger.warning("rate limit exceeded", extra={"path": request.url.path})
        raise RateLimitExceededError()
