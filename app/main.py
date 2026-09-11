"""FastAPI application factory and HTTP routes."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import __version__
from app.catalog import CATEGORIES_BY_KEY
from app.config import Settings, get_settings
from app.dependencies import (
    AppState,
    build_app_state,
    enforce_rate_limit,
    get_dataset,
    get_query_service,
    get_state,
)
from app.exceptions import AppError
from app.logging_config import configure_logging
from app.middleware import BodySizeLimitMiddleware, RequestContextMiddleware
from app.schemas import (
    TRANSFORMATION_NOTICE,
    BoroughSummary,
    CategorySummary,
    ErrorResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SourceResponse,
    SummaryResponse,
    attribution_text,
)
from app.services.dataset import Dataset
from app.services.embeddings import EmbeddingProvider
from app.services.generation import AnswerGenerator
from app.services.query import QueryService

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
ERROR_RESPONSES = {
    400: {"model": ErrorResponse},
    413: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
    429: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
}


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": {"code": code, "message": message}})


def create_app(
    settings: Settings | None = None,
    embedder: EmbeddingProvider | None = None,
    generator: AnswerGenerator | None = None,
) -> FastAPI:
    """Build the application. Overrides exist so tests can inject fakes."""
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logger.info(
            "application starting",
            extra={
                "version": __version__,
                "embedding_model": settings.embedding_model,
                "generation_configured": settings.gemini_configured,
                "retrieval_top_k": settings.retrieval_top_k,
                "retrieval_min_score": settings.retrieval_min_score,
            },
        )
        app.state.container = build_app_state(settings, embedder=embedder, generator=generator)
        yield

    app = FastAPI(
        title="Mexico Offline Retail Intelligence",
        version=__version__,
        description=(
            "Independent demonstration analyzing open INEGI DENUE establishment data for selected "
            "Mexico City boroughs. " + TRANSFORMATION_NOTICE
        ),
        lifespan=lifespan,
        # The interactive docs load scripts from a CDN that the CSP blocks; the
        # API surface is small and documented in the README instead.
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    # Middleware order: the last one added runs outermost.
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_request_body_bytes)
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origin_list,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        expose_headers=["X-Request-ID"],
    )

    _register_error_handlers(app)
    _register_routes(app)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def _register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        return _error_response(exc.status_code, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        location = ".".join(
            str(part) for part in first.get("loc", ()) if isinstance(part, str) and part != "body"
        )
        detail = first.get("msg", "Invalid request.")
        message = f"{location}: {detail}" if location else detail
        return _error_response(422, "validation_error", message)

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = "not_found" if exc.status_code == 404 else "http_error"
        return _error_response(exc.status_code, code, str(exc.detail))


def _register_routes(app: FastAPI) -> None:
    # HEAD is included because Render probes the root with HEAD before marking the deploy live.
    @app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    @app.get(
        "/health",
        response_model=HealthResponse,
        responses={503: {"model": HealthResponse}},
        tags=["operations"],
    )
    async def health(state: AppState = Depends(get_state)) -> JSONResponse:
        """``ok``: dataset loaded and no recent explanation failure (Gemini is optional, so
        running without it is still ``ok``). ``degraded``: dataset loaded but the configured
        Gemini failed recently. ``unavailable`` (HTTP 503): the dataset did not load, so
        platform health checks keep the deploy out of rotation."""
        generation_configured = state.generator.is_configured
        if not state.ready:
            status = "unavailable"
        elif generation_configured and state.generation_status.recent_failure is not None:
            status = "degraded"
        else:
            status = "ok"
        manifest = state.dataset.manifest if state.dataset else None
        body = HealthResponse(
            status=status,
            dataset_loaded=state.dataset is not None,
            establishments=state.dataset.record_count if state.dataset else 0,
            boroughs=manifest.borough_count if manifest else 0,
            categories=manifest.category_count if manifest else 0,
            dataset_complete=manifest.complete if manifest else False,
            embedding_provider=state.embedder.provider_name,
            embedding_model=state.settings.embedding_model,
            generation_provider=state.generator.provider_name if generation_configured else "none",
            generation_configured=generation_configured,
            generation_last_failure=state.generation_status.recent_failure,
            fallback_available=state.ready,
        )
        return JSONResponse(
            status_code=503 if status == "unavailable" else 200, content=body.model_dump(mode="json")
        )

    @app.get(
        "/api/source",
        response_model=SourceResponse,
        responses={503: {"model": ErrorResponse}},
        tags=["dataset"],
    )
    async def source(dataset: Dataset = Depends(get_dataset)) -> SourceResponse:
        manifest = dataset.manifest
        return SourceResponse(
            source_name=manifest.source_name,
            source_url=manifest.source_url,
            attribution=manifest.attribution,
            retrieved_at=manifest.retrieved_at,
            dataset_update_date=manifest.dataset_update_date,
            geographic_scope=manifest.geographic_scope,
            economic_scope=manifest.economic_scope,
            complete=manifest.complete,
            completeness_note=manifest.completeness_note,
            failed_scopes=manifest.failed_scopes,
            transformation_notice=manifest.transformation_notice,
        )

    @app.get(
        "/api/summary",
        response_model=SummaryResponse,
        responses={503: {"model": ErrorResponse}},
        tags=["dataset"],
    )
    async def summary(dataset: Dataset = Depends(get_dataset)) -> SummaryResponse:
        aggregates = dataset.aggregates
        manifest = dataset.manifest
        boroughs = sorted(aggregates.by_borough.items(), key=lambda item: (-item[1], item[0]))
        categories = sorted(aggregates.by_category.items(), key=lambda item: (-item[1], item[0]))
        return SummaryResponse(
            total_establishments=aggregates.total,
            boroughs=[BoroughSummary(name=name, count=count) for name, count in boroughs],
            categories=[
                CategorySummary(
                    key=key,
                    label_en=CATEGORIES_BY_KEY[key].label_en,
                    label_es=CATEGORIES_BY_KEY[key].label_es,
                    count=count,
                )
                for key, count in categories
            ],
            strata=aggregates.by_stratum,
            retrieved_at=manifest.retrieved_at,
            complete=manifest.complete,
            attribution=attribution_text(manifest.retrieved_at[:10]),
            transformation_notice=manifest.transformation_notice,
        )

    @app.post(
        "/api/query",
        response_model=QueryResponse,
        responses=ERROR_RESPONSES,
        tags=["analysis"],
        dependencies=[Depends(enforce_rate_limit)],
    )
    async def query(
        payload: QueryRequest, query_service: QueryService = Depends(get_query_service)
    ) -> QueryResponse:
        return await query_service.answer(payload.question)


app = create_app()
