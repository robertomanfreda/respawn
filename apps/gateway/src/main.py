from contextlib import asynccontextmanager
import logging
from time import perf_counter

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from src import __version__
from src.adapters.mock_backend import MockBackend
from src.adapters.ollama_backend import OllamaBackend
from src.api import chat, health, metrics, models, responses
from src.config import get_settings
from src.observability.logging import configure_logging
from src.observability.metrics import ERRORS, REQUEST_LATENCY, REQUESTS
from src.schemas.errors import OpenAIError, openai_error_handler, request_validation_error_handler
from src.services.id_generator import generate_id
from src.services.prompt_cache import PromptCache
from src.storage import create_engine, create_sessionmaker, create_tables
from src.tools.registry import default_registry


logger = logging.getLogger(__name__)
SUPPORTED_MODEL_BACKENDS = {"mock", "ollama"}


def build_backend(settings):
    """Create the configured model backend and fail fast on typos."""
    backend_name = settings.model_backend.lower()
    if backend_name == "ollama":
        return OllamaBackend(settings.ollama_base_url, settings.backend_timeout_seconds)
    if backend_name == "mock":
        return MockBackend(settings.default_model)
    supported = ", ".join(sorted(SUPPORTED_MODEL_BACKENDS))
    raise ValueError(f"Unsupported MODEL_BACKEND '{settings.model_backend}'. Supported values: {supported}.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    engine = create_engine(settings)
    app.state.settings = settings
    app.state.engine = engine
    app.state.async_session = create_sessionmaker(engine)
    app.state.backend = build_backend(settings)
    app.state.prompt_cache = PromptCache(
        enabled=settings.prompt_cache_enabled,
        min_tokens=settings.prompt_cache_min_tokens,
        max_entries=settings.prompt_cache_max_entries,
        in_memory_ttl_seconds=settings.prompt_cache_in_memory_ttl_seconds,
        extended_ttl_seconds=settings.prompt_cache_extended_ttl_seconds,
        chunk_tokens=settings.prompt_cache_chunk_tokens,
    )
    app.state.tool_registry = default_registry()
    if settings.auto_create_tables:
        await create_tables(engine)
    yield
    await engine.dispose()


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(title="Respawn", version=__version__, lifespan=lifespan)
    app.add_exception_handler(OpenAIError, openai_error_handler)
    app.add_exception_handler(RequestValidationError, request_validation_error_handler)

    @app.middleware("http")
    async def request_observability(request, call_next):
        request_id = request.headers.get("x-request-id") or generate_id("req")
        start = perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            ERRORS.labels(code="unhandled").inc()
            logger.exception(
                "Unhandled request error",
                extra={"request_id": request_id, "method": request.method, "path": request.url.path},
            )
            raise
        elapsed = perf_counter() - start
        path = request.scope.get("route").path if request.scope.get("route") else request.url.path
        REQUEST_LATENCY.observe(elapsed)
        REQUESTS.labels(method=request.method, path=path, status=str(response.status_code)).inc()
        response.headers["x-request-id"] = request_id
        return response

    app.include_router(health.router)
    app.include_router(metrics.router)
    app.include_router(models.router)
    app.include_router(chat.router)
    app.include_router(responses.router)
    return app


app = create_app()
