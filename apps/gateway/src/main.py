from contextlib import asynccontextmanager
import asyncio
import hashlib
import json
import logging
from time import perf_counter

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from src import __version__
from src.adapters.automatic1111_image import Automatic1111ImageGenerationBackend
from src.adapters.comfyui_image import ComfyUIImageGenerationBackend
from src.adapters.image_generation_base import ImageGenerationBackend
from src.adapters.mock_backend import MockBackend
from src.adapters.mock_image import MockImageGenerationBackend
from src.adapters.mock_search import MockSearchBackend
from src.adapters.ollama_backend import OllamaBackend
from src.adapters.searxng_search import SearxngSearchBackend
from src.api import chat, compatibility, files, health, metrics, models, prompt_templates, responses
from src.config import get_settings
from src.observability.logging import TRACE_LEVEL, configure_logging
from src.observability.metrics import (
    ENDPOINT_REQUESTS,
    ERRORS,
    FEATURE_REQUESTS,
    IDEMPOTENCY_REQUESTS,
    OPERATIONAL_FAILURES,
    REQUEST_LATENCY,
    REQUESTS,
)
from src.schemas.errors import OpenAIError, openai_error_handler, request_validation_error_handler
from src.services.id_generator import generate_id
from src.services.idempotency import IdempotencyStore
from src.services.prompt_cache import PromptCache
from src.services.platform_files import run_platform_file_cleanup
from src.services.response_service import resume_background_responses, shutdown_background_responses
from src.storage import create_engine, create_sessionmaker, create_tables


logger = logging.getLogger(__name__)
SUPPORTED_MODEL_BACKENDS = {"mock", "ollama"}
SUPPORTED_WEB_SEARCH_BACKENDS = {"mock", "searxng"}
SUPPORTED_IMAGE_GENERATION_BACKENDS = {"mock", "automatic1111", "comfyui"}


def _unsupported_backend_message(setting_name: str, requested: str, supported_values: set[str]) -> str:
    supported = ", ".join(sorted(supported_values))
    return f"Unsupported {setting_name} '{requested}'. Supported values: {supported}."


def build_backend(settings):
    """Create the configured model backend and fail fast on typos."""
    backend_name = settings.model_backend.lower()
    if backend_name == "ollama":
        return OllamaBackend(settings.ollama_base_url, settings.backend_timeout_seconds)
    if backend_name == "mock":
        return MockBackend(settings.default_model)
    raise ValueError(_unsupported_backend_message("MODEL_BACKEND", settings.model_backend, SUPPORTED_MODEL_BACKENDS))


def build_web_search_backend(settings):
    """Create the optional web search backend when local web search is enabled."""
    if not settings.web_search_enabled:
        return None
    backend_name = settings.web_search_backend.lower()
    if backend_name == "mock":
        return MockSearchBackend()
    if backend_name == "searxng":
        return SearxngSearchBackend(
            base_url=settings.web_search_base_url,
            timeout_seconds=settings.web_search_timeout_seconds,
            user_agent=settings.web_search_user_agent,
        )
    raise ValueError(_unsupported_backend_message("WEB_SEARCH_BACKEND", settings.web_search_backend, SUPPORTED_WEB_SEARCH_BACKENDS))


def build_image_generation_backend(settings) -> ImageGenerationBackend | None:
    """Create the optional image-generation backend when local image generation is enabled."""
    if not settings.image_generation_enabled:
        return None
    backend_name = settings.image_generation_backend.lower()
    if backend_name == "mock":
        return MockImageGenerationBackend()
    if backend_name == "automatic1111":
        return Automatic1111ImageGenerationBackend(
            base_url=settings.image_generation_base_url,
            timeout_seconds=settings.image_generation_timeout_seconds,
            model=settings.image_generation_model,
        )
    if backend_name == "comfyui":
        return ComfyUIImageGenerationBackend(
            base_url=settings.image_generation_base_url,
            timeout_seconds=settings.image_generation_timeout_seconds,
            model=settings.image_generation_model,
        )
    raise ValueError(_unsupported_backend_message("IMAGE_GENERATION_BACKEND", settings.image_generation_backend, SUPPORTED_IMAGE_GENERATION_BACKENDS))


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    engine = create_engine(settings)
    app.state.settings = settings
    app.state.engine = engine
    app.state.async_session = create_sessionmaker(engine)
    app.state.backend = build_backend(settings)
    app.state.web_search_backend = build_web_search_backend(settings)
    app.state.image_generation_backend = build_image_generation_backend(settings)
    app.state.background_tasks = {}
    app.state.idempotency_store = IdempotencyStore(settings.idempotency_cache_max_entries)
    app.state.prompt_cache = PromptCache(
        enabled=settings.prompt_cache_enabled,
        min_tokens=settings.prompt_cache_min_tokens,
        max_entries=settings.prompt_cache_max_entries,
        in_memory_ttl_seconds=settings.prompt_cache_in_memory_ttl_seconds,
        extended_ttl_seconds=settings.prompt_cache_extended_ttl_seconds,
        chunk_tokens=settings.prompt_cache_chunk_tokens,
    )
    if settings.auto_create_tables:
        await create_tables(engine)
    app.state.platform_file_cleanup_task = asyncio.create_task(
        run_platform_file_cleanup(settings=settings, session_factory=app.state.async_session)
    )
    await resume_background_responses(
        settings=settings,
        session_factory=app.state.async_session,
        backend=app.state.backend,
        web_search_backend=app.state.web_search_backend,
        image_generation_backend=app.state.image_generation_backend,
        prompt_cache=app.state.prompt_cache,
        background_tasks=app.state.background_tasks,
    )
    yield
    app.state.platform_file_cleanup_task.cancel()
    await shutdown_background_responses(app.state.background_tasks)
    await asyncio.gather(app.state.platform_file_cleanup_task, return_exceptions=True)
    await engine.dispose()


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(title="Respawn", version=__version__, lifespan=lifespan)
    app.add_exception_handler(OpenAIError, openai_error_handler)
    app.add_exception_handler(RequestValidationError, request_validation_error_handler)

    @app.middleware("http")
    async def request_observability(request, call_next):
        request_id = request.headers.get("x-request-id") or generate_id("req")
        request.state.request_id = request_id
        request.state.tenant_id = None
        start = perf_counter()
        idempotency_key = request.headers.get("idempotency-key")
        idempotency_body_sha256 = None
        idempotency_cache_key = None

        def route_path() -> str:
            route = request.scope.get("route")
            return route.path if route else request.url.path

        def record_response(response: Response, parsed_body: dict | None = None) -> Response:
            elapsed = perf_counter() - start
            path = route_path()
            status = str(response.status_code)
            feature = _feature_family(path)
            error_code = _error_code(parsed_body, response.status_code)
            error_param = _error_param(parsed_body) if response.status_code >= 400 else None
            response_id = _response_id(parsed_body)
            REQUEST_LATENCY.observe(elapsed)
            REQUESTS.labels(method=request.method, path=path, status=status).inc()
            ENDPOINT_REQUESTS.labels(endpoint=path, method=request.method, status=status).inc()
            FEATURE_REQUESTS.labels(feature=feature, status=status).inc()
            if response.status_code >= 400:
                ERRORS.labels(code=error_code).inc()
            if response.status_code >= 500:
                OPERATIONAL_FAILURES.labels(component=_failure_component(error_code, feature, request), code=error_code).inc()
            response.headers["x-request-id"] = request_id
            log_extra = {
                "request_id": request_id,
                "response_id": response_id,
                "tenant": getattr(request.state, "tenant_id", None),
                "feature": feature,
                "backend": getattr(request.app.state.settings, "model_backend", None),
                "method": request.method,
                "path": path,
                "latency_ms": round(elapsed * 1000, 3),
                "status": response.status_code,
                "error_code": error_code if response.status_code >= 400 else None,
                "error_param": error_param,
            }
            if feature == "metrics":
                logger.log(TRACE_LEVEL, "HTTP request completed", extra=log_extra)
            else:
                logger.info("HTTP request completed", extra=log_extra)
            return response

        if request.method == "POST" and idempotency_key is not None:
            if not idempotency_key.strip():
                error = OpenAIError("Idempotency-Key must not be empty.", param="Idempotency-Key", code="invalid_request")
                response = JSONResponse(status_code=error.status_code, content=error.to_response())
                IDEMPOTENCY_REQUESTS.labels(status="invalid").inc()
                return record_response(response, error.to_response())
            body = await request.body()
            idempotency_body_sha256 = hashlib.sha256(body).hexdigest()
            auth_scope = request.headers.get("authorization") or "anonymous"
            idempotency_cache_key = f"{auth_scope}:{request.method}:{request.url.path}:{idempotency_key}"
            try:
                cached = request.app.state.idempotency_store.get(idempotency_cache_key, idempotency_body_sha256)
            except OpenAIError as exc:
                response = JSONResponse(status_code=exc.status_code, content=exc.to_response())
                IDEMPOTENCY_REQUESTS.labels(status=exc.code or "error").inc()
                return record_response(response, exc.to_response())
            if cached is not None:
                response = Response(
                    content=cached.body,
                    status_code=cached.status_code,
                    headers=dict(cached.headers),
                    media_type=cached.media_type,
                )
                response.headers["x-respawn-idempotent-replay"] = "true"
                IDEMPOTENCY_REQUESTS.labels(status="replay").inc()
                return record_response(response, _json_body(cached.body))
            IDEMPOTENCY_REQUESTS.labels(status="miss").inc()
        try:
            response = await call_next(request)
        except Exception:
            ERRORS.labels(code="unhandled").inc()
            OPERATIONAL_FAILURES.labels(component="gateway", code="unhandled").inc()
            logger.exception(
                "Unhandled request error",
                extra={
                    "request_id": request_id,
                    "tenant": getattr(request.state, "tenant_id", None),
                    "method": request.method,
                    "path": request.url.path,
                    "feature": _feature_family(request.url.path),
                    "backend": getattr(request.app.state.settings, "model_backend", None),
                    "error_code": "unhandled",
                },
            )
            raise
        content_type = response.headers.get("content-type", "")
        parsed_body: dict | None = None
        if not content_type.startswith("text/event-stream") and content_type.startswith("application/json"):
            chunks = [chunk async for chunk in response.body_iterator]
            body = b"".join(chunks)
            parsed_body = _json_body(body)
            headers = {
                key: value
                for key, value in response.headers.items()
                if key.lower() not in {"content-length", "x-request-id", "x-respawn-idempotent-replay"}
            }
            if idempotency_cache_key and idempotency_body_sha256 and response.status_code < 500:
                request.app.state.idempotency_store.put(
                    idempotency_cache_key,
                    body_sha256=idempotency_body_sha256,
                    status_code=response.status_code,
                    headers=headers,
                    body=body,
                    media_type=response.media_type,
                )
                IDEMPOTENCY_REQUESTS.labels(status="stored").inc()
            response = Response(content=body, status_code=response.status_code, headers=headers, media_type=response.media_type)
        return record_response(response, parsed_body)

    app.include_router(health.router)
    app.include_router(compatibility.router)
    app.include_router(metrics.router)
    app.include_router(models.router)
    app.include_router(files.router)
    app.include_router(chat.router)
    app.include_router(prompt_templates.router)
    app.include_router(responses.router)
    return app


app = create_app()


def _json_body(body: bytes) -> dict | None:
    try:
        parsed = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _response_id(parsed_body: dict | None) -> str | None:
    if not parsed_body:
        return None
    value = parsed_body.get("id")
    return value if isinstance(value, str) else None


def _error_code(parsed_body: dict | None, status_code: int) -> str:
    error = parsed_body.get("error") if parsed_body else None
    if isinstance(error, dict) and error.get("code"):
        return str(error["code"])
    if isinstance(error, dict) and error.get("type"):
        return str(error["type"])
    return str(status_code)


def _error_param(parsed_body: dict | None) -> str | None:
    error = parsed_body.get("error") if parsed_body else None
    if isinstance(error, dict) and error.get("param") is not None:
        return str(error["param"])
    return None


def _failure_component(error_code: str, feature: str, request) -> str:
    if error_code.startswith("backend_"):
        return getattr(request.app.state.settings, "model_backend", "backend")
    if error_code.startswith("storage_") or error_code.startswith("file_") or feature == "files":
        return "storage"
    if error_code.startswith("database_"):
        return "database"
    return feature


def _feature_family(path: str) -> str:
    if path in {"/healthz", "/readyz"}:
        return "health"
    if path == "/metrics":
        return "metrics"
    if path.startswith("/compatibility/"):
        return "compatibility"
    if path.startswith("/v1/responses/prompts"):
        return "prompt_templates"
    if path.startswith("/v1/responses/prompt_cache"):
        return "prompt_cache"
    if path.startswith("/v1/responses"):
        return "responses"
    if path.startswith("/v1/chat/completions"):
        return "chat_completions"
    if path.startswith("/v1/files"):
        return "files"
    if path.startswith("/v1/models"):
        return "models"
    return "other"
