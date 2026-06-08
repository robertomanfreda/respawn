import asyncio
from pathlib import Path
from time import perf_counter
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text

from src import __version__
from src.observability.metrics import MODEL_BACKEND_MODEL_INFO, OPERATIONAL_FAILURES, READINESS_CHECK_LATENCY, READINESS_CHECKS
from src.services.compatibility_manifest import MANIFEST_VERSION
from src.services.platform_files import PlatformFileStorage

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok", "version": __version__, "compatibility_manifest_version": MANIFEST_VERSION}


@router.get("/readyz")
async def readyz(request: Request):
    settings = request.app.state.settings
    checks = {
        "database": await _run_check("database", lambda: _check_database(request)),
        "ollama": await _run_check("ollama", lambda: _check_backend(request)),
        "worker": await _run_check("worker", lambda: _check_worker(request)),
        "cache": await _run_check("cache", lambda: _check_cache(request)),
        "storage": await _run_check("storage", lambda: _check_storage(settings)),
    }
    ready = all(check["status"] == "ready" for check in checks.values())
    content = {
        "status": "ready" if ready else "not_ready",
        "version": __version__,
        "model_backend": settings.model_backend,
        "default_model": settings.default_model,
        "compatibility_manifest_version": MANIFEST_VERSION,
        "checks": checks,
    }
    return JSONResponse(status_code=200 if ready else 503, content=content)


async def _run_check(name: str, fn) -> dict[str, Any]:
    started_at = perf_counter()
    try:
        details = await fn()
    except Exception as exc:
        READINESS_CHECKS.labels(check=name).set(0)
        READINESS_CHECK_LATENCY.labels(check=name).observe(perf_counter() - started_at)
        OPERATIONAL_FAILURES.labels(component=name, code=exc.__class__.__name__).inc()
        return {"status": "not_ready", "error": exc.__class__.__name__}
    READINESS_CHECKS.labels(check=name).set(1)
    READINESS_CHECK_LATENCY.labels(check=name).observe(perf_counter() - started_at)
    return {"status": "ready", **details}


async def _check_database(request: Request) -> dict[str, Any]:
    async_session = request.app.state.async_session
    async with async_session() as session:
        await session.execute(text("select 1"))
    return {"driver": request.app.state.settings.database_url.split(":", 1)[0]}


async def _check_backend(request: Request) -> dict[str, Any]:
    settings = request.app.state.settings
    timeout = min(max(float(settings.backend_timeout_seconds), 0.1), 5.0)
    models = await asyncio.wait_for(request.app.state.backend.list_models(), timeout=timeout)
    model_ids = [model.id for model in models.data]
    for model_id in model_ids:
        MODEL_BACKEND_MODEL_INFO.labels(backend=settings.model_backend, model=model_id).set(1)
    if settings.default_model not in model_ids:
        raise RuntimeError("Default model is not available from backend.")
    return {
        "backend": settings.model_backend,
        "default_model_present": True,
        "model_count": len(model_ids),
    }


async def _check_worker(request: Request) -> dict[str, Any]:
    background_tasks = getattr(request.app.state, "background_tasks", None)
    cleanup_task = getattr(request.app.state, "platform_file_cleanup_task", None)
    if not isinstance(background_tasks, dict):
        raise RuntimeError("Background task registry is unavailable.")
    if cleanup_task is None or cleanup_task.done():
        raise RuntimeError("Platform file cleanup worker is not running.")
    return {"background_tasks": len(background_tasks), "platform_file_cleanup": "running"}


async def _check_cache(request: Request) -> dict[str, Any]:
    prompt_cache = getattr(request.app.state, "prompt_cache", None)
    if prompt_cache is None:
        raise RuntimeError("Prompt cache is unavailable.")
    return {"enabled": bool(prompt_cache.enabled), "max_entries": int(prompt_cache.max_entries)}


async def _check_storage(settings) -> dict[str, Any]:
    storage = PlatformFileStorage(settings)
    backend = storage.backend
    if backend == "filesystem":
        path = Path(settings.file_storage_path)
        path.mkdir(parents=True, exist_ok=True)
        if not path.is_dir():
            raise RuntimeError("File storage path is not a directory.")
        return {"backend": backend, "path": str(path)}
    return {"backend": backend}
