from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text

from src import __version__
from src.services.compatibility_manifest import MANIFEST_VERSION

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok", "version": __version__, "compatibility_manifest_version": MANIFEST_VERSION}


@router.get("/readyz")
async def readyz(request: Request):
    try:
        async_session = request.app.state.async_session
        async with async_session() as session:
            await session.execute(text("select 1"))
    except Exception as exc:
        return JSONResponse(status_code=503, content={"status": "not_ready", "error": exc.__class__.__name__})
    settings = request.app.state.settings
    return {
        "status": "ready",
        "version": __version__,
        "model_backend": settings.model_backend,
        "default_model": settings.default_model,
        "compatibility_manifest_version": MANIFEST_VERSION,
    }
