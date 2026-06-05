from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(request: Request):
    try:
        async_session = request.app.state.async_session
        async with async_session() as session:
            await session.execute(text("select 1"))
    except Exception as exc:
        return JSONResponse(status_code=503, content={"status": "not_ready", "error": exc.__class__.__name__})
    return {"status": "ready"}
