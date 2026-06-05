from fastapi import APIRouter, Depends, Request

from src.schemas.models import ModelList
from src.security.auth import tenant_id

router = APIRouter(tags=["models"])


@router.get("/v1/models", response_model=ModelList)
async def list_models(request: Request, _: str | None = Depends(tenant_id)) -> ModelList:
    return await request.app.state.backend.list_models()


@router.get("/models", response_model=ModelList, include_in_schema=False)
async def list_models_root(request: Request, _: str | None = Depends(tenant_id)) -> ModelList:
    return await request.app.state.backend.list_models()
