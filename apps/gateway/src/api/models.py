from fastapi import APIRouter, Depends, Request

from src.observability.metrics import MODEL_BACKEND_MODEL_INFO
from src.schemas.models import ModelList
from src.security.auth import tenant_id

router = APIRouter(tags=["models"])


@router.get("/v1/models", response_model=ModelList)
async def list_models(request: Request, _: str | None = Depends(tenant_id)) -> ModelList:
    models = await request.app.state.backend.list_models()
    _record_model_info(request, models)
    return models


@router.get("/models", response_model=ModelList, include_in_schema=False)
async def list_models_root(request: Request, _: str | None = Depends(tenant_id)) -> ModelList:
    models = await request.app.state.backend.list_models()
    _record_model_info(request, models)
    return models


def _record_model_info(request: Request, models: ModelList) -> None:
    backend = request.app.state.settings.model_backend
    for model in models.data:
        MODEL_BACKEND_MODEL_INFO.labels(backend=backend, model=model.id).set(1)
