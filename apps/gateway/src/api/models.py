from fastapi import APIRouter, Depends, Request

from src.observability.metrics import MODEL_BACKEND_MODEL_INFO
from src.schemas.models import ModelList, ModelObject
from src.security.auth import tenant_id
from src.services.context_management import context_window_for_model

router = APIRouter(tags=["models"])


@router.get("/v1/models", response_model=ModelList)
async def list_models(request: Request, _: str | None = Depends(tenant_id)) -> ModelList:
    models = await request.app.state.backend.list_models()
    models = _with_context_metadata(request, models)
    _record_model_info(request, models)
    return models


@router.get("/models", response_model=ModelList, include_in_schema=False)
async def list_models_root(request: Request, _: str | None = Depends(tenant_id)) -> ModelList:
    models = await request.app.state.backend.list_models()
    models = _with_context_metadata(request, models)
    _record_model_info(request, models)
    return models


def _record_model_info(request: Request, models: ModelList) -> None:
    backend = request.app.state.settings.model_backend
    for model in models.data:
        MODEL_BACKEND_MODEL_INFO.labels(backend=backend, model=model.id).set(1)


def _with_context_metadata(request: Request, models: ModelList) -> ModelList:
    settings = request.app.state.settings
    enriched = []
    for model in models.data:
        payload = model.model_dump()
        context_window = int(payload.get("context_window") or context_window_for_model(model.id, settings))
        payload.setdefault("context_window", context_window)
        payload.setdefault("max_context_window", context_window)
        payload.setdefault("effective_context_window_percent", 95)
        enriched.append(ModelObject(**payload))
    return ModelList(data=enriched)
