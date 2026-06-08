from fastapi import APIRouter, Depends, Query, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from src.adapters.base import ModelBackend
from src.config import Settings, get_settings
from src.schemas.errors import OpenAIError
from src.schemas.prompts import PromptCacheDeleted
from src.schemas.responses import ResponseArtifactList, ResponseCompactionObject, ResponseDeleted, ResponseInputItemList, ResponseInputTokenCount, ResponseObject, ResponseRequest
from src.security.auth import tenant_id
from src.services.response_service import ResponseService
from src.services.responses_compat import validate_text_responses_request
from src.storage.repository import ResponseRepository
from src.streaming.sse import sse_response

router = APIRouter(prefix="/v1/responses", tags=["responses"])


def service(request: Request, session: AsyncSession) -> ResponseService:
    return ResponseService(
        settings=request.app.state.settings,
        repository=ResponseRepository(session),
        backend=request.app.state.backend,
        prompt_cache=request.app.state.prompt_cache,
        session_factory=request.app.state.async_session,
        background_tasks=request.app.state.background_tasks,
    )


async def get_session(request: Request) -> AsyncSession:
    async_session = request.app.state.async_session
    async with async_session() as session:
        yield session


@router.post("", response_model=ResponseObject)
async def create_response(
    payload: ResponseRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
    tenant: str | None = Depends(tenant_id),
):
    svc = service(request, session)
    validate_text_responses_request(payload)
    if payload.stream:
        return sse_response(svc.stream(payload, tenant))
    return await svc.create(payload, tenant)


@router.post("/input_tokens", response_model=ResponseInputTokenCount)
async def count_input_tokens(
    payload: ResponseRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
    tenant: str | None = Depends(tenant_id),
):
    return await service(request, session).count_input_tokens(payload, tenant)


@router.post("/compact", response_model=ResponseCompactionObject)
async def compact_response(
    payload: ResponseRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
    tenant: str | None = Depends(tenant_id),
):
    return await service(request, session).compact(payload, tenant)


@router.delete("/prompt_cache", response_model=PromptCacheDeleted)
async def clear_prompt_cache(
    request: Request,
    prompt_cache_key: str | None = Query(default=None),
):
    if prompt_cache_key is not None and not prompt_cache_key.strip():
        raise OpenAIError("prompt_cache_key must not be empty.", param="prompt_cache_key", code="invalid_request")
    deleted = request.app.state.prompt_cache.clear(prompt_cache_key=prompt_cache_key)
    return PromptCacheDeleted(deleted=deleted, prompt_cache_key=prompt_cache_key)


@router.get("/{response_id}", response_model=ResponseObject)
async def retrieve_response(
    response_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
    tenant: str | None = Depends(tenant_id),
):
    return await service(request, session).retrieve(response_id, tenant, include=_query_include(request))


@router.get("/{response_id}/input_items", response_model=ResponseInputItemList)
async def list_response_input_items(
    response_id: str,
    request: Request,
    after: str | None = None,
    before: str | None = None,
    limit: int = Query(default=20, ge=1, le=100),
    order: str = "desc",
    session: AsyncSession = Depends(get_session),
    tenant: str | None = Depends(tenant_id),
):
    return await service(request, session).list_input_items(response_id, tenant, after=after, before=before, limit=limit, order=order, include=_query_include(request))


@router.get("/{response_id}/artifacts", response_model=ResponseArtifactList)
async def list_response_artifacts(
    response_id: str,
    request: Request,
    after: str | None = None,
    before: str | None = None,
    limit: int = Query(default=20, ge=1, le=100),
    order: str = "desc",
    session: AsyncSession = Depends(get_session),
    tenant: str | None = Depends(tenant_id),
):
    return await service(request, session).list_artifacts(response_id, tenant, after=after, before=before, limit=limit, order=order)


@router.get("/{response_id}/artifacts/{artifact_id}/content")
async def retrieve_response_artifact_content(
    response_id: str,
    artifact_id: str,
    session: AsyncSession = Depends(get_session),
    tenant: str | None = Depends(tenant_id),
):
    record = await ResponseRepository(session).require_artifact(response_id=response_id, artifact_id=artifact_id, tenant_id=tenant)
    content = record.content_json or {}
    text = content.get("text") if isinstance(content, dict) else None
    if not isinstance(text, str):
        raise OpenAIError("Artifact content not available.", status_code=404, param="artifact_id", code="not_found")
    media_type = record.mime_type or "text/plain"
    return Response(text.encode("utf-8"), media_type=media_type, headers={"content-disposition": f'attachment; filename="{record.filename or artifact_id}"'})


@router.post("/{response_id}/cancel", response_model=ResponseObject)
async def cancel_response(
    response_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
    tenant: str | None = Depends(tenant_id),
):
    return await service(request, session).cancel(response_id, tenant)


@router.delete("/{response_id}", response_model=ResponseDeleted)
async def delete_response(
    response_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
    tenant: str | None = Depends(tenant_id),
):
    await service(request, session).delete(response_id, tenant)
    return ResponseDeleted(id=response_id)


def _query_include(request: Request) -> list[str] | None:
    values = [*request.query_params.getlist("include"), *request.query_params.getlist("include[]")]
    return values or None
