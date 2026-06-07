from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from src.adapters.base import ModelBackend
from src.config import Settings, get_settings
from src.schemas.responses import ResponseDeleted, ResponseInputItemList, ResponseInputTokenCount, ResponseObject, ResponseRequest
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


@router.get("/{response_id}", response_model=ResponseObject)
async def retrieve_response(
    response_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
    tenant: str | None = Depends(tenant_id),
):
    return await service(request, session).retrieve(response_id, tenant)


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
    return await service(request, session).list_input_items(response_id, tenant, after=after, before=before, limit=limit, order=order)


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
