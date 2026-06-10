from fastapi import APIRouter, Depends, Query, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from src.schemas.errors import OpenAIError
from src.schemas.files import FileDeleted, FileList, FileObject, file_object
from src.security.auth import tenant_id
from src.services.platform_files import PlatformFileService, SUPPORTED_FILE_PURPOSES
from src.storage.repository import ResponseRepository


router = APIRouter(prefix="/v1/files", tags=["files"])


async def get_session(request: Request) -> AsyncSession:
    async_session = request.app.state.async_session
    async with async_session() as session:
        yield session


def service(request: Request, session: AsyncSession) -> PlatformFileService:
    return PlatformFileService(settings=request.app.state.settings, repository=ResponseRepository(session))


@router.post("", response_model=FileObject)
async def create_file(
    request: Request,
    session: AsyncSession = Depends(get_session),
    tenant: str | None = Depends(tenant_id),
):
    record = await service(request, session).create_from_request(request, tenant)
    return file_object(record)


@router.get("", response_model=FileList)
async def list_files(
    purpose: str | None = None,
    after: str | None = None,
    limit: int = Query(default=100, ge=1, le=10_000),
    order: str = Query(default="desc"),
    session: AsyncSession = Depends(get_session),
    tenant: str | None = Depends(tenant_id),
):
    if order not in {"asc", "desc"}:
        raise OpenAIError("order must be 'asc' or 'desc'.", param="order", code="invalid_order")
    if purpose is not None and purpose not in SUPPORTED_FILE_PURPOSES:
        raise OpenAIError("Unsupported file purpose.", param="purpose", code="invalid_file_purpose")
    rows, has_more = await ResponseRepository(session).list_platform_files(tenant, purpose=purpose, after=after, limit=limit, order=order)
    data = [file_object(row) for row in rows]
    return FileList(data=data, first_id=data[0].id if data else None, last_id=data[-1].id if data else None, has_more=has_more)


@router.get("/{file_id}", response_model=FileObject)
async def retrieve_file(
    file_id: str,
    session: AsyncSession = Depends(get_session),
    tenant: str | None = Depends(tenant_id),
):
    record = await ResponseRepository(session).require_platform_file(file_id, tenant)
    return file_object(record)


@router.get("/{file_id}/content")
async def retrieve_file_content(
    file_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
    tenant: str | None = Depends(tenant_id),
):
    record, data = await service(request, session).read_file_content(file_id, tenant)
    return Response(data, media_type=record.mime_type or "application/octet-stream", headers={"content-disposition": f'attachment; filename="{record.filename}"'})


@router.delete("/{file_id}", response_model=FileDeleted)
async def delete_file(
    file_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
    tenant: str | None = Depends(tenant_id),
):
    await service(request, session).delete_file(file_id, tenant)
    return FileDeleted(id=file_id)
