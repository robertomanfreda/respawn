from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from src.observability.metrics import PROMPT_TEMPLATE_REQUESTS
from src.schemas.errors import OpenAIError
from src.schemas.prompts import PromptTemplateCreate, PromptTemplateList, PromptTemplateObject, prompt_template_object
from src.security.auth import tenant_id
from src.services.prompt_templates import prompt_template_from_create
from src.storage.repository import ResponseRepository


router = APIRouter(prefix="/v1/responses/prompts", tags=["responses", "prompt-templates"])


async def get_session(request: Request) -> AsyncSession:
    async_session = request.app.state.async_session
    async with async_session() as session:
        yield session


@router.post("", response_model=PromptTemplateObject)
async def create_prompt_template(
    payload: PromptTemplateCreate,
    session: AsyncSession = Depends(get_session),
    tenant: str | None = Depends(tenant_id),
):
    repository = ResponseRepository(session)
    try:
        prompt_id, version, template, metadata = prompt_template_from_create(payload)
        record = await repository.create_prompt_template(
            prompt_id=prompt_id,
            version=version,
            template_json=template,
            metadata_json=metadata,
            tenant_id=tenant,
        )
        await repository.session.commit()
    except OpenAIError:
        PROMPT_TEMPLATE_REQUESTS.labels(operation="create", status="failed").inc()
        raise
    PROMPT_TEMPLATE_REQUESTS.labels(operation="create", status="success").inc()
    return prompt_template_object(record)


@router.get("", response_model=PromptTemplateList)
async def list_prompt_templates(
    session: AsyncSession = Depends(get_session),
    tenant: str | None = Depends(tenant_id),
):
    repository = ResponseRepository(session)
    records = await repository.list_prompt_templates(tenant)
    PROMPT_TEMPLATE_REQUESTS.labels(operation="list", status="success").inc()
    return PromptTemplateList(data=[prompt_template_object(record) for record in records])


@router.get("/{prompt_id}", response_model=PromptTemplateObject)
async def retrieve_prompt_template(
    prompt_id: str,
    version: str | None = Query(default=None),
    session: AsyncSession = Depends(get_session),
    tenant: str | None = Depends(tenant_id),
):
    repository = ResponseRepository(session)
    try:
        record = await repository.require_prompt_template(prompt_id, tenant, version=version)
    except OpenAIError:
        PROMPT_TEMPLATE_REQUESTS.labels(operation="retrieve", status="failed").inc()
        raise
    PROMPT_TEMPLATE_REQUESTS.labels(operation="retrieve", status="success").inc()
    return prompt_template_object(record)


@router.delete("/{prompt_id}", response_model=PromptTemplateObject)
async def delete_prompt_template(
    prompt_id: str,
    version: str | None = Query(default=None),
    session: AsyncSession = Depends(get_session),
    tenant: str | None = Depends(tenant_id),
):
    repository = ResponseRepository(session)
    try:
        record = await repository.delete_prompt_template(prompt_id, tenant, version=version)
        await repository.session.commit()
    except OpenAIError:
        PROMPT_TEMPLATE_REQUESTS.labels(operation="delete", status="failed").inc()
        raise
    PROMPT_TEMPLATE_REQUESTS.labels(operation="delete", status="success").inc()
    return prompt_template_object(record)
