from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.storage.models import PromptTemplateRecord


class PromptTemplateCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str | None = None
    version: str | None = None
    template: dict[str, Any] | None = None
    instructions: str | None = None
    input: str | list[dict[str, Any]] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PromptTemplateObject(BaseModel):
    id: str
    object: Literal["prompt.template"] = "prompt.template"
    version: str
    template: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: int
    updated_at: int


class PromptTemplateList(BaseModel):
    object: Literal["list"] = "list"
    data: list[PromptTemplateObject]


class PromptCacheDeleted(BaseModel):
    object: Literal["prompt_cache.deleted"] = "prompt_cache.deleted"
    deleted: int
    prompt_cache_key: str | None = None


def prompt_template_object(record: PromptTemplateRecord) -> PromptTemplateObject:
    return PromptTemplateObject(
        id=record.prompt_id,
        version=record.version,
        template=record.template_json or {},
        metadata=record.metadata_json or {},
        created_at=_timestamp(record.created_at),
        updated_at=_timestamp(record.updated_at),
    )


def _timestamp(value: datetime) -> int:
    return int(value.timestamp())
