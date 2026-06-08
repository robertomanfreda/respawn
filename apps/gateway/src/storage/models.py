from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Index, Integer, LargeBinary, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import JSON


JsonType = JSON().with_variant(JSONB, "postgresql")


class Base(AsyncAttrs, DeclarativeBase):
    pass


class ResponseRecord(Base):
    __tablename__ = "responses"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    model: Mapped[str] = mapped_column(Text, nullable=False)
    previous_response_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    input_json: Mapped[Any] = mapped_column(JsonType, nullable=False)
    output_json: Mapped[Any] = mapped_column(JsonType, nullable=False, default=list)
    request_json: Mapped[Any] = mapped_column(JsonType, nullable=False)
    metadata_json: Mapped[Any] = mapped_column(JsonType, nullable=False, default=dict)
    usage_json: Mapped[Any | None] = mapped_column(JsonType, nullable=True)
    error_json: Mapped[Any | None] = mapped_column(JsonType, nullable=True)
    tenant_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ResponseItemRecord(Base):
    __tablename__ = "response_items"
    __table_args__ = (
        UniqueConstraint("response_id", "input_index", name="uq_response_items_input_index"),
        UniqueConstraint("response_id", "output_index", name="uq_response_items_output_index"),
        Index("ix_response_items_response_input", "response_id", "input_index"),
        Index("ix_response_items_response_output", "response_id", "output_index"),
        Index("ix_response_items_response_call", "response_id", "call_id"),
    )

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    response_id: Mapped[str] = mapped_column(ForeignKey("responses.id"), nullable=False)
    type: Mapped[str] = mapped_column(Text, nullable=False)
    role: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_json: Mapped[Any] = mapped_column(JsonType, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    input_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    call_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    name: Mapped[str | None] = mapped_column(Text, nullable=True)
    arguments_json: Mapped[Any | None] = mapped_column(JsonType, nullable=True)
    output_json: Mapped[Any | None] = mapped_column(JsonType, nullable=True)
    summary_json: Mapped[Any | None] = mapped_column(JsonType, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ToolCallRecord(Base):
    __tablename__ = "tool_calls"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    response_id: Mapped[str] = mapped_column(ForeignKey("responses.id"), nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    arguments_json: Mapped[Any] = mapped_column(JsonType, nullable=False)
    output_json: Mapped[Any | None] = mapped_column(JsonType, nullable=True)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class BackgroundJobRecord(Base):
    __tablename__ = "background_jobs"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    response_id: Mapped[str] = mapped_column(ForeignKey("responses.id"), nullable=False, unique=True)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    timeout_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    cancellation_requested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_json: Mapped[Any | None] = mapped_column(JsonType, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class UsageRecord(Base):
    __tablename__ = "usage_records"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    response_id: Mapped[str] = mapped_column(ForeignKey("responses.id"), nullable=False)
    model: Mapped[str] = mapped_column(Text, nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class ResponseContextEventRecord(Base):
    __tablename__ = "response_context_events"
    __table_args__ = (Index("ix_response_context_events_response", "response_id"),)

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    response_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_response_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    type: Mapped[str] = mapped_column(Text, nullable=False)
    strategy: Mapped[str] = mapped_column(Text, nullable=False)
    compacted_item_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_item_ids_json: Mapped[Any] = mapped_column(JsonType, nullable=False, default=list)
    summary_json: Mapped[Any | None] = mapped_column(JsonType, nullable=True)
    input_tokens_before: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    input_tokens_after: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class ResponseArtifactRecord(Base):
    __tablename__ = "response_artifacts"
    __table_args__ = (
        Index("ix_response_artifacts_response", "response_id"),
        Index("ix_response_artifacts_item", "response_id", "item_id"),
        Index("ix_response_artifacts_tenant", "tenant_id"),
    )

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    response_id: Mapped[str] = mapped_column(ForeignKey("responses.id"), nullable=False)
    item_id: Mapped[str] = mapped_column(Text, nullable=False)
    content_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    type: Mapped[str] = mapped_column(Text, nullable=False)
    filename: Mapped[str | None] = mapped_column(Text, nullable=True)
    mime_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    source_json: Mapped[Any] = mapped_column(JsonType, nullable=False, default=dict)
    metadata_json: Mapped[Any] = mapped_column(JsonType, nullable=False, default=dict)
    content_json: Mapped[Any | None] = mapped_column(JsonType, nullable=True)
    tenant_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class PromptTemplateRecord(Base):
    __tablename__ = "prompt_templates"
    __table_args__ = (
        Index("ix_prompt_templates_prompt", "prompt_id", "version"),
        Index("ix_prompt_templates_tenant", "tenant_id"),
    )

    record_id: Mapped[str] = mapped_column(Text, primary_key=True)
    prompt_id: Mapped[str] = mapped_column(Text, nullable=False)
    version: Mapped[str] = mapped_column(Text, nullable=False)
    template_json: Mapped[Any] = mapped_column(JsonType, nullable=False)
    metadata_json: Mapped[Any] = mapped_column(JsonType, nullable=False, default=dict)
    tenant_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class PlatformFileRecord(Base):
    __tablename__ = "platform_files"
    __table_args__ = (
        Index("ix_platform_files_tenant", "tenant_id"),
        Index("ix_platform_files_purpose", "purpose"),
        Index("ix_platform_files_expires", "expires_at"),
    )

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    purpose: Mapped[str] = mapped_column(Text, nullable=False)
    bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    mime_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    sha256: Mapped[str] = mapped_column(Text, nullable=False)
    storage_backend: Mapped[str] = mapped_column(Text, nullable=False)
    storage_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_bytes: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    metadata_json: Mapped[Any] = mapped_column(JsonType, nullable=False, default=dict)
    tenant_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
