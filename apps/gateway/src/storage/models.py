from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
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

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    response_id: Mapped[str] = mapped_column(ForeignKey("responses.id"), nullable=False)
    type: Mapped[str] = mapped_column(Text, nullable=False)
    role: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_json: Mapped[Any] = mapped_column(JsonType, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


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


class UsageRecord(Base):
    __tablename__ = "usage_records"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    response_id: Mapped[str] = mapped_column(ForeignKey("responses.id"), nullable=False)
    model: Mapped[str] = mapped_column(Text, nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
