from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.schemas.errors import OpenAIError
from src.services.id_generator import generate_id
from src.storage.models import ResponseItemRecord, ResponseRecord, ToolCallRecord, UsageRecord


class ResponseRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_response(
        self,
        *,
        response_id: str,
        model: str,
        previous_response_id: str | None,
        input_json: Any,
        request_json: Any,
        metadata_json: dict[str, Any],
        tenant_id: str | None,
        status: str = "in_progress",
    ) -> ResponseRecord:
        record = ResponseRecord(
            id=response_id,
            model=model,
            previous_response_id=previous_response_id,
            status=status,
            input_json=input_json,
            output_json=[],
            request_json=request_json,
            metadata_json=metadata_json,
            tenant_id=tenant_id,
        )
        self.session.add(record)
        await self.session.flush()
        return record

    async def complete_response(self, response_id: str, output_json: list[dict[str, Any]], usage_json: dict[str, Any]) -> ResponseRecord:
        record = await self.require_response(response_id, include_deleted=True)
        record.status = "completed"
        record.output_json = output_json
        record.usage_json = usage_json
        record.completed_at = datetime.now(timezone.utc)
        for item in output_json:
            self.session.add(
                ResponseItemRecord(
                    id=item["id"],
                    response_id=response_id,
                    type=item.get("type", "message"),
                    role=item.get("role"),
                    content_json=item.get("content", item),
                    status=item.get("status", "completed"),
                )
            )
        self.session.add(
            UsageRecord(
                id=generate_id("usage"),
                response_id=response_id,
                model=record.model,
                input_tokens=int(usage_json.get("input_tokens", 0)),
                output_tokens=int(usage_json.get("output_tokens", 0)),
                total_tokens=int(usage_json.get("total_tokens", 0)),
            )
        )
        await self.session.flush()
        return record

    async def fail_response(self, response_id: str, error_json: dict[str, Any]) -> None:
        record = await self.require_response(response_id, include_deleted=True)
        record.status = "failed"
        record.error_json = error_json
        record.completed_at = datetime.now(timezone.utc)
        await self.session.flush()

    async def save_tool_call(self, *, response_id: str, call_id: str, name: str, arguments_json: Any, output_json: Any, status: str) -> None:
        self.session.add(
            ToolCallRecord(
                id=call_id,
                response_id=response_id,
                name=name,
                arguments_json=arguments_json,
                output_json=output_json,
                status=status,
                completed_at=datetime.now(timezone.utc) if status == "completed" else None,
            )
        )
        await self.session.flush()

    async def get_response(self, response_id: str, tenant_id: str | None) -> ResponseRecord | None:
        stmt = select(ResponseRecord).where(ResponseRecord.id == response_id, ResponseRecord.deleted_at.is_(None))
        if tenant_id is not None:
            stmt = stmt.where(ResponseRecord.tenant_id == tenant_id)
        return await self.session.scalar(stmt)

    async def require_response(
        self,
        response_id: str,
        tenant_id: str | None = None,
        *,
        include_deleted: bool = False,
        error_param: str = "response_id",
    ) -> ResponseRecord:
        stmt = select(ResponseRecord).where(ResponseRecord.id == response_id)
        if not include_deleted:
            stmt = stmt.where(ResponseRecord.deleted_at.is_(None))
        if tenant_id is not None:
            stmt = stmt.where(ResponseRecord.tenant_id == tenant_id)
        record = await self.session.scalar(stmt)
        if not record:
            raise OpenAIError("Response not found.", status_code=404, param=error_param, code="not_found")
        return record

    async def soft_delete(self, response_id: str, tenant_id: str | None) -> None:
        record = await self.require_response(response_id, tenant_id)
        record.deleted_at = datetime.now(timezone.utc)
        await self.session.flush()

    async def load_chain(self, response_id: str | None, tenant_id: str | None, max_depth: int) -> list[dict[str, Any]]:
        if response_id is None:
            return []
        chain: list[ResponseRecord] = []
        current_id: str | None = response_id
        for _ in range(max_depth):
            record = await self.require_response(current_id, tenant_id, error_param="previous_response_id")
            chain.append(record)
            current_id = record.previous_response_id
            if current_id is None:
                break
        else:
            raise OpenAIError("Response chain exceeds MAX_CHAIN_DEPTH.", status_code=400, param="previous_response_id", code="chain_too_deep")

        return [
            {
                "id": record.id,
                "request_json": record.request_json,
                "output_json": record.output_json,
                "model": record.model,
            }
            for record in reversed(chain)
        ]
