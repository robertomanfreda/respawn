from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.schemas.errors import OpenAIError
from src.services.id_generator import generate_id
from src.storage.models import BackgroundJobRecord, ResponseItemRecord, ResponseRecord, ToolCallRecord, UsageRecord


TERMINAL_RESPONSE_STATUSES = {"completed", "failed", "cancelled", "incomplete"}
TERMINAL_JOB_STATUSES = {"completed", "failed", "cancelled", "incomplete", "timeout"}


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

    async def complete_response(self, response_id: str, output_json: list[dict[str, Any]], usage_json: dict[str, Any], *, status: str = "completed") -> ResponseRecord:
        record = await self.require_response(response_id, include_deleted=True)
        record.status = status
        record.output_json = output_json
        record.usage_json = usage_json
        record.completed_at = datetime.now(timezone.utc)
        await self.save_output_items(response_id=response_id, output_items=output_json)
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
        rows = await self.session.scalars(
            select(ResponseItemRecord).where(
                ResponseItemRecord.response_id == response_id,
                ResponseItemRecord.output_index.is_not(None),
                ResponseItemRecord.status == "in_progress",
            )
        )
        for row in rows:
            row.status = "failed"
            row.completed_at = record.completed_at
        await self.session.flush()

    async def cancel_response(self, response_id: str) -> ResponseRecord:
        record = await self.require_response(response_id, include_deleted=True)
        record.status = "cancelled"
        record.completed_at = datetime.now(timezone.utc)
        rows = await self.session.scalars(
            select(ResponseItemRecord).where(
                ResponseItemRecord.response_id == response_id,
                ResponseItemRecord.output_index.is_not(None),
                ResponseItemRecord.status == "in_progress",
            )
        )
        for row in rows:
            row.status = "cancelled"
            row.completed_at = record.completed_at
        await self.session.flush()
        return record

    async def create_input_items(self, *, response_id: str, input_items: list[dict[str, Any]]) -> None:
        for index, item in enumerate(input_items):
            self.session.add(_record_from_item(response_id=response_id, item=item, input_index=index))
        await self.session.flush()

    async def create_output_item(self, *, response_id: str, item: dict[str, Any], output_index: int) -> None:
        self.session.add(_record_from_item(response_id=response_id, item=item, output_index=output_index))
        await self.session.flush()

    async def update_output_item(self, *, response_id: str, item: dict[str, Any], output_index: int) -> None:
        existing = await self.session.scalar(
            select(ResponseItemRecord).where(
                ResponseItemRecord.response_id == response_id,
                ResponseItemRecord.id == item["id"],
            )
        )
        if existing is None:
            self.session.add(_record_from_item(response_id=response_id, item=item, output_index=output_index))
        else:
            _update_record_from_item(existing, item, output_index=output_index)
        await self.session.flush()

    async def save_output_items(self, *, response_id: str, output_items: list[dict[str, Any]]) -> None:
        for index, item in enumerate(output_items):
            await self.update_output_item(response_id=response_id, item=item, output_index=index)

    async def list_input_items(self, response_id: str, tenant_id: str | None) -> list[dict[str, Any]]:
        await self.require_response(response_id, tenant_id)
        rows = await self.session.scalars(
            select(ResponseItemRecord)
            .where(ResponseItemRecord.response_id == response_id, ResponseItemRecord.input_index.is_not(None))
            .order_by(ResponseItemRecord.input_index, ResponseItemRecord.created_at)
        )
        return [_input_item_from_record(row) for row in rows]

    async def list_output_items(self, response_id: str, tenant_id: str | None) -> list[dict[str, Any]]:
        await self.require_response(response_id, tenant_id)
        return await self._output_items_for_response(response_id)

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

    async def create_background_job(self, *, response_id: str, timeout_at: datetime | None) -> BackgroundJobRecord:
        job = BackgroundJobRecord(
            id=generate_id("job"),
            response_id=response_id,
            status="queued",
            attempts=0,
            timeout_at=timeout_at,
        )
        self.session.add(job)
        await self.session.flush()
        return job

    async def get_background_job(self, response_id: str) -> BackgroundJobRecord | None:
        return await self.session.scalar(select(BackgroundJobRecord).where(BackgroundJobRecord.response_id == response_id))

    async def require_background_job(self, response_id: str, tenant_id: str | None) -> BackgroundJobRecord:
        await self.require_response(response_id, tenant_id)
        job = await self.get_background_job(response_id)
        if job is None:
            raise OpenAIError("Only background responses can be cancelled.", status_code=400, param="response_id", code="invalid_request")
        return job

    async def start_background_job(self, response_id: str) -> BackgroundJobRecord:
        job = await self.get_background_job(response_id)
        if job is None:
            raise OpenAIError("Background job not found.", status_code=404, param="response_id", code="not_found")
        if job.status in TERMINAL_JOB_STATUSES:
            return job
        now = datetime.now(timezone.utc)
        if job.cancellation_requested_at is not None:
            job.status = "cancelled"
            job.completed_at = now
            await self.cancel_response(response_id)
            await self.session.flush()
            return job
        job.status = "in_progress"
        job.attempts += 1
        job.started_at = job.started_at or now
        job.heartbeat_at = now
        response = await self.require_response(response_id, include_deleted=True)
        response.status = "in_progress"
        await self.session.flush()
        return job

    async def heartbeat_background_job(self, response_id: str) -> None:
        job = await self.get_background_job(response_id)
        if job is None or job.status in TERMINAL_JOB_STATUSES:
            return
        job.heartbeat_at = datetime.now(timezone.utc)
        await self.session.flush()

    async def request_background_cancel(self, response_id: str, tenant_id: str | None) -> BackgroundJobRecord:
        job = await self.require_background_job(response_id, tenant_id)
        if job.status in TERMINAL_JOB_STATUSES:
            return job
        now = datetime.now(timezone.utc)
        if job.cancellation_requested_at is None:
            job.cancellation_requested_at = now
        job.status = "cancelled"
        job.completed_at = now
        await self.cancel_response(response_id)
        await self.session.flush()
        return job

    async def complete_background_job(self, response_id: str, *, status: str, error_json: dict[str, Any] | None = None) -> BackgroundJobRecord | None:
        job = await self.get_background_job(response_id)
        if job is None:
            return None
        job.status = status
        job.error_json = error_json
        job.completed_at = datetime.now(timezone.utc)
        job.heartbeat_at = job.completed_at
        await self.session.flush()
        return job

    async def list_runnable_background_jobs(self) -> list[dict[str, str | None]]:
        rows = await self.session.execute(
            select(BackgroundJobRecord.response_id, ResponseRecord.tenant_id)
            .join(ResponseRecord, ResponseRecord.id == BackgroundJobRecord.response_id)
            .where(
                BackgroundJobRecord.status.in_(("queued", "in_progress")),
                ResponseRecord.deleted_at.is_(None),
            )
        )
        return [{"response_id": response_id, "tenant_id": tenant_id} for response_id, tenant_id in rows]

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

        loaded = []
        for record in reversed(chain):
            loaded.append(
                {
                    "id": record.id,
                    "request_json": record.request_json,
                    "input_items": await self._input_items_for_response(record.id),
                    "output_json": await self._output_items_for_response(record.id) or record.output_json,
                    "model": record.model,
                }
            )
        return loaded

    async def _input_items_for_response(self, response_id: str) -> list[dict[str, Any]]:
        rows = await self.session.scalars(
            select(ResponseItemRecord)
            .where(ResponseItemRecord.response_id == response_id, ResponseItemRecord.input_index.is_not(None))
            .order_by(ResponseItemRecord.input_index, ResponseItemRecord.created_at)
        )
        return [_input_item_from_record(row) for row in rows]

    async def _output_items_for_response(self, response_id: str) -> list[dict[str, Any]]:
        rows = await self.session.scalars(
            select(ResponseItemRecord)
            .where(ResponseItemRecord.response_id == response_id, ResponseItemRecord.output_index.is_not(None))
            .order_by(ResponseItemRecord.output_index, ResponseItemRecord.created_at)
        )
        return [_output_item_from_record(row) for row in rows]


def _record_from_item(
    *,
    response_id: str,
    item: dict[str, Any],
    input_index: int | None = None,
    output_index: int | None = None,
) -> ResponseItemRecord:
    now = datetime.now(timezone.utc)
    status = item.get("status", "completed")
    return ResponseItemRecord(
        id=item["id"],
        response_id=response_id,
        type=item.get("type", "message"),
        role=item.get("role"),
        content_json=item.get("content", []),
        status=status,
        input_index=input_index,
        output_index=output_index,
        call_id=item.get("call_id"),
        name=item.get("name"),
        arguments_json=item.get("arguments"),
        output_json=item.get("output"),
        summary_json=item.get("summary"),
        completed_at=now if status in {"completed", "incomplete", "failed"} else None,
    )


def _update_record_from_item(row: ResponseItemRecord, item: dict[str, Any], *, output_index: int | None = None) -> None:
    status = item.get("status", row.status)
    row.type = item.get("type", row.type)
    row.role = item.get("role")
    row.content_json = item.get("content", [])
    row.status = status
    row.output_index = output_index
    row.call_id = item.get("call_id")
    row.name = item.get("name")
    row.arguments_json = item.get("arguments")
    row.output_json = item.get("output")
    row.summary_json = item.get("summary")
    if status in {"completed", "incomplete", "failed"}:
        row.completed_at = datetime.now(timezone.utc)


def _input_item_from_record(row: ResponseItemRecord) -> dict[str, Any]:
    if row.type == "reasoning":
        return {
            "id": row.id,
            "type": "reasoning",
            "summary": row.summary_json or [],
            "status": row.status,
        }
    if row.type == "function_call":
        return {
            "id": row.id,
            "type": "function_call",
            "call_id": row.call_id,
            "name": row.name,
            "arguments": row.arguments_json if row.arguments_json is not None else "{}",
            "status": row.status,
        }
    if row.type == "function_call_output":
        return {
            "id": row.id,
            "type": "function_call_output",
            "call_id": row.call_id,
            "output": row.output_json if row.output_json is not None else "",
            "status": row.status,
        }
    item = {
        "id": row.id,
        "type": row.type,
        "role": row.role or "user",
        "content": row.content_json or [],
    }
    if row.status:
        item["status"] = row.status
    return item


def _output_item_from_record(row: ResponseItemRecord) -> dict[str, Any]:
    if row.type == "reasoning":
        return {
            "id": row.id,
            "type": "reasoning",
            "summary": row.summary_json or [],
            "status": row.status,
        }
    if row.type == "function_call":
        return {
            "id": row.id,
            "type": "function_call",
            "status": row.status,
            "call_id": row.call_id,
            "name": row.name,
            "arguments": row.arguments_json if row.arguments_json is not None else "{}",
        }
    item = {
        "id": row.id,
        "type": row.type,
        "status": row.status,
        "role": row.role or "assistant",
        "content": row.content_json or [],
    }
    if row.call_id:
        item["call_id"] = row.call_id
    if row.name:
        item["name"] = row.name
    if row.arguments_json is not None:
        item["arguments"] = row.arguments_json
    if row.output_json is not None:
        item["output"] = row.output_json
    return item
