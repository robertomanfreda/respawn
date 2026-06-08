import asyncio
import json
import secrets
import time
from collections.abc import AsyncIterator
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from typing import Any

from src.adapters.base import ChatCompletionResult, ModelBackend
from src.config import Settings
from src.observability.metrics import (
    BACKGROUND_JOB_LATENCY,
    BACKGROUND_JOB_RUNNING,
    BACKGROUND_JOBS,
    CONTEXT_COMPACTION_RATIO,
    CONTEXT_COMPACTION_TOKENS,
    CONTEXT_COMPACTIONS,
    CONTEXT_OVERFLOWS,
    CONTEXT_TRUNCATIONS,
    FUNCTION_TOOL_CALLS,
    FUNCTION_TOOL_CAPABILITY_ERRORS,
    FUNCTION_TOOL_OUTPUTS,
    FUNCTION_TOOL_REQUESTS,
    INCLUDE_EXPANSIONS,
    INCLUDE_EXPANSION_BYTES,
    IN_FLIGHT_RESPONSES,
    MODEL_TOKEN_USAGE,
    PROMPT_CACHE_HIT_RATIO,
    PROMPT_CACHE_REQUESTS,
    PROMPT_CACHE_TOKENS,
    REASONING_HEAVY_REQUESTS,
    REASONING_REQUESTS,
    REASONING_TOKENS,
    RESPONSE_LATENCY,
    RESPONSES,
    STREAMING_RESPONSES_RUNNING,
    TOKEN_USAGE,
)
from src.schemas.errors import OpenAIError
from src.schemas.responses import ResponseArtifactList, ResponseCompactionObject, ResponseInputItemList, ResponseInputTokenCount, ResponseObject, ResponseRequest, ResponseUsage
from src.services.context_management import ContextPlan, ContextPlanner, compact_response_window
from src.services.include_expansions import (
    OUTPUT_TEXT_LOGPROBS,
    input_image_url_requested,
    logprobs_requested,
    requested_includes,
    validate_include_capabilities,
    validate_include_values,
)
from src.services.response_history_builder import (
    assistant_text_to_output,
    build_messages,
)
from src.services.id_generator import generate_id
from src.services.multimodal_inputs import prepare_multimodal_request
from src.services.prompt_cache import PromptCache, PromptCacheMatch
from src.services.prompt_templates import PromptTemplateRenderer
from src.services.structured_outputs import repair_instruction, schema_from_response_format, validate_text_against_schema
from src.services.responses_compat import (
    estimate_input_tokens,
    estimate_text_tokens,
    backend_function_tools,
    backend_tool_choice,
    function_call_output_ids,
    input_items_from_request,
    normalized_text_config,
    paginate_items,
    reasoning_encrypted_content_requested,
    reasoning_output_item,
    reasoning_requested,
    response_output_text,
    stream_obfuscation_enabled,
    tool_choice_instruction,
    validate_function_call_outputs_match,
    validate_reasoning_capabilities,
    validate_text_responses_request,
)
from src.services.reasoning_encryption import seal_reasoning_content
from src.services.reasoning_summaries import DeterministicReasoningSummaryProvider
from src.services.usage_meter import enrich_response_usage, normalize_usage
from src.storage.repository import ResponseRepository
from src.streaming.events import make_event


class ResponseService:
    """Orchestrates Responses API requests across storage and one configured backend."""

    def __init__(
        self,
        *,
        settings: Settings,
        repository: ResponseRepository,
        backend: ModelBackend,
        prompt_cache: PromptCache,
        session_factory: Any | None = None,
        background_tasks: dict[str, asyncio.Task] | None = None,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.backend = backend
        self.prompt_cache = prompt_cache
        self.session_factory = session_factory
        self.background_tasks = background_tasks
        self.reasoning_summary_provider = DeterministicReasoningSummaryProvider()
        self.context_planner = ContextPlanner(settings)
        self.prompt_template_renderer = PromptTemplateRenderer(repository)

    async def create(self, request: ResponseRequest, tenant_id: str | None) -> ResponseObject:
        validate_text_responses_request(request)
        request = await self._render_prompt_request(request, tenant_id)
        validate_text_responses_request(request)
        model = request.model or self.settings.default_model
        validate_include_capabilities(request, model=model, settings=self.settings)
        validate_reasoning_capabilities(request, model=model, settings=self.settings)
        request = await prepare_multimodal_request(request, model=model, settings=self.settings, repository=self.repository, tenant_id=tenant_id)
        if request.background:
            return await self.create_background(request, tenant_id)
        response_id = generate_id("resp")
        request = self._stamp_input_artifacts(request, response_id=response_id)
        should_store = self._should_store(request)
        mode = "blocking"
        status = "failed"
        started_at = time.perf_counter()
        response_created = False
        IN_FLIGHT_RESPONSES.labels(model=model, mode=mode).inc()
        if request.tools:
            FUNCTION_TOOL_REQUESTS.labels(model=model, status="started").inc()
        try:
            if should_store:
                input_items = self._input_items(request)
                await self.repository.create_response(
                    response_id=response_id,
                    model=model,
                    previous_response_id=request.previous_response_id,
                    input_json=self._stored_input(request),
                    request_json=self._request_json(request),
                    metadata_json=request.metadata,
                    tenant_id=tenant_id,
                )
                await self.repository.create_input_items(response_id=response_id, input_items=input_items)
                await self.repository.create_input_artifacts(response_id=response_id, input_items=input_items, tenant_id=tenant_id)
                response_created = True
            output, usage, response_status, incomplete_details = await self._generate_response_output(
                response_id=response_id,
                model=model,
                request=request,
                tenant_id=tenant_id,
                should_store=should_store,
            )
            if should_store:
                await self.repository.complete_response(response_id, output, usage.model_dump(), status=response_status)
                await self.repository.session.commit()
            status = response_status
            if request.tools:
                FUNCTION_TOOL_REQUESTS.labels(model=model, status=response_status).inc()
            self._record_reasoning_metrics(model=model, request=request, usage=usage, status=response_status)
            return self._response_object(response_id, model, response_status, output, usage, request.metadata, request=request, should_store=should_store, incomplete_details=incomplete_details)
        except Exception as exc:
            if should_store and response_created:
                await self.repository.fail_response(response_id, _error_json(exc))
                await self.repository.session.commit()
            if isinstance(exc, OpenAIError):
                if request.tools:
                    FUNCTION_TOOL_REQUESTS.labels(model=model, status="failed").inc()
                self._record_reasoning_metrics(model=model, request=request, usage=ResponseUsage(), status="failed")
                raise
            if request.tools:
                FUNCTION_TOOL_REQUESTS.labels(model=model, status="failed").inc()
            self._record_reasoning_metrics(model=model, request=request, usage=ResponseUsage(), status="failed")
            raise OpenAIError("Internal gateway error.", status_code=500, type="server_error", code="internal_error") from exc
        finally:
            self._record_response_metrics(model=model, mode=mode, status=status, should_store=should_store, started_at=started_at)

    async def create_background(self, request: ResponseRequest, tenant_id: str | None) -> ResponseObject:
        model = request.model or self.settings.default_model
        validate_include_capabilities(request, model=model, settings=self.settings)
        response_id = generate_id("resp")
        request = self._stamp_input_artifacts(request, response_id=response_id)
        should_store = self._should_store(request)
        if not should_store:
            raise OpenAIError("Background mode requires store=true.", param="store", code="invalid_request")

        mode = "background"
        status = "queued"
        started_at = time.perf_counter()
        IN_FLIGHT_RESPONSES.labels(model=model, mode=mode).inc()
        if request.tools:
            FUNCTION_TOOL_REQUESTS.labels(model=model, status="queued").inc()
        try:
            input_items = self._input_items(request)
            await self.repository.create_response(
                response_id=response_id,
                model=model,
                previous_response_id=request.previous_response_id,
                input_json=self._stored_input(request),
                request_json=self._request_json(request),
                metadata_json=request.metadata,
                tenant_id=tenant_id,
                status="queued",
            )
            await self.repository.create_input_items(response_id=response_id, input_items=input_items)
            await self.repository.create_input_artifacts(response_id=response_id, input_items=input_items, tenant_id=tenant_id)
            timeout_at = datetime.now(timezone.utc) + timedelta(seconds=self.settings.background_job_timeout_seconds)
            await self.repository.create_background_job(response_id=response_id, timeout_at=timeout_at)
            await self.repository.session.commit()
            BACKGROUND_JOBS.labels(status="queued").inc()
            self._schedule_background_response(response_id, tenant_id)
            return self._response_object(response_id, model, "queued", [], ResponseUsage(), request.metadata, request=request, should_store=True)
        except Exception as exc:
            status = "failed"
            if isinstance(exc, OpenAIError):
                raise
            raise OpenAIError("Internal gateway error.", status_code=500, type="server_error", code="internal_error") from exc
        finally:
            self._record_response_metrics(model=model, mode=mode, status=status, should_store=True, started_at=started_at)

    async def run_background_response(self, response_id: str, tenant_id: str | None) -> None:
        job_started_at = time.perf_counter()
        running = False
        terminal_status = "failed"
        heartbeat_task: asyncio.Task | None = None
        try:
            job = await self.repository.start_background_job(response_id)
            if job.status == "cancelled":
                await self.repository.session.commit()
                terminal_status = "cancelled"
                BACKGROUND_JOBS.labels(status="cancelled").inc()
                return
            await self.repository.session.commit()
            running = True
            BACKGROUND_JOB_RUNNING.inc()
            BACKGROUND_JOBS.labels(status="in_progress").inc()
            heartbeat_task = self._start_heartbeat(response_id)

            record = await self.repository.require_response(response_id, tenant_id)
            request = ResponseRequest.model_validate(record.request_json or {})
            if request.tools:
                FUNCTION_TOOL_REQUESTS.labels(model=record.model, status="started").inc()
            output, usage, response_status, _ = await asyncio.wait_for(
                self._generate_response_output(
                    response_id=response_id,
                    model=record.model,
                    request=request,
                    tenant_id=tenant_id,
                    should_store=True,
                ),
                timeout=self.settings.background_job_timeout_seconds,
            )

            current_job = await self.repository.get_background_job(response_id)
            if current_job is not None:
                await self.repository.session.refresh(current_job)
            if current_job is not None and (current_job.cancellation_requested_at is not None or current_job.status == "cancelled"):
                await self.repository.cancel_response(response_id)
                await self.repository.complete_background_job(response_id, status="cancelled")
                await self.repository.session.commit()
                terminal_status = "cancelled"
                BACKGROUND_JOBS.labels(status="cancelled").inc()
                return

            await self.repository.complete_response(response_id, output, usage.model_dump(), status=response_status)
            await self.repository.complete_background_job(response_id, status=response_status)
            await self.repository.session.commit()
            terminal_status = response_status
            if request.tools:
                FUNCTION_TOOL_REQUESTS.labels(model=record.model, status=response_status).inc()
            self._record_reasoning_metrics(model=record.model, request=request, usage=usage, status=response_status)
            BACKGROUND_JOBS.labels(status=response_status).inc()
        except asyncio.TimeoutError as exc:
            terminal_status = "timeout"
            error = {"message": "Background response timed out.", "type": "server_error", "param": None, "code": "background_timeout"}
            await self.repository.fail_response(response_id, error)
            await self.repository.complete_background_job(response_id, status="timeout", error_json=error)
            await self.repository.session.commit()
            BACKGROUND_JOBS.labels(status="timeout").inc()
            return
        except asyncio.CancelledError:
            terminal_status = "cancelled"
            await self.repository.cancel_response(response_id)
            await self.repository.complete_background_job(response_id, status="cancelled")
            await self.repository.session.commit()
            BACKGROUND_JOBS.labels(status="cancelled").inc()
            raise
        except Exception as exc:
            terminal_status = "failed"
            error = _error_json(exc)
            await self.repository.fail_response(response_id, error)
            await self.repository.complete_background_job(response_id, status="failed", error_json=error)
            await self.repository.session.commit()
            with suppress(Exception):
                record = await self.repository.require_response(response_id, tenant_id, include_deleted=True)
                request = ResponseRequest.model_validate(record.request_json or {})
                if request.tools:
                    FUNCTION_TOOL_REQUESTS.labels(model=record.model, status="failed").inc()
            BACKGROUND_JOBS.labels(status="failed").inc()
            if isinstance(exc, OpenAIError):
                return
            return
        finally:
            if heartbeat_task is not None:
                heartbeat_task.cancel()
                with suppress(asyncio.CancelledError):
                    await heartbeat_task
            if running:
                BACKGROUND_JOB_RUNNING.dec()
            BACKGROUND_JOB_LATENCY.labels(status=terminal_status).observe(time.perf_counter() - job_started_at)

    async def cancel(self, response_id: str, tenant_id: str | None) -> ResponseObject:
        record = await self.repository.require_response(response_id, tenant_id)
        job = await self.repository.get_background_job(response_id)
        if job is None:
            raise OpenAIError("Only background responses can be cancelled.", status_code=400, param="response_id", code="invalid_request")
        await self.repository.request_background_cancel(response_id, tenant_id)
        await self.repository.session.commit()
        task = self.background_tasks.get(response_id) if self.background_tasks is not None else None
        if task is not None and not task.done():
            task.cancel()
        if record.status not in {"completed", "failed", "incomplete"}:
            BACKGROUND_JOBS.labels(status="cancel_requested").inc()
        return await self.retrieve(response_id, tenant_id)

    async def stream(self, request: ResponseRequest, tenant_id: str | None) -> AsyncIterator[str]:
        validate_text_responses_request(request)
        request = await self._render_prompt_request(request, tenant_id)
        validate_text_responses_request(request)
        model = request.model or self.settings.default_model
        validate_include_capabilities(request, model=model, settings=self.settings)
        validate_reasoning_capabilities(request, model=model, settings=self.settings)
        request = await prepare_multimodal_request(request, model=model, settings=self.settings, repository=self.repository, tenant_id=tenant_id)
        response_id = generate_id("resp")
        request = self._stamp_input_artifacts(request, response_id=response_id)
        created_at = int(time.time())
        output_text = ""
        reasoning_text = ""
        usage = ResponseUsage()
        sequence_number = 0
        mode = "stream"
        status = "cancelled"
        started_at = time.perf_counter()
        include_obfuscation = stream_obfuscation_enabled(request)

        def event(name: str, data: dict[str, Any]) -> str:
            nonlocal sequence_number
            rendered = make_event(name, data, sequence_number, event_id=f"{response_id}-{sequence_number}")
            sequence_number += 1
            return rendered

        def delta_event_data(data: dict[str, Any]) -> dict[str, Any]:
            if include_obfuscation:
                return {**data, "obfuscation": secrets.token_urlsafe(8)}
            return data

        should_store = self._should_store(request)
        response_created = False
        IN_FLIGHT_RESPONSES.labels(model=model, mode=mode).inc()
        STREAMING_RESPONSES_RUNNING.labels(model=model).inc()
        if request.tools:
            FUNCTION_TOOL_REQUESTS.labels(model=model, status="started").inc()
        try:
            if should_store:
                input_items = self._input_items(request)
                await self.repository.create_response(
                    response_id=response_id,
                    model=model,
                    previous_response_id=request.previous_response_id,
                    input_json=self._stored_input(request),
                    request_json=self._request_json(request),
                    metadata_json=request.metadata,
                    tenant_id=tenant_id,
                )
                await self.repository.create_input_items(response_id=response_id, input_items=input_items)
                await self.repository.create_input_artifacts(response_id=response_id, input_items=input_items, tenant_id=tenant_id)
                response_created = True
            in_progress_response = self._response_object(response_id, model, "in_progress", [], usage, request.metadata, request=request, should_store=should_store, created_at=created_at).model_dump()
            yield event("response.created", {"response": in_progress_response})
            chain = await self.repository.load_chain(request.previous_response_id, tenant_id, self.settings.max_chain_depth)
            validate_function_call_outputs_match(chain, request.input)
            submitted_tool_outputs = function_call_output_ids(request.input)
            if submitted_tool_outputs:
                FUNCTION_TOOL_OUTPUTS.labels(model=model).inc(len(submitted_tool_outputs))
            plan = await self._plan_context(model=model, request=request, chain=chain, response_id=response_id, should_store=should_store, mode=mode)
            payload = self._payload(model=model, request=plan.request, chain=plan.chain)
            cache_match = self.prompt_cache.inspect(payload, prompt_cache_key=request.prompt_cache_key, retention=request.prompt_cache_retention)
            self._record_prompt_cache_metrics(cache_match)
            output_index = 0
            emitted_items: list[tuple[int, dict[str, Any]]] = []
            if plan.compaction_item is not None:
                compaction_item = {**plan.compaction_item, "status": "completed"}
                if should_store:
                    await self.repository.create_output_item(response_id=response_id, output_index=output_index, item=compaction_item)
                yield event(
                    "response.output_item.added",
                    {
                        "output_index": output_index,
                        "item": compaction_item,
                    },
                )
                yield event("response.output_item.done", {"output_index": output_index, "item": compaction_item})
                emitted_items.append((output_index, compaction_item))
                output_index += 1
            reasoning_id: str | None = None
            finish_reason: str | None = None
            yield event("response.in_progress", {"response": in_progress_response})
            if reasoning_requested(request):
                reasoning_id = generate_id("rs")
                if should_store:
                    await self.repository.create_output_item(
                        response_id=response_id,
                        output_index=output_index,
                        item={"id": reasoning_id, "type": "reasoning", "summary": [], "status": "in_progress"},
                    )
                yield event(
                    "response.output_item.added",
                    {
                        "output_index": output_index,
                        "item": {"id": reasoning_id, "type": "reasoning", "summary": [], "status": "in_progress"},
                    },
                )
                output_index += 1
            output_id: str | None = None
            text_output_index: int | None = None
            async for chunk in self.backend.create_chat_completion_stream(payload):
                if chunk.get("type") == "reasoning_delta":
                    reasoning_text += chunk.get("delta", "")
                elif chunk.get("type") == "delta":
                    delta = chunk.get("delta", "")
                    if output_id is None:
                        output_id = generate_id("msg")
                        text_output_index = output_index
                        if should_store:
                            await self.repository.create_output_item(
                                response_id=response_id,
                                output_index=text_output_index,
                                item={"id": output_id, "type": "message", "role": "assistant", "status": "in_progress", "content": []},
                            )
                        yield event(
                            "response.output_item.added",
                            {
                                "output_index": text_output_index,
                                "item": {"id": output_id, "type": "message", "role": "assistant", "status": "in_progress", "content": []},
                            },
                        )
                        yield event(
                            "response.content_part.added",
                            {
                                "item_id": output_id,
                                "output_index": text_output_index,
                                "content_index": 0,
                                "part": {"type": "output_text", "text": "", "annotations": [], "logprobs": []},
                            },
                        )
                        output_index += 1
                    output_text += delta
                    yield event(
                        "response.output_text.delta",
                        delta_event_data(
                            {
                                "item_id": output_id,
                                "output_index": text_output_index,
                                "content_index": 0,
                                "delta": delta,
                                "logprobs": [],
                            }
                        ),
                    )
                elif chunk.get("type") == "tool_calls":
                    for item in self._function_call_output_items(chunk.get("tool_calls") or [], request):
                        current_output_index = output_index
                        added_item = {**item, "arguments": "", "status": "in_progress"}
                        if should_store:
                            await self.repository.create_output_item(
                                response_id=response_id,
                                output_index=current_output_index,
                                item=added_item,
                            )
                        yield event(
                            "response.output_item.added",
                            {"response_id": response_id, "output_index": current_output_index, "item": added_item},
                        )
                        if item["arguments"]:
                            yield event(
                                "response.function_call_arguments.delta",
                                delta_event_data(
                                    {
                                        "response_id": response_id,
                                        "item_id": item["id"],
                                        "output_index": current_output_index,
                                        "delta": item["arguments"],
                                    }
                                ),
                            )
                        yield event(
                            "response.function_call_arguments.done",
                            {
                                "response_id": response_id,
                                "item_id": item["id"],
                                "output_index": current_output_index,
                                "arguments": item["arguments"],
                            },
                        )
                        yield event("response.output_item.done", {"response_id": response_id, "output_index": current_output_index, "item": item})
                        emitted_items.append((current_output_index, item))
                        output_index += 1
                elif chunk.get("type") == "done":
                    usage = normalize_usage(chunk.get("usage"))
                    finish_reason = chunk.get("finish_reason")
            streamed_tool_calls = [self._backend_tool_call_from_item(item) for _, item in emitted_items if item.get("type") == "function_call"]
            self._validate_tool_result(ChatCompletionResult(content=output_text, reasoning=reasoning_text, finish_reason=finish_reason, tool_calls=streamed_tool_calls), request, model)
            if output_text or not streamed_tool_calls:
                validate_text_against_schema(output_text, schema_from_response_format(self._structured_format(request)))
            usage = enrich_response_usage(
                usage,
                cached_tokens=cache_match.cached_tokens,
                reasoning_tokens=self._reasoning_tokens(reasoning_text, usage),
                minimum_output_tokens=estimate_text_tokens(output_text),
            )
            output = []
            if reasoning_id is not None:
                reasoning_item = self._reasoning_output_item(reasoning_id, reasoning_text, request, response_id=response_id)
                output.append(reasoning_item)
                if reasoning_item["summary"]:
                    summary_part = reasoning_item["summary"][0]
                    yield event(
                        "response.reasoning_summary_part.added",
                        {"item_id": reasoning_id, "output_index": 0, "summary_index": 0, "part": {"type": "summary_text", "text": ""}},
                    )
                    yield event(
                        "response.reasoning_summary_text.delta",
                        delta_event_data({"item_id": reasoning_id, "output_index": 0, "summary_index": 0, "delta": summary_part["text"]}),
                    )
                    yield event(
                        "response.reasoning_summary_text.done",
                        {"item_id": reasoning_id, "output_index": 0, "summary_index": 0, "text": summary_part["text"]},
                    )
                    yield event(
                        "response.reasoning_summary_part.done",
                        {"item_id": reasoning_id, "output_index": 0, "summary_index": 0, "part": summary_part},
                    )
                yield event("response.output_item.done", {"output_index": 0, "item": reasoning_item})
            if output_id is None and not emitted_items:
                output_id = generate_id("msg")
                text_output_index = output_index
                if should_store:
                    await self.repository.create_output_item(
                        response_id=response_id,
                        output_index=text_output_index,
                        item={"id": output_id, "type": "message", "role": "assistant", "status": "in_progress", "content": []},
                    )
                yield event(
                    "response.output_item.added",
                    {
                        "output_index": text_output_index,
                        "item": {"id": output_id, "type": "message", "role": "assistant", "status": "in_progress", "content": []},
                    },
                )
                yield event(
                    "response.content_part.added",
                    {
                        "item_id": output_id,
                        "output_index": text_output_index,
                        "content_index": 0,
                        "part": {"type": "output_text", "text": "", "annotations": [], "logprobs": []},
                    },
                )
            if output_id is not None and text_output_index is not None:
                emitted_items.append((text_output_index, assistant_text_to_output(output_id, output_text, annotations=self._output_annotations(request))))
            output.extend(item for _, item in sorted(emitted_items, key=lambda pair: pair[0]))
            self._record_token_usage(model, usage)
            response_status, incomplete_details = self._completion_status(finish_reason)
            self._record_reasoning_metrics(model=model, request=request, usage=usage, status=response_status)
            self.prompt_cache.store(payload, prompt_cache_key=request.prompt_cache_key, retention=request.prompt_cache_retention)
            if response_status == "incomplete" and output:
                output[-1]["status"] = "incomplete"
            if should_store:
                await self.repository.complete_response(response_id, output, usage.model_dump(), status=response_status)
                await self.repository.session.commit()
            text_item = next((item for item in output if item.get("id") == output_id and item.get("type") == "message"), None)
            if text_item is not None and text_output_index is not None:
                yield event(
                    "response.output_text.done",
                    {"item_id": output_id, "output_index": text_output_index, "content_index": 0, "text": output_text, "logprobs": []},
                )
                yield event(
                    "response.content_part.done",
                    {
                        "item_id": output_id,
                        "output_index": text_output_index,
                        "content_index": 0,
                        "part": {"type": "output_text", "text": output_text, "annotations": self._output_annotations(request), "logprobs": []},
                    },
                )
                yield event("response.output_item.done", {"output_index": text_output_index, "item": text_item})
            completed = self._response_object(response_id, model, response_status, output, usage, request.metadata, request=request, should_store=should_store, incomplete_details=incomplete_details, created_at=created_at).model_dump()
            status = response_status
            if request.tools:
                FUNCTION_TOOL_REQUESTS.labels(model=model, status=response_status).inc()
            yield event("response.incomplete" if response_status == "incomplete" else "response.completed", {"response": completed})
        except asyncio.CancelledError:
            status = "cancelled"
            if request.tools:
                FUNCTION_TOOL_REQUESTS.labels(model=model, status="cancelled").inc()
            if should_store and response_created:
                await self.repository.cancel_response(response_id)
                await self.repository.session.commit()
            raise
        except Exception as exc:
            status = "failed"
            if request.tools:
                FUNCTION_TOOL_REQUESTS.labels(model=model, status="failed").inc()
            self._record_reasoning_metrics(model=model, request=request, usage=usage, status="failed")
            if should_store and response_created:
                await self.repository.fail_response(response_id, _error_json(exc))
                await self.repository.session.commit()
            yield event("response.failed", {"response": self._response_object(response_id, model, "failed", [], usage, request.metadata, request=request, should_store=should_store, error=_error_json(exc), created_at=created_at).model_dump()})
            yield event("error", {"error": _error_json(exc)})
        finally:
            STREAMING_RESPONSES_RUNNING.labels(model=model).dec()
            self._record_response_metrics(model=model, mode=mode, status=status, should_store=should_store, started_at=started_at)

    async def retrieve(self, response_id: str, tenant_id: str | None, *, include: list[str] | None = None) -> ResponseObject:
        if include is not None:
            validate_include_values(include)
        record = await self.repository.require_response(response_id, tenant_id)
        output = await self.repository.list_output_items(response_id, tenant_id)
        artifacts = await self.repository.list_artifacts(response_id, tenant_id)
        request_json = record.request_json or {}
        effective_include = self._effective_includes(request_json, include)
        self._validate_retrieve_include_capabilities(record.model, output or record.output_json or [], effective_include)
        return self._response_object(
            record.id,
            record.model,
            record.status,
            output or record.output_json or [],
            normalize_usage(record.usage_json),
            record.metadata_json or {},
            request_json=request_json,
            should_store=bool((record.request_json or {}).get("store", self.settings.store_default)),
            error=record.error_json,
            incomplete_details=self._stored_incomplete_details(record.status),
            created_at=int(record.created_at.timestamp()),
            include=effective_include,
            artifacts=artifacts,
        )

    async def list_input_items(self, response_id: str, tenant_id: str | None, *, after: str | None, before: str | None, limit: int, order: str, include: list[str] | None = None) -> ResponseInputItemList:
        if include is not None:
            validate_include_values(include)
        record = await self.repository.require_response(response_id, tenant_id)
        items = await self.repository.list_input_items(response_id, tenant_id)
        if not items:
            items = input_items_from_request(record.request_json or {})
        page, has_more = paginate_items(items, after=after, before=before, limit=limit, order=order)
        return ResponseInputItemList(
            data=page,
            first_id=page[0]["id"] if page else None,
            last_id=page[-1]["id"] if page else None,
            has_more=has_more,
        )

    async def list_artifacts(self, response_id: str, tenant_id: str | None, *, after: str | None, before: str | None, limit: int, order: str) -> ResponseArtifactList:
        await self.repository.require_response(response_id, tenant_id)
        artifacts = await self.repository.list_artifacts(response_id, tenant_id)
        page, has_more = paginate_items(artifacts, after=after, before=before, limit=limit, order=order)
        return ResponseArtifactList(
            data=[_public_artifact(artifact, include_content=False) for artifact in page],
            first_id=page[0]["id"] if page else None,
            last_id=page[-1]["id"] if page else None,
            has_more=has_more,
        )

    async def count_input_tokens(self, request: ResponseRequest, tenant_id: str | None) -> ResponseInputTokenCount:
        validate_text_responses_request(request)
        request = await self._render_prompt_request(request, tenant_id)
        validate_text_responses_request(request)
        model = request.model or self.settings.default_model
        validate_reasoning_capabilities(request, model=model, settings=self.settings)
        request = await prepare_multimodal_request(request, model=model, settings=self.settings, repository=self.repository, tenant_id=tenant_id)
        chain = await self.repository.load_chain(request.previous_response_id, tenant_id, self.settings.max_chain_depth)
        validate_function_call_outputs_match(chain, request.input)
        payload = self._payload(model=model, request=request, chain=chain)
        cache_match = self.prompt_cache.inspect(payload, prompt_cache_key=request.prompt_cache_key, retention=request.prompt_cache_retention)
        self._record_prompt_cache_metrics(cache_match)
        input_tokens = max(cache_match.input_tokens, self.context_planner.count_payload_tokens(payload), estimate_input_tokens(request, chain))
        return ResponseInputTokenCount(input_tokens=input_tokens, input_tokens_details={"cached_tokens": min(cache_match.cached_tokens, input_tokens)})

    async def compact(self, request: ResponseRequest, tenant_id: str | None) -> ResponseCompactionObject:
        validate_text_responses_request(request)
        request = await self._render_prompt_request(request, tenant_id)
        validate_text_responses_request(request)
        if request.previous_response_id is not None:
            raise OpenAIError("responses/compact is stateless and does not accept previous_response_id.", param="previous_response_id", code="invalid_request")
        model = request.model or self.settings.default_model
        validate_reasoning_capabilities(request, model=model, settings=self.settings)
        request = await prepare_multimodal_request(request, model=model, settings=self.settings, repository=self.repository, tenant_id=tenant_id)
        response_id = generate_id("resp")
        output, after_request, summary, before, after = compact_response_window(input_value=request.input, model=model, settings=self.settings)
        usage = ResponseUsage(input_tokens=before, output_tokens=after, total_tokens=before + after)
        await self.repository.save_context_event(
            response_id=response_id,
            source_response_id=None,
            type="compaction",
            strategy="standalone_compact",
            compacted_item_id=output[-1].get("id") if output else None,
            source_item_ids=summary.get("source_item_ids", []),
            summary_json=summary,
            input_tokens_before=before,
            input_tokens_after=after,
        )
        await self.repository.session.commit()
        self._record_compaction_metrics(model=model, mode="standalone", before=before, after=after)
        return ResponseCompactionObject(id=response_id, created_at=int(time.time()), output=output, usage=usage)

    async def delete(self, response_id: str, tenant_id: str | None) -> None:
        await self.repository.soft_delete(response_id, tenant_id)
        await self.repository.session.commit()

    async def _generate_response_output(
        self,
        *,
        response_id: str,
        model: str,
        request: ResponseRequest,
        tenant_id: str | None,
        should_store: bool,
    ) -> tuple[list[dict[str, Any]], ResponseUsage, str, dict[str, Any] | None]:
        chain = await self.repository.load_chain(request.previous_response_id, tenant_id, self.settings.max_chain_depth)
        validate_function_call_outputs_match(chain, request.input)
        submitted_tool_outputs = function_call_output_ids(request.input)
        if submitted_tool_outputs:
            FUNCTION_TOOL_OUTPUTS.labels(model=model).inc(len(submitted_tool_outputs))
        plan = await self._plan_context(model=model, request=request, chain=chain, response_id=response_id, should_store=should_store, mode="create")
        payload = self._payload(model=model, request=plan.request, chain=plan.chain)
        cache_match = self.prompt_cache.inspect(payload, prompt_cache_key=request.prompt_cache_key, retention=request.prompt_cache_retention)
        self._record_prompt_cache_metrics(cache_match)
        result = await self._run_with_structured_repair(response_id=response_id, payload=payload, request=plan.request, should_store=should_store)
        self._validate_logprobs_result(result, request, model)
        self._validate_tool_result(result, request, model)
        usage = self._usage(result, cache_match)
        response_status, incomplete_details = self._completion_status(result.finish_reason)
        output = self._output_items(result, request, response_id=response_id, response_status=response_status)
        if plan.compaction_item is not None:
            output.insert(0, {**plan.compaction_item, "status": "completed"})
        self._record_token_usage(model, usage)
        self.prompt_cache.store(payload, prompt_cache_key=request.prompt_cache_key, retention=request.prompt_cache_retention)
        return output, usage, response_status, incomplete_details

    async def _plan_context(
        self,
        *,
        model: str,
        request: ResponseRequest,
        chain: list[dict[str, Any]],
        response_id: str,
        should_store: bool,
        mode: str,
    ) -> ContextPlan:
        try:
            plan = self.context_planner.plan(model=model, request=request, chain=chain)
        except OpenAIError as exc:
            if exc.code == "context_length_exceeded":
                CONTEXT_OVERFLOWS.labels(model=model, truncation=request.truncation).inc()
            raise

        if plan.strategy == "truncation_auto":
            CONTEXT_TRUNCATIONS.labels(model=model, reason="context_window").inc(plan.truncated_items or 1)
            if should_store:
                await self.repository.save_context_event(
                    response_id=response_id,
                    source_response_id=request.previous_response_id,
                    type="truncation",
                    strategy=plan.strategy,
                    compacted_item_id=None,
                    source_item_ids=[],
                    summary_json=None,
                    input_tokens_before=plan.input_tokens_before,
                    input_tokens_after=plan.input_tokens_after,
                )
        elif plan.compaction_item is not None:
            self._record_compaction_metrics(model=model, mode=mode, before=plan.input_tokens_before, after=plan.input_tokens_after)
            if should_store:
                await self.repository.save_context_event(
                    response_id=response_id,
                    source_response_id=request.previous_response_id,
                    type="compaction",
                    strategy=plan.strategy,
                    compacted_item_id=plan.compaction_item.get("id"),
                    source_item_ids=plan.source_item_ids or [],
                    summary_json=plan.compaction_summary,
                    input_tokens_before=plan.input_tokens_before,
                    input_tokens_after=plan.input_tokens_after,
                )
        return plan

    def _record_compaction_metrics(self, *, model: str, mode: str, before: int, after: int) -> None:
        CONTEXT_COMPACTIONS.labels(model=model, mode=mode).inc()
        CONTEXT_COMPACTION_TOKENS.labels(model=model, mode=mode, phase="before").inc(before)
        CONTEXT_COMPACTION_TOKENS.labels(model=model, mode=mode, phase="after").inc(after)
        CONTEXT_TRUNCATIONS.labels(model=model, reason="context_window").inc(0)
        CONTEXT_OVERFLOWS.labels(model=model, truncation="disabled").inc(0)
        ratio = after / before if before > 0 else 0.0
        CONTEXT_COMPACTION_RATIO.labels(model=model, mode=mode).observe(ratio)

    async def _render_prompt_request(self, request: ResponseRequest, tenant_id: str | None) -> ResponseRequest:
        return await self.prompt_template_renderer.render_request(request, tenant_id)

    def _record_prompt_cache_metrics(self, cache_match: PromptCacheMatch) -> None:
        status = "hit" if cache_match.cached_tokens > 0 else "miss"
        PROMPT_CACHE_REQUESTS.labels(retention=cache_match.retention, status=status).inc()
        PROMPT_CACHE_TOKENS.labels(kind="input").inc(cache_match.input_tokens)
        PROMPT_CACHE_TOKENS.labels(kind="cached").inc(cache_match.cached_tokens)
        ratio = cache_match.cached_tokens / cache_match.input_tokens if cache_match.input_tokens > 0 else 0.0
        PROMPT_CACHE_HIT_RATIO.labels(retention=cache_match.retention).set(ratio)

    def _schedule_background_response(self, response_id: str, tenant_id: str | None) -> None:
        if self.session_factory is None or self.background_tasks is None:
            return
        schedule_background_response(
            settings=self.settings,
            session_factory=self.session_factory,
            backend=self.backend,
            prompt_cache=self.prompt_cache,
            background_tasks=self.background_tasks,
            response_id=response_id,
            tenant_id=tenant_id,
        )

    def _start_heartbeat(self, response_id: str) -> asyncio.Task | None:
        if self.session_factory is None:
            return None
        return asyncio.create_task(
            _heartbeat_background_job(
                response_id=response_id,
                session_factory=self.session_factory,
                interval_seconds=self.settings.background_job_heartbeat_seconds,
            )
        )

    def _payload(self, *, model: str, request: ResponseRequest, chain: list[dict[str, Any]]) -> dict[str, Any]:
        messages = build_messages(
            instructions=request.instructions,
            chain=chain,
            input_value=request.input,
            compaction_key=self.settings.reasoning_encryption_key,
        )
        instruction = tool_choice_instruction(request)
        if instruction:
            messages.insert(0, {"role": "system", "content": instruction})
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_output_tokens or self.settings.max_output_tokens_default,
        }
        tools = backend_function_tools(request)
        if request.tool_choice == "none":
            tools = []
        if tools:
            payload["tools"] = tools
        tool_choice = backend_tool_choice(request)
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if request.parallel_tool_calls is not None:
            payload["parallel_tool_calls"] = request.parallel_tool_calls
        if request.max_tool_calls is not None:
            payload["max_tool_calls"] = request.max_tool_calls
        structured_format = self._structured_format(request)
        if structured_format:
            payload["response_format"] = structured_format
        if request.reasoning is not None:
            payload["reasoning"] = request.reasoning
        if request.top_logprobs is not None or OUTPUT_TEXT_LOGPROBS in requested_includes(request):
            payload["top_logprobs"] = request.top_logprobs or 0
        return {k: v for k, v in payload.items() if v is not None}

    def _output_items(self, result: ChatCompletionResult, request: ResponseRequest, *, response_id: str, response_status: str = "completed") -> list[dict[str, Any]]:
        output = [*result.output_items]
        if reasoning_requested(request):
            item_id = generate_id("rs")
            output.insert(0, self._reasoning_output_item(item_id, result.reasoning, request, response_id=response_id))
        function_calls = self._function_call_output_items(result.tool_calls, request)
        output.extend(function_calls)
        if result.content or not function_calls:
            item = assistant_text_to_output(
                generate_id("msg"),
                result.content,
                annotations=self._output_annotations(request),
                logprobs=result.content_logprobs,
            )
            if response_status == "incomplete":
                item["status"] = "incomplete"
            output.append(item)
        return output

    def _function_call_output_items(self, tool_calls: list[dict[str, Any]], request: ResponseRequest) -> list[dict[str, Any]]:
        items = []
        for call in tool_calls:
            function = dict(call.get("function") or {})
            call_id = str(call.get("id") or generate_id("call"))
            item = {
                "id": generate_id("fc"),
                "type": "function_call",
                "status": "completed",
                "call_id": call_id,
                "name": str(function.get("name") or ""),
                "arguments": self._arguments_to_string(function.get("arguments", "{}")),
            }
            items.append(item)
        return items

    def _backend_tool_call_from_item(self, item: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": item["call_id"],
            "type": "function",
            "function": {
                "name": item["name"],
                "arguments": item["arguments"],
            },
        }

    def _validate_tool_result(self, result: ChatCompletionResult, request: ResponseRequest, model: str) -> None:
        tool_call_count = len(result.tool_calls)
        if request.parallel_tool_calls is False and tool_call_count > 1:
            self._tool_capability_error(model, "parallel_tool_calls", "Configured backend/model returned parallel function calls while parallel_tool_calls=false.")
        if request.max_tool_calls is not None and tool_call_count > request.max_tool_calls:
            self._tool_capability_error(model, "max_tool_calls", "Configured backend/model returned more function calls than max_tool_calls.")
        if self._tool_choice_requires_output(request) and tool_call_count == 0:
            self._tool_capability_error(model, "required_tool_choice", "Configured backend/model did not produce a function_call for the required tool choice.")
        forced_name = self._forced_tool_name(request)
        if forced_name is not None:
            called_names = [str((call.get("function") or {}).get("name") or "") for call in result.tool_calls]
            if called_names != [forced_name]:
                self._tool_capability_error(model, "forced_tool_choice", "Configured backend/model did not call the requested function.")
        if tool_call_count:
            FUNCTION_TOOL_CALLS.labels(model=model).inc(tool_call_count)

    def _validate_logprobs_result(self, result: ChatCompletionResult, request: ResponseRequest, model: str) -> None:
        if not (request.top_logprobs is not None or OUTPUT_TEXT_LOGPROBS in requested_includes(request)):
            return
        if not result.content or result.content_logprobs:
            return
        INCLUDE_EXPANSIONS.labels(include=OUTPUT_TEXT_LOGPROBS, status="missing_backend_data").inc()
        raise OpenAIError(
            "Configured backend/model did not return output text logprobs.",
            status_code=400,
            param="include" if OUTPUT_TEXT_LOGPROBS in requested_includes(request) else "top_logprobs",
            code="unsupported_model_capability",
        )

    def _tool_choice_requires_output(self, request: ResponseRequest) -> bool:
        tool_choice = request.tool_choice
        if tool_choice == "required":
            return True
        if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            return True
        if isinstance(tool_choice, dict) and tool_choice.get("type") == "allowed_tools" and tool_choice.get("mode") == "required":
            return True
        return False

    def _forced_tool_name(self, request: ResponseRequest) -> str | None:
        tool_choice = request.tool_choice
        if not isinstance(tool_choice, dict) or tool_choice.get("type") != "function":
            return None
        function = tool_choice.get("function") if isinstance(tool_choice.get("function"), dict) else {}
        return str(tool_choice.get("name") or function.get("name") or "")

    def _tool_capability_error(self, model: str, reason: str, message: str) -> None:
        FUNCTION_TOOL_CAPABILITY_ERRORS.labels(model=model, reason=reason).inc()
        raise OpenAIError(message, status_code=400, param="tools", code="unsupported_model_capability")

    def _usage(self, result: ChatCompletionResult, cache_match: PromptCacheMatch) -> ResponseUsage:
        usage = normalize_usage(result.usage)
        return enrich_response_usage(
            usage,
            cached_tokens=cache_match.cached_tokens,
            reasoning_tokens=self._reasoning_tokens(result.reasoning, usage),
            minimum_output_tokens=estimate_text_tokens(result.content),
        )

    def _reasoning_tokens(self, reasoning_text: str, usage: ResponseUsage) -> int:
        reported = usage.output_tokens_details.reasoning_tokens
        estimated = estimate_text_tokens(reasoning_text)
        return max(reported, estimated)

    async def _run_with_structured_repair(
        self,
        *,
        response_id: str,
        payload: dict[str, Any],
        request: ResponseRequest,
        should_store: bool,
    ):
        schema = schema_from_response_format(self._structured_format(request))
        result = await self.backend.create_chat_completion(payload)
        try:
            validate_text_against_schema(result.content, schema)
            return result
        except OpenAIError as exc:
            if not schema or exc.code != "structured_output_validation_failed":
                raise

        repair_payload = dict(payload)
        repair_payload["messages"] = [
            *payload["messages"],
            {"role": "assistant", "content": result.content},
            {"role": "system", "content": repair_instruction(schema)},
        ]
        repaired = await self.backend.create_chat_completion(repair_payload)
        validate_text_against_schema(repaired.content, schema)
        return repaired

    def _should_store(self, request: ResponseRequest) -> bool:
        return self.settings.store_default if request.store is None else request.store

    def _structured_format(self, request: ResponseRequest) -> dict[str, Any] | None:
        text_config = normalized_text_config(request)
        format_config = text_config.get("format")
        if isinstance(format_config, dict) and format_config.get("type") == "text":
            return None
        return format_config if isinstance(format_config, dict) else None

    def _input_items(self, request: ResponseRequest) -> list[dict[str, Any]]:
        input_value = request.input
        if input_value is None:
            return []
        if isinstance(input_value, str):
            return [
                {
                    "id": generate_id("msg"),
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": input_value}],
                    "status": "completed",
                }
            ]

        items: list[dict[str, Any]] = []
        for item in input_value:
            item_type = item.get("type")
            role = item.get("role")
            if item_type == "compaction":
                items.append(
                    {
                        "id": str(item.get("id") or generate_id("cmp")),
                        "type": "compaction",
                        "encrypted_content": str(item.get("encrypted_content") or ""),
                        "status": item.get("status", "completed"),
                    }
                )
                continue
            if item_type == "reasoning":
                reasoning_item = {
                    "id": str(item.get("id") or generate_id("rs")),
                    "type": "reasoning",
                    "summary": item.get("summary", []),
                    "status": item.get("status", "completed"),
                }
                if item.get("encrypted_content") is not None:
                    reasoning_item["encrypted_content"] = item.get("encrypted_content")
                items.append(reasoning_item)
                continue
            if item_type == "function_call":
                items.append(
                    {
                        "id": generate_id("fc"),
                        "type": "function_call",
                        "call_id": str(item["call_id"]),
                        "name": str(item["name"]),
                        "arguments": self._arguments_to_string(item.get("arguments", "{}")),
                        "status": item.get("status", "completed"),
                    }
                )
                continue
            if item_type == "function_call_output":
                items.append(
                    {
                        "id": generate_id("fco"),
                        "type": "function_call_output",
                        "call_id": str(item["call_id"]),
                        "output": item.get("output", ""),
                        "status": item.get("status", "completed"),
                    }
                )
                continue
            if item_type == "message" or role in {"user", "assistant", "system", "developer"}:
                items.append(
                    {
                        "id": generate_id("msg"),
                        "type": "message",
                        "role": role or "user",
                        "content": self._input_content_parts(item.get("content", "")),
                        "status": item.get("status", "completed"),
                    }
                )
        return items

    def _input_content_parts(self, content: Any) -> list[dict[str, Any]]:
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") in {"input_image", "input_file"}:
                        parts.append(dict(part))
                        continue
                    text = part.get("text", part.get("output_text", ""))
                else:
                    text = str(part)
                parts.append({"type": "input_text", "text": str(text)})
            return parts
        if isinstance(content, dict) and content.get("type") in {"input_image", "input_file"}:
            return [dict(content)]
        if isinstance(content, dict):
            return [{"type": "input_text", "text": str(content.get("text", content))}]
        return [{"type": "input_text", "text": str(content)}]

    def _arguments_to_string(self, arguments: Any) -> str:
        if isinstance(arguments, str):
            value = arguments
        else:
            value = json.dumps(arguments, separators=(",", ":"), ensure_ascii=False)
        try:
            json.loads(value or "{}")
        except json.JSONDecodeError as exc:
            raise OpenAIError("Backend returned malformed function call arguments.", status_code=502, type="server_error", param="tools", code="backend_tool_arguments_invalid") from exc
        return value

    def _stored_input(self, request: ResponseRequest) -> Any:
        return request.input if request.input is not None else []

    def _request_json(self, request: ResponseRequest) -> dict[str, Any]:
        return request.model_dump(exclude_none=True)

    def _completion_status(self, finish_reason: str | None) -> tuple[str, dict[str, Any] | None]:
        if finish_reason in {"length", "max_tokens"}:
            return "incomplete", {"reason": "max_tokens"}
        return "completed", None

    def _stored_incomplete_details(self, status: str) -> dict[str, Any] | None:
        if status == "incomplete":
            return {"reason": "max_tokens"}
        return None

    def _request_snapshot(self, *, request: ResponseRequest | None, request_json: dict[str, Any] | None) -> dict[str, Any]:
        snapshot = request.model_dump(exclude_none=True) if request is not None else dict(request_json or {})
        snapshot.setdefault("tool_choice", "auto")
        snapshot.setdefault("tools", [])
        snapshot.setdefault("truncation", "disabled")
        snapshot.setdefault("service_tier", "auto")
        return snapshot

    def _response_reasoning(self, request_snapshot: dict[str, Any]) -> dict[str, Any]:
        reasoning = request_snapshot.get("reasoning")
        if isinstance(reasoning, dict):
            return {"effort": reasoning.get("effort"), "summary": reasoning.get("summary")}
        return {"effort": None, "summary": None}

    def _effective_includes(self, request_snapshot: dict[str, Any], include: list[str] | None) -> list[str]:
        values: list[str] = []
        for value in request_snapshot.get("include") or []:
            if isinstance(value, str) and value not in values:
                values.append(value)
        for value in include or []:
            if isinstance(value, str) and value not in values:
                values.append(value)
        return values

    def _validate_retrieve_include_capabilities(self, model: str, output: list[dict[str, Any]], include: list[str]) -> None:
        if OUTPUT_TEXT_LOGPROBS not in include or self._output_has_logprobs(output):
            return
        validate_include_capabilities({"include": [OUTPUT_TEXT_LOGPROBS]}, model=model, settings=self.settings)

    def _output_has_logprobs(self, output: list[dict[str, Any]]) -> bool:
        for item in output or []:
            for part in item.get("content") or []:
                if isinstance(part, dict) and part.get("type") == "output_text" and part.get("logprobs"):
                    return True
        return False

    def _response_input(self, input_value: Any, include: list[str], artifacts: list[dict[str, Any]] | None) -> Any:
        if not isinstance(input_value, list):
            return input_value
        artifacts_by_id = {artifact["id"]: artifact for artifact in artifacts or [] if isinstance(artifact, dict) and artifact.get("id")}
        if not artifacts_by_id:
            artifacts_by_id = self._artifact_map_from_input(input_value)
        return [self._response_input_item(item, include, artifacts_by_id) for item in input_value]

    def _response_input_item(self, item: Any, include: list[str], artifacts_by_id: dict[str, dict[str, Any]]) -> Any:
        if not isinstance(item, dict):
            return item
        next_item = {key: value for key, value in item.items() if not key.startswith("_respawn_")}
        content = next_item.get("content")
        if isinstance(content, list):
            next_item["content"] = [self._response_content_part(part, include, artifacts_by_id) for part in content]
        elif isinstance(content, dict):
            next_item["content"] = self._response_content_part(content, include, artifacts_by_id)
        elif item.get("type") in {"input_image", "input_file"}:
            next_item = self._response_content_part(item, include, artifacts_by_id)
        return next_item

    def _response_content_part(self, part: Any, include: list[str], artifacts_by_id: dict[str, dict[str, Any]]) -> Any:
        if not isinstance(part, dict):
            return part
        next_part = {key: value for key, value in part.items() if not key.startswith("_respawn_")}
        artifact_id = part.get("_respawn_artifact_id")
        artifact = artifacts_by_id.get(artifact_id) if isinstance(artifact_id, str) else None
        if part.get("type") == "input_image" and input_image_url_requested(include):
            if artifact is not None:
                next_part["artifact"] = _public_artifact(artifact, include_content=False)
            if not next_part.get("image_url") and artifact is not None:
                source = artifact.get("source") or {}
                if source.get("type") == "url":
                    next_part["image_url"] = source.get("url")
        return next_part

    def _response_output(self, output: list[dict[str, Any]], include: list[str]) -> list[dict[str, Any]]:
        include_logprobs = logprobs_requested(include)
        serialized = []
        for item in output or []:
            next_item = dict(item)
            content = next_item.get("content")
            if isinstance(content, list):
                next_content = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "output_text":
                        next_part = dict(part)
                        if not include_logprobs:
                            next_part["logprobs"] = []
                        next_content.append(next_part)
                    else:
                        next_content.append(part)
                next_item["content"] = next_content
            serialized.append(next_item)
        return serialized

    def _enforce_include_payload_limit(self, include: list[str], *, input_value: Any, output: list[dict[str, Any]]) -> None:
        if not include:
            return
        size = len(json.dumps({"input": input_value, "output": output}, default=str, separators=(",", ":")).encode("utf-8"))
        if size > self.settings.include_expansion_max_bytes:
            raise OpenAIError(
                f"Expanded include payload exceeds the {self.settings.include_expansion_max_bytes} byte limit.",
                param="include",
                code="payload_too_large",
            )
        for value in include:
            INCLUDE_EXPANSIONS.labels(include=value, status="expanded").inc()
            INCLUDE_EXPANSION_BYTES.labels(include=value).inc(size)

    def _stamp_input_artifacts(self, request: ResponseRequest, *, response_id: str) -> ResponseRequest:
        if not isinstance(request.input, list):
            return request
        stamped = [self._stamp_input_item_artifacts(item, response_id=response_id) for item in request.input]
        return request.model_copy(update={"input": stamped})

    def _stamp_input_item_artifacts(self, item: Any, *, response_id: str) -> Any:
        if not isinstance(item, dict):
            return item
        item_type = item.get("type")
        role = item.get("role")
        if item_type in {"input_image", "input_file"}:
            return self._stamp_content_part_artifact(item, response_id=response_id)
        if item_type != "message" and role not in {"user", "assistant", "system", "developer"}:
            return item
        content = item.get("content")
        if isinstance(content, list):
            return {**item, "content": [self._stamp_content_part_artifact(part, response_id=response_id) for part in content]}
        if isinstance(content, dict):
            return {**item, "content": self._stamp_content_part_artifact(content, response_id=response_id)}
        return item

    def _stamp_content_part_artifact(self, part: Any, *, response_id: str) -> Any:
        if not isinstance(part, dict) or part.get("type") not in {"input_image", "input_file"}:
            return part
        artifact_id = part.get("_respawn_artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id:
            artifact_id = generate_id("art")
        return {**part, "_respawn_artifact_id": artifact_id, "_respawn_response_id": response_id}

    def _output_annotations(self, request: ResponseRequest) -> list[dict[str, Any]]:
        annotations: list[dict[str, Any]] = []
        for index, part in enumerate(self._input_file_parts(request.input)):
            artifact_id = part.get("_respawn_artifact_id")
            if not isinstance(artifact_id, str) or not artifact_id:
                continue
            annotations.append(
                {
                    "type": "file_citation",
                    "file_id": artifact_id,
                    "filename": str(part.get("filename") or "input_file"),
                    "index": index,
                }
            )
        return annotations

    def _input_file_parts(self, input_value: Any) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = []
        if not isinstance(input_value, list):
            return parts
        for item in input_value:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "input_file":
                parts.append(item)
                continue
            content = item.get("content")
            if isinstance(content, dict) and content.get("type") == "input_file":
                parts.append(content)
            elif isinstance(content, list):
                parts.extend(part for part in content if isinstance(part, dict) and part.get("type") == "input_file")
        return parts

    def _artifact_map_from_input(self, input_value: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        artifacts: dict[str, dict[str, Any]] = {}
        for item in input_value:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if item.get("type") in {"input_image", "input_file"}:
                candidate_parts = [item]
            elif isinstance(content, dict):
                candidate_parts = [content]
            elif isinstance(content, list):
                candidate_parts = [part for part in content if isinstance(part, dict)]
            else:
                candidate_parts = []
            for part in candidate_parts:
                if part.get("type") not in {"input_image", "input_file"}:
                    continue
                artifact_id = part.get("_respawn_artifact_id")
                if not isinstance(artifact_id, str) or not artifact_id:
                    continue
                artifacts[artifact_id] = {
                    "id": artifact_id,
                    "object": "response.artifact",
                    "type": part.get("type"),
                    "filename": part.get("filename"),
                    "mime_type": part.get("mime_type"),
                    "size_bytes": _safe_int(part.get("size_bytes")),
                    "source": _source_reference(part),
                    "content": {"text": part.get("text")} if part.get("type") == "input_file" and isinstance(part.get("text"), str) else None,
                }
        return artifacts

    def _record_token_usage(self, model: str, usage: ResponseUsage) -> None:
        TOKEN_USAGE.labels(model=model, kind="input").inc(usage.input_tokens)
        TOKEN_USAGE.labels(model=model, kind="output").inc(usage.output_tokens)
        TOKEN_USAGE.labels(model=model, kind="total").inc(usage.total_tokens)
        TOKEN_USAGE.labels(model=model, kind="cached_input").inc(usage.input_tokens_details.cached_tokens)
        TOKEN_USAGE.labels(model=model, kind="reasoning").inc(usage.output_tokens_details.reasoning_tokens)
        MODEL_TOKEN_USAGE.labels(api="responses", model=model, kind="input").inc(usage.input_tokens)
        MODEL_TOKEN_USAGE.labels(api="responses", model=model, kind="output").inc(usage.output_tokens)
        MODEL_TOKEN_USAGE.labels(api="responses", model=model, kind="total").inc(usage.total_tokens)
        MODEL_TOKEN_USAGE.labels(api="responses", model=model, kind="cached_input").inc(usage.input_tokens_details.cached_tokens)
        MODEL_TOKEN_USAGE.labels(api="responses", model=model, kind="reasoning").inc(usage.output_tokens_details.reasoning_tokens)

    def _record_reasoning_metrics(self, *, model: str, request: ResponseRequest, usage: ResponseUsage, status: str) -> None:
        if not reasoning_requested(request):
            return
        reasoning = request.reasoning or {}
        effort = str(reasoning.get("effort") or "auto")
        summary = str(reasoning.get("summary") or "none")
        encrypted_content = str(reasoning_encrypted_content_requested(request)).lower()
        tokens = usage.output_tokens_details.reasoning_tokens
        REASONING_REQUESTS.labels(model=model, effort=effort, summary=summary, encrypted_content=encrypted_content, status=status).inc()
        REASONING_TOKENS.labels(model=model, effort=effort).inc(tokens)
        REASONING_HEAVY_REQUESTS.labels(model=model, effort=effort).inc(0)
        if tokens >= self.settings.reasoning_heavy_token_threshold:
            REASONING_HEAVY_REQUESTS.labels(model=model, effort=effort).inc()

    def _reasoning_output_item(self, item_id: str, reasoning_text: str, request: ResponseRequest, *, response_id: str) -> dict[str, Any]:
        encrypted_content = None
        if reasoning_encrypted_content_requested(request):
            encrypted_content = seal_reasoning_content(
                reasoning_text,
                key=self.settings.reasoning_encryption_key,
                response_id=response_id,
                item_id=item_id,
            )
        return reasoning_output_item(
            item_id,
            reasoning_text,
            request,
            encrypted_content=encrypted_content,
            summary_provider=self.reasoning_summary_provider,
        )

    def _record_response_metrics(self, *, model: str, mode: str, status: str, should_store: bool, started_at: float) -> None:
        RESPONSE_LATENCY.labels(model=model, mode=mode).observe(time.perf_counter() - started_at)
        RESPONSES.labels(model=model, mode=mode, status=status, stored=str(should_store).lower()).inc()
        IN_FLIGHT_RESPONSES.labels(model=model, mode=mode).dec()

    def _response_object(
        self,
        response_id: str,
        model: str,
        status: str,
        output: list[dict[str, Any]],
        usage: ResponseUsage,
        metadata: dict[str, Any],
        *,
        request: ResponseRequest | None = None,
        request_json: dict[str, Any] | None = None,
        should_store: bool = True,
        error: dict[str, Any] | None = None,
        incomplete_details: dict[str, Any] | None = None,
        created_at: int | None = None,
        include: list[str] | None = None,
        artifacts: list[dict[str, Any]] | None = None,
    ) -> ResponseObject:
        request_snapshot = self._request_snapshot(request=request, request_json=request_json)
        effective_include = self._effective_includes(request_snapshot, include)
        serialized_input = self._response_input(request_snapshot.get("input"), effective_include, artifacts)
        serialized_output = self._response_output(output, effective_include)
        self._enforce_include_payload_limit(effective_include, input_value=serialized_input, output=serialized_output)
        return ResponseObject(
            id=response_id,
            created_at=created_at or int(time.time()),
            status=status,
            error=error,
            incomplete_details=incomplete_details,
            input=serialized_input,
            background=bool(request_snapshot.get("background", False)),
            instructions=request_snapshot.get("instructions"),
            max_output_tokens=request_snapshot.get("max_output_tokens"),
            max_tool_calls=request_snapshot.get("max_tool_calls"),
            model=model,
            output=serialized_output,
            output_text=response_output_text(serialized_output),
            parallel_tool_calls=bool(request_snapshot.get("parallel_tool_calls", bool(request_snapshot.get("tools")))),
            previous_response_id=request_snapshot.get("previous_response_id"),
            prompt=request_snapshot.get("prompt"),
            prompt_cache_key=request_snapshot.get("prompt_cache_key"),
            prompt_cache_retention=request_snapshot.get("prompt_cache_retention"),
            reasoning=self._response_reasoning(request_snapshot),
            safety_identifier=request_snapshot.get("safety_identifier"),
            service_tier=request_snapshot.get("service_tier", "auto"),
            store=should_store,
            temperature=request_snapshot.get("temperature", 1),
            text=normalized_text_config(request_snapshot),
            tool_choice=request_snapshot.get("tool_choice", "auto"),
            tools=request_snapshot.get("tools", []),
            top_logprobs=request_snapshot.get("top_logprobs"),
            top_p=request_snapshot.get("top_p", 1),
            truncation=request_snapshot.get("truncation", "disabled"),
            usage=usage,
            user=request_snapshot.get("user"),
            metadata=metadata,
        )


def _error_json(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, OpenAIError):
        return exc.to_response()["error"]
    return {"message": str(exc), "type": "server_error", "param": None, "code": "internal_error"}


def _public_artifact(artifact: dict[str, Any], *, include_content: bool) -> dict[str, Any]:
    public = {
        "id": artifact.get("id"),
        "object": artifact.get("object", "response.artifact"),
        "type": artifact.get("type"),
        "filename": artifact.get("filename"),
        "mime_type": artifact.get("mime_type"),
        "size_bytes": artifact.get("size_bytes", 0),
        "source": artifact.get("source") or {},
    }
    if include_content:
        public["content"] = artifact.get("content")
    return public


def _source_reference(part: dict[str, Any]) -> dict[str, Any]:
    source = part.get("source") or part.get("image_url")
    if not isinstance(source, str) or not source:
        return {"type": "unknown"}
    if source.startswith(("http://", "https://")):
        return {"type": "url", "url": source}
    if source.startswith("data:"):
        return {"type": "data_url", "redacted": True}
    if source == "base64":
        return {"type": "base64", "redacted": True}
    return {"type": "local_reference", "label": source}


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def schedule_background_response(
    *,
    settings: Settings,
    session_factory: Any,
    backend: ModelBackend,
    prompt_cache: PromptCache,
    background_tasks: dict[str, asyncio.Task],
    response_id: str,
    tenant_id: str | None,
) -> None:
    existing = background_tasks.get(response_id)
    if existing is not None and not existing.done():
        return
    task = asyncio.create_task(
        _run_background_response_task(
            settings=settings,
            session_factory=session_factory,
            backend=backend,
            prompt_cache=prompt_cache,
            background_tasks=background_tasks,
            response_id=response_id,
            tenant_id=tenant_id,
        )
    )
    background_tasks[response_id] = task
    task.add_done_callback(lambda _: background_tasks.pop(response_id, None))


async def resume_background_responses(
    *,
    settings: Settings,
    session_factory: Any,
    backend: ModelBackend,
    prompt_cache: PromptCache,
    background_tasks: dict[str, asyncio.Task],
) -> None:
    async with session_factory() as session:
        repository = ResponseRepository(session)
        jobs = await repository.list_runnable_background_jobs()
    for job in jobs:
        schedule_background_response(
            settings=settings,
            session_factory=session_factory,
            backend=backend,
            prompt_cache=prompt_cache,
            background_tasks=background_tasks,
            response_id=str(job["response_id"]),
            tenant_id=job["tenant_id"],
        )


async def shutdown_background_responses(background_tasks: dict[str, asyncio.Task]) -> None:
    tasks = [task for task in background_tasks.values() if not task.done()]
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    background_tasks.clear()


async def _run_background_response_task(
    *,
    settings: Settings,
    session_factory: Any,
    backend: ModelBackend,
    prompt_cache: PromptCache,
    background_tasks: dict[str, asyncio.Task],
    response_id: str,
    tenant_id: str | None,
) -> None:
    async with session_factory() as session:
        repository = ResponseRepository(session)
        service = ResponseService(
            settings=settings,
            repository=repository,
            backend=backend,
            prompt_cache=prompt_cache,
            session_factory=session_factory,
            background_tasks=background_tasks,
        )
        await service.run_background_response(response_id, tenant_id)


async def _heartbeat_background_job(*, response_id: str, session_factory: Any, interval_seconds: float) -> None:
    interval = max(interval_seconds, 0.1)
    while True:
        await asyncio.sleep(interval)
        async with session_factory() as session:
            repository = ResponseRepository(session)
            await repository.heartbeat_background_job(response_id)
            await session.commit()
