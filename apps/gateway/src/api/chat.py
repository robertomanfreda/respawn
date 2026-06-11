from collections.abc import AsyncIterator
import json
import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from src.adapters.base import ModelBackend
from src.observability.metrics import CHAT_COMPLETION_LATENCY, CHAT_COMPLETIONS, MODEL_TOKEN_USAGE
from src.observability.model_io import log_model_request, log_model_response, log_model_stream_chunk
from src.security.auth import tenant_id
from src.services.id_generator import generate_id

router = APIRouter(prefix="/v1/chat/completions", tags=["chat"])
logger = logging.getLogger(__name__)


@router.post("", response_model=None)
async def chat_completions(
    payload: dict[str, Any],
    request: Request,
    _: str | None = Depends(tenant_id),
) -> dict[str, Any] | StreamingResponse:
    backend_payload = dict(payload)
    backend_payload.setdefault("model", request.app.state.settings.default_model)
    model = backend_payload["model"]
    if backend_payload.get("stream"):
        return StreamingResponse(
            _chat_completion_stream(backend_payload, request.app.state.backend),
            media_type="text/event-stream",
        )

    mode = "blocking"
    status = "failed"
    started_at = time.perf_counter()
    try:
        log_model_request(logger, api="chat_completions", phase="blocking", payload=backend_payload)
        result = await request.app.state.backend.create_chat_completion(backend_payload)
        log_model_response(logger, api="chat_completions", phase="blocking", result=result)
        _record_token_usage(model, result.usage)
        status = "completed"
        return {
            "id": "chatcmpl_local",
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result.content,
                        "tool_calls": result.tool_calls or None,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": result.usage,
        }
    finally:
        _record_chat_metrics(model=model, mode=mode, status=status, started_at=started_at)


async def _chat_completion_stream(payload: dict[str, Any], backend: ModelBackend) -> AsyncIterator[str]:
    completion_id = generate_id("chatcmpl")
    created = int(time.time())
    model = payload["model"]
    mode = "stream"
    status = "cancelled"
    started_at = time.perf_counter()

    try:
        yield _chat_stream_event(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
        )

        usage: dict[str, int] = {}
        log_model_request(logger, api="chat_completions", phase="stream", payload=payload)
        async for chunk in backend.create_chat_completion_stream(payload):
            log_model_stream_chunk(logger, api="chat_completions", phase="stream", chunk=chunk)
            if chunk.get("type") == "delta":
                yield _chat_stream_event(
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": chunk.get("delta", "")},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
            elif chunk.get("type") == "done":
                usage = chunk.get("usage") or {}

        done_chunk: dict[str, Any] = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        if usage or (payload.get("stream_options") or {}).get("include_usage"):
            done_chunk["usage"] = usage
        _record_token_usage(model, usage)
        status = "completed"
        yield _chat_stream_event(done_chunk)
        yield "data: [DONE]\n\n"
    except Exception:
        status = "failed"
        raise
    finally:
        _record_chat_metrics(model=model, mode=mode, status=status, started_at=started_at)


def _chat_stream_event(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data, separators=(',', ':'))}\n\n"


def _record_token_usage(model: str, usage: dict[str, int]) -> None:
    input_tokens = int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
    output_tokens = int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0)
    total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens) or 0)
    MODEL_TOKEN_USAGE.labels(api="chat", model=model, kind="input").inc(input_tokens)
    MODEL_TOKEN_USAGE.labels(api="chat", model=model, kind="output").inc(output_tokens)
    MODEL_TOKEN_USAGE.labels(api="chat", model=model, kind="total").inc(total_tokens)


def _record_chat_metrics(*, model: str, mode: str, status: str, started_at: float) -> None:
    CHAT_COMPLETION_LATENCY.labels(model=model, mode=mode).observe(time.perf_counter() - started_at)
    CHAT_COMPLETIONS.labels(model=model, mode=mode, status=status).inc()
