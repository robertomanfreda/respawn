import json
import logging
from typing import Any

from pydantic import BaseModel


TRACE_LEVEL = 5


def log_model_request(logger: logging.Logger, *, api: str, phase: str, payload: dict[str, Any]) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return
    logger.debug(
        "Model request payload",
        extra={
            "api": api,
            "phase": phase,
            "model": payload.get("model"),
            "model_payload": _jsonable(payload),
        },
    )


def log_model_response(logger: logging.Logger, *, api: str, phase: str, result: Any) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return
    logger.debug(
        "Model response result",
        extra={
            "api": api,
            "phase": phase,
            "model_result": _jsonable(result),
        },
    )


def log_model_stream_chunk(logger: logging.Logger, *, api: str, phase: str, chunk: dict[str, Any]) -> None:
    if not logger.isEnabledFor(TRACE_LEVEL):
        return
    logger.log(
        TRACE_LEVEL,
        "Model stream chunk",
        extra={
            "api": api,
            "phase": phase,
            "model_chunk": _jsonable(chunk),
        },
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        value = value.model_dump()
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
    except (TypeError, ValueError):
        return str(value)
