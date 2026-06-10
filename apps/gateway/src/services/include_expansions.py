from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.config import Settings
from src.observability.metrics import INCLUDE_CAPABILITY_ERRORS
from src.schemas.errors import OpenAIError
from src.services.model_capabilities import capabilities_for_model


REASONING_ENCRYPTED_CONTENT = "reasoning.encrypted_content"
OUTPUT_TEXT_LOGPROBS = "message.output_text.logprobs"
INPUT_IMAGE_URL = "message.input_image.image_url"
WEB_SEARCH_ACTION_SOURCES = "web_search_call.action.sources"

HOSTED_TOOL_INCLUDES = {
    "file_search_call.results",
    "web_search_call.results",
    "computer_call_output.output.image_url",
    "code_interpreter_call.outputs",
}


@dataclass(frozen=True)
class IncludeFeature:
    value: str
    feature_id: str
    local: bool
    capability: str | None = None


INCLUDE_REGISTRY = {
    REASONING_ENCRYPTED_CONTENT: IncludeFeature(REASONING_ENCRYPTED_CONTENT, "io.reasoning_encrypted_content", True),
    INPUT_IMAGE_URL: IncludeFeature(INPUT_IMAGE_URL, "io.input_image_url_include", True),
    OUTPUT_TEXT_LOGPROBS: IncludeFeature(OUTPUT_TEXT_LOGPROBS, "io.output_text_logprobs", True, capability="logprobs"),
    WEB_SEARCH_ACTION_SOURCES: IncludeFeature(WEB_SEARCH_ACTION_SOURCES, "io.web_search_call_sources", True),
    **{
        value: IncludeFeature(value, f"hosted.{value}", False)
        for value in HOSTED_TOOL_INCLUDES
    },
}


def validate_include_values(include: list[str]) -> None:
    if not isinstance(include, list):
        raise OpenAIError("include must be a list of strings.", param="include")
    for index, value in enumerate(include):
        if not isinstance(value, str):
            raise OpenAIError("include entries must be strings.", param=f"include.{index}")
        feature = INCLUDE_REGISTRY.get(value)
        if feature is None:
            _unsupported(f"include.{index}", f"Include value '{value}' is not supported.")
        if not feature.local:
            _unsupported(
                f"include.{index}",
                f"Include value '{value}' requires OpenAI-hosted tool execution, which Respawn does not provide.",
            )


def validate_include_capabilities(request: Any, *, model: str, settings: Settings) -> None:
    requested = requested_includes(request)
    needs_logprobs = OUTPUT_TEXT_LOGPROBS in requested or getattr(request, "top_logprobs", None) is not None
    if not needs_logprobs:
        return
    if bool(getattr(request, "stream", False)):
        raise OpenAIError(
            "Streaming output text logprobs are not supported by Respawn yet.",
            status_code=400,
            param="include" if OUTPUT_TEXT_LOGPROBS in requested else "top_logprobs",
            code="unsupported_parameter",
        )
    if "logprobs" in capabilities_for_model(model, settings):
        return
    INCLUDE_CAPABILITY_ERRORS.labels(model=model, include=OUTPUT_TEXT_LOGPROBS).inc()
    raise OpenAIError(
        f"Model '{model}' is not configured with the logprobs capability required for output text logprobs.",
        status_code=400,
        param="include" if OUTPUT_TEXT_LOGPROBS in requested else "top_logprobs",
        code="unsupported_model_capability",
    )


def requested_includes(request: Any) -> set[str]:
    include = getattr(request, "include", None)
    if include is None and isinstance(request, dict):
        include = request.get("include")
    return {value for value in include or [] if isinstance(value, str)}


def logprobs_requested(include: set[str] | list[str]) -> bool:
    return OUTPUT_TEXT_LOGPROBS in set(include)


def input_image_url_requested(include: set[str] | list[str]) -> bool:
    return INPUT_IMAGE_URL in set(include)


def _unsupported(param: str, message: str) -> None:
    raise OpenAIError(message, status_code=400, param=param, code="unsupported_parameter")
