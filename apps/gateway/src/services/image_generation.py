from __future__ import annotations

import base64
import logging
import re
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from src.adapters.image_generation_base import (
    ImageGenerationBackend,
    ImageGenerationBackendError,
    ImageGenerationError,
    ImageGenerationRequest,
    ImageGenerationResult,
)
from src.config import Settings
from src.observability.metrics import IMAGE_GENERATION_ERRORS, IMAGE_GENERATION_LATENCY, IMAGE_GENERATION_PIXELS, IMAGE_GENERATION_REQUESTS
from src.schemas.errors import OpenAIError
from src.schemas.responses import ResponseRequest
from src.services.id_generator import generate_id
from src.services.response_history_builder import content_to_text
from src.services.responses_compat import image_generation_disabled_by_choice, image_generation_requested, image_generation_required, image_generation_tools
from src.services.tool_call_arguments import tool_call_arguments


logger = logging.getLogger(__name__)

SIZE_RE = re.compile(r"^([1-9][0-9]{1,4})x([1-9][0-9]{1,4})$")
QUALITY_STEPS = {"low": 8, "medium": 16, "high": 24}


@dataclass(frozen=True)
class ImageGenerationExecution:
    result: ImageGenerationResult
    output_item: dict[str, Any]


def validate_image_generation_configuration(request: ResponseRequest, *, settings: Settings, backend: ImageGenerationBackend | None) -> None:
    tools = image_generation_tools(request)
    if not tools:
        return
    first_param = str(tools[0].get("_param") or "tools.0")
    if not settings.image_generation_enabled:
        raise OpenAIError(
            "The image_generation tool is disabled. Set IMAGE_GENERATION_ENABLED=true and configure IMAGE_GENERATION_BACKEND to enable local image generation.",
            status_code=400,
            param=f"{first_param}.type",
            code="unsupported_parameter",
        )
    if backend is None:
        raise OpenAIError(
            "The image_generation tool is enabled but no image generation backend is configured.",
            status_code=503,
            type="server_error",
            param=f"{first_param}.type",
            code="image_generation_unavailable",
        )
    for tool in tools:
        _resolved_size(tool, settings=settings)


class ImageGenerationService:
    def __init__(self, *, settings: Settings, backend: ImageGenerationBackend | None) -> None:
        self.settings = settings
        self.backend = backend

    async def execute_if_needed(self, request: ResponseRequest, *, response_id: str) -> ImageGenerationExecution | None:
        if not image_generation_requested(request) or image_generation_disabled_by_choice(request):
            return None
        tools = image_generation_tools(request)
        if not tools:
            return None
        validate_image_generation_configuration(request, settings=self.settings, backend=self.backend)
        tool = tools[0]
        if not image_generation_required(request):
            return None
        prompt = derive_image_generation_prompt(request.input)
        if not prompt:
            raise OpenAIError(
                "image_generation requires non-empty user text to derive an image prompt.",
                param=f"{tool.get('_param', 'tools.0')}.type",
                code="invalid_request",
            )

        return await self.execute_prompt(prompt, request=request, response_id=response_id)

    async def execute_tool_call(self, tool_call: dict[str, Any], request: ResponseRequest, *, response_id: str) -> ImageGenerationExecution:
        tools = image_generation_tools(request)
        if not tools:
            raise OpenAIError("image_generation tool call requires an image_generation tool in the request.", param="tools", code="invalid_tool_call")
        validate_image_generation_configuration(request, settings=self.settings, backend=self.backend)
        prompt = _prompt_from_tool_call(tool_call)
        if not prompt:
            raise OpenAIError(
                "image_generation tool call requires a non-empty prompt argument.",
                param="tools.0.type",
                code="invalid_tool_call",
            )
        return await self.execute_prompt(prompt, request=request, response_id=response_id)

    async def execute_prompt(self, prompt: str, request: ResponseRequest, *, response_id: str) -> ImageGenerationExecution:
        tools = image_generation_tools(request)
        if not tools:
            raise OpenAIError("image_generation requires an image_generation tool in the request.", param="tools", code="invalid_request")
        validate_image_generation_configuration(request, settings=self.settings, backend=self.backend)
        tool = tools[0]
        provider = self.backend.provider_name if self.backend is not None else "unconfigured"
        started_at = perf_counter()
        try:
            width, height, size = _resolved_size(tool, settings=self.settings)
            quality = str(tool.get("quality") or "auto")
            generation_request = ImageGenerationRequest(
                prompt=prompt,
                size=size,
                width=width,
                height=height,
                quality=quality,
                output_format=str(tool.get("output_format") or self.settings.image_generation_output_format),
                negative_prompt=self.settings.image_generation_negative_prompt,
                steps=self._steps_for_quality(quality),
                cfg_scale=float(self.settings.image_generation_default_cfg_scale),
                sampler=self.settings.image_generation_sampler,
                metadata=dict(request.metadata),
            )
            assert self.backend is not None
            result = await self.backend.generate(generation_request)
            _validate_base64(result.image_base64)
            IMAGE_GENERATION_REQUESTS.labels(provider=provider, status="completed").inc()
            IMAGE_GENERATION_PIXELS.labels(provider=provider).inc(width * height)
            IMAGE_GENERATION_LATENCY.labels(provider=provider).observe(perf_counter() - started_at)
            self._log_run(response_id=response_id, provider=provider, status="completed", size=size, latency_ms=result.latency_ms)
            return ImageGenerationExecution(result=result, output_item=self._output_item(result))
        except ImageGenerationError as exc:
            IMAGE_GENERATION_REQUESTS.labels(provider=provider, status="failed").inc()
            IMAGE_GENERATION_ERRORS.labels(provider=provider, code=exc.code).inc()
            IMAGE_GENERATION_LATENCY.labels(provider=provider).observe(perf_counter() - started_at)
            self._log_run(response_id=response_id, provider=provider, status="failed", size=str(tool.get("size") or "auto"), latency_ms=(perf_counter() - started_at) * 1000, code=exc.code)
            raise OpenAIError(
                exc.message,
                status_code=exc.status_code,
                type=exc.type,
                param=f"{tool.get('_param', 'tools.0')}.type",
                code=exc.code,
            ) from exc

    def _steps_for_quality(self, quality: str) -> int:
        requested = QUALITY_STEPS.get(quality, int(self.settings.image_generation_default_steps))
        return max(1, min(requested, int(self.settings.image_generation_max_steps)))

    def _output_item(self, result: ImageGenerationResult) -> dict[str, Any]:
        item: dict[str, Any] = {
            "id": generate_id("ig"),
            "type": "image_generation_call",
            "status": "completed",
            "result": result.image_base64,
            "revised_prompt": result.revised_prompt,
            "size": result.size,
            "quality": result.quality,
            "output_format": result.output_format,
        }
        if result.seed is not None:
            item["seed"] = result.seed
        return {key: value for key, value in item.items() if value is not None}

    def _log_run(self, *, response_id: str, provider: str, status: str, size: str, latency_ms: float, code: str | None = None) -> None:
        logger.info(
            "Image generation completed",
            extra={
                "feature": "image_generation",
                "response_id": response_id,
                "provider": provider,
                "status": status,
                "size": size,
                "latency_ms": round(latency_ms, 3),
                "error_code": code,
            },
        )


def derive_image_generation_prompt(input_value: Any) -> str:
    if isinstance(input_value, str):
        return _normalize_prompt(input_value)
    if not isinstance(input_value, list):
        return ""
    for item in reversed(input_value):
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        if item.get("type") == "message" or role in {"user", "developer"}:
            text = content_to_text(item.get("content", ""))
            if text.strip():
                return _normalize_prompt(text)
        if item.get("type") in {"input_image", "input_file"}:
            text = content_to_text(item)
            if text.strip():
                return _normalize_prompt(text)
    return ""


def _normalize_prompt(prompt: str) -> str:
    return " ".join(prompt.split())[:2000]


def _resolved_size(tool: dict[str, Any], *, settings: Settings) -> tuple[int, int, str]:
    raw = str(tool.get("size") or "auto")
    if raw == "auto":
        raw = settings.image_generation_default_size
    match = SIZE_RE.fullmatch(raw)
    if match is None:
        raise OpenAIError("image_generation size must be 'auto' or WIDTHxHEIGHT.", param=f"{tool.get('_param', 'tools.0')}.size", code="invalid_request")
    width = int(match.group(1))
    height = int(match.group(2))
    if width * height > int(settings.image_generation_max_pixels):
        raise OpenAIError(
            f"image_generation size exceeds IMAGE_GENERATION_MAX_PIXELS={settings.image_generation_max_pixels}.",
            param=f"{tool.get('_param', 'tools.0')}.size",
            code="invalid_request",
        )
    return width, height, f"{width}x{height}"


def _prompt_from_tool_call(tool_call: dict[str, Any]) -> str:
    return _normalize_prompt(str(tool_call_arguments(tool_call).get("prompt") or ""))


def _validate_base64(value: str) -> None:
    try:
        base64.b64decode(value, validate=True)
    except Exception as exc:
        raise ImageGenerationBackendError("Image generation provider returned invalid base64 image data.") from exc
