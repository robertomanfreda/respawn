import json
from time import perf_counter
from typing import Any

import httpx

from src.adapters.image_generation_base import (
    ImageGenerationBackend,
    ImageGenerationBackendError,
    ImageGenerationRequest,
    ImageGenerationResult,
    ImageGenerationTimeoutError,
    ImageGenerationUnavailableError,
)
from src.services.id_generator import generate_id


class Automatic1111ImageGenerationBackend(ImageGenerationBackend):
    provider_name = "automatic1111"

    def __init__(self, *, base_url: str, timeout_seconds: float, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = max(float(timeout_seconds), 0.1)
        self.model = model

    async def generate(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        started_at = perf_counter()
        payload = {
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "steps": request.steps,
            "cfg_scale": request.cfg_scale,
            "width": request.width,
            "height": request.height,
            "sampler_name": request.sampler,
            "seed": request.seed if request.seed is not None else -1,
            "batch_size": 1,
            "n_iter": 1,
        }
        response_payload = await self._post_json("/sdapi/v1/txt2img", payload)
        images = response_payload.get("images")
        if not isinstance(images, list) or not images or not isinstance(images[0], str):
            raise ImageGenerationBackendError("Image generation provider returned no image.")
        info = _parse_info(response_payload.get("info"))
        seed = _safe_int(info.get("seed"))
        return ImageGenerationResult(
            id=generate_id("igr"),
            prompt=request.prompt,
            revised_prompt=request.prompt,
            image_base64=_strip_data_url_prefix(images[0]),
            output_format=request.output_format,
            size=request.size,
            quality=request.quality,
            seed=seed if seed is not None else request.seed,
            provider=self.provider_name,
            latency_ms=(perf_counter() - started_at) * 1000,
        )

    async def check_ready(self) -> dict[str, Any]:
        models = await self._get_json("/sdapi/v1/sd-models")
        details: dict[str, Any] = {"backend": self.provider_name, "base_url": self.base_url}
        if isinstance(models, list):
            names = [str(model.get("model_name") or model.get("title") or "") for model in models if isinstance(model, dict)]
            details["model_count"] = len(names)
            if self.model and self.model not in {"sd-v1-5", "sd1.5", "sd-v1-5.safetensors"}:
                details["configured_model_present"] = any(self.model in name for name in names)
        return details

    async def _get_json(self, path: str) -> Any:
        if not self.base_url:
            raise ImageGenerationUnavailableError("IMAGE_GENERATION_BASE_URL is required for the Automatic1111 backend.")
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.get(f"{self.base_url}{path}")
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException as exc:
            raise ImageGenerationTimeoutError() from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise ImageGenerationUnavailableError() from exc

    async def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.base_url:
            raise ImageGenerationUnavailableError("IMAGE_GENERATION_BASE_URL is required for the Automatic1111 backend.")
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(f"{self.base_url}{path}", json=payload)
            response.raise_for_status()
            data = response.json()
        except httpx.TimeoutException as exc:
            raise ImageGenerationTimeoutError() from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise ImageGenerationBackendError() from exc
        if not isinstance(data, dict):
            raise ImageGenerationBackendError("Image generation provider returned a malformed response.")
        return data


def _strip_data_url_prefix(value: str) -> str:
    if "," in value and value.strip().lower().startswith("data:image/"):
        return value.split(",", 1)[1]
    return value


def _parse_info(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
