from __future__ import annotations

import asyncio
import base64
import random
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


class ComfyUIImageGenerationBackend(ImageGenerationBackend):
    provider_name = "comfyui"

    def __init__(self, *, base_url: str, timeout_seconds: float, model: str, poll_interval_seconds: float = 0.5) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = max(float(timeout_seconds), 0.1)
        self.model = model
        self.poll_interval_seconds = max(float(poll_interval_seconds), 0.05)

    async def generate(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        started_at = perf_counter()
        seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)
        workflow = _build_txt2img_workflow(request, checkpoint=self.model, seed=seed)
        queued = await self._post_json("/prompt", {"prompt": workflow, "client_id": generate_id("client")})
        prompt_id = queued.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ImageGenerationBackendError(_comfyui_error_message(queued, "ComfyUI did not return a prompt_id."))

        image_ref = await self._wait_for_first_image(prompt_id)
        image_bytes = await self._get_bytes("/view", params=image_ref)
        if not image_bytes:
            raise ImageGenerationBackendError("ComfyUI returned an empty image.")

        return ImageGenerationResult(
            id=generate_id("igr"),
            prompt=request.prompt,
            revised_prompt=request.prompt,
            image_base64=base64.b64encode(image_bytes).decode("ascii"),
            output_format=request.output_format,
            size=request.size,
            quality=request.quality,
            seed=seed,
            provider=self.provider_name,
            latency_ms=(perf_counter() - started_at) * 1000,
        )

    async def check_ready(self) -> dict[str, Any]:
        stats = await self._get_json("/system_stats", unavailable=True)
        checkpoints = await self._get_json("/models/checkpoints", unavailable=True)
        checkpoint_names = [str(name) for name in checkpoints] if isinstance(checkpoints, list) else []
        details: dict[str, Any] = {"backend": self.provider_name, "base_url": self.base_url}
        if isinstance(stats, dict):
            devices = stats.get("devices")
            if isinstance(devices, list):
                details["device_count"] = len(devices)
        details["model_count"] = len(checkpoint_names)
        if self.model:
            details["configured_model_present"] = _checkpoint_matches(self.model, checkpoint_names)
        return details

    async def _wait_for_first_image(self, prompt_id: str) -> dict[str, str]:
        deadline = perf_counter() + self.timeout_seconds
        while perf_counter() < deadline:
            history = await self._get_json(f"/history/{prompt_id}")
            if isinstance(history, dict) and prompt_id in history:
                entry = history[prompt_id]
                if isinstance(entry, dict):
                    image_ref = _first_image_reference(entry)
                    if image_ref is not None:
                        return image_ref
                    status = entry.get("status")
                    if _history_failed(status):
                        raise ImageGenerationBackendError(_status_error_message(status))
                    if _history_completed(status):
                        raise ImageGenerationBackendError("ComfyUI completed the prompt without producing an image.")
            await asyncio.sleep(self.poll_interval_seconds)
        raise ImageGenerationTimeoutError("ComfyUI image generation timed out.")

    async def _get_json(self, path: str, *, unavailable: bool = False) -> Any:
        if not self.base_url:
            raise ImageGenerationUnavailableError("IMAGE_GENERATION_BASE_URL is required for the ComfyUI backend.")
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.get(f"{self.base_url}{path}")
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException as exc:
            raise ImageGenerationTimeoutError() from exc
        except (httpx.HTTPError, ValueError) as exc:
            if unavailable:
                raise ImageGenerationUnavailableError() from exc
            raise ImageGenerationBackendError() from exc

    async def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.base_url:
            raise ImageGenerationUnavailableError("IMAGE_GENERATION_BASE_URL is required for the ComfyUI backend.")
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
            raise ImageGenerationBackendError("ComfyUI returned a malformed response.")
        if data.get("error") or data.get("node_errors"):
            raise ImageGenerationBackendError(_comfyui_error_message(data, "ComfyUI rejected the workflow."))
        return data

    async def _get_bytes(self, path: str, *, params: dict[str, str]) -> bytes:
        if not self.base_url:
            raise ImageGenerationUnavailableError("IMAGE_GENERATION_BASE_URL is required for the ComfyUI backend.")
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.get(f"{self.base_url}{path}", params=params)
            response.raise_for_status()
            return response.content
        except httpx.TimeoutException as exc:
            raise ImageGenerationTimeoutError() from exc
        except httpx.HTTPError as exc:
            raise ImageGenerationBackendError() from exc


def _build_txt2img_workflow(request: ImageGenerationRequest, *, checkpoint: str, seed: int) -> dict[str, dict[str, Any]]:
    return {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": request.steps,
                "cfg": request.cfg_scale,
                "sampler_name": _comfy_sampler_name(request.sampler),
                "scheduler": "normal",
                "denoise": 1,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
            },
        },
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": request.width, "height": request.height, "batch_size": 1}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": request.prompt, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": request.negative_prompt, "clip": ["4", 1]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "respawn", "images": ["8", 0]}},
    }


def _comfy_sampler_name(name: str) -> str:
    normalized = name.strip().lower().replace(" ", "_")
    aliases = {
        "euler_a": "euler_ancestral",
        "euler_ancestral": "euler_ancestral",
        "euler": "euler",
        "ddim": "ddim",
        "dpm++_2m": "dpmpp_2m",
        "dpmpp_2m": "dpmpp_2m",
        "dpm++_2m_karras": "dpmpp_2m",
        "dpmpp_2m_karras": "dpmpp_2m",
    }
    return aliases.get(normalized, normalized or "euler")


def _first_image_reference(history_entry: dict[str, Any]) -> dict[str, str] | None:
    outputs = history_entry.get("outputs")
    if not isinstance(outputs, dict):
        return None
    for node_output in outputs.values():
        if not isinstance(node_output, dict):
            continue
        images = node_output.get("images")
        if not isinstance(images, list):
            continue
        for image in images:
            if not isinstance(image, dict):
                continue
            filename = image.get("filename")
            if isinstance(filename, str) and filename:
                return {
                    "filename": filename,
                    "subfolder": str(image.get("subfolder") or ""),
                    "type": str(image.get("type") or "output"),
                }
    return None


def _history_completed(status: Any) -> bool:
    if not isinstance(status, dict):
        return False
    return bool(status.get("completed"))


def _history_failed(status: Any) -> bool:
    if not isinstance(status, dict):
        return False
    status_str = str(status.get("status_str") or "").lower()
    return status_str in {"error", "failed", "failure"}


def _status_error_message(status: Any) -> str:
    if not isinstance(status, dict):
        return "ComfyUI failed while executing the workflow."
    messages = status.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if not isinstance(message, (list, tuple)) or len(message) < 2:
                continue
            payload = message[1]
            if isinstance(payload, dict):
                exception_message = payload.get("exception_message")
                if isinstance(exception_message, str) and exception_message:
                    return f"ComfyUI failed while executing the workflow: {exception_message}"
    return "ComfyUI failed while executing the workflow."


def _comfyui_error_message(payload: dict[str, Any], fallback: str) -> str:
    error = payload.get("error")
    if isinstance(error, dict):
        message = error.get("message") or error.get("details")
        if isinstance(message, str) and message:
            return f"ComfyUI rejected the workflow: {message}"
    if isinstance(error, str) and error:
        return f"ComfyUI rejected the workflow: {error}"
    node_errors = payload.get("node_errors")
    if isinstance(node_errors, dict) and node_errors:
        return f"ComfyUI rejected the workflow: {next(iter(node_errors))}"
    return fallback


def _checkpoint_matches(configured: str, checkpoint_names: list[str]) -> bool:
    configured_lower = configured.lower()
    return any(configured_lower == name.lower() or configured_lower in name.lower() for name in checkpoint_names)
