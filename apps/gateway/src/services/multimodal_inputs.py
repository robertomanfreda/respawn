from __future__ import annotations

import base64
import binascii
import io
import mimetypes
import re
from pathlib import Path
from typing import Any
from urllib.parse import unquote_to_bytes, urlparse

import httpx

from src.config import Settings
from src.schemas.errors import OpenAIError
from src.schemas.responses import ResponseRequest


TEXT_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cs",
    ".css",
    ".csv",
    ".go",
    ".h",
    ".hpp",
    ".html",
    ".ini",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".kt",
    ".lua",
    ".m",
    ".md",
    ".markdown",
    ".php",
    ".py",
    ".rb",
    ".rs",
    ".sh",
    ".sql",
    ".swift",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
TEXT_MIME_TYPES = {
    "application/json",
    "application/toml",
    "application/xml",
    "application/x-yaml",
    "text/csv",
    "text/markdown",
    "text/plain",
    "text/tab-separated-values",
    "text/xml",
}
DATA_URL_RE = re.compile(r"^data:(?P<mime>[^;,]+)?(?P<base64>;base64)?,(?P<data>.*)$", re.DOTALL)


async def prepare_multimodal_request(request: ResponseRequest, *, model: str, settings: Settings) -> ResponseRequest:
    """Normalize supported image/file inputs before storage and backend calls."""

    input_value = request.input
    if not isinstance(input_value, list):
        return request

    caps = _capabilities_for_model(model, settings)
    needs_image = _contains_part_type(input_value, "input_image")
    needs_file = _contains_part_type(input_value, "input_file")
    if _contains_part_type(input_value, "input_audio"):
        _unsupported("input", "Audio input is not supported by Respawn.")
    if needs_image and "vision" not in caps:
        raise OpenAIError(
            f"Model '{model}' is not configured with the vision capability required for input_image.",
            param="model",
            code="unsupported_model_capability",
        )
    if needs_file and "file-text" not in caps:
        raise OpenAIError(
            f"Model '{model}' is not configured with the file-text capability required for input_file.",
            param="model",
            code="unsupported_model_capability",
        )

    normalized = []
    changed = False
    for index, item in enumerate(input_value):
        next_item, item_changed = await _normalize_input_item(item, settings=settings, param=f"input.{index}")
        normalized.append(next_item)
        changed = changed or item_changed
    if not changed:
        return request
    return request.model_copy(update={"input": normalized})


def model_capabilities(settings: Settings) -> dict[str, set[str]]:
    capabilities: dict[str, set[str]] = {}
    for entry in settings.model_capabilities.split(";"):
        entry = entry.strip()
        if not entry or "=" not in entry:
            continue
        model, values = entry.split("=", 1)
        capabilities[model.strip().lower()] = {value.strip().lower() for value in values.split(",") if value.strip()}
    return capabilities


def _capabilities_for_model(model: str, settings: Settings) -> set[str]:
    configured = model_capabilities(settings)
    model_key = model.lower()
    if model_key in configured:
        return configured[model_key]
    if ":" in model_key:
        base = model_key.split(":", 1)[0]
        if base in configured:
            return configured[base]
    return {"text"}


async def _normalize_input_item(item: dict[str, Any], *, settings: Settings, param: str) -> tuple[dict[str, Any], bool]:
    if not isinstance(item, dict):
        return item, False
    item_type = item.get("type")
    role = item.get("role")
    if item_type in {"input_image", "input_file", "input_audio"}:
        normalized = await _normalize_content_part(item, settings=settings, param=param)
        return normalized, True
    if item_type != "message" and role not in {"user", "assistant", "system", "developer"}:
        return item, False
    content, changed = await _normalize_content(item.get("content", ""), settings=settings, param=f"{param}.content")
    if not changed:
        return item, False
    return {**item, "content": content}, True


async def _normalize_content(content: Any, *, settings: Settings, param: str) -> tuple[Any, bool]:
    if isinstance(content, list):
        normalized = []
        changed = False
        for index, part in enumerate(content):
            if isinstance(part, dict):
                next_part = await _normalize_content_part(part, settings=settings, param=f"{param}.{index}")
                normalized.append(next_part)
                changed = changed or next_part is not part
            else:
                normalized.append(part)
        return normalized, changed
    if isinstance(content, dict):
        normalized = await _normalize_content_part(content, settings=settings, param=param)
        return normalized, normalized is not content
    return content, False


async def _normalize_content_part(part: dict[str, Any], *, settings: Settings, param: str) -> dict[str, Any]:
    part_type = part.get("type")
    if part_type == "input_image":
        return await _normalize_input_image(part, settings=settings, param=param)
    if part_type == "input_file":
        return await _normalize_input_file(part, settings=settings, param=param)
    if part_type == "input_audio":
        _unsupported(f"{param}.type", "Audio input is not supported by Respawn.")
    return part


async def _normalize_input_image(part: dict[str, Any], *, settings: Settings, param: str) -> dict[str, Any]:
    _reject_file_id(part, param)
    source = _part_source(part, keys=("image_url", "image_data", "data"))
    payload = await _load_bytes(source, part=part, settings=settings, param=f"{param}.image_url", max_bytes=settings.multimodal_max_image_bytes)
    mime_type = payload["mime_type"] or part.get("mime_type") or mimetypes.guess_type(payload["source_label"])[0] or "application/octet-stream"
    if not str(mime_type).lower().startswith("image/"):
        raise OpenAIError("input_image must reference an image MIME type.", param=f"{param}.image_url", code="invalid_image")
    return {
        "type": "input_image",
        "image_url": payload["source_label"],
        "image_base64": base64.b64encode(payload["data"]).decode("ascii"),
        "mime_type": mime_type,
        "detail": _image_detail(part.get("detail"), param=param),
    }


async def _normalize_input_file(part: dict[str, Any], *, settings: Settings, param: str) -> dict[str, Any]:
    _reject_file_id(part, param)
    source = _part_source(part, keys=("file_url", "file_data", "data"))
    payload = await _load_bytes(source, part=part, settings=settings, param=f"{param}.file_data", max_bytes=settings.multimodal_max_file_bytes)
    filename = str(part.get("filename") or _filename_from_source(payload["source_label"]) or "input_file")
    mime_type = payload["mime_type"] or part.get("mime_type") or mimetypes.guess_type(filename)[0] or "application/octet-stream"
    text = _extract_file_text(payload["data"], filename=filename, mime_type=str(mime_type), param=param)
    return {
        "type": "input_file",
        "filename": filename,
        "mime_type": mime_type,
        "text": text,
        "size_bytes": len(payload["data"]),
        "source": payload["source_label"],
    }


async def _load_bytes(source: Any, *, part: dict[str, Any], settings: Settings, param: str, max_bytes: int) -> dict[str, Any]:
    if isinstance(source, dict):
        source = source.get("url") or source.get("data") or source.get("file_data")
    if not isinstance(source, str) or not source:
        raise OpenAIError("A URL, data URL, or base64 payload is required for multimodal input.", param=param)

    match = DATA_URL_RE.match(source)
    if match:
        mime_type = match.group("mime") or part.get("mime_type")
        raw = match.group("data")
        if match.group("base64"):
            data = _decode_base64(raw, param=param)
        else:
            data = unquote_to_bytes(raw)
        _enforce_size(data, max_bytes=max_bytes, param=param)
        return {"data": data, "mime_type": mime_type, "source_label": source}

    if source.startswith(("http://", "https://")):
        try:
            async with httpx.AsyncClient(timeout=settings.multimodal_download_timeout_seconds, follow_redirects=True) as client:
                response = await client.get(source)
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise OpenAIError("Timed out while downloading multimodal input.", param=param, code="file_download_timeout") from exc
        except httpx.HTTPStatusError as exc:
            raise OpenAIError(f"Failed to download multimodal input: HTTP {exc.response.status_code}.", param=param, code="file_download_failed") from exc
        except httpx.HTTPError as exc:
            raise OpenAIError("Failed to download multimodal input.", param=param, code="file_download_failed") from exc
        data = response.content
        _enforce_size(data, max_bytes=max_bytes, param=param)
        mime_type = response.headers.get("content-type", "").split(";", 1)[0] or part.get("mime_type")
        return {"data": data, "mime_type": mime_type, "source_label": source}

    data = _decode_base64(source, param=param)
    _enforce_size(data, max_bytes=max_bytes, param=param)
    return {"data": data, "mime_type": part.get("mime_type"), "source_label": part.get("filename") or "base64"}


def _extract_file_text(data: bytes, *, filename: str, mime_type: str, param: str) -> str:
    extension = Path(filename).suffix.lower()
    normalized_mime = mime_type.lower()
    if extension == ".pdf" or normalized_mime == "application/pdf":
        return _extract_pdf_text(data)
    if extension in TEXT_EXTENSIONS or normalized_mime.startswith("text/") or normalized_mime in TEXT_MIME_TYPES:
        return _decode_text(data)
    raise OpenAIError(
        f"Unsupported input_file type for '{filename}'. Respawn currently extracts text, code, CSV, JSON, Markdown, and PDF files.",
        param=f"{param}.file_data",
        code="unsupported_file_type",
    )


def _extract_pdf_text(data: bytes) -> str:
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
        if text:
            return text
    except Exception:
        pass
    decoded = data.decode("latin-1", errors="ignore")
    matches = re.findall(r"\(([^()]*)\)\s*Tj", decoded)
    if matches:
        return "\n".join(matches)
    printable = "".join(char if char.isprintable() or char in "\n\t" else " " for char in decoded)
    return re.sub(r"\s+", " ", printable).strip()


def _decode_text(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="replace")


def _part_source(part: dict[str, Any], *, keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in part:
            return part[key]
    return None


def _filename_from_source(source: str) -> str | None:
    if source.startswith(("http://", "https://")):
        path = urlparse(source).path
        name = Path(path).name
        return name or None
    if not source.startswith("data:"):
        name = Path(source).name
        return name or None
    return None


def _decode_base64(value: str, *, param: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise OpenAIError("Multimodal base64 payload is invalid.", param=param, code="invalid_base64") from exc


def _enforce_size(data: bytes, *, max_bytes: int, param: str) -> None:
    if len(data) > max_bytes:
        raise OpenAIError(f"Multimodal input exceeds the {max_bytes} byte limit.", param=param, code="file_too_large")


def _image_detail(value: Any, *, param: str) -> str:
    detail = "auto" if value is None else str(value)
    if detail not in {"auto", "low", "high", "original"}:
        raise OpenAIError("input_image.detail must be one of auto, low, high, or original.", param=f"{param}.detail")
    return detail


def _reject_file_id(part: dict[str, Any], param: str) -> None:
    if part.get("file_id"):
        _unsupported(f"{param}.file_id", "input_file/input_image file_id is not supported until Respawn has a local Files API.")


def _contains_part_type(input_value: list[dict[str, Any]], part_type: str) -> bool:
    for item in input_value:
        if not isinstance(item, dict):
            continue
        if item.get("type") == part_type:
            return True
        content = item.get("content")
        if isinstance(content, dict) and content.get("type") == part_type:
            return True
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == part_type:
                    return True
    return False


def _unsupported(param: str, message: str) -> None:
    raise OpenAIError(message, param=param, code="unsupported_parameter")
