from __future__ import annotations

import hashlib
import asyncio
import json
import mimetypes
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
from typing import Any

from fastapi import Request

from src.config import Settings
from src.observability.metrics import STORAGE_OPERATIONS
from src.schemas.errors import OpenAIError
from src.storage.models import PlatformFileRecord
from src.storage.repository import ResponseRepository


SUPPORTED_FILE_PURPOSES = {"assistants", "batch", "fine-tune", "vision", "user_data", "evals"}
INPUT_FILE_PURPOSES = {"assistants", "user_data"}
MIN_EXPIRES_AFTER_SECONDS = 3600
MAX_EXPIRES_AFTER_SECONDS = 2_592_000
EICAR_SIGNATURE = b"X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR"
FILENAME_RE = re.compile(r'filename="(?P<value>(?:\\.|[^"])*)"')
NAME_RE = re.compile(r'name="(?P<value>(?:\\.|[^"])*)"')


@dataclass(frozen=True)
class UploadedFilePayload:
    filename: str
    purpose: str
    data: bytes
    mime_type: str | None
    expires_after_seconds: int | None = None


class PlatformFileStorage:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def backend(self) -> str:
        backend = self.settings.file_storage_backend.lower()
        if backend not in {"database", "filesystem"}:
            raise OpenAIError(
                f"Unsupported FILE_STORAGE_BACKEND '{self.settings.file_storage_backend}'.",
                status_code=500,
                type="server_error",
                code="invalid_file_storage_backend",
            )
        return backend

    def store(self, *, file_id: str, tenant_id: str | None, data: bytes) -> tuple[str, str | None, bytes | None]:
        backend = "unknown"
        status = "failed"
        try:
            backend = self.backend
            if backend == "database":
                status = "completed"
                return "database", None, data
            path = self._path(file_id=file_id, tenant_id=tenant_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            status = "completed"
            return "filesystem", str(path), None
        finally:
            STORAGE_OPERATIONS.labels(backend=backend, operation="store", status=status).inc()

    def read(self, record: PlatformFileRecord) -> bytes:
        backend = record.storage_backend or "unknown"
        status = "failed"
        try:
            if record.storage_backend == "database":
                status = "completed"
                return record.content_bytes or b""
            if not record.storage_path:
                raise OpenAIError("File content is missing.", status_code=404, param="file_id", code="not_found")
            path = Path(record.storage_path)
            if not path.exists():
                raise OpenAIError("File content is missing.", status_code=404, param="file_id", code="not_found")
            data = path.read_bytes()
            status = "completed"
            return data
        finally:
            STORAGE_OPERATIONS.labels(backend=backend, operation="read", status=status).inc()

    def delete(self, record: PlatformFileRecord) -> None:
        backend = record.storage_backend or "unknown"
        status = "failed"
        try:
            if record.storage_backend != "filesystem" or not record.storage_path:
                status = "completed"
                return
            path = Path(record.storage_path)
            if path.exists():
                path.unlink()
            status = "completed"
        finally:
            STORAGE_OPERATIONS.labels(backend=backend, operation="delete", status=status).inc()

    def _path(self, *, file_id: str, tenant_id: str | None) -> Path:
        tenant_label = _safe_path_segment(tenant_id or "global")
        return Path(self.settings.file_storage_path) / tenant_label / file_id


class PlatformFileService:
    def __init__(self, *, settings: Settings, repository: ResponseRepository) -> None:
        self.settings = settings
        self.repository = repository
        self.storage = PlatformFileStorage(settings)

    async def create_from_request(self, request: Request, tenant_id: str | None) -> PlatformFileRecord:
        payload = await parse_file_upload_request(request)
        return await self.create_file(payload, tenant_id)

    async def create_file(self, payload: UploadedFilePayload, tenant_id: str | None) -> PlatformFileRecord:
        _validate_upload(payload, self.settings)
        active_bytes = await self.repository.total_platform_file_bytes(tenant_id)
        if active_bytes + len(payload.data) > self.settings.file_storage_quota_bytes:
            raise OpenAIError("Local file storage quota exceeded.", param="file", code="storage_quota_exceeded")

        file_id = await self.repository.next_platform_file_id()
        storage_backend, storage_path, content_bytes = self.storage.store(file_id=file_id, tenant_id=tenant_id, data=payload.data)
        now = datetime.now(timezone.utc)
        expires_at = _expires_at(payload.expires_after_seconds, settings=self.settings, now=now)
        record = await self.repository.create_platform_file(
            file_id=file_id,
            filename=payload.filename,
            purpose=payload.purpose,
            bytes_count=len(payload.data),
            mime_type=payload.mime_type or mimetypes.guess_type(payload.filename)[0] or "application/octet-stream",
            sha256=hashlib.sha256(payload.data).hexdigest(),
            storage_backend=storage_backend,
            storage_path=storage_path,
            content_bytes=content_bytes,
            metadata_json={},
            tenant_id=tenant_id,
            created_at=now,
            expires_at=expires_at,
        )
        await self.repository.session.commit()
        return record

    async def read_file_content(self, file_id: str, tenant_id: str | None) -> tuple[PlatformFileRecord, bytes]:
        record = await self.repository.require_platform_file(file_id, tenant_id)
        return record, self.storage.read(record)

    async def delete_file(self, file_id: str, tenant_id: str | None) -> PlatformFileRecord:
        record = await self.repository.delete_platform_file(file_id, tenant_id)
        self.storage.delete(record)
        record.storage_path = None
        await self.repository.session.commit()
        return record

    async def cleanup_expired_files(self) -> int:
        status = "failed"
        try:
            records = await self.repository.mark_expired_platform_files_deleted()
            for record in records:
                self.storage.delete(record)
                record.storage_path = None
            await self.repository.session.commit()
            status = "completed"
            return len(records)
        finally:
            STORAGE_OPERATIONS.labels(backend=self.storage.settings.file_storage_backend.lower(), operation="cleanup", status=status).inc()


async def run_platform_file_cleanup(*, settings: Settings, session_factory: Any) -> None:
    interval = max(float(settings.file_cleanup_interval_seconds), 0.1)
    while True:
        await asyncio.sleep(interval)
        async with session_factory() as session:
            service = PlatformFileService(settings=settings, repository=ResponseRepository(session))
            await service.cleanup_expired_files()


async def parse_file_upload_request(request: Request) -> UploadedFilePayload:
    content_type = request.headers.get("content-type", "")
    body = await request.body()
    if content_type.startswith("multipart/form-data"):
        return _parse_multipart_upload(body, content_type)
    if content_type.startswith("application/json"):
        return _parse_json_upload(body)
    raise OpenAIError("Files must be uploaded as multipart/form-data.", param="file", code="invalid_request")


def _parse_multipart_upload(body: bytes, content_type: str) -> UploadedFilePayload:
    boundary = _multipart_boundary(content_type)
    fields: dict[str, str] = {}
    file_data: bytes | None = None
    filename = ""
    mime_type: str | None = None
    for headers, data in _multipart_parts(body, boundary):
        disposition = headers.get("content-disposition", "")
        name = _disposition_value(disposition, NAME_RE)
        if not name:
            continue
        if name == "file":
            file_data = data
            filename = _disposition_value(disposition, FILENAME_RE) or "upload"
            mime_type = headers.get("content-type")
        else:
            fields[name] = data.decode("utf-8", errors="replace")
    if file_data is None:
        raise OpenAIError("file is required.", param="file", code="invalid_request")
    purpose = fields.get("purpose", "")
    expires_after = _expires_after_from_field(fields.get("expires_after"))
    return UploadedFilePayload(filename=filename, purpose=purpose, data=file_data, mime_type=mime_type, expires_after_seconds=expires_after)


def _parse_json_upload(body: bytes) -> UploadedFilePayload:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OpenAIError("Invalid JSON file upload body.", param="file", code="invalid_request") from exc
    data_value = payload.get("file_data") or payload.get("data")
    if not isinstance(data_value, str):
        raise OpenAIError("file_data is required.", param="file_data", code="invalid_request")
    import base64

    try:
        data = base64.b64decode(data_value.removeprefix("data:").split(",", 1)[-1], validate=True)
    except Exception as exc:
        raise OpenAIError("file_data must be base64 encoded.", param="file_data", code="invalid_base64") from exc
    expires_after = payload.get("expires_after")
    seconds = _expires_after_from_field(json.dumps(expires_after)) if isinstance(expires_after, dict) else None
    return UploadedFilePayload(
        filename=str(payload.get("filename") or "upload"),
        purpose=str(payload.get("purpose") or ""),
        data=data,
        mime_type=payload.get("mime_type"),
        expires_after_seconds=seconds,
    )


def _multipart_boundary(content_type: str) -> bytes:
    for part in content_type.split(";"):
        part = part.strip()
        if part.startswith("boundary="):
            return part.removeprefix("boundary=").strip('"').encode("utf-8")
    raise OpenAIError("multipart boundary is missing.", param="file", code="invalid_request")


def _multipart_parts(body: bytes, boundary: bytes) -> list[tuple[dict[str, str], bytes]]:
    marker = b"--" + boundary
    parts = []
    for raw in body.split(marker):
        raw = raw.strip()
        if not raw or raw == b"--":
            continue
        if raw.endswith(b"--"):
            raw = raw[:-2].strip()
        header_blob, separator, data = raw.partition(b"\r\n\r\n")
        if not separator:
            continue
        headers: dict[str, str] = {}
        for line in header_blob.split(b"\r\n"):
            key, _, value = line.decode("utf-8", errors="replace").partition(":")
            if key:
                headers[key.lower()] = value.strip()
        if data.endswith(b"\r\n"):
            data = data[:-2]
        parts.append((headers, data))
    return parts


def _disposition_value(disposition: str, pattern: re.Pattern[str]) -> str | None:
    match = pattern.search(disposition)
    if not match:
        return None
    return match.group("value").replace('\\"', '"')


def _expires_after_from_field(value: str | None) -> int | None:
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise OpenAIError("expires_after must be a JSON object.", param="expires_after", code="invalid_request") from exc
    if not isinstance(parsed, dict):
        raise OpenAIError("expires_after must be an object.", param="expires_after", code="invalid_request")
    if parsed.get("anchor") != "created_at":
        raise OpenAIError("expires_after.anchor must be 'created_at'.", param="expires_after.anchor", code="invalid_request")
    seconds = parsed.get("seconds")
    if not isinstance(seconds, int) or seconds < MIN_EXPIRES_AFTER_SECONDS or seconds > MAX_EXPIRES_AFTER_SECONDS:
        raise OpenAIError("expires_after.seconds must be between 3600 and 2592000.", param="expires_after.seconds", code="invalid_request")
    return seconds


def _validate_upload(payload: UploadedFilePayload, settings: Settings) -> None:
    if not payload.filename.strip():
        raise OpenAIError("filename is required.", param="file", code="invalid_request")
    if payload.purpose not in SUPPORTED_FILE_PURPOSES:
        raise OpenAIError("Unsupported file purpose.", param="purpose", code="invalid_file_purpose")
    if len(payload.data) == 0:
        raise OpenAIError("Uploaded file must not be empty.", param="file", code="invalid_request")
    if len(payload.data) > settings.file_upload_max_bytes:
        raise OpenAIError(f"Uploaded file exceeds the {settings.file_upload_max_bytes} byte limit.", param="file", code="file_too_large")
    if settings.file_malware_scan_enabled and EICAR_SIGNATURE in payload.data:
        raise OpenAIError("Uploaded file failed local malware validation.", param="file", code="file_malware_detected")


def _expires_at(seconds: int | None, *, settings: Settings, now: datetime) -> datetime | None:
    ttl = seconds if seconds is not None else settings.file_default_ttl_seconds
    if ttl <= 0:
        return None
    return now + timedelta(seconds=ttl)


def _safe_path_segment(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return safe or "tenant"
