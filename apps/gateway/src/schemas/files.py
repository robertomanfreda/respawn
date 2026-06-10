from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from src.storage.models import PlatformFileRecord


class FileObject(BaseModel):
    id: str
    object: Literal["file"] = "file"
    bytes: int
    created_at: int
    filename: str
    purpose: str
    status: Literal["uploaded", "processed", "error"] = "processed"
    status_details: str | None = None
    expires_at: int | None = None


class FileList(BaseModel):
    object: Literal["list"] = "list"
    data: list[FileObject] = Field(default_factory=list)
    first_id: str | None = None
    last_id: str | None = None
    has_more: bool = False


class FileDeleted(BaseModel):
    id: str
    object: Literal["file"] = "file"
    deleted: bool = True


def file_object(record: PlatformFileRecord) -> FileObject:
    return FileObject(
        id=record.id,
        bytes=record.bytes,
        created_at=_timestamp(record.created_at),
        filename=record.filename,
        purpose=record.purpose,
        expires_at=_timestamp(record.expires_at) if record.expires_at is not None else None,
    )


def _timestamp(value: datetime) -> int:
    return int(value.timestamp())
