from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from src.schemas.errors import OpenAIError


@dataclass(frozen=True)
class IdempotencyEntry:
    body_sha256: str
    status_code: int
    headers: dict[str, str]
    body: bytes
    media_type: str | None


class IdempotencyStore:
    def __init__(self, max_entries: int = 1024) -> None:
        self.max_entries = max(max_entries, 1)
        self._entries: OrderedDict[str, IdempotencyEntry] = OrderedDict()

    def get(self, cache_key: str, body_sha256: str) -> IdempotencyEntry | None:
        entry = self._entries.get(cache_key)
        if entry is None:
            return None
        self._entries.move_to_end(cache_key)
        if entry.body_sha256 != body_sha256:
            raise OpenAIError(
                "Idempotency-Key was reused with a different request body.",
                status_code=409,
                param="Idempotency-Key",
                code="idempotency_conflict",
            )
        return entry

    def put(
        self,
        cache_key: str,
        *,
        body_sha256: str,
        status_code: int,
        headers: dict[str, str],
        body: bytes,
        media_type: str | None,
    ) -> None:
        self._entries[cache_key] = IdempotencyEntry(
            body_sha256=body_sha256,
            status_code=status_code,
            headers=headers,
            body=body,
            media_type=media_type,
        )
        self._entries.move_to_end(cache_key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
