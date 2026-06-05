from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import re
from time import monotonic
from typing import Any


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


@dataclass(frozen=True)
class PromptCacheMatch:
    input_tokens: int
    cached_tokens: int
    cache_key: str
    retention: str


class PromptCache:
    """Small in-process prompt prefix cache used for OpenAI-like usage accounting."""

    def __init__(
        self,
        *,
        enabled: bool,
        min_tokens: int,
        max_entries: int,
        in_memory_ttl_seconds: int,
        extended_ttl_seconds: int,
        chunk_tokens: int,
    ) -> None:
        self.enabled = enabled
        self.min_tokens = max(1, min_tokens)
        self.max_entries = max(1, max_entries)
        self.in_memory_ttl_seconds = max(1, in_memory_ttl_seconds)
        self.extended_ttl_seconds = max(1, extended_ttl_seconds)
        self.chunk_tokens = max(1, chunk_tokens)
        self._entries: OrderedDict[str, float] = OrderedDict()

    def inspect(self, payload: dict[str, Any], *, prompt_cache_key: str | None, retention: str | None) -> PromptCacheMatch:
        cache_key = prompt_cache_key or "default"
        retention_policy = retention or "in_memory"
        tokens = _prompt_tokens(payload)
        cached_tokens = self._cached_tokens(tokens, cache_key) if self.enabled else 0
        return PromptCacheMatch(
            input_tokens=len(tokens),
            cached_tokens=cached_tokens,
            cache_key=cache_key,
            retention=retention_policy,
        )

    def store(self, payload: dict[str, Any], *, prompt_cache_key: str | None, retention: str | None) -> None:
        if not self.enabled:
            return
        cache_key = prompt_cache_key or "default"
        retention_policy = retention or "in_memory"
        tokens = _prompt_tokens(payload)
        if len(tokens) < self.min_tokens:
            return

        now = monotonic()
        ttl = self.extended_ttl_seconds if retention_policy == "24h" else self.in_memory_ttl_seconds
        expires_at = now + ttl
        self._prune(now)
        for length in self._prefix_lengths(len(tokens)):
            key = _prefix_key(cache_key, tokens, length)
            self._entries[key] = expires_at
            self._entries.move_to_end(key)
        self._trim()

    def _cached_tokens(self, tokens: list[str], cache_key: str) -> int:
        if len(tokens) < self.min_tokens:
            return 0
        now = monotonic()
        self._prune(now)
        for length in reversed(self._prefix_lengths(len(tokens))):
            key = _prefix_key(cache_key, tokens, length)
            expires_at = self._entries.get(key)
            if expires_at and expires_at > now:
                self._entries.move_to_end(key)
                return length
        return 0

    def _prefix_lengths(self, total_tokens: int) -> list[int]:
        if total_tokens < self.min_tokens:
            return []
        lengths = list(range(self.min_tokens, total_tokens + 1, self.chunk_tokens))
        if not lengths or lengths[-1] != total_tokens:
            lengths.append(total_tokens)
        return lengths

    def _prune(self, now: float) -> None:
        expired = [key for key, expires_at in self._entries.items() if expires_at <= now]
        for key in expired:
            self._entries.pop(key, None)

    def _trim(self) -> None:
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)


def _prompt_tokens(payload: dict[str, Any]) -> list[str]:
    prompt_surface = {
        "model": payload.get("model"),
        "messages": payload.get("messages") or [],
        "tools": payload.get("tools") or [],
        "response_format": payload.get("response_format"),
    }
    text = json.dumps(prompt_surface, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return TOKEN_PATTERN.findall(text)


def _prefix_key(cache_key: str, tokens: list[str], length: int) -> str:
    digest = hashlib.sha256()
    digest.update(cache_key.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(length).encode("ascii"))
    digest.update(b"\0")
    digest.update("\n".join(tokens[:length]).encode("utf-8"))
    return digest.hexdigest()
