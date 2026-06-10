import re
from typing import Protocol


class ReasoningSummaryProvider(Protocol):
    def summarize(self, reasoning_text: str, *, mode: str | None = None) -> str:
        raise NotImplementedError


class DeterministicReasoningSummaryProvider:
    def summarize(self, reasoning_text: str, *, mode: str | None = None) -> str:
        tokens = estimate_text_tokens(reasoning_text)
        if tokens <= 0:
            return "No reasoning trace was returned by the local backend."
        return f"Local backend returned a reasoning trace before the final answer. Estimated reasoning tokens: {tokens}. Raw reasoning content is intentionally not exposed by Respawn."


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))
