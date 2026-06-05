from typing import Any

from src.schemas.responses import ResponseUsage


def normalize_usage(raw: dict[str, Any] | None) -> ResponseUsage:
    raw = raw or {}
    input_tokens = int(raw.get("input_tokens", raw.get("prompt_tokens", 0)) or 0)
    output_tokens = int(raw.get("output_tokens", raw.get("completion_tokens", 0)) or 0)
    total_tokens = int(raw.get("total_tokens", input_tokens + output_tokens) or 0)
    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens
    input_details = raw.get("input_tokens_details") if isinstance(raw.get("input_tokens_details"), dict) else {}
    output_details = raw.get("output_tokens_details") if isinstance(raw.get("output_tokens_details"), dict) else {}
    return ResponseUsage(
        input_tokens=input_tokens,
        input_tokens_details={"cached_tokens": int(input_details.get("cached_tokens", 0) or 0)},
        output_tokens=output_tokens,
        output_tokens_details={"reasoning_tokens": int(output_details.get("reasoning_tokens", 0) or 0)},
        total_tokens=total_tokens,
    )


def enrich_response_usage(
    usage: ResponseUsage,
    *,
    cached_tokens: int = 0,
    reasoning_tokens: int = 0,
    minimum_output_tokens: int = 0,
) -> ResponseUsage:
    cached = max(0, min(cached_tokens, usage.input_tokens))
    reasoning = max(0, reasoning_tokens)
    output_tokens = usage.output_tokens
    if reasoning:
        output_tokens = max(output_tokens, minimum_output_tokens + reasoning)
    return ResponseUsage(
        input_tokens=usage.input_tokens,
        input_tokens_details={"cached_tokens": cached},
        output_tokens=output_tokens,
        output_tokens_details={"reasoning_tokens": reasoning},
        total_tokens=usage.input_tokens + output_tokens,
    )
