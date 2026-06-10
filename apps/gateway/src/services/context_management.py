import base64
import hashlib
import hmac
import json
import re
import secrets
from dataclasses import dataclass
from typing import Any

from src.config import Settings
from src.schemas.errors import OpenAIError
from src.schemas.responses import ResponseRequest
from src.services.response_history_builder import build_messages, content_to_text
from src.services.id_generator import generate_id


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
COMPACTION_ENVELOPE_VERSION = 1
COMPACTION_ENVELOPE_ALGORITHM = "respawn-context-compaction-hmac-sha256-stream"


@dataclass(frozen=True)
class ContextPlan:
    chain: list[dict[str, Any]]
    request: ResponseRequest
    input_tokens_before: int
    input_tokens_after: int
    limit: int
    truncated_items: int = 0
    compaction_item: dict[str, Any] | None = None
    compaction_summary: dict[str, Any] | None = None
    source_item_ids: list[str] | None = None
    strategy: str = "none"

    @property
    def changed(self) -> bool:
        return self.strategy != "none"


class RegexTokenizer:
    """Deterministic local tokenizer used when the backend exposes no tokenizer."""

    name = "regex"

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        return len(TOKEN_PATTERN.findall(text))

    def count_payload(self, payload: dict[str, Any]) -> int:
        surface = {
            "model": payload.get("model"),
            "messages": payload.get("messages") or [],
            "tools": payload.get("tools") or [],
            "response_format": payload.get("response_format"),
        }
        return self.count_text(json.dumps(surface, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


class ContextPlanner:
    def __init__(self, settings: Settings, tokenizer: RegexTokenizer | None = None) -> None:
        self.settings = settings
        self.tokenizer = tokenizer or RegexTokenizer()

    def count_payload_tokens(self, payload: dict[str, Any]) -> int:
        return self.tokenizer.count_payload(payload)

    def plan(self, *, model: str, request: ResponseRequest, chain: list[dict[str, Any]]) -> ContextPlan:
        before = self._count(model=model, request=request, chain=chain)
        limit = self.input_token_limit(model, request)
        threshold = compact_threshold(request.context_management)

        if threshold is not None and before >= threshold and (chain or _input_item_count(request.input) > 1):
            return self._compact(model=model, request=request, chain=chain, before=before, limit=limit)

        if before <= limit:
            return ContextPlan(chain=chain, request=request, input_tokens_before=before, input_tokens_after=before, limit=limit)

        if request.truncation == "disabled":
            raise_context_overflow(before=before, limit=limit, truncation=request.truncation)

        return self._truncate_auto(model=model, request=request, chain=chain, before=before, limit=limit)

    def input_token_limit(self, model: str, request: ResponseRequest) -> int:
        context_window = context_window_for_model(model, self.settings)
        output_budget = request.max_output_tokens or self.settings.max_output_tokens_default
        return max(1, context_window - output_budget - max(0, self.settings.context_token_margin))

    def _count(self, *, model: str, request: ResponseRequest, chain: list[dict[str, Any]]) -> int:
        payload = {
            "model": model,
            "messages": build_messages(
                instructions=request.instructions,
                chain=chain,
                input_value=request.input,
                compaction_key=self.settings.reasoning_encryption_key,
            ),
            "tools": request.tools,
        }
        return self.tokenizer.count_payload(payload)

    def _truncate_auto(self, *, model: str, request: ResponseRequest, chain: list[dict[str, Any]], before: int, limit: int) -> ContextPlan:
        planned_chain = list(chain)
        planned_request = request
        truncated = 0

        while planned_chain and self._count(model=model, request=planned_request, chain=planned_chain) > limit:
            truncated += response_item_count(planned_chain.pop(0))

        if self._count(model=model, request=planned_request, chain=planned_chain) > limit and isinstance(request.input, list):
            input_items = list(request.input)
            while len(input_items) > 1:
                input_items.pop(0)
                truncated += 1
                planned_request = request.model_copy(update={"input": input_items})
                if self._count(model=model, request=planned_request, chain=planned_chain) <= limit:
                    break

        after = self._count(model=model, request=planned_request, chain=planned_chain)
        if after > limit:
            raise_context_overflow(before=before, limit=limit, truncation=request.truncation)

        return ContextPlan(
            chain=planned_chain,
            request=planned_request,
            input_tokens_before=before,
            input_tokens_after=after,
            limit=limit,
            truncated_items=truncated,
            strategy="truncation_auto",
        )

    def _compact(self, *, model: str, request: ResponseRequest, chain: list[dict[str, Any]], before: int, limit: int) -> ContextPlan:
        summary = summarize_context(chain=chain, input_value=request.input)
        item_id = generate_compaction_item_id(summary["text"])
        encrypted_content = seal_compaction_content(
            summary,
            key=self.settings.reasoning_encryption_key,
            item_id=item_id,
            source_item_ids=summary["source_item_ids"],
        )
        compaction_item = {
            "id": item_id,
            "type": "compaction",
            "encrypted_content": encrypted_content,
        }

        retained_input = _retained_tail(request.input)
        planned_request = request.model_copy(update={"input": retained_input})
        planned_chain = [
            {
                "id": "resp_context_compacted",
                "request_json": {"input": [compaction_item]},
                "input_items": [compaction_item],
                "output_json": [],
                "model": model,
            }
        ]

        after = self._count(model=model, request=planned_request, chain=planned_chain)
        if after > limit and request.truncation == "disabled":
            raise_context_overflow(before=after, limit=limit, truncation=request.truncation)
        if after > limit:
            return self._truncate_auto(model=model, request=planned_request, chain=planned_chain, before=before, limit=limit)

        return ContextPlan(
            chain=planned_chain,
            request=planned_request,
            input_tokens_before=before,
            input_tokens_after=after,
            limit=limit,
            compaction_item=compaction_item,
            compaction_summary=summary,
            source_item_ids=summary["source_item_ids"],
            strategy="context_management_compaction",
        )


def compact_response_window(
    *,
    input_value: str | list[dict[str, Any]] | None,
    model: str,
    settings: Settings,
) -> tuple[list[dict[str, Any]], ResponseRequest, dict[str, Any], int, int]:
    request = ResponseRequest(model=model, input=input_value)
    planner = ContextPlanner(settings)
    before = planner._count(model=model, request=request, chain=[])
    summary = summarize_context(chain=[], input_value=input_value)
    item_id = generate_compaction_item_id(summary["text"])
    compaction_item = {
        "id": item_id,
        "type": "compaction",
        "encrypted_content": seal_compaction_content(
            summary,
            key=settings.reasoning_encryption_key,
            item_id=item_id,
            source_item_ids=summary["source_item_ids"],
        ),
    }
    retained = _retained_tail(input_value)
    output = _items_for_compact_output(retained, compaction_item)
    after_request = ResponseRequest(model=model, input=output)
    after = planner._count(model=model, request=after_request, chain=[])
    return output, after_request, summary, before, after


def compact_threshold(context_management: list[dict[str, Any]] | None) -> int | None:
    if not context_management:
        return None
    thresholds = [
        int(entry["compact_threshold"])
        for entry in context_management
        if isinstance(entry, dict) and entry.get("type") == "compaction" and entry.get("compact_threshold") is not None
    ]
    return min(thresholds) if thresholds else None


def validate_context_management(context_management: Any) -> None:
    if context_management is None:
        return
    if not isinstance(context_management, list):
        raise OpenAIError("context_management must be a list.", param="context_management")
    if len(context_management) > 8:
        raise OpenAIError("context_management must contain at most 8 entries.", param="context_management")
    for index, entry in enumerate(context_management):
        if not isinstance(entry, dict):
            raise OpenAIError("context_management entries must be objects.", param=f"context_management.{index}")
        entry_type = entry.get("type")
        if entry_type != "compaction":
            raise OpenAIError("Only context_management entries with type='compaction' are supported.", status_code=400, param=f"context_management.{index}.type", code="unsupported_parameter")
        threshold = entry.get("compact_threshold")
        if threshold is not None:
            if not isinstance(threshold, int) or threshold < 1000:
                raise OpenAIError("context_management compact_threshold must be an integer greater than or equal to 1000.", param=f"context_management.{index}.compact_threshold")
        unsupported = sorted(set(entry) - {"type", "compact_threshold"})
        if unsupported:
            raise OpenAIError(f"context_management field '{unsupported[0]}' is not supported.", status_code=400, param=f"context_management.{index}.{unsupported[0]}", code="unsupported_parameter")


def validate_compaction_item(item: dict[str, Any], *, param: str) -> None:
    encrypted_content = item.get("encrypted_content")
    if not isinstance(encrypted_content, str) or not encrypted_content:
        raise OpenAIError("compaction input items require encrypted_content.", param=f"{param}.encrypted_content")
    unsupported = sorted(set(item) - {"id", "type", "encrypted_content", "status"})
    if unsupported:
        raise OpenAIError(f"Compaction item field '{unsupported[0]}' is not supported.", status_code=400, param=f"{param}.{unsupported[0]}", code="unsupported_parameter")


def compaction_item_to_message(item: dict[str, Any], *, key: str | None) -> dict[str, str] | None:
    if not key:
        return None
    encrypted_content = item.get("encrypted_content")
    if not isinstance(encrypted_content, str):
        return None
    try:
        payload = unseal_compaction_content(encrypted_content, key=key)
    except ValueError:
        return None
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return None
    text = summary.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    return {"role": "system", "content": f"Compacted prior context:\n{text}"}


def summarize_context(*, chain: list[dict[str, Any]], input_value: str | list[dict[str, Any]] | None) -> dict[str, Any]:
    entries: list[str] = []
    source_item_ids: list[str] = []
    for response in chain:
        for item in response.get("input_items") or []:
            _append_summary_entry(entries, source_item_ids, item, prefix="input")
        for item in response.get("output_json") or []:
            _append_summary_entry(entries, source_item_ids, item, prefix="output")
    for item in _input_value_to_items(input_value):
        _append_summary_entry(entries, source_item_ids, item, prefix="input")

    text = "\n".join(entries)
    if len(text) > 4000:
        text = text[:3997].rstrip() + "..."
    return {
        "text": text or "No prior textual context.",
        "source_item_ids": source_item_ids,
    }


def context_window_for_model(model: str, settings: Settings) -> int:
    configured = _parse_model_context_windows(settings.model_context_windows)
    model_key = model.lower()
    if model_key in configured:
        return configured[model_key]
    if ":" in model_key:
        base = model_key.split(":", 1)[0]
        if base in configured:
            return configured[base]
    return max(1, settings.context_window_default_tokens)


def response_item_count(response: dict[str, Any]) -> int:
    return len(response.get("input_items") or []) + len(response.get("output_json") or [])


def generate_compaction_item_id(summary_text: str) -> str:
    return generate_id("cmp")


def seal_compaction_content(summary: dict[str, Any], *, key: str, item_id: str, source_item_ids: list[str]) -> str:
    key_bytes = _key_bytes(key)
    nonce = secrets.token_bytes(16)
    plaintext = json.dumps(
        {
            "summary": summary,
            "item_id": item_id,
            "source_item_ids": source_item_ids,
        },
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    ciphertext = _xor(plaintext, _keystream(key_bytes, nonce, len(plaintext)))
    envelope = {
        "v": COMPACTION_ENVELOPE_VERSION,
        "alg": COMPACTION_ENVELOPE_ALGORITHM,
        "kid": hashlib.sha256(key_bytes).hexdigest()[:16],
        "nonce": _b64url(nonce),
        "ciphertext": _b64url(ciphertext),
    }
    envelope["tag"] = _b64url(hmac.new(key_bytes, _tag_payload(envelope), hashlib.sha256).digest())
    return _b64url(json.dumps(envelope, separators=(",", ":")).encode("utf-8"))


def unseal_compaction_content(blob: str, *, key: str) -> dict[str, Any]:
    key_bytes = _key_bytes(key)
    try:
        envelope = json.loads(_b64url_decode(blob))
        tag = _b64url_decode(str(envelope.pop("tag")))
        expected = hmac.new(key_bytes, _tag_payload(envelope), hashlib.sha256).digest()
        if not hmac.compare_digest(tag, expected):
            raise ValueError("compaction encrypted_content tag mismatch")
        nonce = _b64url_decode(str(envelope["nonce"]))
        ciphertext = _b64url_decode(str(envelope["ciphertext"]))
        plaintext = _xor(ciphertext, _keystream(key_bytes, nonce, len(ciphertext)))
        return json.loads(plaintext)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("invalid compaction encrypted_content envelope") from exc


def raise_context_overflow(*, before: int, limit: int, truncation: str) -> None:
    raise OpenAIError(
        f"Input context contains {before} estimated tokens, exceeding the local limit of {limit} tokens for truncation={truncation}.",
        status_code=400,
        param="input",
        code="context_length_exceeded",
    )


def _append_summary_entry(entries: list[str], source_item_ids: list[str], item: dict[str, Any], *, prefix: str) -> None:
    item_type = item.get("type")
    if item_type == "compaction":
        return
    role = item.get("role") or ("assistant" if prefix == "output" else "user")
    text = ""
    if item_type == "reasoning":
        text = content_to_text(item.get("summary", ""))
    elif item_type == "function_call":
        text = f"Function call {item.get('name')} with arguments {item.get('arguments', '{}')}"
    elif item_type == "function_call_output":
        text = f"Function result for {item.get('call_id')}: {item.get('output', '')}"
    else:
        text = content_to_text(item.get("content", item.get("output", "")))
    text = " ".join(text.split())
    if not text:
        return
    entries.append(f"{role}: {text}")
    if item.get("id"):
        source_item_ids.append(str(item["id"]))


def _input_value_to_items(input_value: str | list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if input_value is None:
        return []
    if isinstance(input_value, str):
        return [{"id": "input_text", "type": "message", "role": "user", "content": input_value}]
    return [item for item in input_value if isinstance(item, dict)]


def _input_item_count(input_value: str | list[dict[str, Any]] | None) -> int:
    if isinstance(input_value, list):
        return len(input_value)
    return 1 if input_value else 0


def _retained_tail(input_value: str | list[dict[str, Any]] | None) -> str | list[dict[str, Any]] | None:
    if not isinstance(input_value, list):
        return input_value
    if len(input_value) <= 1:
        return input_value
    return input_value[-1:]


def _items_for_compact_output(retained: str | list[dict[str, Any]] | None, compaction_item: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    if isinstance(retained, str) and retained:
        output.append(
            {
                "id": "msg_compact_retained",
                "type": "message",
                "status": "completed",
                "role": "user",
                "content": [{"type": "input_text", "text": retained}],
            }
        )
    elif isinstance(retained, list):
        for index, item in enumerate(retained):
            if not isinstance(item, dict):
                continue
            normalized = dict(item)
            normalized.setdefault("id", f"msg_compact_retained_{index}")
            normalized.setdefault("status", "completed")
            output.append(normalized)
    output.append(compaction_item)
    return output


def _parse_model_context_windows(raw: str) -> dict[str, int]:
    windows: dict[str, int] = {}
    for entry in raw.split(";"):
        entry = entry.strip()
        if not entry or "=" not in entry:
            continue
        model, value = entry.split("=", 1)
        try:
            windows[model.strip().lower()] = max(1, int(value.strip()))
        except ValueError:
            continue
    return windows


def _key_bytes(key: str) -> bytes:
    if key.startswith("base64:"):
        return base64.b64decode(key.removeprefix("base64:"))
    return key.encode("utf-8")


def _tag_payload(envelope: dict[str, Any]) -> bytes:
    return json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _keystream(key: bytes, nonce: bytes, size: int) -> bytes:
    stream = bytearray()
    counter = 0
    while len(stream) < size:
        stream.extend(hmac.new(key, nonce + counter.to_bytes(8, "big"), hashlib.sha256).digest())
        counter += 1
    return bytes(stream[:size])


def _xor(left: bytes, right: bytes) -> bytes:
    return bytes(left_byte ^ right_byte for left_byte, right_byte in zip(left, right, strict=True))


def _b64url(payload: bytes) -> str:
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _b64url_decode(payload: str) -> bytes:
    padding = "=" * (-len(payload) % 4)
    return base64.urlsafe_b64decode(payload + padding)
