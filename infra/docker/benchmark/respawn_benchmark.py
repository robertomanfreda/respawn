#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BASE_URL = os.getenv("RESPAWN_BASE_URL", "http://respawn:8080").rstrip("/")
API_KEY = os.getenv("RESPAWN_API_KEY", "local-dev-key")
MODEL = os.getenv("RESPAWN_BENCHMARK_MODEL", "gpt-oss:120b")
RUNS = int(os.getenv("RESPAWN_BENCHMARK_RUNS", "3"))
TIMEOUT_SECONDS = float(os.getenv("RESPAWN_BENCHMARK_TIMEOUT_SECONDS", "180"))
MAX_OUTPUT_TOKENS = int(os.getenv("RESPAWN_BENCHMARK_MAX_OUTPUT_TOKENS", "64"))
EXPECT_OLLAMA_METRICS = os.getenv("RESPAWN_BENCHMARK_EXPECT_OLLAMA_METRICS", "true").lower() in {"1", "true", "yes", "on"}
OUTPUT_PATH = os.getenv("RESPAWN_BENCHMARK_OUTPUT", "")


@dataclass
class CaseResult:
    name: str
    ok: bool
    latency_ms: float
    message: str = ""


@dataclass
class BenchmarkState:
    cases: list[CaseResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stored_response_id: str | None = None


def main() -> int:
    state = BenchmarkState()
    print(f"Respawn benchmark target={BASE_URL} model={MODEL} runs={RUNS}")

    wait_for_ready()

    run_case(state, "healthz", case_healthz)
    run_case(state, "readyz", case_readyz)
    run_case(state, "models", case_models)
    run_case(state, "responses.blocking", lambda: case_responses_blocking(state))
    run_case(state, "responses.retrieve", lambda: case_responses_retrieve(state))
    run_case(state, "responses.input_items", lambda: case_responses_input_items(state))
    run_case(state, "responses.input_tokens", case_responses_input_tokens)
    run_case(state, "responses.prompt_cache", case_responses_prompt_cache)
    run_case(state, "responses.reasoning", case_responses_reasoning)
    run_case(state, "responses.previous_response_id", lambda: case_responses_previous_response(state))
    run_case(state, "responses.store_false", case_responses_store_false)
    run_case(state, "responses.structured_output", case_responses_structured_output)
    run_case(state, "responses.function_tool", case_responses_function_tool)
    run_case(state, "responses.stream", case_responses_stream)
    run_case(state, "responses.unsupported_field", case_responses_unsupported_field)
    run_case(state, "responses.unsupported_multimodal", case_responses_unsupported_multimodal)
    run_case(state, "chat.completions", case_chat_completions)
    run_case(state, "chat.completions.stream", case_chat_completions_stream)
    run_case(state, "metrics", lambda: case_metrics(state))
    run_case(state, "responses.delete", lambda: case_responses_delete(state))

    response_samples = run_latency_samples(
        "latency.responses.blocking",
        lambda: post_json(
            "/v1/responses",
            {
                "model": MODEL,
                "input": "Benchmark latency sample. Reply with one short sentence.",
                "max_output_tokens": min(MAX_OUTPUT_TOKENS, 32),
                "store": False,
            },
        ),
    )
    chat_samples = run_latency_samples(
        "latency.chat.completions",
        lambda: post_json(
            "/v1/chat/completions",
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": "Benchmark latency sample. Reply with one short sentence."}],
                "max_tokens": min(MAX_OUTPUT_TOKENS, 32),
            },
        ),
    )

    failed = [case for case in state.cases if not case.ok]
    print_summary(state, response_samples, chat_samples)
    write_report(state, response_samples, chat_samples)
    return 1 if failed else 0


def run_case(state: BenchmarkState, name: str, fn) -> None:
    started = time.perf_counter()
    try:
        message = fn() or ""
    except Exception as exc:
        latency_ms = elapsed_ms(started)
        state.cases.append(CaseResult(name=name, ok=False, latency_ms=latency_ms, message=str(exc)))
        print(f"FAIL {name:<34} {latency_ms:9.1f} ms  {exc}")
        return

    latency_ms = elapsed_ms(started)
    state.cases.append(CaseResult(name=name, ok=True, latency_ms=latency_ms, message=message))
    suffix = f"  {message}" if message else ""
    print(f"OK   {name:<34} {latency_ms:9.1f} ms{suffix}")


def wait_for_ready() -> None:
    deadline = time.monotonic() + TIMEOUT_SECONDS
    last_error = ""
    while time.monotonic() < deadline:
        try:
            body, _, _ = request_json("GET", "/readyz")
            if body.get("status") == "ready":
                return
            last_error = json.dumps(body)
        except Exception as exc:
            last_error = str(exc)
        time.sleep(2)
    raise RuntimeError(f"Respawn did not become ready within {TIMEOUT_SECONDS:.0f}s: {last_error}")


def case_healthz() -> str:
    body, _, _ = request_json("GET", "/healthz")
    expect(body.get("status") == "ok", f"unexpected healthz body: {body}")
    return body["status"]


def case_readyz() -> str:
    body, _, _ = request_json("GET", "/readyz")
    expect(body.get("status") == "ready", f"unexpected readyz body: {body}")
    return body["status"]


def case_models() -> str:
    body, _, _ = request_json("GET", "/v1/models")
    models = body.get("data") or []
    expect(isinstance(models, list), f"models response has no data list: {body}")
    expect(any(model.get("id") == MODEL for model in models if isinstance(model, dict)), f"{MODEL} not found in /v1/models")
    return f"{len(models)} model(s)"


def case_responses_blocking(state: BenchmarkState) -> str:
    body, _, _ = post_json(
        "/v1/responses",
        {
            "model": MODEL,
            "input": "Respawn benchmark blocking response. Reply with one short sentence.",
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "store": True,
            "metadata": {"benchmark": "true"},
        },
    )
    expect_response_completed(body)
    state.stored_response_id = body["id"]
    return body["id"]


def case_responses_retrieve(state: BenchmarkState) -> str:
    expect(state.stored_response_id, "no stored response id from previous case")
    body, _, _ = request_json("GET", f"/v1/responses/{state.stored_response_id}")
    expect(body.get("id") == state.stored_response_id, f"retrieved wrong response: {body}")
    return body["status"]


def case_responses_input_items(state: BenchmarkState) -> str:
    expect(state.stored_response_id, "no stored response id from previous case")
    body, _, _ = request_json("GET", f"/v1/responses/{state.stored_response_id}/input_items?order=asc")
    expect(body.get("object") == "list", f"unexpected input item list: {body}")
    expect(body.get("data"), f"input item list is empty: {body}")
    first = body["data"][0]
    expect(first.get("type") == "message", f"unexpected first input item: {first}")
    expect(first.get("content"), f"input item has no content: {first}")
    return f"{len(body['data'])} item(s)"


def case_responses_input_tokens() -> str:
    body, _, _ = post_json(
        "/v1/responses/input_tokens",
        {
            "model": MODEL,
            "input": "Count this benchmark input.",
            "tools": [],
        },
    )
    expect(body.get("object") == "response.input_tokens", f"unexpected input token object: {body}")
    expect(body.get("input_tokens", 0) > 0, f"input token count missing: {body}")
    return f"{body['input_tokens']} token(s)"


def case_responses_prompt_cache() -> str:
    prefix = " ".join(f"cache-token-{index}" for index in range(1100))
    payload = {
        "model": MODEL,
        "input": f"{prefix} first request",
        "prompt_cache_key": "respawn-benchmark",
        "prompt_cache_retention": "in_memory",
        "max_output_tokens": 8,
        "store": False,
    }
    first, _, _ = post_json("/v1/responses", payload)
    second, _, _ = post_json("/v1/responses", {**payload, "input": f"{prefix} second request"})
    expect_response_completed(first)
    expect_response_completed(second)
    first_cached = first["usage"]["input_tokens_details"].get("cached_tokens", 0)
    second_cached = second["usage"]["input_tokens_details"].get("cached_tokens", 0)
    expect(first_cached == 0, f"first prompt cache request should be cold: {first['usage']}")
    expect(second_cached > 0, f"second prompt cache request should report cached tokens: {second['usage']}")
    return f"{second_cached} cached token(s)"


def case_responses_reasoning() -> str:
    body, _, _ = post_json(
        "/v1/responses",
        {
            "model": MODEL,
            "input": "Use low reasoning and answer with exactly one short sentence.",
            "reasoning": {"effort": "low", "summary": "auto"},
            "max_output_tokens": min(MAX_OUTPUT_TOKENS, 32),
            "store": False,
        },
    )
    expect_response_completed(body)
    output = body.get("output") or []
    expect(output and output[0].get("type") == "reasoning", f"reasoning item missing: {output}")
    usage = body.get("usage") or {}
    reasoning_tokens = usage.get("output_tokens_details", {}).get("reasoning_tokens", 0)
    expect(reasoning_tokens >= 0, f"reasoning token details missing: {usage}")
    return f"{reasoning_tokens} reasoning token(s)"


def case_responses_previous_response(state: BenchmarkState) -> str:
    expect(state.stored_response_id, "no previous response id from previous case")
    body, _, _ = post_json(
        "/v1/responses",
        {
            "model": MODEL,
            "previous_response_id": state.stored_response_id,
            "input": "Continue the benchmark conversation in one short sentence.",
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "store": False,
        },
    )
    expect_response_completed(body)
    return body["status"]


def case_responses_store_false() -> str:
    body, _, _ = post_json(
        "/v1/responses",
        {
            "model": MODEL,
            "input": "Ephemeral benchmark response.",
            "max_output_tokens": min(MAX_OUTPUT_TOKENS, 16),
            "store": False,
        },
    )
    expect_response_completed(body)
    _, status, _ = request_raw("GET", f"/v1/responses/{body['id']}")
    expect(status == 404, f"store=false response should not be retrievable, got HTTP {status}")
    return "not retrievable"


def case_responses_structured_output() -> str:
    body, _, _ = post_json(
        "/v1/responses",
        {
            "model": MODEL,
            "input": "Return a JSON object with status set to ok.",
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "store": False,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "benchmark_status",
                    "schema": {
                        "type": "object",
                        "properties": {"status": {"type": "string"}},
                        "required": ["status"],
                        "additionalProperties": False,
                    },
                }
            },
        },
    )
    expect_response_completed(body)
    text = output_text(body)
    parsed = json.loads(text)
    expect(parsed.get("status"), f"structured response missing status: {text}")
    return text


def case_responses_function_tool() -> str:
    body, _, _ = post_json(
        "/v1/responses",
        {
            "model": MODEL,
            "instructions": "You must call the calculator tool when arithmetic is requested.",
            "input": "Compute 2+2 using the calculator tool.",
            "temperature": 0,
            "max_output_tokens": max(MAX_OUTPUT_TOKENS, 128),
            "store": False,
            "tools": [
                {
                    "type": "function",
                    "name": "calculator",
                    "description": "Evaluate a simple arithmetic expression.",
                    "parameters": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                        "required": ["expression"],
                    },
                }
            ],
        },
    )
    expect_response_completed(body)
    item_types = [item.get("type") for item in body.get("output", [])]
    expect("function_call" in item_types, f"tool call item missing in output: {body.get('output')}")
    text = output_text(body)
    expect("4" in text, f"calculator result not found in output text: {text!r}")
    return text


def case_responses_stream() -> str:
    text, status, _ = request_raw(
        "POST",
        "/v1/responses",
        {
            "model": MODEL,
            "input": "Respawn benchmark streaming response. Reply briefly.",
            "stream": True,
            "max_output_tokens": min(MAX_OUTPUT_TOKENS, 32),
            "store": False,
        },
    )
    expect(status == 200, f"stream response returned HTTP {status}: {text[:200]}")
    expect("event: response.created" in text, "missing response.created SSE event")
    expect("event: response.completed" in text, "missing response.completed SSE event")
    return "SSE completed"


def case_responses_unsupported_field() -> str:
    body, status, _ = request_json_error("POST", "/v1/responses", {"model": MODEL, "input": "hello", "background": True})
    expect(status == 400, f"unsupported background returned HTTP {status}: {body}")
    error = body.get("error") or {}
    expect(error.get("code") == "unsupported_parameter", f"unexpected unsupported error: {body}")
    expect(error.get("param") == "background", f"unexpected unsupported param: {body}")
    return error["param"]


def case_responses_unsupported_multimodal() -> str:
    body, status, _ = request_json_error(
        "POST",
        "/v1/responses",
        {
            "model": MODEL,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "describe"},
                        {"type": "input_image", "image_url": "https://example.com/image.png"},
                    ],
                }
            ],
        },
    )
    expect(status == 400, f"unsupported multimodal returned HTTP {status}: {body}")
    error = body.get("error") or {}
    expect(error.get("code") == "unsupported_parameter", f"unexpected multimodal error: {body}")
    return error.get("param", "")


def case_chat_completions() -> str:
    body, _, _ = post_json(
        "/v1/chat/completions",
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Respawn benchmark chat completion. Reply briefly."}],
            "max_tokens": min(MAX_OUTPUT_TOKENS, 32),
        },
    )
    choices = body.get("choices") or []
    expect(body.get("object") == "chat.completion", f"unexpected chat object: {body}")
    expect(choices, f"chat completion has no choices: {body}")
    return body.get("id", "chatcmpl")


def case_chat_completions_stream() -> str:
    text, status, _ = request_raw(
        "POST",
        "/v1/chat/completions",
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Respawn benchmark chat stream. Reply briefly."}],
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": min(MAX_OUTPUT_TOKENS, 32),
        },
    )
    expect(status == 200, f"chat stream returned HTTP {status}: {text[:200]}")
    expect("data: [DONE]" in text, "missing chat stream DONE marker")
    expect('"usage"' in text, "chat stream did not include usage")
    return "SSE completed"


def case_metrics(state: BenchmarkState) -> str:
    text, status, _ = request_raw("GET", "/metrics")
    expect(status == 200, f"/metrics returned HTTP {status}")
    required = [
        "gateway_requests_total",
        "gateway_responses_total",
        "gateway_model_token_usage_total",
        "gateway_backend_requests_total",
    ]
    for metric in required:
        expect(metric in text, f"missing metric {metric}")
    if EXPECT_OLLAMA_METRICS and "gateway_ollama_eval_tokens_per_second" not in text:
        raise AssertionError("missing Ollama native throughput metrics")
    if not EXPECT_OLLAMA_METRICS and "gateway_ollama_eval_tokens_per_second" not in text:
        state.warnings.append("Ollama native throughput metrics were not present.")
    return "metrics present"


def case_responses_delete(state: BenchmarkState) -> str:
    expect(state.stored_response_id, "no stored response id from previous case")
    body, _, _ = request_json("DELETE", f"/v1/responses/{state.stored_response_id}")
    expect(body.get("deleted") is True, f"delete response did not return deleted=true: {body}")
    _, status, _ = request_raw("GET", f"/v1/responses/{state.stored_response_id}")
    expect(status == 404, f"deleted response should return 404, got HTTP {status}")
    return state.stored_response_id


def run_latency_samples(name: str, fn) -> list[float]:
    samples = []
    for index in range(max(RUNS, 0)):
        started = time.perf_counter()
        body, _, _ = fn()
        expect(body, f"{name} sample {index + 1} returned empty body")
        samples.append(elapsed_ms(started))
    if samples:
        summary = summarize(samples)
        print(
            f"STAT {name:<34} p50={summary['p50_ms']:8.1f} ms "
            f"p95={summary['p95_ms']:8.1f} ms min={summary['min_ms']:8.1f} ms max={summary['max_ms']:8.1f} ms"
        )
    return samples


def post_json(path: str, payload: dict[str, Any]) -> tuple[dict[str, Any], int, float]:
    return request_json("POST", path, payload)


def request_json(method: str, path: str, payload: dict[str, Any] | None = None) -> tuple[dict[str, Any], int, float]:
    text, status, latency_ms = request_raw(method, path, payload)
    expect(200 <= status < 300, f"{method} {path} returned HTTP {status}: {text[:500]}")
    try:
        return json.loads(text), status, latency_ms
    except json.JSONDecodeError as exc:
        raise AssertionError(f"{method} {path} returned invalid JSON: {text[:500]}") from exc


def request_json_error(method: str, path: str, payload: dict[str, Any] | None = None) -> tuple[dict[str, Any], int, float]:
    text, status, latency_ms = request_raw(method, path, payload)
    try:
        return json.loads(text), status, latency_ms
    except json.JSONDecodeError as exc:
        raise AssertionError(f"{method} {path} returned invalid JSON error body: {text[:500]}") from exc


def request_raw(method: str, path: str, payload: dict[str, Any] | None = None) -> tuple[str, int, float]:
    url = f"{BASE_URL}{path}"
    data = None
    headers = {}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    if payload is not None:
        data = json.dumps(payload, separators=(",", ":")).encode()
        headers["Content-Type"] = "application/json"

    request = Request(url, data=data, headers=headers, method=method)
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            body = response.read().decode()
            return body, response.status, elapsed_ms(started)
    except HTTPError as exc:
        return exc.read().decode(), exc.code, elapsed_ms(started)
    except URLError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc}") from exc


def expect_response_completed(body: dict[str, Any]) -> None:
    expect(body.get("object") == "response", f"unexpected response object: {body}")
    expect(body.get("status") == "completed", f"response did not complete: {body}")
    expect(body.get("id", "").startswith("resp_"), f"unexpected response id: {body.get('id')}")
    expect(isinstance(body.get("output"), list), f"response output is not a list: {body}")
    expect("output_text" in body, f"response missing output_text: {body}")
    usage = body.get("usage") or {}
    expect(isinstance(usage.get("total_tokens", 0), int), f"usage total_tokens is missing: {usage}")
    expect("input_tokens_details" in usage, f"usage missing input token details: {usage}")
    expect("output_tokens_details" in usage, f"usage missing output token details: {usage}")


def output_text(body: dict[str, Any]) -> str:
    parts = []
    for item in body.get("output") or []:
        for content in item.get("content") or []:
            if content.get("type") in {"output_text", "text"}:
                parts.append(str(content.get("text", "")))
    return "".join(parts)


def summarize(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {"count": 0, "min_ms": 0.0, "max_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0}
    ordered = sorted(samples)
    p95_index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1))
    return {
        "count": len(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "p50_ms": statistics.median(samples),
        "p95_ms": ordered[p95_index],
        "mean_ms": statistics.mean(samples),
    }


def print_summary(state: BenchmarkState, response_samples: list[float], chat_samples: list[float]) -> None:
    passed = sum(1 for case in state.cases if case.ok)
    failed = len(state.cases) - passed
    print()
    print(f"Feature cases: passed={passed} failed={failed}")
    for warning in state.warnings:
        print(f"WARN {warning}")
    if failed:
        for case in state.cases:
            if not case.ok:
                print(f"FAILURE {case.name}: {case.message}")

    if response_samples:
        print(f"Responses latency: {json.dumps(summarize(response_samples), sort_keys=True)}")
    if chat_samples:
        print(f"Chat latency:      {json.dumps(summarize(chat_samples), sort_keys=True)}")


def write_report(state: BenchmarkState, response_samples: list[float], chat_samples: list[float]) -> None:
    if not OUTPUT_PATH:
        return
    report = {
        "target": BASE_URL,
        "model": MODEL,
        "runs": RUNS,
        "cases": [case.__dict__ for case in state.cases],
        "warnings": state.warnings,
        "latency": {
            "responses_blocking": summarize(response_samples),
            "chat_completions": summarize(chat_samples),
        },
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Report written to {OUTPUT_PATH}")


def expect(condition: Any, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


if __name__ == "__main__":
    sys.exit(main())
