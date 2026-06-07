#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _parse_csv_set(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


BASE_URL = os.getenv("RESPAWN_BASE_URL", "http://respawn:8080").rstrip("/")
API_KEY = os.getenv("RESPAWN_API_KEY", "local-dev-key")
SECONDARY_API_KEY = os.getenv("RESPAWN_SECONDARY_API_KEY", "respawn-other-key")
MODEL = os.getenv("RESPAWN_BENCHMARK_MODEL", "gpt-oss:120b")
TEXT_MODEL = os.getenv("RESPAWN_BENCHMARK_TEXT_MODEL", MODEL)
VISION_MODEL = os.getenv("RESPAWN_BENCHMARK_VISION_MODEL", "moondream:latest")
ASSET_BASE_URL = os.getenv("RESPAWN_BENCHMARK_ASSET_BASE_URL", "http://benchmark-assets:8000").rstrip("/")
MODEL_BACKEND = os.getenv("RESPAWN_BENCHMARK_MODEL_BACKEND", "ollama")
OLLAMA_BASE_URL = os.getenv("RESPAWN_BENCHMARK_OLLAMA_BASE_URL", "")
DATABASE_DRIVER = os.getenv("RESPAWN_BENCHMARK_DATABASE_DRIVER", "")
GIT_SHA = os.getenv("RESPAWN_BENCHMARK_GIT_SHA", "")
RUNS = int(os.getenv("RESPAWN_BENCHMARK_RUNS", "3"))
TIMEOUT_SECONDS = float(os.getenv("RESPAWN_BENCHMARK_TIMEOUT_SECONDS", "180"))
MAX_OUTPUT_TOKENS = int(os.getenv("RESPAWN_BENCHMARK_MAX_OUTPUT_TOKENS", "64"))
COMPLETION_OUTPUT_TOKENS = int(os.getenv("RESPAWN_BENCHMARK_COMPLETION_OUTPUT_TOKENS", str(max(MAX_OUTPUT_TOKENS, 128))))
EXPECT_OLLAMA_METRICS = os.getenv("RESPAWN_BENCHMARK_EXPECT_OLLAMA_METRICS", "true").lower() in {"1", "true", "yes", "on"}
OUTPUT_PATH = os.getenv("RESPAWN_BENCHMARK_OUTPUT", "")
INCLUDE_TAGS = _parse_csv_set(os.getenv("RESPAWN_BENCHMARK_INCLUDE_TAGS", ""))
EXCLUDE_TAGS = _parse_csv_set(os.getenv("RESPAWN_BENCHMARK_EXCLUDE_TAGS", ""))
COVERAGE_GATE = os.getenv("RESPAWN_BENCHMARK_COVERAGE_GATE", "true").lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    tags: set[str]
    feature_ids: set[str]
    fn: Callable[["BenchmarkState"], str]


@dataclass
class CaseResult:
    name: str
    ok: bool
    latency_ms: float
    message: str = ""
    status: str = "passed"
    tags: list[str] = field(default_factory=list)
    feature_ids: list[str] = field(default_factory=list)
    skip_reason: str | None = None


@dataclass
class BenchmarkState:
    cases: list[CaseResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stored_response_id: str | None = None
    item_response_id: str | None = None
    item_ids: list[str] = field(default_factory=list)
    compatibility_manifest: dict[str, Any] | None = None
    compatibility_coverage: dict[str, Any] | None = None
    server_health: dict[str, Any] | None = None
    server_ready: dict[str, Any] | None = None
    background_completed_response_id: str | None = None
    background_cancelled_response_id: str | None = None
    function_call_response_id: str | None = None
    function_call_item: dict[str, Any] | None = None
    function_followup_response_id: str | None = None


def main() -> int:
    state = BenchmarkState()
    cases = benchmark_cases()
    print(f"Respawn benchmark target={BASE_URL} model={MODEL} backend={MODEL_BACKEND} runs={RUNS}")
    if INCLUDE_TAGS:
        print(f"Tag include filter: {','.join(sorted(INCLUDE_TAGS))}")
    if EXCLUDE_TAGS:
        print(f"Tag exclude filter: {','.join(sorted(EXCLUDE_TAGS))}")

    wait_for_ready()

    for case in cases:
        run_case(state, case)

    response_samples = run_latency_samples(
        "latency.responses.blocking",
        {"core"},
        lambda: post_json(
            "/v1/responses",
            {
                "model": MODEL,
                "input": "Benchmark latency sample. Reply with one short sentence.",
                "max_output_tokens": completion_output_tokens(),
                "store": False,
            },
        ),
    )
    chat_samples = run_latency_samples(
        "latency.chat.completions",
        {"core"},
        lambda: post_json(
            "/v1/chat/completions",
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": "Benchmark latency sample. Reply with one short sentence."}],
                "max_tokens": min(MAX_OUTPUT_TOKENS, 32),
            },
        ),
    )

    failed = [case for case in state.cases if case.status == "failed"]
    print_summary(state, response_samples, chat_samples)
    write_report(state, response_samples, chat_samples)
    return 1 if failed else 0


def benchmark_cases() -> list[BenchmarkCase]:
    return [
        BenchmarkCase("healthz", {"core"}, set(), lambda state: case_healthz(state)),
        BenchmarkCase("readyz", {"core"}, set(), lambda state: case_readyz(state)),
        BenchmarkCase("compatibility.manifest", {"core", "observability"}, set(), case_compatibility_manifest),
        BenchmarkCase("compatibility.coverage", {"core", "observability"}, set(), lambda state: case_compatibility_coverage(state, benchmark_cases())),
        BenchmarkCase("models", {"core"}, {"endpoints.models.list"}, lambda state: case_models()),
        BenchmarkCase("responses.blocking", {"core", "state"}, {"endpoints.responses.create", "request.model_and_text_input", "request.sampling_and_limits", "request.truncation_disabled", "response.core_shape"}, case_responses_blocking),
        BenchmarkCase("responses.shape.blocking_text", {"core", "state"}, {"request.text_format", "response.request_settings", "response.output_content_shape"}, lambda state: case_responses_shape_blocking_text()),
        BenchmarkCase("responses.shape.metadata_retrieve", {"core", "state"}, {"request.metadata", "request.service_tier", "request.safety_identifier"}, lambda state: case_responses_shape_metadata_retrieve()),
        BenchmarkCase("responses.shape.max_output_incomplete", {"core"}, {"response.incomplete_status"}, lambda state: case_responses_shape_max_output_incomplete()),
        BenchmarkCase("responses.shape.unsupported_future_field", {"core", "state"}, {"request.future_unsupported_fields"}, lambda state: case_responses_shape_unsupported_future_field()),
        BenchmarkCase("responses.tools.function_call", {"tools"}, {"request.function_tools"}, case_responses_tools_function_call),
        BenchmarkCase("responses.tools.client_output_followup", {"tools", "state"}, {"io.function_call_items"}, case_responses_tools_client_output_followup),
        BenchmarkCase("responses.tools.previous_response_replay", {"tools", "state"}, {"state.function_tool_previous_response_replay"}, case_responses_tools_previous_response_replay),
        BenchmarkCase("responses.tools.retrieve_function_call", {"tools", "state"}, {"state.function_tool_item_storage"}, case_responses_tools_retrieve_function_call),
        BenchmarkCase("responses.tools.input_items_function_output", {"tools", "state"}, {"state.function_tool_input_item_listing"}, case_responses_tools_input_items_function_output),
        BenchmarkCase("responses.tools.stream_arguments", {"tools", "streaming"}, {"streaming.function_call_arguments"}, case_responses_tools_stream_arguments),
        BenchmarkCase("responses.tools.tool_choice_forced_function", {"tools"}, {"request.tool_choice"}, case_responses_tools_tool_choice_forced_function),
        BenchmarkCase("responses.tools.parallel_or_capability_error", {"tools"}, {"request.parallel_and_max_tool_calls"}, case_responses_tools_parallel_or_capability_error),
        BenchmarkCase("responses.tools.unsupported_builtin_tools", {"tools"}, {"request.unsupported_tool_categories", "io.legacy_tool_result_unsupported"}, lambda state: case_responses_tools_unsupported_builtin_tools()),
        BenchmarkCase("responses.tools.no_internal_execution", {"tools"}, set(), case_responses_tools_no_internal_execution),
        BenchmarkCase("responses.retrieve", {"core", "state"}, {"endpoints.responses.retrieve"}, case_responses_retrieve),
        BenchmarkCase("responses.input_items", {"state"}, {"endpoints.responses.input_items"}, case_responses_input_items),
        BenchmarkCase("responses.items.input_storage", {"state"}, {"state.input_item_storage"}, case_responses_items_input_storage),
        BenchmarkCase("responses.items.pagination_after", {"state"}, {"state.input_item_pagination"}, case_responses_items_pagination_after),
        BenchmarkCase("responses.items.store_false_hidden", {"state"}, {"state.input_items_store_false_hidden"}, lambda state: case_responses_items_store_false_hidden()),
        BenchmarkCase("responses.items.tenant_scope", {"state"}, {"state.input_items_tenant_scope"}, case_responses_items_tenant_scope),
        BenchmarkCase("responses.input_tokens", {"core"}, {"endpoints.responses.input_tokens"}, lambda state: case_responses_input_tokens()),
        BenchmarkCase("responses.prompt_cache", {"core"}, {"request.prompt_cache", "response.cached_tokens"}, lambda state: case_responses_prompt_cache()),
        BenchmarkCase("responses.reasoning", {"reasoning"}, {"request.reasoning", "io.reasoning_items", "response.reasoning_tokens"}, lambda state: case_responses_reasoning()),
        BenchmarkCase("responses.previous_response_id", {"state"}, {"request.previous_response_id"}, case_responses_previous_response),
        BenchmarkCase("responses.store_false", {"state"}, {"request.store"}, lambda state: case_responses_store_false()),
        BenchmarkCase("responses.background.create_poll_complete", {"background", "state"}, {"request.background", "state.background_polling"}, case_responses_background_create_poll_complete),
        BenchmarkCase("responses.background.cancel", {"background", "state"}, {"endpoints.responses.cancel", "state.background_cancellation"}, case_responses_background_cancel),
        BenchmarkCase("responses.background.retrieve_terminal", {"background", "state"}, {"state.background_terminal_retrieve", "response.background_shape"}, case_responses_background_retrieve_terminal),
        BenchmarkCase("responses.background.store_false_invalid", {"background", "state"}, {"request.background_store_requirement"}, lambda state: case_responses_background_store_false_invalid()),
        BenchmarkCase("responses.background.timeout", {"background", "state"}, {"state.background_timeout"}, lambda state: case_responses_background_timeout()),
        BenchmarkCase("responses.structured_output", {"core"}, {"request.instructions", "request.structured_output"}, lambda state: case_responses_structured_output()),
        BenchmarkCase("responses.input_message_list", {"core"}, {"io.text_messages"}, lambda state: case_responses_input_message_list()),
        BenchmarkCase("responses.stream.lifecycle_text", {"streaming"}, {"request.stream", "request.stream_options", "streaming.text_lifecycle", "streaming.event_sequence_and_sse_id"}, lambda state: case_responses_stream_lifecycle_text()),
        BenchmarkCase("responses.stream.reasoning", {"streaming", "reasoning"}, {"streaming.reasoning_summary"}, lambda state: case_responses_stream_reasoning()),
        BenchmarkCase("responses.stream.failure", {"streaming"}, {"streaming.failure"}, lambda state: case_responses_stream_failure()),
        BenchmarkCase("responses.stream.incomplete", {"streaming"}, {"streaming.incomplete"}, lambda state: case_responses_stream_incomplete()),
        BenchmarkCase("responses.stream.sdk_parse", {"streaming"}, {"streaming.sdk_parse"}, lambda state: case_responses_stream_sdk_parse()),
        BenchmarkCase("responses.unsupported_field", {"state"}, {"request.unsupported_fields"}, lambda state: case_responses_unsupported_field()),
        BenchmarkCase("responses.multimodal.input_file_text", {"multimodal"}, {"io.input_file"}, lambda state: case_responses_multimodal_input_file_text()),
        BenchmarkCase("responses.multimodal.input_file_csv", {"multimodal"}, set(), lambda state: case_responses_multimodal_input_file_csv()),
        BenchmarkCase("responses.multimodal.input_file_pdf", {"multimodal"}, set(), lambda state: case_responses_multimodal_input_file_pdf()),
        BenchmarkCase("responses.multimodal.input_image_vision", {"multimodal"}, {"io.input_image"}, lambda state: case_responses_multimodal_input_image_vision()),
        BenchmarkCase("responses.multimodal.input_image_unsupported_model", {"multimodal"}, {"io.input_image_capability_errors"}, lambda state: case_responses_multimodal_input_image_unsupported_model()),
        BenchmarkCase("responses.multimodal.input_audio_unsupported", {"multimodal"}, {"io.input_audio_unsupported"}, lambda state: case_responses_multimodal_input_audio_unsupported()),
        BenchmarkCase("responses.multimodal.file_limits", {"multimodal"}, {"io.input_file_limits"}, lambda state: case_responses_multimodal_file_limits()),
        BenchmarkCase("chat.completions", {"core"}, {"endpoints.chat_completions.create"}, lambda state: case_chat_completions()),
        BenchmarkCase("chat.completions.stream", {"streaming"}, set(), lambda state: case_chat_completions_stream()),
        BenchmarkCase("metrics", {"observability"}, {"observability.metrics"}, case_metrics),
        BenchmarkCase("metrics.background_jobs", {"background", "observability"}, {"observability.background_metrics"}, case_background_metrics),
        BenchmarkCase("metrics.function_tools", {"tools", "observability"}, {"observability.function_tool_metrics"}, case_function_tool_metrics),
        BenchmarkCase("responses.delete", {"core", "state"}, {"endpoints.responses.delete"}, case_responses_delete),
    ]


def run_case(state: BenchmarkState, case: BenchmarkCase) -> None:
    skip_reason = selected_case_skip_reason(case, INCLUDE_TAGS, EXCLUDE_TAGS)
    if skip_reason:
        state.cases.append(
            CaseResult(
                name=case.name,
                ok=True,
                latency_ms=0.0,
                status="skipped",
                tags=sorted(case.tags),
                feature_ids=sorted(case.feature_ids),
                skip_reason=skip_reason,
            )
        )
        print(f"SKIP {case.name:<34} {skip_reason}")
        return

    started = time.perf_counter()
    try:
        message = case.fn(state) or ""
    except Exception as exc:
        latency_ms = elapsed_ms(started)
        state.cases.append(
            CaseResult(
                name=case.name,
                ok=False,
                latency_ms=latency_ms,
                message=str(exc),
                status="failed",
                tags=sorted(case.tags),
                feature_ids=sorted(case.feature_ids),
            )
        )
        print(f"FAIL {case.name:<34} {latency_ms:9.1f} ms  {exc}")
        return

    latency_ms = elapsed_ms(started)
    state.cases.append(
        CaseResult(
            name=case.name,
            ok=True,
            latency_ms=latency_ms,
            message=message,
            status="passed",
            tags=sorted(case.tags),
            feature_ids=sorted(case.feature_ids),
        )
    )
    suffix = f"  {message}" if message else ""
    print(f"OK   {case.name:<34} {latency_ms:9.1f} ms{suffix}")


def selected_case_skip_reason(case: BenchmarkCase, include_tags: set[str], exclude_tags: set[str]) -> str | None:
    return selected_tags_skip_reason(case.tags, include_tags, exclude_tags)


def selected_tags_skip_reason(tags: set[str], include_tags: set[str], exclude_tags: set[str]) -> str | None:
    if include_tags and not tags.intersection(include_tags):
        return f"no tags in include filter {','.join(sorted(include_tags))}"
    excluded = tags.intersection(exclude_tags)
    if excluded:
        return f"excluded tag(s): {','.join(sorted(excluded))}"
    return None


def parse_sse_events(text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for block in text.strip().split("\n\n"):
        if not block:
            continue
        parsed: dict[str, Any] = {"id": None, "event": None, "data": None}
        for line in block.splitlines():
            if line.startswith("id: "):
                parsed["id"] = line.removeprefix("id: ").strip()
            elif line.startswith("event: "):
                parsed["event"] = line.removeprefix("event: ").strip()
            elif line.startswith("data: "):
                try:
                    parsed["data"] = json.loads(line.removeprefix("data: ").strip())
                except json.JSONDecodeError as exc:
                    raise AssertionError(f"malformed SSE JSON: {line}") from exc
        expect(parsed["event"], f"SSE block missing event line: {block}")
        expect(isinstance(parsed["data"], dict), f"SSE block missing JSON data: {block}")
        events.append(parsed)
    expect(events, "SSE stream emitted no events")
    return events


def manifest_coverage(manifest: dict[str, Any], cases: list[BenchmarkCase]) -> dict[str, Any]:
    registered_cases = {case.name for case in cases}
    feature_ids_by_case: dict[str, set[str]] = {}
    for case in cases:
        feature_ids_by_case.setdefault(case.name, set()).update(case.feature_ids)

    supported_features = [
        feature
        for feature in manifest.get("features", [])
        if str(feature.get("status", "")).startswith("supported") and feature.get("benchmark_required", True)
    ]
    covered: list[str] = []
    missing: list[dict[str, Any]] = []
    for feature in supported_features:
        feature_id = str(feature.get("id", ""))
        benchmark_case = feature.get("benchmark_case")
        if benchmark_case in registered_cases and feature_id in feature_ids_by_case.get(str(benchmark_case), set()):
            covered.append(feature_id)
            continue
        missing.append({"id": feature_id, "benchmark_case": benchmark_case})

    return {
        "covered_supported_features": sorted(covered),
        "missing_supported_features": missing,
        "registered_cases": sorted(registered_cases),
    }


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


def case_healthz(state: BenchmarkState) -> str:
    body, _, _ = request_json("GET", "/healthz")
    expect(body.get("status") == "ok", f"unexpected healthz body: {body}")
    state.server_health = body
    return body["status"]


def case_readyz(state: BenchmarkState) -> str:
    body, _, _ = request_json("GET", "/readyz")
    expect(body.get("status") == "ready", f"unexpected readyz body: {body}")
    state.server_ready = body
    return body["status"]


def case_compatibility_manifest(state: BenchmarkState) -> str:
    body, _, _ = request_json("GET", "/compatibility/responses")
    expect(body.get("object") == "respawn.responses_compatibility_manifest", f"unexpected compatibility manifest: {body}")
    features = body.get("features") or []
    expect(isinstance(features, list) and features, f"manifest has no features: {body}")
    for feature in features:
        expect(feature.get("id"), f"manifest feature missing id: {feature}")
        expect(feature.get("status"), f"manifest feature missing status: {feature}")
    state.compatibility_manifest = body
    summary = body.get("summary") or {}
    return f"{summary.get('supported', 0)} supported / {summary.get('unsupported', 0)} unsupported"


def case_compatibility_coverage(state: BenchmarkState, cases: list[BenchmarkCase]) -> str:
    expect(state.compatibility_manifest, "compatibility manifest case must run before coverage")
    coverage = manifest_coverage(state.compatibility_manifest, cases)
    state.compatibility_coverage = coverage
    missing = coverage["missing_supported_features"]
    if not COVERAGE_GATE:
        if missing:
            state.warnings.append(f"Compatibility coverage gate disabled with {len(missing)} missing supported feature(s).")
        return f"coverage gate disabled; missing={len(missing)}"
    expect(not missing, f"supported manifest features without benchmark coverage: {missing}")
    return f"{len(coverage['covered_supported_features'])} supported feature(s) covered"


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
            "temperature": 0,
            "top_p": 1,
            "max_output_tokens": completion_output_tokens(),
            "truncation": "disabled",
            "store": True,
            "metadata": {"benchmark": "true"},
        },
    )
    expect_response_final(body)
    state.stored_response_id = body["id"]
    return body["id"]



def case_responses_shape_blocking_text() -> str:
    body, _, _ = post_json(
        "/v1/responses",
        {
            "model": MODEL,
            "input": "Respawn benchmark Phase 1 text shape. Reply with one short sentence.",
            "text": {"format": {"type": "text"}},
            "temperature": 0,
            "top_p": 1,
            "max_output_tokens": completion_output_tokens(),
            "store": False,
        },
    )
    expect_response_completed(body)
    expect(body.get("text") == {"format": {"type": "text"}}, f"text format did not round-trip: {body.get('text')}")
    expect(body.get("parallel_tool_calls") is False, f"parallel_tool_calls should be false: {body}")
    expect(body.get("temperature") == 0, f"temperature did not round-trip: {body.get('temperature')}")
    expect(body.get("top_p") == 1, f"top_p did not round-trip: {body.get('top_p')}")
    expect(body.get("store") is False, f"store did not round-trip: {body.get('store')}")
    content = first_output_text_part(body)
    expect(isinstance(content.get("annotations"), list), f"output text part missing annotations: {content}")
    expect(isinstance(content.get("logprobs"), list), f"output text part missing logprobs: {content}")
    return body["status"]


def case_responses_shape_metadata_retrieve() -> str:
    payload = {
        "model": MODEL,
        "input": "Respawn benchmark Phase 1 metadata retrieve. Reply briefly.",
        "metadata": {"phase": "one", "kind": "retrieve"},
        "temperature": 0,
        "top_p": 1,
        "max_output_tokens": completion_output_tokens(),
        "service_tier": "default",
        "safety_identifier": "respawn-benchmark",
        "text": {"format": {"type": "text"}},
        "store": True,
    }
    created, _, _ = post_json("/v1/responses", payload)
    expect_response_final(created)
    retrieved, _, _ = request_json("GET", f"/v1/responses/{created['id']}")
    for body in (created, retrieved):
        expect(body.get("metadata") == payload["metadata"], f"metadata did not round-trip: {body.get('metadata')}")
        expect(body.get("service_tier") == "default", f"service_tier did not round-trip: {body.get('service_tier')}")
        expect(body.get("safety_identifier") == "respawn-benchmark", f"safety_identifier did not round-trip: {body.get('safety_identifier')}")
        expect(body.get("text") == {"format": {"type": "text"}}, f"text did not round-trip: {body.get('text')}")
        expect(body.get("temperature") == 0, f"temperature did not round-trip: {body.get('temperature')}")
        expect(body.get("top_p") == 1, f"top_p did not round-trip: {body.get('top_p')}")
    return retrieved["id"]


def case_responses_shape_max_output_incomplete() -> str:
    body, _, _ = post_json(
        "/v1/responses",
        {
            "model": MODEL,
            "input": "Respawn benchmark Phase 1 max output behavior. Produce several words.",
            "max_output_tokens": 1,
            "store": False,
        },
    )
    expect(body.get("object") == "response", f"unexpected response object: {body}")
    expect(body.get("status") in {"completed", "incomplete"}, f"unexpected status for max output behavior: {body}")
    if body.get("status") == "incomplete":
        details = body.get("incomplete_details") or {}
        expect(details.get("reason"), f"incomplete response missing reason: {body}")
    else:
        usage = body.get("usage") or {}
        expect(usage.get("output_tokens", 0) <= 1 or len(output_text(body).split()) <= 1, f"bounded completed response exceeded max output: {body}")
    return body["status"]


def case_responses_shape_unsupported_future_field() -> str:
    body, status, _ = request_json_error("POST", "/v1/responses", {"model": MODEL, "input": "hello", "truncation": "auto"})
    expect(status == 400, f"unsupported future field returned HTTP {status}: {body}")
    error = body.get("error") or {}
    expect(error.get("code") == "unsupported_parameter", f"unexpected unsupported future error: {body}")
    expect(error.get("param") == "truncation", f"unexpected unsupported future param: {body}")
    return error["param"]


def calculator_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "name": "calculator",
        "description": "Evaluate a small arithmetic expression.",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
    }


def ensure_function_call_response(state: BenchmarkState) -> tuple[str, dict[str, Any]]:
    if state.function_call_response_id and state.function_call_item:
        return state.function_call_response_id, state.function_call_item
    body, _, _ = post_json(
        "/v1/responses",
        {
            "model": MODEL,
            "input": "Use the calculator function to evaluate 2+2. Do not answer in text; call the function.",
            "tools": [calculator_tool()],
            "tool_choice": "required",
            "max_output_tokens": completion_output_tokens(64),
            "store": True,
        },
    )
    item = expect_function_call_item(body, name="calculator")
    state.function_call_response_id = body["id"]
    state.function_call_item = item
    return body["id"], item


def case_responses_tools_function_call(state: BenchmarkState) -> str:
    _, item = ensure_function_call_response(state)
    return item["call_id"]


def case_responses_tools_client_output_followup(state: BenchmarkState) -> str:
    _, call = ensure_function_call_response(state)
    body, _, _ = post_json(
        "/v1/responses",
        {
            "model": MODEL,
            "input": [
                call,
                {"type": "function_call_output", "call_id": call["call_id"], "output": "{\"result\":4}"},
            ],
            "tools": [calculator_tool()],
            "max_output_tokens": completion_output_tokens(64),
            "store": True,
        },
    )
    expect_response_final(body)
    expect((body.get("output_text") or "").strip(), f"function output follow-up did not produce text: {body}")
    state.function_followup_response_id = body["id"]
    return body["status"]


def case_responses_tools_previous_response_replay(state: BenchmarkState) -> str:
    response_id, call = ensure_function_call_response(state)
    body, _, _ = post_json(
        "/v1/responses",
        {
            "model": MODEL,
            "previous_response_id": response_id,
            "input": [{"type": "function_call_output", "call_id": call["call_id"], "output": "{\"result\":4}"}],
            "tools": [calculator_tool()],
            "max_output_tokens": completion_output_tokens(64),
            "store": True,
        },
    )
    expect_response_final(body)
    expect((body.get("output_text") or "").strip(), f"previous_response_id function follow-up did not produce text: {body}")
    state.function_followup_response_id = body["id"]
    return body["status"]


def case_responses_tools_retrieve_function_call(state: BenchmarkState) -> str:
    response_id, call = ensure_function_call_response(state)
    retrieved, _, _ = request_json("GET", f"/v1/responses/{response_id}")
    retrieved_call = expect_function_call_item(retrieved, name=call["name"])
    expect(retrieved_call["call_id"] == call["call_id"], f"retrieved function call changed call_id: {retrieved_call} vs {call}")
    expect(retrieved_call["arguments"] == call["arguments"], f"retrieved function call changed arguments: {retrieved_call} vs {call}")
    return retrieved_call["call_id"]


def case_responses_tools_input_items_function_output(state: BenchmarkState) -> str:
    if not state.function_followup_response_id:
        case_responses_tools_previous_response_replay(state)
    body, _, _ = request_json("GET", f"/v1/responses/{state.function_followup_response_id}/input_items?order=asc")
    items = body.get("data") or []
    output_items = [item for item in items if item.get("type") == "function_call_output"]
    expect(output_items, f"input_items did not include function_call_output: {body}")
    expect(output_items[0].get("call_id"), f"function_call_output missing call_id: {output_items[0]}")
    return output_items[0]["call_id"]


def case_responses_tools_tool_choice_forced_function(state: BenchmarkState) -> str:
    body, status, _ = request_json_error(
        "POST",
        "/v1/responses",
        {
            "model": MODEL,
            "input": "Call the calculator function for 2+2.",
            "tools": [calculator_tool()],
            "tool_choice": {"type": "function", "name": "calculator"},
            "max_output_tokens": completion_output_tokens(64),
            "store": False,
        },
    )
    if 200 <= status < 300:
        item = expect_function_call_item(body, name="calculator")
        return item["call_id"]
    error = body.get("error") or {}
    expect(error.get("code") == "unsupported_model_capability", f"unexpected forced tool_choice error: {body}")
    return error["code"]


def case_responses_tools_parallel_or_capability_error(state: BenchmarkState) -> str:
    body, status, _ = request_json_error(
        "POST",
        "/v1/responses",
        {
            "model": MODEL,
            "input": "Use the calculator function once for 2+2.",
            "tools": [calculator_tool()],
            "tool_choice": "required",
            "parallel_tool_calls": False,
            "max_tool_calls": 1,
            "max_output_tokens": completion_output_tokens(64),
            "store": False,
        },
    )
    if 200 <= status < 300:
        calls = [item for item in body.get("output") or [] if item.get("type") == "function_call"]
        expect(len(calls) == 1, f"expected exactly one function_call with parallel_tool_calls=false: {body}")
        return calls[0]["call_id"]
    error = body.get("error") or {}
    expect(error.get("code") == "unsupported_model_capability", f"unexpected parallel/max tool error: {body}")
    return error["code"]


def case_responses_tools_no_internal_execution(state: BenchmarkState) -> str:
    body, status, _ = request_json_error(
        "POST",
        "/v1/responses",
        {
            "model": MODEL,
            "input": "Loop forever by calling echo again and again.",
            "tools": [
                {
                    "type": "function",
                    "name": "echo",
                    "description": "Echo text.",
                    "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
                }
            ],
            "tool_choice": "required",
            "max_output_tokens": completion_output_tokens(64),
            "store": False,
        },
    )
    if status == 400:
        error = body.get("error") or {}
        expect(error.get("code") == "unsupported_model_capability", f"unexpected no-internal-execution capability error: {body}")
        return error["code"]
    expect(200 <= status < 300, f"no-internal-execution request returned HTTP {status}: {body}")
    item = expect_function_call_item(body, name="echo")
    expect(body.get("output_text") == "", f"Respawn should not execute function tools internally: {body}")
    return item["name"]


def case_responses_tools_unsupported_builtin_tools() -> str:
    cases = [
        ({"model": MODEL, "input": "search", "tools": [{"type": "web_search"}]}, "tools.0.type"),
        ({"model": MODEL, "input": [{"type": "tool_result", "call_id": "call_1", "output": "4"}]}, "input.0.type"),
    ]
    params = []
    for payload, expected_param in cases:
        body, status, _ = request_json_error("POST", "/v1/responses", payload)
        expect(status == 400, f"unsupported tool category returned HTTP {status}: {body}")
        error = body.get("error") or {}
        expect(error.get("code") == "unsupported_parameter", f"unexpected unsupported tool category error: {body}")
        expect(error.get("param") == expected_param, f"unexpected unsupported tool category param: {body}")
        params.append(expected_param)
    return ",".join(params)


def case_responses_tools_stream_arguments(state: BenchmarkState) -> str:
    text, status, _ = request_raw(
        "POST",
        "/v1/responses",
        {
            "model": MODEL,
            "input": "Use the calculator function to evaluate 2+2. Do not answer in text; call the function.",
            "tools": [calculator_tool()],
            "tool_choice": "required",
            "stream": True,
            "max_output_tokens": completion_output_tokens(64),
            "store": True,
        },
    )
    expect(status == 200, f"function-call stream returned HTTP {status}: {text[:500]}")
    events = parse_sse_events(text)
    event_types = [event["event"] for event in events]
    expect("response.output_item.added" in event_types, f"missing function_call item added event: {event_types}")
    expect("response.function_call_arguments.delta" in event_types, f"missing function_call arguments delta event: {event_types}")
    expect("response.function_call_arguments.done" in event_types, f"missing function_call arguments done event: {event_types}")
    expect("response.output_item.done" in event_types, f"missing function_call item done event: {event_types}")
    expect(event_types[-1] in {"response.completed", "response.incomplete"}, f"missing terminal function-call SSE event: {event_types}")

    added = next(
        event
        for event in events
        if event["event"] == "response.output_item.added"
        and (event["data"].get("item") or {}).get("type") == "function_call"
    )
    done = next(event for event in events if event["event"] == "response.function_call_arguments.done")
    call_done = next(
        event
        for event in events
        if event["event"] == "response.output_item.done"
        and (event["data"].get("item") or {}).get("type") == "function_call"
    )
    deltas = [
        str(event["data"].get("delta", ""))
        for event in events
        if event["event"] == "response.function_call_arguments.delta" and event["data"].get("item_id") == done["data"].get("item_id")
    ]
    arguments = str(done["data"].get("arguments") or "")
    expect("".join(deltas) == arguments, f"function_call argument deltas did not reconstruct done arguments: {events}")
    json.loads(arguments or "{}")
    item = call_done["data"].get("item") or {}
    expect(item.get("id") == added["data"]["item"]["id"], f"function_call stream item id changed: {added} vs {call_done}")
    expect(item.get("arguments") == arguments, f"function_call done item arguments mismatch: {item} vs {arguments}")

    terminal_response = events[-1]["data"].get("response") or {}
    retrieved, _, _ = request_json("GET", f"/v1/responses/{terminal_response['id']}")
    retrieved_call = expect_function_call_item(retrieved, name=item.get("name") or "calculator")
    expect(retrieved_call["call_id"] == item["call_id"], f"stream retrieve changed call_id: {retrieved_call} vs {item}")
    expect(retrieved_call["arguments"] == arguments, f"stream retrieve changed arguments: {retrieved_call} vs {arguments}")
    return item["call_id"]


def ensure_stored_response(state: BenchmarkState) -> str:
    if not state.stored_response_id:
        case_responses_blocking(state)
    expect(state.stored_response_id, "no stored response id available")
    return state.stored_response_id


def case_responses_retrieve(state: BenchmarkState) -> str:
    response_id = ensure_stored_response(state)
    body, _, _ = request_json("GET", f"/v1/responses/{response_id}")
    expect(body.get("id") == response_id, f"retrieved wrong response: {body}")
    return body["status"]


def case_responses_input_items(state: BenchmarkState) -> str:
    response_id = ensure_stored_response(state)
    body, _, _ = request_json("GET", f"/v1/responses/{response_id}/input_items?order=asc")
    expect(body.get("object") == "list", f"unexpected input item list: {body}")
    expect(body.get("data"), f"input item list is empty: {body}")
    first = body["data"][0]
    expect(first.get("type") == "message", f"unexpected first input item: {first}")
    expect(first.get("content"), f"input item has no content: {first}")
    return f"{len(body['data'])} item(s)"


def ensure_item_response(state: BenchmarkState) -> tuple[str, list[dict[str, Any]]]:
    if state.item_response_id and state.item_ids:
        body, _, _ = request_json("GET", f"/v1/responses/{state.item_response_id}/input_items?order=asc&limit=10")
        return state.item_response_id, body.get("data") or []

    response, _, _ = post_json(
        "/v1/responses",
        {
            "model": MODEL,
            "input": [
                {"role": "user", "content": "First canonical input item."},
                {"role": "user", "content": "Second canonical input item."},
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "Stored local reasoning marker."}]},
            ],
            "max_output_tokens": completion_output_tokens(),
            "store": True,
        },
    )
    expect_response_final(response)
    body, _, _ = request_json("GET", f"/v1/responses/{response['id']}/input_items?order=asc&limit=10")
    items = body.get("data") or []
    expect(len(items) == 3, f"expected three canonical input items: {body}")
    state.item_response_id = response["id"]
    state.item_ids = [item["id"] for item in items]
    return response["id"], items


def case_responses_items_input_storage(state: BenchmarkState) -> str:
    response_id, first_read = ensure_item_response(state)
    second, _, _ = request_json("GET", f"/v1/responses/{response_id}/input_items?order=asc&limit=10")
    second_read = second.get("data") or []
    expect([item["id"] for item in first_read] == [item["id"] for item in second_read], "input item ids changed between reads")
    expect(first_read[0].get("content") == [{"type": "input_text", "text": "First canonical input item."}], f"first item content mismatch: {first_read[0]}")
    expect(first_read[1].get("content") == [{"type": "input_text", "text": "Second canonical input item."}], f"second item content mismatch: {first_read[1]}")
    expect(first_read[2].get("type") == "reasoning", f"reasoning item missing: {first_read}")
    return f"{len(first_read)} stored input item(s)"


def case_responses_items_pagination_after(state: BenchmarkState) -> str:
    response_id, items = ensure_item_response(state)
    first_id = items[0]["id"]
    second_id = items[1]["id"]
    after, _, _ = request_json("GET", f"/v1/responses/{response_id}/input_items?order=asc&after={first_id}&limit=1")
    expect((after.get("data") or [])[0]["id"] == second_id, f"after pagination mismatch: {after}")
    before, _, _ = request_json("GET", f"/v1/responses/{response_id}/input_items?order=asc&before={second_id}&limit=10")
    expect([item["id"] for item in before.get("data") or []] == [first_id], f"before pagination mismatch: {before}")
    return "after/before pagination ok"


def case_responses_items_store_false_hidden() -> str:
    body, _, _ = post_json(
        "/v1/responses",
        {
            "model": MODEL,
            "input": "Hidden benchmark input item.",
            "max_output_tokens": completion_output_tokens(),
            "store": False,
        },
    )
    expect_response_final(body)
    _, status, _ = request_raw("GET", f"/v1/responses/{body['id']}/input_items")
    expect(status == 404, f"store=false input_items should return 404, got HTTP {status}")
    return "hidden"


def case_responses_items_tenant_scope(state: BenchmarkState) -> str:
    response_id, _ = ensure_item_response(state)
    _, primary_status, _ = request_raw("GET", f"/v1/responses/{response_id}/input_items", api_key=API_KEY)
    expect(primary_status == 200, f"primary tenant should read input_items, got HTTP {primary_status}")
    body, other_status, _ = request_json_error("GET", f"/v1/responses/{response_id}/input_items", api_key=SECONDARY_API_KEY)
    expect(other_status == 404, f"other tenant should not read input_items, got HTTP {other_status}: {body}")
    error = body.get("error") or {}
    expect(error.get("code") == "not_found", f"unexpected tenant isolation error: {body}")
    return "tenant isolated"


def case_responses_input_tokens() -> str:
    body, _, _ = post_json(
        "/v1/responses/input_tokens",
        {
            "model": MODEL,
            "input": "Count this benchmark input.",
        },
    )
    expect(body.get("object") == "response.input_tokens", f"unexpected input token object: {body}")
    expect(body.get("input_tokens", 0) > 0, f"input token count missing: {body}")
    return f"{body['input_tokens']} token(s)"


def case_responses_prompt_cache() -> str:
    prefix = " ".join(f"cache-token-{index}" for index in range(1100))
    cache_key = f"respawn-benchmark-{os.getpid()}-{time.monotonic_ns()}"
    payload = {
        "model": MODEL,
        "input": f"{prefix} first request",
        "prompt_cache_key": cache_key,
        "prompt_cache_retention": "in_memory",
        "max_output_tokens": completion_output_tokens(64),
        "store": False,
    }
    first, _, _ = post_json("/v1/responses", payload)
    second, _, _ = post_json("/v1/responses", {**payload, "input": f"{prefix} second request"})
    expect_response_final(first)
    expect_response_final(second)
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
            "max_output_tokens": completion_output_tokens(128),
            "store": False,
        },
    )
    expect_response_final(body)
    output = body.get("output") or []
    expect(output and output[0].get("type") == "reasoning", f"reasoning item missing: {output}")
    usage = body.get("usage") or {}
    reasoning_tokens = usage.get("output_tokens_details", {}).get("reasoning_tokens", 0)
    expect(reasoning_tokens >= 0, f"reasoning token details missing: {usage}")
    return f"{reasoning_tokens} reasoning token(s)"


def case_responses_previous_response(state: BenchmarkState) -> str:
    response_id = ensure_stored_response(state)
    body, _, _ = post_json(
        "/v1/responses",
        {
            "model": MODEL,
            "previous_response_id": response_id,
            "input": "Continue the benchmark response chain in one short sentence.",
            "max_output_tokens": completion_output_tokens(),
            "store": False,
        },
    )
    expect_response_final(body)
    return body["status"]


def case_responses_store_false() -> str:
    body, _, _ = post_json(
        "/v1/responses",
        {
            "model": MODEL,
            "input": "Ephemeral benchmark response.",
            "max_output_tokens": completion_output_tokens(64),
            "store": False,
        },
    )
    expect_response_final(body)
    _, status, _ = request_raw("GET", f"/v1/responses/{body['id']}")
    expect(status == 404, f"store=false response should not be retrievable, got HTTP {status}")
    return "not retrievable"


def case_responses_background_create_poll_complete(state: BenchmarkState) -> str:
    created, _, _ = post_json(
        "/v1/responses",
        {
            "model": MODEL,
            "input": "Respawn benchmark background response. Reply with one short sentence.",
            "background": True,
            "max_output_tokens": completion_output_tokens(64),
            "store": True,
        },
    )
    expect_response_object(created)
    expect(created.get("background") is True, f"background flag did not round-trip: {created}")
    expect(created.get("status") in {"queued", "in_progress", "completed", "incomplete"}, f"unexpected initial background status: {created}")
    if created.get("status") in {"queued", "in_progress"}:
        expect(created.get("output") == [], f"nonterminal background response should not expose final output yet: {created}")
    terminal = poll_response_terminal(created["id"], expected={"completed", "incomplete"})
    expect_response_final(terminal)
    expect(terminal.get("background") is True, f"retrieved background flag missing: {terminal}")
    state.background_completed_response_id = created["id"]
    return terminal["status"]


def case_responses_background_cancel(state: BenchmarkState) -> str:
    created, _, _ = post_json(
        "/v1/responses",
        {
            "model": MODEL,
            "input": "Respawn benchmark background slow cancellation. Produce a careful but concise answer.",
            "background": True,
            "max_output_tokens": completion_output_tokens(128),
            "store": True,
        },
    )
    expect_response_object(created)
    cancelled, _, _ = post_json(f"/v1/responses/{created['id']}/cancel", {})
    expect_response_object(cancelled)
    expect(cancelled.get("status") in {"cancelled", "completed", "incomplete"}, f"unexpected cancel result: {cancelled}")
    if cancelled.get("status") == "cancelled":
        time.sleep(0.2)
        retrieved, _, _ = request_json("GET", f"/v1/responses/{created['id']}")
        expect(retrieved.get("status") == "cancelled", f"cancelled background response changed status: {retrieved}")
    state.background_cancelled_response_id = created["id"]
    return cancelled["status"]


def case_responses_background_retrieve_terminal(state: BenchmarkState) -> str:
    if not state.background_completed_response_id:
        case_responses_background_create_poll_complete(state)
    if not state.background_cancelled_response_id:
        case_responses_background_cancel(state)
    completed, _, _ = request_json("GET", f"/v1/responses/{state.background_completed_response_id}")
    cancelled, _, _ = request_json("GET", f"/v1/responses/{state.background_cancelled_response_id}")
    expect(completed.get("status") in {"completed", "incomplete"}, f"completed background retrieve not terminal: {completed}")
    expect(cancelled.get("status") in {"cancelled", "completed", "incomplete"}, f"cancelled background retrieve not terminal: {cancelled}")
    expect(completed.get("background") is True, f"completed background flag missing: {completed}")
    expect(cancelled.get("background") is True, f"cancelled background flag missing: {cancelled}")
    return f"{completed['status']}/{cancelled['status']}"


def case_responses_background_store_false_invalid() -> str:
    body, status, _ = request_json_error(
        "POST",
        "/v1/responses",
        {"model": MODEL, "input": "hello", "background": True, "store": False},
    )
    expect(status == 400, f"background store=false returned HTTP {status}: {body}")
    error = body.get("error") or {}
    expect(error.get("code") == "invalid_request", f"unexpected background store=false error: {body}")
    expect(error.get("param") == "store", f"unexpected background store=false param: {body}")
    return error["param"]


def case_responses_background_timeout() -> str:
    if MODEL_BACKEND == "mock":
        created, _, _ = post_json(
            "/v1/responses",
            {
                "model": MODEL,
                "input": "background timeout",
                "background": True,
                "max_output_tokens": completion_output_tokens(64),
                "store": True,
            },
        )
        failed = poll_response_terminal(created["id"], expected={"failed"})
        error = failed.get("error") or {}
        expect(error.get("code") == "background_timeout", f"unexpected timeout error: {failed}")
        return "failed/background_timeout"

    created, _, _ = post_json(
        "/v1/responses",
        {
            "model": MODEL,
            "input": "Respawn benchmark background timeout guard smoke. Reply briefly.",
            "background": True,
            "max_output_tokens": min(MAX_OUTPUT_TOKENS, 16),
            "store": True,
        },
    )
    terminal = poll_response_terminal(created["id"], expected={"completed", "incomplete", "failed"})
    expect(terminal.get("background") is True, f"background timeout guard response missing background flag: {terminal}")
    return f"guard path {terminal['status']}"


def case_responses_structured_output() -> str:
    body, _, _ = post_json(
        "/v1/responses",
        {
            "model": MODEL,
            "instructions": "Return only compact JSON. Do not include markdown, prose, or code fences.",
            "input": "Return exactly this JSON object: {\"status\":\"ok\"}",
            "temperature": 0,
            "max_output_tokens": max(MAX_OUTPUT_TOKENS, 128),
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


def case_responses_input_message_list() -> str:
    body, _, _ = post_json(
        "/v1/responses",
        {
            "model": MODEL,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Respawn benchmark input_text list. Reply with ok."}],
                }
            ],
            "max_output_tokens": completion_output_tokens(),
            "store": False,
        },
    )
    expect_response_completed(body)
    return body["status"]


def case_responses_stream_lifecycle_text() -> str:
    text, status, _ = request_raw(
        "POST",
        "/v1/responses",
        {
            "model": MODEL,
            "input": "Respawn benchmark streaming response. Reply briefly.",
            "stream": True,
            "max_output_tokens": completion_output_tokens(),
            "store": True,
        },
    )
    expect(status == 200, f"stream response returned HTTP {status}: {text[:200]}")
    events = parse_sse_events(text)
    event_types = [event["event"] for event in events]
    expect(event_types[:2] == ["response.created", "response.in_progress"], f"unexpected lifecycle start: {event_types}")
    for required in [
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
    ]:
        expect(required in event_types, f"missing {required} SSE event: {event_types}")
    expect(event_types[-1] in {"response.completed", "response.incomplete"}, f"missing terminal SSE event: {event_types}")
    expect([event["data"]["sequence_number"] for event in events] == list(range(len(events))), "SSE sequence numbers are not monotonic")
    expect(all(event["id"] for event in events), "SSE id missing from one or more events")
    expect(all(event["data"].get("type") == event["event"] for event in events), "SSE data.type does not match event name")
    delta_events = [event for event in events if event["event"] == "response.output_text.delta"]
    expect(delta_events, "missing output text delta event")
    expect(all(isinstance(event["data"].get("obfuscation"), str) for event in delta_events), "default stream obfuscation field missing")

    no_obfuscation_text, no_obfuscation_status, _ = request_raw(
        "POST",
        "/v1/responses",
        {
            "model": MODEL,
            "input": "Respawn benchmark streaming response without obfuscation. Reply briefly.",
            "stream": True,
            "stream_options": {"include_obfuscation": False},
            "max_output_tokens": completion_output_tokens(),
            "store": False,
        },
    )
    expect(no_obfuscation_status == 200, f"stream_options response returned HTTP {no_obfuscation_status}: {no_obfuscation_text[:200]}")
    no_obfuscation_deltas = [event for event in parse_sse_events(no_obfuscation_text) if event["event"] == "response.output_text.delta"]
    expect(no_obfuscation_deltas, "missing output text delta event with stream_options")
    expect(all("obfuscation" not in event["data"] for event in no_obfuscation_deltas), "include_obfuscation=false did not suppress obfuscation fields")

    terminal_response = events[-1]["data"]["response"]
    retrieved, _, _ = request_json("GET", f"/v1/responses/{terminal_response['id']}")
    expect(retrieved.get("status") == terminal_response.get("status"), f"stream terminal status differs from retrieve: {terminal_response} vs {retrieved}")
    expect(retrieved.get("output_text") == terminal_response.get("output_text"), "stream terminal output_text differs from retrieve")
    return event_types[-1]


def case_responses_stream_reasoning() -> str:
    text, status, _ = request_raw(
        "POST",
        "/v1/responses",
        {
            "model": MODEL,
            "input": "Respawn benchmark reasoning stream. Reply briefly.",
            "stream": True,
            "reasoning": {"effort": "low", "summary": "auto"},
            "max_output_tokens": completion_output_tokens(),
            "store": False,
        },
    )
    expect(status == 200, f"reasoning stream response returned HTTP {status}: {text[:200]}")
    event_types = [event["event"] for event in parse_sse_events(text)]
    expect("response.output_item.added" in event_types, "missing reasoning output item event")
    expect("response.reasoning_summary_part.added" in event_types, "missing reasoning summary part added event")
    expect("response.reasoning_summary_text.delta" in event_types, "missing reasoning summary delta event")
    expect("response.reasoning_summary_text.done" in event_types, "missing reasoning summary text done event")
    expect("response.reasoning_summary_part.done" in event_types, "missing reasoning summary part done event")
    expect(event_types[-1] in {"response.completed", "response.incomplete"}, "missing terminal reasoning SSE event")
    return "reasoning SSE final"


def case_responses_stream_failure() -> str:
    text, status, _ = request_raw(
        "POST",
        "/v1/responses",
        {
            "model": MODEL,
            "input": "Respawn benchmark stream failure.",
            "stream": True,
            "text": {"format": {"type": "json_schema", "name": "impossible_schema", "schema": {"not": {}}}},
            "max_output_tokens": 4,
            "store": False,
        },
    )
    expect(status == 200, f"stream failure case returned HTTP {status}: {text[:200]}")
    event_types = [event["event"] for event in parse_sse_events(text)]
    expect(event_types[-2:] == ["response.failed", "error"], f"missing failure terminal SSE events: {event_types}")
    return "failed SSE emitted"


def case_responses_stream_incomplete() -> str:
    text, status, _ = request_raw(
        "POST",
        "/v1/responses",
        {
            "model": MODEL,
            "input": "Respawn benchmark incomplete stream. Reply with several words.",
            "stream": True,
            "max_output_tokens": 1,
            "store": False,
        },
    )
    expect(status == 200, f"incomplete stream response returned HTTP {status}: {text[:200]}")
    events = parse_sse_events(text)
    expect(events[-1]["event"] == "response.incomplete", f"missing response.incomplete terminal event: {[event['event'] for event in events]}")
    response = events[-1]["data"].get("response") or {}
    expect(response.get("status") == "incomplete", f"incomplete terminal response has wrong status: {response}")
    expect((response.get("incomplete_details") or {}).get("reason") == "max_tokens", f"incomplete details missing: {response}")
    return "response.incomplete"


def case_responses_stream_sdk_parse() -> str:
    text, status, _ = request_raw(
        "POST",
        "/v1/responses",
        {
            "model": MODEL,
            "input": "Respawn benchmark SDK parse stream. Reply briefly.",
            "stream": True,
            "max_output_tokens": completion_output_tokens(),
            "store": False,
        },
    )
    expect(status == 200, f"SDK parse stream response returned HTTP {status}: {text[:200]}")
    events = parse_sse_events(text)
    expect(events, "SSE parser found no events")
    for event in events:
        expect(event["event"] == event["data"].get("type"), f"event name/type mismatch: {event}")
        expect(isinstance(event["data"].get("sequence_number"), int), f"sequence_number missing: {event}")
    expect(events[-1]["event"] in {"response.completed", "response.incomplete"}, f"SDK parse stream has no terminal response event: {events[-1]}")
    return f"{len(events)} event(s)"


def case_responses_unsupported_field() -> str:
    body, status, _ = request_json_error("POST", "/v1/responses", {"model": MODEL, "input": "hello", "include": ["file_search_call.results"]})
    expect(status == 400, f"unsupported include returned HTTP {status}: {body}")
    error = body.get("error") or {}
    expect(error.get("code") == "unsupported_parameter", f"unexpected unsupported error: {body}")
    expect(error.get("param") == "include", f"unexpected unsupported param: {body}")
    return error["param"]


def case_responses_multimodal_input_file_text() -> str:
    body, _, _ = request_json(
        "POST",
        "/v1/responses",
        {
            "model": TEXT_MODEL,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_file", "filename": "facts.txt", "file_url": f"{ASSET_BASE_URL}/facts.txt"},
                        {"type": "input_text", "text": "Answer with only the marker word from the file."},
                    ],
                }
            ],
            "max_output_tokens": completion_output_tokens(512),
            "store": True,
        },
    )
    expect_response_final(body)
    text = output_text(body).lower()
    expect("cobalt" in text, f"text file response did not use extracted file text: {body}")
    items, _, _ = request_json("GET", f"/v1/responses/{body['id']}/input_items?order=asc")
    file_part = items["data"][0]["content"][0]
    expect(file_part.get("type") == "input_file", f"input_file was not stored as an input_file part: {items}")
    expect("cobalt" in (file_part.get("text") or "").lower(), f"stored input_file did not include extracted text: {items}")
    return text


def case_responses_multimodal_input_file_csv() -> str:
    body, _, _ = request_json(
        "POST",
        "/v1/responses",
        {
            "model": TEXT_MODEL,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_file", "filename": "table.csv", "file_url": f"{ASSET_BASE_URL}/table.csv"},
                        {"type": "input_text", "text": "Which item has count 5? Answer with only the item name."},
                    ],
                }
            ],
            "max_output_tokens": completion_output_tokens(128),
            "store": True,
        },
    )
    expect_response_final(body)
    text = output_text(body).lower()
    expect("beta" in text, f"CSV response did not use extracted CSV text: {body}")
    items, _, _ = request_json("GET", f"/v1/responses/{body['id']}/input_items?order=asc")
    file_part = items["data"][0]["content"][0]
    expect(file_part.get("type") == "input_file", f"CSV input_file was not stored: {items}")
    expect("beta,5" in (file_part.get("text") or ""), f"stored CSV input_file did not include extracted text: {items}")
    return text


def case_responses_multimodal_input_file_pdf() -> str:
    body, _, _ = request_json(
        "POST",
        "/v1/responses",
        {
            "model": TEXT_MODEL,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_file", "filename": "tiny.pdf", "file_url": f"{ASSET_BASE_URL}/tiny.pdf"},
                        {"type": "input_text", "text": "What is the PDF marker word? Answer with only that word."},
                    ],
                }
            ],
            "max_output_tokens": completion_output_tokens(512),
            "store": True,
        },
    )
    expect_response_final(body)
    text = output_text(body).lower()
    expect("quartz" in text, f"PDF response did not use extracted PDF text: {body}")
    items, _, _ = request_json("GET", f"/v1/responses/{body['id']}/input_items?order=asc")
    file_part = items["data"][0]["content"][0]
    expect(file_part.get("type") == "input_file", f"PDF input_file was not stored: {items}")
    expect("quartz" in (file_part.get("text") or "").lower(), f"stored PDF input_file did not include extracted text: {items}")
    return text


def case_responses_multimodal_input_image_vision() -> str:
    body, _, _ = request_json(
        "POST",
        "/v1/responses",
        {
            "model": VISION_MODEL,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Describe this image in one short phrase."},
                        {"type": "input_image", "image_url": f"{ASSET_BASE_URL}/tiny-red.png", "detail": "low"},
                    ],
                }
            ],
            "max_output_tokens": completion_output_tokens(64),
            "store": True,
        },
    )
    expect_response_final(body)
    text = output_text(body).lower()
    expect("red" in text, f"image response did not answer the visual question: {body}")
    items, _, _ = request_json("GET", f"/v1/responses/{body['id']}/input_items?order=asc")
    image_part = items["data"][0]["content"][1]
    expect(image_part.get("type") == "input_image", f"input_image was not stored: {items}")
    expect(image_part.get("image_base64"), f"stored input_image did not include normalized base64: {items}")
    return text


def case_responses_multimodal_input_image_unsupported_model() -> str:
    body, status, _ = request_json_error(
        "POST",
        "/v1/responses",
        {
            "model": TEXT_MODEL,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "describe"},
                        {"type": "input_image", "image_url": f"{ASSET_BASE_URL}/tiny-red.png"},
                    ],
                }
            ],
        },
    )
    expect(status == 400, f"text model image request returned HTTP {status}: {body}")
    error = body.get("error") or {}
    expect(error.get("code") == "unsupported_model_capability", f"unexpected image capability error: {body}")
    expect(error.get("param") == "model", f"unexpected image capability param: {body}")
    return error["code"]


def case_responses_multimodal_input_audio_unsupported() -> str:
    body, status, _ = request_json_error(
        "POST",
        "/v1/responses",
        {
            "model": TEXT_MODEL,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "wav"}},
                    ],
                }
            ],
        },
    )
    expect(status == 400, f"audio request returned HTTP {status}: {body}")
    error = body.get("error") or {}
    expect(error.get("code") == "unsupported_parameter", f"unexpected audio error: {body}")
    return error.get("param", "")


def case_responses_multimodal_file_limits() -> str:
    oversized = "x" * 2_000_001
    # Use a base64 payload that decodes above the default 2 MB file limit.
    import base64

    large_data = base64.b64encode(oversized.encode("utf-8")).decode("ascii")
    body, status, _ = request_json_error(
        "POST",
        "/v1/responses",
        {
            "model": TEXT_MODEL,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_file", "filename": "too-large.txt", "file_data": f"data:text/plain;base64,{large_data}"},
                    ],
                }
            ],
        },
    )
    expect(status == 400, f"oversized file returned HTTP {status}: {body}")
    error = body.get("error") or {}
    expect(error.get("code") == "file_too_large", f"unexpected oversized file error: {body}")

    body, status, _ = request_json_error(
        "POST",
        "/v1/responses",
        {
            "model": TEXT_MODEL,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_file", "filename": "missing.txt", "file_url": "http://127.0.0.1:1/missing.txt"},
                    ],
                }
            ],
        },
    )
    expect(status == 400, f"failed file URL returned HTTP {status}: {body}")
    error = body.get("error") or {}
    expect(error.get("code") in {"file_download_failed", "file_download_timeout"}, f"unexpected download error: {body}")
    return error["code"]


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


def case_background_metrics(state: BenchmarkState) -> str:
    if not state.background_completed_response_id:
        case_responses_background_create_poll_complete(state)
    text, status, _ = request_raw("GET", "/metrics")
    expect(status == 200, f"/metrics returned HTTP {status}")
    required = [
        "gateway_background_jobs_total",
        "gateway_background_jobs_running",
        "gateway_background_job_latency_seconds_bucket",
    ]
    for metric in required:
        expect(metric in text, f"missing background metric {metric}")
    return "background metrics present"


def case_function_tool_metrics(state: BenchmarkState) -> str:
    ensure_function_call_response(state)
    if not state.function_followup_response_id:
        case_responses_tools_previous_response_replay(state)
    text, status, _ = request_raw("GET", "/metrics")
    expect(status == 200, f"/metrics returned HTTP {status}")
    required = [
        "gateway_function_tool_requests_total",
        "gateway_function_tool_calls_total",
        "gateway_function_tool_outputs_total",
        "gateway_function_tool_unsupported_total",
        "gateway_function_tool_capability_errors_total",
    ]
    for metric in required:
        expect(metric in text, f"missing function tool metric {metric}")
    return "function tool metrics present"


def case_responses_delete(state: BenchmarkState) -> str:
    response_id = ensure_stored_response(state)
    body, _, _ = request_json("DELETE", f"/v1/responses/{response_id}")
    expect(body.get("deleted") is True, f"delete response did not return deleted=true: {body}")
    _, status, _ = request_raw("GET", f"/v1/responses/{response_id}")
    expect(status == 404, f"deleted response should return 404, got HTTP {status}")
    return response_id


def run_latency_samples(name: str, tags: set[str], fn) -> list[float]:
    skip_reason = selected_tags_skip_reason(tags, INCLUDE_TAGS, EXCLUDE_TAGS)
    if skip_reason:
        print(f"SKIP {name:<34} {skip_reason}")
        return []
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


def completion_output_tokens(minimum: int = 128) -> int:
    return max(COMPLETION_OUTPUT_TOKENS, minimum)


def post_json(path: str, payload: dict[str, Any], *, api_key: str | None = None) -> tuple[dict[str, Any], int, float]:
    return request_json("POST", path, payload, api_key=api_key)


def request_json(method: str, path: str, payload: dict[str, Any] | None = None, *, api_key: str | None = None) -> tuple[dict[str, Any], int, float]:
    text, status, latency_ms = request_raw(method, path, payload, api_key=api_key)
    expect(200 <= status < 300, f"{method} {path} returned HTTP {status}: {text[:500]}")
    try:
        return json.loads(text), status, latency_ms
    except json.JSONDecodeError as exc:
        raise AssertionError(f"{method} {path} returned invalid JSON: {text[:500]}") from exc


def request_json_error(method: str, path: str, payload: dict[str, Any] | None = None, *, api_key: str | None = None) -> tuple[dict[str, Any], int, float]:
    text, status, latency_ms = request_raw(method, path, payload, api_key=api_key)
    try:
        return json.loads(text), status, latency_ms
    except json.JSONDecodeError as exc:
        raise AssertionError(f"{method} {path} returned invalid JSON error body: {text[:500]}") from exc


def request_raw(method: str, path: str, payload: dict[str, Any] | None = None, *, api_key: str | None = None) -> tuple[str, int, float]:
    url = f"{BASE_URL}{path}"
    data = None
    headers = {}
    selected_api_key = API_KEY if api_key is None else api_key
    if selected_api_key:
        headers["Authorization"] = f"Bearer {selected_api_key}"
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
    expect_response_object(body)
    expect(body.get("status") == "completed", f"response did not complete: {body}")


def expect_response_final(body: dict[str, Any]) -> None:
    expect_response_object(body)
    expect(body.get("status") in {"completed", "incomplete"}, f"response was not final: {body}")


def expect_response_object(body: dict[str, Any]) -> None:
    expect(body.get("object") == "response", f"unexpected response object: {body}")
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


def poll_response_terminal(response_id: str, *, expected: set[str], timeout_seconds: float | None = None) -> dict[str, Any]:
    deadline = time.monotonic() + (timeout_seconds or TIMEOUT_SECONDS)
    last_body: dict[str, Any] = {}
    terminal_statuses = {"completed", "failed", "cancelled", "incomplete"}
    while time.monotonic() < deadline:
        body, _, _ = request_json("GET", f"/v1/responses/{response_id}")
        last_body = body
        status = str(body.get("status", ""))
        if status in expected:
            return body
        if status in terminal_statuses:
            raise AssertionError(f"response {response_id} reached unexpected terminal status {status}: {body}")
        time.sleep(1)
    raise AssertionError(f"response {response_id} did not reach {sorted(expected)} within timeout; last body: {last_body}")



def first_output_text_part(body: dict[str, Any]) -> dict[str, Any]:
    for item in body.get("output") or []:
        for content in item.get("content") or []:
            if content.get("type") in {"output_text", "text"}:
                return content
    raise AssertionError(f"response has no output text part: {body}")


def expect_function_call_item(body: dict[str, Any], *, name: str | None = None) -> dict[str, Any]:
    expect_response_object(body)
    calls = [item for item in body.get("output") or [] if item.get("type") == "function_call"]
    expect(calls, f"response output did not include a function_call item: {body}")
    item = next((candidate for candidate in calls if name is None or candidate.get("name") == name), calls[0])
    if name is not None:
        expect(item.get("name") == name, f"function_call used unexpected name: {item}")
    expect(str(item.get("id", "")).startswith("fc_"), f"function_call id should start with fc_: {item}")
    expect(str(item.get("call_id", "")).startswith("call_"), f"function_call call_id should start with call_: {item}")
    expect(item.get("status") in {"completed", "in_progress", "incomplete"}, f"function_call status missing: {item}")
    expect(isinstance(item.get("arguments"), str), f"function_call arguments should be a JSON string: {item}")
    json.loads(item.get("arguments") or "{}")
    return item


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
    passed = sum(1 for case in state.cases if case.status == "passed")
    failed = sum(1 for case in state.cases if case.status == "failed")
    skipped = sum(1 for case in state.cases if case.status == "skipped")
    print()
    print(f"Feature cases: passed={passed} failed={failed} skipped={skipped}")
    for warning in state.warnings:
        print(f"WARN {warning}")
    if failed:
        for case in state.cases:
            if case.status == "failed":
                print(f"FAILURE {case.name}: {case.message}")

    compatibility = compatibility_report(state)
    surfaces = compatibility["surfaces"]
    print(
        "Compatibility surfaces: "
        f"supported={len(surfaces['supported'])} unsupported={len(surfaces['unsupported'])} skipped={len(surfaces['skipped'])}"
    )
    if state.compatibility_coverage:
        missing = state.compatibility_coverage.get("missing_supported_features") or []
        print(f"Compatibility coverage: missing={len(missing)} covered={len(state.compatibility_coverage.get('covered_supported_features') or [])}")

    if response_samples:
        print(f"Responses latency: {json.dumps(summarize(response_samples), sort_keys=True)}")
    if chat_samples:
        print(f"Chat latency:      {json.dumps(summarize(chat_samples), sort_keys=True)}")


def write_report(state: BenchmarkState, response_samples: list[float], chat_samples: list[float]) -> None:
    if not OUTPUT_PATH:
        return
    report = {
        "metadata": benchmark_metadata(state),
        "cases": [case.__dict__ for case in state.cases],
        "warnings": state.warnings,
        "compatibility": compatibility_report(state),
        "latency": {
            "responses_blocking": summarize(response_samples),
            "chat_completions": summarize(chat_samples),
        },
    }
    output_dir = os.path.dirname(OUTPUT_PATH)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Report written to {OUTPUT_PATH}")


def benchmark_metadata(state: BenchmarkState) -> dict[str, Any]:
    return {
        "target": BASE_URL,
        "model": MODEL,
        "text_model": TEXT_MODEL,
        "vision_model": VISION_MODEL,
        "asset_base_url": ASSET_BASE_URL,
        "model_backend": MODEL_BACKEND,
        "ollama_base_url": OLLAMA_BASE_URL,
        "database_driver": DATABASE_DRIVER,
        "git_sha": GIT_SHA,
        "runs": RUNS,
        "timeout_seconds": TIMEOUT_SECONDS,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "completion_output_tokens": COMPLETION_OUTPUT_TOKENS,
        "expect_ollama_metrics": EXPECT_OLLAMA_METRICS,
        "coverage_gate": COVERAGE_GATE,
        "include_tags": sorted(INCLUDE_TAGS),
        "exclude_tags": sorted(EXCLUDE_TAGS),
        "server_health": state.server_health or {},
        "server_ready": state.server_ready or {},
    }


def compatibility_report(state: BenchmarkState) -> dict[str, Any]:
    manifest = state.compatibility_manifest or {"features": [], "summary": {}}
    features = manifest.get("features") or []
    skipped_feature_ids = {
        feature_id
        for case in state.cases
        if case.status == "skipped"
        for feature_id in case.feature_ids
    }
    supported = [_surface_summary(feature) for feature in features if str(feature.get("status", "")).startswith("supported")]
    unsupported = [_surface_summary(feature) for feature in features if feature.get("status") == "unsupported"]
    skipped = [_surface_summary(feature) for feature in features if feature.get("id") in skipped_feature_ids]
    return {
        "manifest": manifest,
        "coverage": state.compatibility_coverage or {},
        "surfaces": {
            "supported": supported,
            "unsupported": unsupported,
            "skipped": skipped,
        },
    }


def _surface_summary(feature: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": feature.get("id"),
        "category": feature.get("category"),
        "surface": feature.get("surface"),
        "status": feature.get("status"),
        "benchmark_case": feature.get("benchmark_case"),
        "tags": feature.get("tags") or [],
    }


def expect(condition: Any, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


if __name__ == "__main__":
    sys.exit(main())
