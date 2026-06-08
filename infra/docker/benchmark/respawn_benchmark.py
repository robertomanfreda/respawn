#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
COMPARE_TO_PATH = os.getenv("RESPAWN_BENCHMARK_COMPARE_TO", "")
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
    reasoning_response_id: str | None = None
    reasoning_item: dict[str, Any] | None = None
    context_compacted_response_id: str | None = None
    compacted_window: list[dict[str, Any]] | None = None
    benchmark_comparison: dict[str, Any] | None = None


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
    if COMPARE_TO_PATH:
        state.benchmark_comparison = compare_report_to_previous(build_report(state, response_samples, chat_samples), COMPARE_TO_PATH)
    print_summary(state, response_samples, chat_samples)
    write_report(state, response_samples, chat_samples)
    return 1 if failed else 0


def benchmark_cases() -> list[BenchmarkCase]:
    return [
        BenchmarkCase("healthz", {"core"}, set(), lambda state: case_healthz(state)),
        BenchmarkCase("readyz", {"core", "ops"}, {"ops.readiness_checks"}, lambda state: case_readyz(state)),
        BenchmarkCase("compatibility.manifest", {"core", "observability"}, set(), case_compatibility_manifest),
        BenchmarkCase("compatibility.coverage", {"core", "observability", "ops"}, {"ops.release_certification"}, lambda state: case_compatibility_coverage(state, benchmark_cases())),
        BenchmarkCase("models", {"core"}, {"endpoints.models.list"}, lambda state: case_models()),
        BenchmarkCase("responses.blocking", {"core", "state"}, {"endpoints.responses.create", "request.model_and_text_input", "request.sampling_and_limits", "request.truncation_disabled", "response.core_shape"}, case_responses_blocking),
        BenchmarkCase("responses.shape.blocking_text", {"core", "state"}, {"request.text_format", "response.request_settings", "response.output_content_shape"}, lambda state: case_responses_shape_blocking_text()),
        BenchmarkCase("responses.shape.metadata_retrieve", {"core", "state"}, {"request.metadata", "request.service_tier", "request.safety_identifier"}, lambda state: case_responses_shape_metadata_retrieve()),
        BenchmarkCase("responses.shape.max_output_incomplete", {"core"}, {"response.incomplete_status"}, lambda state: case_responses_shape_max_output_incomplete()),
        BenchmarkCase("responses.shape.unsupported_user_field", {"core", "state"}, {"request.unsupported_fields"}, lambda state: case_responses_shape_unsupported_user_field()),
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
        BenchmarkCase("responses.input_tokens.model_aware", {"core", "context"}, {"endpoints.responses.input_tokens"}, lambda state: case_responses_input_tokens()),
        BenchmarkCase("files.create_retrieve_delete", {"files", "state"}, {"endpoints.files.create_retrieve_delete"}, lambda state: case_files_create_retrieve_delete()),
        BenchmarkCase("responses.input_file.file_id", {"files", "multimodal"}, {"endpoints.responses.artifact_content", "io.input_file_file_id"}, lambda state: case_responses_input_file_file_id()),
        BenchmarkCase("platform_objects.tenant_scope", {"files", "state"}, {"state.platform_objects_tenant_scope"}, lambda state: case_platform_objects_tenant_scope()),
        BenchmarkCase("sdk.responses.create_retrieve_delete", {"sdk", "core", "state"}, {"sdk.python_create_retrieve_delete", "sdk.request_id_headers", "state.files_pagination"}, lambda state: case_sdk_responses_create_retrieve_delete()),
        BenchmarkCase("sdk.responses.stream", {"sdk", "streaming"}, {"sdk.python_stream"}, lambda state: case_sdk_responses_stream()),
        BenchmarkCase("sdk.responses.background", {"sdk", "background", "state"}, {"sdk.python_background"}, lambda state: case_sdk_responses_background()),
        BenchmarkCase("sdk.errors", {"sdk", "state"}, {"request.idempotency_key", "sdk.python_errors"}, lambda state: case_sdk_errors()),
        BenchmarkCase("responses.prompt.template_render", {"core", "prompt", "state"}, {"request.prompt_templates"}, lambda state: case_responses_prompt_template_render()),
        BenchmarkCase("responses.prompt.template_missing", {"core", "prompt", "state"}, {"request.prompt_template_errors"}, lambda state: case_responses_prompt_template_missing()),
        BenchmarkCase("responses.prompt_cache.in_memory", {"core", "prompt"}, {"request.prompt_cache", "response.cached_tokens"}, lambda state: case_responses_prompt_cache_in_memory()),
        BenchmarkCase("responses.context.truncation_disabled_overflow", {"context"}, {"request.truncation_disabled_overflow"}, lambda state: case_responses_context_truncation_disabled_overflow()),
        BenchmarkCase("responses.context.truncation_auto", {"context"}, {"request.truncation_auto", "state.context_truncation_records"}, lambda state: case_responses_context_truncation_auto()),
        BenchmarkCase("responses.context.compaction", {"context", "state"}, {"request.context_management", "io.compaction_items", "state.context_compaction_records"}, case_responses_context_compaction),
        BenchmarkCase("responses.compact", {"context", "state"}, {"endpoints.responses.compact", "response.compaction_object"}, lambda state: case_responses_compact()),
        BenchmarkCase("responses.compact.followup_memory", {"context", "state"}, {"state.compaction_followup_memory"}, lambda state: case_responses_compact_followup_memory()),
        BenchmarkCase("responses.reasoning", {"reasoning"}, {"request.reasoning", "io.reasoning_items", "response.reasoning_tokens"}, lambda state: case_responses_reasoning()),
        BenchmarkCase("responses.reasoning.effort_matrix", {"reasoning"}, {"request.reasoning_effort_matrix"}, lambda state: case_responses_reasoning_effort_matrix()),
        BenchmarkCase("responses.reasoning.summary", {"reasoning"}, {"response.reasoning_summary"}, lambda state: case_responses_reasoning_summary()),
        BenchmarkCase("responses.reasoning.previous_response_carryover", {"reasoning", "state"}, {"state.reasoning_previous_response_carryover"}, case_responses_reasoning_previous_response_carryover),
        BenchmarkCase("responses.reasoning.encrypted_roundtrip", {"reasoning", "state"}, {"io.reasoning_encrypted_content"}, lambda state: case_responses_reasoning_encrypted_roundtrip()),
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
        BenchmarkCase("responses.include.file_artifacts", {"include", "multimodal", "state"}, {"request.include_input_image_url", "state.response_artifacts"}, lambda state: case_responses_include_file_artifacts()),
        BenchmarkCase("responses.include.annotations", {"include", "multimodal", "state"}, {"endpoints.responses.artifact_content", "endpoints.responses.artifacts", "io.output_text_file_annotations", "state.response_artifact_pagination"}, lambda state: case_responses_include_annotations()),
        BenchmarkCase("responses.include.unsupported_logprobs", {"include", "core"}, {"request.include_output_text_logprobs", "response.output_text_logprobs"}, lambda state: case_responses_include_unsupported_logprobs()),
        BenchmarkCase("responses.include.hosted_tool_unsupported", {"include", "tools"}, {"request.include_hosted_tool_expansions"}, lambda state: case_responses_include_hosted_tool_unsupported()),
        BenchmarkCase("responses.retrieve.include", {"include", "state"}, {"request.include_input_image_url"}, lambda state: case_responses_retrieve_include()),
        BenchmarkCase("chat.completions", {"core"}, {"endpoints.chat_completions.create"}, lambda state: case_chat_completions()),
        BenchmarkCase("chat.completions.stream", {"streaming"}, set(), lambda state: case_chat_completions_stream()),
        BenchmarkCase("metrics", {"observability"}, {"observability.metrics"}, case_metrics),
        BenchmarkCase("metrics.background_jobs", {"background", "observability"}, {"observability.background_metrics"}, case_background_metrics),
        BenchmarkCase("metrics.function_tools", {"tools", "observability"}, {"observability.function_tool_metrics"}, case_function_tool_metrics),
        BenchmarkCase("metrics.reasoning", {"reasoning", "observability"}, {"observability.reasoning_metrics"}, case_reasoning_metrics),
        BenchmarkCase("metrics.context_management", {"context", "observability"}, {"observability.context_metrics"}, case_context_management_metrics),
        BenchmarkCase("metrics.include_expansions", {"include", "observability"}, {"observability.include_metrics"}, case_include_metrics),
        BenchmarkCase("metrics.prompt_cache", {"prompt", "observability"}, {"observability.prompt_cache_metrics"}, case_prompt_cache_metrics),
        BenchmarkCase("metrics.full_surface", {"observability", "ops"}, {"observability.full_surface_metrics"}, case_metrics_full_surface),
        BenchmarkCase("ops.ollama_unavailable", {"ops", "observability"}, {"ops.ollama_unavailable"}, case_ops_ollama_unavailable),
        BenchmarkCase("ops.concurrent_streaming", {"ops", "streaming"}, {"ops.concurrent_streaming"}, case_ops_concurrent_streaming),
        BenchmarkCase("ops.concurrent_background", {"ops", "background"}, {"ops.concurrent_background"}, case_ops_concurrent_background),
        BenchmarkCase("benchmark.history_compare", {"benchmark", "ops"}, {"benchmark.history_compare"}, lambda state: case_benchmark_history_compare()),
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


def case_responses_shape_unsupported_user_field() -> str:
    body, status, _ = request_json_error("POST", "/v1/responses", {"model": MODEL, "input": "hello", "user": "legacy-user"})
    expect(status == 400, f"unsupported user field returned HTTP {status}: {body}")
    error = body.get("error") or {}
    expect(error.get("code") == "unsupported_parameter", f"unexpected unsupported user error: {body}")
    expect(error.get("param") == "user", f"unexpected unsupported user param: {body}")
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


def case_files_create_retrieve_delete() -> str:
    created = upload_benchmark_text_file()
    file_id = created["id"]
    listed, _, _ = request_json("GET", "/v1/files")
    expect(any(item.get("id") == file_id for item in listed.get("data", [])), f"uploaded file missing from list: {listed}")
    retrieved, _, _ = request_json("GET", f"/v1/files/{file_id}")
    expect(retrieved.get("filename") == "facts.txt", f"retrieved file filename mismatch: {retrieved}")
    content, status, _ = request_raw("GET", f"/v1/files/{file_id}/content")
    expect(status == 200, f"file content returned HTTP {status}: {content[:200]}")
    expect("cobalt" in content.lower(), f"file content did not include marker: {content[:200]}")
    deleted, _, _ = request_json("DELETE", f"/v1/files/{file_id}")
    expect(deleted.get("deleted") is True, f"delete file did not return deleted=true: {deleted}")
    body, missing_status, _ = request_json_error("GET", f"/v1/files/{file_id}")
    expect(missing_status == 404, f"deleted file should be hidden, got HTTP {missing_status}: {body}")
    return file_id


def case_responses_input_file_file_id() -> str:
    created = upload_benchmark_text_file()
    body, _, _ = post_json(
        "/v1/responses",
        {
            "model": MODEL,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_file", "file_id": created["id"]},
                        {"type": "input_text", "text": "What is the file marker word? Reply with only that word."},
                    ],
                }
            ],
            "max_output_tokens": completion_output_tokens(256),
            "reasoning": {"effort": "none"},
            "temperature": 0,
            "store": True,
        },
    )
    expect_response_final(body)
    expect("cobalt" in output_text(body).lower(), f"file_id response did not use uploaded file: {body}")
    content = body.get("output", [{}])[-1].get("content", [{}])[0]
    annotations = content.get("annotations") or []
    expect(annotations, f"file_id response did not include local file annotations: {body}")
    annotation = annotations[0]
    artifact_content, artifact_status, _ = request_raw("GET", f"/v1/responses/{body['id']}/artifacts/{annotation['file_id']}/content")
    expect(artifact_status == 200, f"artifact content returned HTTP {artifact_status}: {artifact_content[:200]}")
    expect("cobalt" in artifact_content.lower(), f"artifact content did not include uploaded file text: {artifact_content[:200]}")
    return body["id"]


def case_platform_objects_tenant_scope() -> str:
    created = upload_benchmark_text_file(api_key=API_KEY)
    file_id = created["id"]
    _, primary_status, _ = request_raw("GET", f"/v1/files/{file_id}", api_key=API_KEY)
    expect(primary_status == 200, f"primary tenant should retrieve file, got HTTP {primary_status}")
    body, other_status, _ = request_json_error("GET", f"/v1/files/{file_id}", api_key=SECONDARY_API_KEY)
    expect(other_status == 404, f"other tenant should not retrieve file, got HTTP {other_status}: {body}")
    response_body, response_status, _ = request_json_error(
        "POST",
        "/v1/responses",
        {
            "model": MODEL,
            "input": [{"role": "user", "content": [{"type": "input_file", "file_id": file_id}]}],
            "max_output_tokens": 8,
        },
        api_key=SECONDARY_API_KEY,
    )
    expect(response_status == 404, f"other tenant should not use file_id, got HTTP {response_status}: {response_body}")
    return "tenant isolated"


def case_sdk_responses_create_retrieve_delete() -> str:
    sdk = openai_sdk_client()
    raw = sdk.responses.with_raw_response.create(
        model=MODEL,
        input=[
            {"role": "user", "content": "first SDK benchmark item"},
            {"role": "user", "content": "second SDK benchmark item"},
        ],
        max_output_tokens=completion_output_tokens(256),
        reasoning={"effort": "none"},
        text={"format": {"type": "text"}},
        extra_headers={"x-request-id": "req_benchmark_sdk"},
    )
    created = raw.parse()
    expect(raw.headers.get("x-request-id") == "req_benchmark_sdk", f"SDK raw response missing request id: {raw.headers}")
    expect(getattr(created, "_request_id", None) == "req_benchmark_sdk", f"SDK parsed object missing request id: {created}")
    retrieved = sdk.responses.retrieve(created.id)
    expect(retrieved.id == created.id, f"SDK retrieve returned wrong response: {retrieved}")
    items = sdk.responses.input_items.list(created.id, order="asc", limit=1)
    expect(len(items.data) == 1, f"SDK input_items list did not parse page: {items}")
    expect(items.model_dump().get("first_id") == items.data[0].id, f"SDK input_items first_id mismatch: {items.model_dump()}")

    uploaded = sdk.files.create(file=("sdk-benchmark.txt", b"Respawn SDK benchmark file marker."), purpose="user_data")
    file_page = sdk.files.list(order="asc", limit=1)
    file_page_body = file_page.model_dump()
    expect(file_page.data, f"SDK files list did not return data: {file_page_body}")
    expect(file_page_body.get("first_id"), f"SDK files list missing first_id: {file_page_body}")
    expect("Respawn SDK" in sdk.files.content(uploaded.id).text, "SDK files.content did not return uploaded text")
    deleted_file = sdk.files.delete(uploaded.id)
    expect(deleted_file.deleted is True, f"SDK files.delete did not parse deleted object: {deleted_file}")

    deleted_response = sdk.responses.delete(created.id)
    expect(deleted_response is None, f"SDK responses.delete should return None, got {deleted_response}")
    return created.id


def case_sdk_responses_stream() -> str:
    sdk = openai_sdk_client()
    with sdk.responses.stream(
        model=MODEL,
        input="Respawn SDK benchmark streaming. Reply briefly.",
        max_output_tokens=completion_output_tokens(256),
        reasoning={"effort": "none"},
        store=False,
    ) as stream:
        events = list(stream)
    event_types = [event.type for event in events]
    expect("response.output_text.delta" in event_types, f"SDK stream missing text deltas: {event_types}")
    expect(event_types[-1] in {"response.completed", "response.incomplete"}, f"SDK stream missing terminal event: {event_types}")
    expect([getattr(event, "sequence_number", None) for event in events] == list(range(len(events))), f"SDK stream sequence mismatch: {event_types}")
    return event_types[-1]


def case_sdk_responses_background() -> str:
    sdk = openai_sdk_client()
    created = sdk.responses.create(
        model=MODEL,
        input="Respawn SDK benchmark background cancellation. Produce a concise answer.",
        background=True,
        max_output_tokens=completion_output_tokens(256),
        reasoning={"effort": "none"},
        store=True,
    )
    expect(created.background is True, f"SDK background flag missing: {created}")
    cancelled = sdk.responses.cancel(created.id)
    expect(cancelled.status in {"cancelled", "completed", "incomplete"}, f"SDK cancel returned unexpected status: {cancelled}")
    retrieved = sdk.responses.retrieve(created.id)
    expect(retrieved.id == created.id, f"SDK background retrieve wrong id: {retrieved}")
    return cancelled.status


def case_sdk_errors() -> str:
    from openai import BadRequestError, ConflictError, InternalServerError, NotFoundError, UnprocessableEntityError

    sdk = openai_sdk_client()
    headers = {"Idempotency-Key": "benchmark-sdk-idempotency"}
    first = sdk.responses.create(model=MODEL, input="SDK idempotent body", max_output_tokens=completion_output_tokens(256), reasoning={"effort": "none"}, extra_headers=headers)
    replayed = sdk.responses.create(model=MODEL, input="SDK idempotent body", max_output_tokens=completion_output_tokens(256), reasoning={"effort": "none"}, extra_headers=headers)
    expect(replayed.id == first.id, f"SDK idempotent replay changed response id: {first.id} vs {replayed.id}")
    try:
        sdk.responses.create(model=MODEL, input="SDK changed idempotent body", max_output_tokens=8, extra_headers=headers)
    except ConflictError as exc:
        expect(exc.status_code == 409, f"SDK conflict status mismatch: {exc.status_code}")
    else:
        raise AssertionError("SDK idempotency conflict did not raise ConflictError")

    try:
        sdk.responses.create(model=MODEL, input="bad", user="legacy-user")
    except BadRequestError as exc:
        expect(exc.status_code == 400, f"SDK bad request status mismatch: {exc.status_code}")
    else:
        raise AssertionError("SDK unsupported user field did not raise BadRequestError")

    try:
        sdk.responses.retrieve("resp_missing")
    except NotFoundError as exc:
        expect(exc.status_code == 404, f"SDK not found status mismatch: {exc.status_code}")
    else:
        raise AssertionError("SDK missing response did not raise NotFoundError")

    try:
        sdk.responses.create(model=MODEL, input="invalid temperature", temperature=3)
    except UnprocessableEntityError as exc:
        expect(exc.status_code == 422, f"SDK validation status mismatch: {exc.status_code}")
    else:
        raise AssertionError("SDK validation error did not raise UnprocessableEntityError")

    if MODEL_BACKEND != "mock":
        try:
            sdk.responses.create(model="respawn-missing-model-for-sdk-error", input="backend error", max_output_tokens=8)
        except InternalServerError as exc:
            expect(exc.status_code in {500, 502, 503, 504}, f"SDK server error status mismatch: {exc.status_code}")
        else:
            raise AssertionError("SDK backend error did not raise InternalServerError")
    return "sdk errors mapped"


def case_responses_prompt_template_render() -> str:
    prompt_id = f"pmpt_benchmark_{os.getpid()}_{time.monotonic_ns()}"
    created, _, _ = post_json(
        "/v1/responses/prompts",
        {
            "id": prompt_id,
            "version": "1",
            "input": "Reply with exactly this prompt template marker word and no extra words: {{word}}.",
            "metadata": {"benchmark": "responses.prompt.template_render"},
        },
    )
    expect(created.get("id") == prompt_id, f"prompt template create returned wrong id: {created}")
    body, _, _ = post_json(
        "/v1/responses",
        {
            "model": MODEL,
            "prompt": {"id": prompt_id, "version": "1", "variables": {"word": "vermilion"}},
            "max_output_tokens": completion_output_tokens(96),
            "store": False,
        },
    )
    expect_response_final(body)
    expect("vermilion" in output_text(body).lower(), f"rendered prompt output did not reflect variable: {body}")
    prompt = body.get("prompt") or {}
    expect(prompt.get("version") == "1", f"resolved prompt version missing: {body}")
    return prompt_id


def case_responses_prompt_template_missing() -> str:
    prompt_id = f"pmpt_missing_{os.getpid()}_{time.monotonic_ns()}"
    body, status, _ = request_json_error("POST", "/v1/responses", {"model": MODEL, "prompt": {"id": prompt_id}, "max_output_tokens": 8})
    expect(status == 404, f"missing prompt template returned HTTP {status}: {body}")
    error = body.get("error") or {}
    expect(error.get("code") == "not_found", f"unexpected missing prompt template code: {body}")
    expect(error.get("param") == "prompt.id", f"unexpected missing prompt template param: {body}")
    return error["code"]


def case_responses_prompt_cache_in_memory() -> str:
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


def case_responses_context_truncation_disabled_overflow() -> str:
    long_input = " ".join(f"overflow-token-{index}" for index in range(9000))
    body, status, _ = request_json_error(
        "POST",
        "/v1/responses",
        {
            "model": MODEL,
            "input": long_input,
            "max_output_tokens": 1,
            "store": False,
        },
    )
    expect(status == 400, f"truncation=disabled overflow returned HTTP {status}: {body}")
    error = body.get("error") or {}
    expect(error.get("code") == "context_length_exceeded", f"unexpected overflow error: {body}")
    expect(error.get("param") == "input", f"unexpected overflow param: {body}")
    return "overflow rejected"


def case_responses_context_truncation_auto() -> str:
    old_items = [{"role": "user", "content": f"old context item {index} " * 8} for index in range(900)]
    body, _, _ = post_json(
        "/v1/responses",
        {
            "model": MODEL,
            "input": [*old_items, {"role": "user", "content": "Reply with exactly: auto truncation ok"}],
            "truncation": "auto",
            "max_output_tokens": completion_output_tokens(32),
            "store": False,
        },
    )
    expect_response_final(body)
    expect(body.get("truncation") == "auto", f"truncation did not round-trip: {body}")
    expect(body.get("usage", {}).get("input_tokens", 0) > 0, f"truncation auto usage missing: {body}")
    return body["status"]


def case_responses_context_compaction(state: BenchmarkState) -> str:
    if state.context_compacted_response_id:
        return state.context_compacted_response_id
    filler = " ".join(f"context-filler-{index}" for index in range(1200))
    first, _, _ = post_json(
        "/v1/responses",
        {
            "model": MODEL,
            "input": f"The preserved marker word is amethyst. {filler}\nReply with one short acknowledgement.",
            "max_output_tokens": completion_output_tokens(16),
            "store": True,
        },
    )
    expect_response_final(first)
    second, _, _ = post_json(
        "/v1/responses",
        {
            "model": MODEL,
            "previous_response_id": first["id"],
            "input": "What is the preserved marker word? Reply with only that word.",
            "context_management": [{"type": "compaction", "compact_threshold": 1000}],
            "max_output_tokens": completion_output_tokens(32),
            "store": True,
        },
    )
    expect_response_final(second)
    output = second.get("output") or []
    expect(output and output[0].get("type") == "compaction", f"server-side compaction item missing: {second}")
    expect(isinstance(output[0].get("encrypted_content"), str), f"compaction encrypted_content missing: {second}")
    state.context_compacted_response_id = second["id"]
    return second["id"]


def case_responses_compact() -> str:
    body, _, _ = post_json(
        "/v1/responses/compact",
        {
            "model": MODEL,
            "input": [
                {"role": "user", "content": "The preserved marker word is amethyst."},
                {"role": "assistant", "content": "Noted."},
            ],
        },
    )
    expect(body.get("object") == "response.compaction", f"unexpected compaction object: {body}")
    expect(body.get("id", "").startswith("resp_"), f"unexpected compaction id: {body}")
    output = body.get("output") or []
    expect(output and output[-1].get("type") == "compaction", f"compaction output item missing: {body}")
    expect(isinstance(output[-1].get("encrypted_content"), str), f"compaction encrypted_content missing: {body}")
    usage = body.get("usage") or {}
    expect(usage.get("input_tokens", 0) > 0, f"compaction usage missing input tokens: {body}")
    return body["id"]


def case_responses_compact_followup_memory() -> str:
    compacted, _, _ = post_json(
        "/v1/responses/compact",
        {
            "model": MODEL,
            "input": [
                {"role": "user", "content": "The preserved marker word is amethyst. Keep this fact."},
                {"role": "assistant", "content": "I will remember the marker word."},
            ],
        },
    )
    output = compacted.get("output") or []
    body, _, _ = post_json(
        "/v1/responses",
        {
            "model": MODEL,
            "input": [*output, {"role": "user", "content": "What is the preserved marker word? Reply with only that word."}],
            "max_output_tokens": completion_output_tokens(128),
            "store": False,
        },
    )
    expect_response_final(body)
    text = (body.get("output_text") or "").lower()
    expect("amethyst" in text, f"compacted follow-up did not preserve marker fact: {body}")
    return "amethyst"


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


def case_responses_reasoning_effort_matrix() -> str:
    accepted = []
    for effort in ("none", "minimal", "low", "medium", "high"):
        body, _, _ = post_json(
            "/v1/responses",
            {
                "model": MODEL,
                "input": f"Respawn benchmark reasoning effort {effort}. Reply briefly.",
                "reasoning": {"effort": effort},
                "max_output_tokens": completion_output_tokens(64),
                "store": False,
            },
        )
        expect_response_final(body)
        expect(body.get("reasoning", {}).get("effort") == effort, f"reasoning effort did not round-trip: {body}")
        accepted.append(effort)

    body, status, _ = request_json_error(
        "POST",
        "/v1/responses",
        {
            "model": MODEL,
            "input": "Respawn benchmark xhigh reasoning capability check.",
            "reasoning": {"effort": "xhigh"},
            "max_output_tokens": completion_output_tokens(64),
            "store": False,
        },
    )
    if 200 <= status < 300:
        expect_response_final(body)
        accepted.append("xhigh")
    else:
        error = body.get("error") or {}
        expect(status == 400, f"xhigh reasoning returned HTTP {status}: {body}")
        expect(error.get("code") == "unsupported_model_capability", f"unexpected xhigh capability error: {body}")
        expect(error.get("param") == "reasoning.effort", f"unexpected xhigh capability param: {body}")
    return ",".join(accepted)


def case_responses_reasoning_summary() -> str:
    body, _, _ = post_json(
        "/v1/responses",
        {
            "model": MODEL,
            "input": "Respawn benchmark reasoning summary. Reply with one concise sentence.",
            "reasoning": {"effort": "low", "summary": "auto"},
            "max_output_tokens": completion_output_tokens(96),
            "store": False,
        },
    )
    expect_response_final(body)
    reasoning = expect_reasoning_item(body)
    summary = reasoning.get("summary") or []
    expect(summary and summary[0].get("type") == "summary_text", f"reasoning summary missing: {reasoning}")
    text = str(summary[0].get("text") or "")
    expect("Raw reasoning content is intentionally not exposed" in text or "No reasoning trace" in text, f"unexpected summary text: {text}")
    expect("encrypted_content" not in reasoning, f"encrypted_content should be opt-in: {reasoning}")
    return "summary_text"


def case_responses_reasoning_previous_response_carryover(state: BenchmarkState) -> str:
    if not state.reasoning_response_id or not state.reasoning_item:
        first, _, _ = post_json(
            "/v1/responses",
            {
                "model": MODEL,
                "input": "Respawn benchmark stored reasoning. Remember the marker word cobalt.",
                "reasoning": {"effort": "low", "summary": "auto"},
                "max_output_tokens": completion_output_tokens(96),
                "store": True,
            },
        )
        expect_response_final(first)
        state.reasoning_response_id = first["id"]
        state.reasoning_item = expect_reasoning_item(first)

    second, _, _ = post_json(
        "/v1/responses",
        {
            "model": MODEL,
            "previous_response_id": state.reasoning_response_id,
            "input": "Continue after the stored reasoning response in one short sentence.",
            "reasoning": {"effort": "low", "summary": "auto"},
            "max_output_tokens": completion_output_tokens(96),
            "store": True,
        },
    )
    expect_response_final(second)
    retrieved, _, _ = request_json("GET", f"/v1/responses/{state.reasoning_response_id}")
    retrieved_reasoning = expect_reasoning_item(retrieved)
    expect(retrieved_reasoning.get("id") == state.reasoning_item.get("id"), f"stored reasoning item changed: {retrieved_reasoning} vs {state.reasoning_item}")
    expect(second.get("previous_response_id") == state.reasoning_response_id, f"previous_response_id did not round-trip: {second}")
    return second["status"]


def case_responses_reasoning_encrypted_roundtrip() -> str:
    first, _, _ = post_json(
        "/v1/responses",
        {
            "model": MODEL,
            "input": "Respawn benchmark encrypted reasoning. Reply briefly.",
            "reasoning": {"effort": "low", "summary": "auto"},
            "include": ["reasoning.encrypted_content"],
            "max_output_tokens": completion_output_tokens(96),
            "store": False,
        },
    )
    expect_response_final(first)
    reasoning = expect_reasoning_item(first)
    encrypted_content = reasoning.get("encrypted_content")
    expect(isinstance(encrypted_content, str) and encrypted_content, f"reasoning encrypted_content missing: {reasoning}")
    expect(encrypted_content not in first.get("output_text", ""), "encrypted_content leaked into output_text")

    second, _, _ = post_json(
        "/v1/responses",
        {
            "model": MODEL,
            "input": [
                reasoning,
                {"role": "user", "content": "Continue after the encrypted reasoning item in one short sentence."},
            ],
            "reasoning": {"effort": "low", "summary": "auto"},
            "include": ["reasoning.encrypted_content"],
            "max_output_tokens": completion_output_tokens(96),
            "store": False,
        },
    )
    expect_response_final(second)
    second_reasoning = expect_reasoning_item(second)
    expect(isinstance(second_reasoning.get("encrypted_content"), str), f"second encrypted_content missing: {second_reasoning}")
    return "encrypted_content"


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
    expect_response_final(body)
    input_items = body.get("input") or []
    expect(input_items and input_items[0].get("type") == "message", f"message-list input did not round-trip: {body}")
    content = input_items[0].get("content") or []
    expect(content and content[0].get("type") == "input_text", f"input_text content did not round-trip: {body}")
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
    expect(error.get("param") == "include.0", f"unexpected unsupported param: {body}")
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


def case_responses_include_file_artifacts() -> str:
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
            "include": ["message.input_image.image_url"],
            "max_output_tokens": completion_output_tokens(64),
            "store": True,
        },
    )
    expect_response_final(body)
    image_part = body.get("input", [{}])[0].get("content", [{}, {}])[1]
    artifact = image_part.get("artifact") or {}
    expect(artifact.get("id", "").startswith("art_"), f"image include did not expose artifact metadata: {body}")
    expect(artifact.get("source", {}).get("type") == "url", f"image artifact source was not URL metadata: {artifact}")
    expect("content" not in artifact, f"image artifact leaked content bytes: {artifact}")
    return artifact["id"]


def case_responses_include_annotations() -> str:
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
                        {"type": "input_text", "text": "Answer with one short sentence using the file marker word."},
                    ],
                }
            ],
            "max_output_tokens": completion_output_tokens(128),
            "store": True,
        },
    )
    expect_response_final(body)
    content = body.get("output", [{}])[-1].get("content", [{}])[0]
    annotations = content.get("annotations") or []
    expect(annotations, f"output_text did not include file annotations: {body}")
    annotation = annotations[0]
    expect(annotation.get("type") == "file_citation", f"unexpected annotation type: {annotation}")
    expect(annotation.get("file_id", "").startswith("art_"), f"annotation did not point at a local artifact: {annotation}")
    retrieved, _, _ = request_json("GET", f"/v1/responses/{body['id']}")
    retrieved_annotations = retrieved.get("output", [{}])[-1].get("content", [{}])[0].get("annotations") or []
    expect(retrieved_annotations == annotations, f"retrieve did not preserve annotations: {retrieved}")
    artifact_content, artifact_status, _ = request_raw("GET", f"/v1/responses/{body['id']}/artifacts/{annotation['file_id']}/content")
    expect(artifact_status == 200, f"artifact content returned HTTP {artifact_status}: {artifact_content[:200]}")
    expect("cobalt" in artifact_content.lower(), f"artifact content did not include file text: {artifact_content[:200]}")
    artifacts, _, _ = request_json("GET", f"/v1/responses/{body['id']}/artifacts?order=asc&limit=1")
    expect(artifacts.get("object") == "list", f"artifact list did not return a list object: {artifacts}")
    expect((artifacts.get("data") or [{}])[0].get("id") == annotation["file_id"], f"artifact list did not include cited artifact: {artifacts}")
    return annotation["file_id"]


def case_responses_include_unsupported_logprobs() -> str:
    body, status, _ = request_json_error(
        "POST",
        "/v1/responses",
        {
            "model": TEXT_MODEL,
            "input": "Logprob capability check.",
            "include": ["message.output_text.logprobs"],
            "top_logprobs": 1,
            "max_output_tokens": min(MAX_OUTPUT_TOKENS, 32),
            "store": False,
        },
    )
    expect(status == 400, f"logprobs request returned HTTP {status}: {body}")
    error = body.get("error") or {}
    expect(error.get("code") == "unsupported_model_capability", f"unexpected logprobs error: {body}")
    expect(error.get("param") in {"include", "top_logprobs"}, f"unexpected logprobs param: {body}")
    return error["code"]


def case_responses_include_hosted_tool_unsupported() -> str:
    body, status, _ = request_json_error("POST", "/v1/responses", {"model": MODEL, "input": "hello", "include": ["file_search_call.results"]})
    expect(status == 400, f"hosted include returned HTTP {status}: {body}")
    error = body.get("error") or {}
    expect(error.get("code") == "unsupported_parameter", f"unexpected hosted include error: {body}")
    expect(error.get("param") == "include.0", f"unexpected hosted include param: {body}")
    return error["param"]


def case_responses_retrieve_include() -> str:
    created, _, _ = request_json(
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
    expect_response_final(created)
    image_part = created.get("input", [{}])[0].get("content", [{}, {}])[1]
    expect("artifact" not in image_part, f"create without include should not expand artifact metadata: {created}")
    retrieved, _, _ = request_json("GET", f"/v1/responses/{created['id']}?include[]=message.input_image.image_url")
    artifact = retrieved.get("input", [{}])[0].get("content", [{}, {}])[1].get("artifact") or {}
    expect(artifact.get("id", "").startswith("art_"), f"retrieve include did not expand artifact metadata: {retrieved}")
    return artifact["id"]


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
    if EXPECT_OLLAMA_METRICS and "gateway_backend_eval_tokens_per_second" not in text:
        raise AssertionError("missing generic backend native throughput metrics")
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


def case_reasoning_metrics(state: BenchmarkState) -> str:
    case_responses_reasoning()
    text, status, _ = request_raw("GET", "/metrics")
    expect(status == 200, f"/metrics returned HTTP {status}")
    required = [
        "gateway_reasoning_requests_total",
        "gateway_reasoning_tokens_total",
        "gateway_reasoning_heavy_requests_total",
    ]
    for metric in required:
        expect(metric in text, f"missing reasoning metric {metric}")
    return "reasoning metrics present"


def case_context_management_metrics(state: BenchmarkState) -> str:
    case_responses_compact()
    case_responses_context_truncation_auto()
    text, status, _ = request_raw("GET", "/metrics")
    expect(status == 200, f"/metrics returned HTTP {status}")
    required = [
        "gateway_context_compactions_total",
        "gateway_context_compaction_tokens_total",
        "gateway_context_compaction_ratio_bucket",
        "gateway_context_truncations_total",
        "gateway_context_overflows_total",
    ]
    for metric in required:
        expect(metric in text, f"missing context metric {metric}")
    return "context metrics present"


def case_include_metrics(state: BenchmarkState) -> str:
    case_responses_include_file_artifacts()
    case_responses_include_unsupported_logprobs()
    text, status, _ = request_raw("GET", "/metrics")
    expect(status == 200, f"/metrics returned HTTP {status}")
    required = [
        "gateway_response_include_expansions_total",
        "gateway_response_include_expansion_bytes_total",
        "gateway_response_include_capability_errors_total",
    ]
    for metric in required:
        expect(metric in text, f"missing include metric {metric}")
    return "include metrics present"


def case_prompt_cache_metrics(state: BenchmarkState) -> str:
    case_responses_prompt_template_render()
    case_responses_prompt_cache_in_memory()
    text, status, _ = request_raw("GET", "/metrics")
    expect(status == 200, f"/metrics returned HTTP {status}")
    required = [
        "gateway_prompt_template_requests_total",
        "gateway_prompt_cache_requests_total",
        "gateway_prompt_cache_tokens_total",
        "gateway_prompt_cache_hit_ratio",
    ]
    for metric in required:
        expect(metric in text, f"missing prompt/cache metric {metric}")
    return "prompt/cache metrics present"


def case_metrics_full_surface(state: BenchmarkState) -> str:
    request_json("GET", "/readyz")
    case_responses_blocking(state)
    stream_text, stream_status, _ = request_raw(
        "POST",
        "/v1/responses",
        {
            "model": MODEL,
            "input": "Respawn full-surface metrics stream.",
            "stream": True,
            "max_output_tokens": completion_output_tokens(16),
            "store": False,
        },
    )
    expect(stream_status == 200, f"metrics full-surface stream returned HTTP {stream_status}: {stream_text[:500]}")
    created_file = upload_benchmark_text_file()
    request_raw("GET", f"/v1/files/{created_file['id']}/content")
    request_json("DELETE", f"/v1/files/{created_file['id']}")
    text, status, _ = request_raw("GET", "/metrics")
    expect(status == 200, f"/metrics returned HTTP {status}")
    required = [
        "gateway_requests_total",
        "gateway_endpoint_requests_total",
        "gateway_feature_requests_total",
        "gateway_idempotency_requests_total",
        "gateway_errors_total",
        "gateway_operational_failures_total",
        "gateway_responses_total",
        "gateway_response_latency_seconds_bucket",
        "gateway_inflight_responses",
        "gateway_streaming_responses_running",
        "gateway_background_jobs_total",
        "gateway_background_job_latency_seconds_bucket",
        "gateway_background_jobs_running",
        "gateway_context_compactions_total",
        "gateway_response_include_expansions_total",
        "gateway_prompt_cache_requests_total",
        "gateway_storage_operations_total",
        "gateway_backend_model_info",
        "gateway_backend_requests_total",
        "gateway_backend_model_requests_total",
        "gateway_backend_eval_tokens_total",
        "gateway_backend_eval_duration_seconds_total",
        "gateway_backend_eval_tokens_per_second",
        "gateway_model_token_usage_total",
        "gateway_readiness_check",
        "gateway_readiness_check_latency_seconds_bucket",
    ]
    for metric in required:
        expect(metric in text, f"missing full-surface metric {metric}")
    return f"{len(required)} metric families present"


def case_ops_ollama_unavailable(state: BenchmarkState) -> str:
    if MODEL_BACKEND == "mock":
        return "mock backend has no external Ollama dependency"
    missing_model = os.getenv("RESPAWN_BENCHMARK_UNAVAILABLE_MODEL", f"respawn-missing-model-{os.getpid()}-{time.monotonic_ns()}")
    body, status, _ = request_json_error(
        "POST",
        "/v1/responses",
        {
            "model": missing_model,
            "input": "Respawn outage injection. This model should not exist.",
            "max_output_tokens": 8,
            "store": False,
        },
    )
    expect(status in {500, 502, 503, 504}, f"backend outage injection returned HTTP {status}: {body}")
    error = body.get("error") or {}
    expect(error.get("code") in {"backend_error", "backend_timeout", "internal_error"}, f"unexpected backend outage error: {body}")
    text, metrics_status, _ = request_raw("GET", "/metrics")
    expect(metrics_status == 200, f"/metrics returned HTTP {metrics_status}")
    expect("gateway_backend_requests_total" in text, "missing backend request metric after outage injection")
    expect("gateway_operational_failures_total" in text, "missing operational failure metric after outage injection")
    return str(error.get("code"))


def case_ops_concurrent_streaming(state: BenchmarkState) -> str:
    def run_stream(index: int) -> dict[str, Any]:
        text, status, _ = request_raw(
            "POST",
            "/v1/responses",
            {
                "model": MODEL,
                "input": f"Respawn concurrent stream {index}. Reply with a short sentence containing marker stream-{index}.",
                "stream": True,
                "max_output_tokens": completion_output_tokens(64),
                "store": True,
            },
        )
        expect(status == 200, f"stream {index} returned HTTP {status}: {text[:500]}")
        events = parse_sse_events(text)
        response_id = response_id_from_events(events)
        item_ids = output_item_ids_from_events(events)
        expect(response_id.startswith("resp_"), f"stream {index} missing response id: {events[-3:]}")
        return {"index": index, "response_id": response_id, "item_ids": item_ids, "text": text}

    results = run_concurrently(run_stream, range(2))
    response_ids = [result["response_id"] for result in results]
    expect(len(set(response_ids)) == len(response_ids), f"concurrent streams reused response ids: {response_ids}")
    for result in results:
        for other_id in response_ids:
            if other_id != result["response_id"]:
                expect(other_id not in result["text"], f"stream {result['index']} leaked response id {other_id}")
    item_sets = [set(result["item_ids"]) for result in results]
    if all(item_sets):
        expect(item_sets[0].isdisjoint(item_sets[1]), f"concurrent streams reused output item ids: {item_sets}")
    return f"{len(results)} streams isolated"


def case_ops_concurrent_background(state: BenchmarkState) -> str:
    def create_background(index: int) -> dict[str, Any]:
        body, _, _ = post_json(
            "/v1/responses",
            {
                "model": MODEL,
                "input": f"Respawn concurrent background {index}. Reply with marker background-{index}.",
                "background": True,
                "max_output_tokens": completion_output_tokens(64),
                "store": True,
            },
        )
        expect(body.get("background") is True, f"background flag missing: {body}")
        return body

    created = run_concurrently(create_background, range(2))
    response_ids = [body["id"] for body in created]
    expect(len(set(response_ids)) == len(response_ids), f"concurrent background reused response ids: {response_ids}")
    terminals = [poll_response_terminal(response_id, expected={"completed", "incomplete"}) for response_id in response_ids]
    item_sets = [
        {str(item.get("id")) for item in terminal.get("output") or [] if item.get("id")}
        for terminal in terminals
    ]
    if all(item_sets):
        expect(item_sets[0].isdisjoint(item_sets[1]), f"concurrent background reused output item ids: {item_sets}")
    return f"{len(terminals)} background jobs isolated"


def case_benchmark_history_compare() -> str:
    previous = {
        "cases": [
            {"name": "case.fast", "status": "passed", "latency_ms": 10.0},
            {"name": "case.old_failure", "status": "failed", "latency_ms": 20.0},
        ],
        "latency": {"responses_blocking": {"p50_ms": 100.0}, "chat_completions": {"p50_ms": 50.0}},
    }
    current = {
        "cases": [
            {"name": "case.fast", "status": "passed", "latency_ms": 15.0},
            {"name": "case.old_failure", "status": "passed", "latency_ms": 18.0},
            {"name": "case.new", "status": "passed", "latency_ms": 5.0},
        ],
        "latency": {"responses_blocking": {"p50_ms": 90.0}, "chat_completions": {"p50_ms": 75.0}},
    }
    comparison = compare_reports(current, previous)
    expect(comparison["case_counts"]["failed_delta"] == -1, f"unexpected failed delta: {comparison}")
    expect(comparison["case_counts"]["new_cases"] == ["case.new"], f"unexpected new cases: {comparison}")
    expect(comparison["latency"]["responses_blocking"]["p50_delta_ms"] == -10.0, f"unexpected latency delta: {comparison}")
    return "history comparison ok"


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


def upload_benchmark_text_file(*, api_key: str | None = None, content: str = "Respawn file marker word: cobalt.") -> dict[str, Any]:
    body, _, _ = request_multipart_file(
        "/v1/files",
        filename="facts.txt",
        content=content.encode(),
        content_type="text/plain",
        fields={"purpose": "user_data"},
        api_key=api_key,
    )
    expect(body.get("object") == "file", f"unexpected file upload object: {body}")
    expect(str(body.get("id", "")).startswith("file_"), f"uploaded file id did not look local: {body}")
    return body


def openai_sdk_client() -> Any:
    from openai import OpenAI

    return OpenAI(base_url=f"{BASE_URL}/v1", api_key=API_KEY or "local-dev-key", max_retries=0, timeout=TIMEOUT_SECONDS)


def request_multipart_file(
    path: str,
    *,
    filename: str,
    content: bytes,
    content_type: str,
    fields: dict[str, str],
    api_key: str | None = None,
) -> tuple[dict[str, Any], int, float]:
    boundary = f"respawn-boundary-{os.getpid()}-{time.monotonic_ns()}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                str(value).encode(),
                b"\r\n",
            ]
        )
    chunks.extend(
        [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode(),
            f"Content-Type: {content_type}\r\n\r\n".encode(),
            content,
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    data = b"".join(chunks)
    text, status, latency_ms = request_raw_bytes(
        "POST",
        path,
        data,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        api_key=api_key,
    )
    expect(200 <= status < 300, f"POST {path} returned HTTP {status}: {text[:500]}")
    try:
        return json.loads(text), status, latency_ms
    except json.JSONDecodeError as exc:
        raise AssertionError(f"POST {path} returned invalid JSON: {text[:500]}") from exc


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
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload, separators=(",", ":")).encode()
        headers["Content-Type"] = "application/json"
    return request_raw_bytes(method, path, data, headers=headers, api_key=api_key)


def request_raw_bytes(method: str, path: str, data: bytes | None, *, headers: dict[str, str] | None = None, api_key: str | None = None) -> tuple[str, int, float]:
    url = f"{BASE_URL}{path}"
    request_headers = dict(headers or {})
    selected_api_key = API_KEY if api_key is None else api_key
    if selected_api_key:
        request_headers["Authorization"] = f"Bearer {selected_api_key}"
    request = Request(url, data=data, headers=request_headers, method=method)
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


def run_concurrently(fn, values) -> list[Any]:
    value_list = list(values)
    results: list[Any] = [None] * len(value_list)
    with ThreadPoolExecutor(max_workers=len(value_list)) as executor:
        futures = {executor.submit(fn, value): index for index, value in enumerate(value_list)}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return results


def response_id_from_events(events: list[dict[str, Any]]) -> str:
    for event in reversed(events):
        response = (event.get("data") or {}).get("response")
        if isinstance(response, dict) and isinstance(response.get("id"), str):
            return response["id"]
    raise AssertionError(f"SSE events did not include a response id: {events[-3:]}")


def output_item_ids_from_events(events: list[dict[str, Any]]) -> list[str]:
    ids = []
    for event in events:
        item = (event.get("data") or {}).get("item")
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            ids.append(item["id"])
    return ids



def first_output_text_part(body: dict[str, Any]) -> dict[str, Any]:
    for item in body.get("output") or []:
        for content in item.get("content") or []:
            if content.get("type") in {"output_text", "text"}:
                return content
    raise AssertionError(f"response has no output text part: {body}")


def expect_reasoning_item(body: dict[str, Any]) -> dict[str, Any]:
    expect_response_object(body)
    reasoning_items = [item for item in body.get("output") or [] if item.get("type") == "reasoning"]
    expect(reasoning_items, f"response output did not include a reasoning item: {body}")
    item = reasoning_items[0]
    expect(str(item.get("id", "")).startswith("rs_"), f"reasoning item id should start with rs_: {item}")
    expect(isinstance(item.get("summary", []), list), f"reasoning summary should be a list: {item}")
    expect(item.get("status") in {"completed", "in_progress", "incomplete"}, f"reasoning status missing: {item}")
    return item


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
    if state.benchmark_comparison:
        status = state.benchmark_comparison.get("status")
        counts = state.benchmark_comparison.get("case_counts") or {}
        print(
            "Benchmark comparison: "
            f"status={status} passed_delta={counts.get('passed_delta', 0)} failed_delta={counts.get('failed_delta', 0)}"
        )

    if response_samples:
        print(f"Responses latency: {json.dumps(summarize(response_samples), sort_keys=True)}")
    if chat_samples:
        print(f"Chat latency:      {json.dumps(summarize(chat_samples), sort_keys=True)}")


def write_report(state: BenchmarkState, response_samples: list[float], chat_samples: list[float]) -> None:
    if not OUTPUT_PATH:
        return
    report = build_report(state, response_samples, chat_samples)
    output_dir = os.path.dirname(OUTPUT_PATH)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Report written to {OUTPUT_PATH}")


def build_report(state: BenchmarkState, response_samples: list[float], chat_samples: list[float]) -> dict[str, Any]:
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
    if state.benchmark_comparison:
        report["comparison"] = state.benchmark_comparison
    return report


def compare_report_to_previous(current_report: dict[str, Any], previous_path: str) -> dict[str, Any]:
    if not previous_path:
        return {"status": "not_configured"}
    if not os.path.exists(previous_path):
        return {"status": "missing_previous", "previous_path": previous_path}
    with open(previous_path, encoding="utf-8") as handle:
        previous_report = json.load(handle)
    comparison = compare_reports(current_report, previous_report)
    comparison["previous_path"] = previous_path
    return comparison


def compare_reports(current_report: dict[str, Any], previous_report: dict[str, Any]) -> dict[str, Any]:
    current_cases = _case_map(current_report)
    previous_cases = _case_map(previous_report)
    current_names = set(current_cases)
    previous_names = set(previous_cases)
    current_counts = _case_status_counts(current_cases)
    previous_counts = _case_status_counts(previous_cases)
    matched = sorted(current_names & previous_names)
    regressions = []
    improvements = []
    for name in matched:
        current_case = current_cases[name]
        previous_case = previous_cases[name]
        delta = float(current_case.get("latency_ms", 0.0) or 0.0) - float(previous_case.get("latency_ms", 0.0) or 0.0)
        row = {
            "name": name,
            "latency_delta_ms": round(delta, 3),
            "previous_status": previous_case.get("status"),
            "current_status": current_case.get("status"),
        }
        if delta > 0:
            regressions.append(row)
        elif delta < 0:
            improvements.append(row)

    return {
        "status": "compared",
        "case_counts": {
            "current": current_counts,
            "previous": previous_counts,
            "passed_delta": current_counts.get("passed", 0) - previous_counts.get("passed", 0),
            "failed_delta": current_counts.get("failed", 0) - previous_counts.get("failed", 0),
            "skipped_delta": current_counts.get("skipped", 0) - previous_counts.get("skipped", 0),
            "new_cases": sorted(current_names - previous_names),
            "removed_cases": sorted(previous_names - current_names),
        },
        "latency": _latency_comparison(current_report.get("latency") or {}, previous_report.get("latency") or {}),
        "slowest_case_regressions": sorted(regressions, key=lambda row: row["latency_delta_ms"], reverse=True)[:10],
        "largest_case_improvements": sorted(improvements, key=lambda row: row["latency_delta_ms"])[:10],
    }


def _case_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(case.get("name")): case for case in report.get("cases", []) if isinstance(case, dict) and case.get("name")}


def _case_status_counts(cases: dict[str, dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases.values():
        status = str(case.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def _latency_comparison(current_latency: dict[str, Any], previous_latency: dict[str, Any]) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    for name in sorted(set(current_latency) | set(previous_latency)):
        current = current_latency.get(name) or {}
        previous = previous_latency.get(name) or {}
        current_p50 = float(current.get("p50_ms", 0.0) or 0.0)
        previous_p50 = float(previous.get("p50_ms", 0.0) or 0.0)
        comparison[name] = {
            "current_p50_ms": current_p50,
            "previous_p50_ms": previous_p50,
            "p50_delta_ms": round(current_p50 - previous_p50, 3),
        }
    return comparison


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
