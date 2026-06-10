# Respawn Compatibility Matrix

This document is the single human-readable compatibility matrix for the current Respawn gateway. The machine-readable version is exposed by the running gateway at `GET /compatibility/responses` and is implemented in `apps/gateway/src/services/compatibility_manifest.py`.

The matrix describes what Respawn actually supports today: one Respawn instance connected to one configured model backend, currently Ollama in the Docker stack. It does not claim hosted OpenAI service parity for product areas that Respawn deliberately excludes.

## Summary

- Manifest version: `phase-18`
- Manifest source: `docs/COMPATIBILITY.md`
- Total tracked features: `128`
- Supported or conditionally supported: `121`
- Explicitly unsupported: `7`

| Status | Count | Meaning |
| --- | ---: | --- |
| `supported` | 50 | OpenAI-shaped behavior is implemented and benchmarked. |
| `supported_backend_capability` | 4 | The request/response shape is supported, but success depends on configured backend/model capability. |
| `supported_estimated` | 1 | Respawn returns a deterministic local estimate rather than hosted-provider authoritative data. |
| `supported_local` | 65 | Respawn implements the behavior with local single-instance semantics. |
| `supported_text_only` | 1 | Implemented for the current text-only scope. |
| `unsupported` | 7 | Rejected explicitly or deliberately outside Respawn scope. |

## Explicit Product Boundaries

- Respawn does not implement the OpenAI Conversations API. Continuity is modeled with stored Responses and `previous_response_id`.
- Respawn supports function tool calling, including namespace-wrapped function tools, as protocol data only. Clients execute functions and return `function_call_output`; Respawn does not execute function tools locally.
- Respawn supports query-style `web_search` as an opt-in local feature through a configured search provider.
- Respawn supports text-to-image `image_generation` as an opt-in local feature through a configured ComfyUI, Automatic1111, or mock image backend. Image editing, partial-image streaming, browser actions, page clicking, screenshots, form input, hosted MCP, shell/filesystem/git/workspace/code-computer tool execution, and hosted tool result expansion remain outside scope.
- Audio and realtime/transcription surfaces are not implemented.
- Multi-deployment orchestration, distributed prompt caches, dynamic backend routing, and multi-replica consistency are outside the current product model.
- Backend-specific capabilities such as vision or token logprobs depend on the configured backend/model capability metadata.

## Compatibility Matrix

### Endpoints

| Feature ID | Surface | Status | Benchmark | Notes |
| --- | --- | --- | --- | --- |
| `endpoints.responses.create` | POST /v1/responses | `supported` | `responses.blocking` |  |
| `endpoints.responses.retrieve` | GET /v1/responses/{response_id} | `supported` | `responses.retrieve` |  |
| `endpoints.responses.delete` | DELETE /v1/responses/{response_id} | `supported` | `responses.delete` |  |
| `endpoints.responses.input_items` | GET /v1/responses/{response_id}/input_items | `supported` | `responses.input_items` |  |
| `endpoints.responses.artifact_content` | GET /v1/responses/{response_id}/artifacts/{artifact_id}/content | `supported_local` | `responses.input_file.file_id` | Downloads tenant-scoped local response artifact text content created from input_file normalization. |
| `endpoints.responses.artifacts` | GET /v1/responses/{response_id}/artifacts | `supported_local` | `responses.include.annotations` | Lists tenant-scoped local response artifact metadata with after/before cursors. |
| `endpoints.responses.input_tokens` | POST /v1/responses/input_tokens | `supported_estimated` | `responses.input_tokens.model_aware` |  |
| `endpoints.responses.cancel` | POST /v1/responses/{response_id}/cancel | `supported` | `responses.background.cancel` | Cancels background Responses jobs in the current single Respawn instance. |
| `endpoints.responses.compact` | POST /v1/responses/compact | `supported_local` | `responses.compact` | Returns a response.compaction object with a local opaque compaction item and deterministic token accounting. |
| `endpoints.chat_completions.create` | POST /v1/chat/completions | `supported` | `chat.completions` |  |
| `endpoints.models.list` | GET /v1/models | `supported` | `models` |  |
| `endpoints.files.create_retrieve_delete` | POST/GET/DELETE /v1/files and GET /v1/files/{file_id}/content | `supported_local` | `files.create_retrieve_delete` | Implements the local Files API subset needed by Responses file_id inputs, with tenant scope, quota, TTL, and database/filesystem storage backends. |

### Request Fields

| Feature ID | Surface | Status | Benchmark | Notes |
| --- | --- | --- | --- | --- |
| `request.model_and_text_input` | model, input string, input text parts | `supported` | `responses.blocking` |  |
| `request.instructions` | instructions | `supported` | `responses.structured_output` |  |
| `request.previous_response_id` | previous_response_id | `supported` | `responses.previous_response_id` |  |
| `request.store` | store | `supported` | `responses.store_false` |  |
| `request.stream` | stream | `supported` | `responses.stream.lifecycle_text` |  |
| `request.stream_options` | stream_options.include_obfuscation | `supported_local` | `responses.stream.lifecycle_text` | Respawn validates stream_options and emits obfuscation fields on text delta events by default. include_obfuscation=false suppresses them. |
| `request.background` | background=true | `supported` | `responses.background.create_poll_complete` | Requires store=true; streaming background responses are rejected until streaming replay/resume is implemented. |
| `request.background_store_requirement` | background=true with store=false | `supported` | `responses.background.store_false_invalid` | Rejected with an OpenAI-shaped invalid_request error because background responses must be pollable. |
| `request.sampling_and_limits` | temperature, top_p, max_output_tokens | `supported` | `responses.blocking` |  |
| `request.function_tools` | tools entries with type=function and namespace-wrapped function tools | `supported` | `responses.tools.function_call` | Function and namespace-wrapped function tools are protocol data only; Respawn never executes them. |
| `request.web_search_tool` | tools entries with type=web_search or web_search_preview | `supported_local` | `responses.web_search.basic` | Opt-in via `WEB_SEARCH_ENABLED=true`; Respawn executes query-style search through the configured provider and injects bounded result context before backend generation. |
| `request.web_search_filters` | web_search filters.allowed_domains and filters.blocked_domains | `supported_local` | `responses.web_search.filters` | Per-request filters are validated and enforced locally after provider results return. Operator-level block lists always win. |
| `request.web_search_disabled_error` | disabled or cache-only web_search error paths | `supported_local` | `responses.web_search.disabled` | Disabled web search and `external_web_access=false` without a cache provider return explicit OpenAI-shaped unsupported_parameter errors. |
| `request.image_generation_tool` | tools entries with type=image_generation for local text-to-image | `supported_local` | `responses.image_generation.basic` | Opt-in via `IMAGE_GENERATION_ENABLED=true`; Respawn executes text-to-image generation through the configured local image backend. |
| `request.image_generation_disabled_error` | disabled, unsupported, or malformed image_generation error paths | `supported_local` | `responses.image_generation.disabled` | Disabled image generation and unsupported image_generation fields return explicit OpenAI-shaped errors. |
| `request.tool_choice` | tool_choice auto, none, required, forced function, allowed_tools, web_search, and image_generation choices | `supported_local` | `responses.tools.tool_choice_forced_function` | Function choices are mapped to the configured backend where possible. `web_search` and `image_generation` required/none choices are enforced locally before backend generation. |
| `request.parallel_and_max_tool_calls` | parallel_tool_calls, max_tool_calls | `supported_local` | `responses.tools.parallel_or_capability_error` | Respawn validates and enforces these limits around backend output instead of silently ignoring them. |
| `request.unsupported_tool_categories` | hosted MCP, custom free-form, shell, apply_patch, file/code/computer/internal tools, image edit/partial-image modes, and browser actions | `unsupported` | `responses.tools.unsupported_builtin_tools` | Function protocol data, local query-style `web_search`, and local text-to-image `image_generation` are supported. Other hosted or local tool execution remains out of scope. |
| `request.structured_output` | response_format, text.format | `supported` | `responses.structured_output` |  |
| `request.text_format` | text.format={type:text\|json_object\|json_schema} | `supported` | `responses.shape.blocking_text` |  |
| `request.metadata` | metadata | `supported` | `responses.shape.metadata_retrieve` |  |
| `request.service_tier` | service_tier | `supported_local` | `responses.shape.metadata_retrieve` | Accepted and round-tripped as local metadata; it does not change local scheduling. |
| `request.safety_identifier` | safety_identifier | `supported_local` | `responses.shape.metadata_retrieve` | Accepted and round-tripped for SDK shape compatibility. |
| `request.client_metadata` | client_metadata | `supported_local` | `responses.shape.metadata_retrieve` | Accepted as opaque client telemetry metadata for Codex/SDK compatibility; stored with the request snapshot but not forwarded to the backend or exposed on response objects. |
| `request.prompt_templates` | prompt={id, version, variables} | `supported_local` | `responses.prompt.template_render` | Respawn supports local API-managed prompt templates under /v1/responses/prompts. Templates render instructions/input with {{variable}} placeholders before local context planning and backend calls. |
| `request.prompt_template_errors` | missing prompt templates and missing prompt variables | `supported_local` | `responses.prompt.template_missing` | Missing templates and missing variables return deterministic OpenAI-shaped errors. |
| `request.idempotency_key` | Idempotency-Key request header for POST requests | `supported_local` | `sdk.errors` | Single-instance in-memory replay cache. Reusing a key with a different body returns 409 idempotency_conflict. |
| `request.prompt_cache` | prompt_cache_key, prompt_cache_retention | `supported_local` | `responses.prompt_cache.in_memory` | In-process single-instance prefix accounting with in_memory and 24h TTL semantics plus manual local invalidation. |
| `request.reasoning` | reasoning | `supported_local` | `responses.reasoning` | Supports local reasoning requests, summaries, and backend-mapped effort settings subject to configured model capabilities. |
| `request.reasoning_effort_matrix` | reasoning.effort none, minimal, low, medium, high, xhigh | `supported_backend_capability` | `responses.reasoning.effort_matrix` | xhigh is accepted only for models configured with reasoning-effort-xhigh and maps to the closest local backend setting where needed. |
| `request.context_management` | context_management=[{type:compaction, compact_threshold}] | `supported_local` | `responses.context.compaction` | Runs deterministic local compaction when the estimated rendered context crosses compact_threshold. |
| `request.truncation_disabled` | truncation=disabled | `supported` | `responses.blocking` |  |
| `request.truncation_disabled_overflow` | truncation=disabled overflow handling | `supported_local` | `responses.context.truncation_disabled_overflow` | Respawn fails with context_length_exceeded before calling the configured backend when local estimates exceed the configured context window. |
| `request.truncation_auto` | truncation=auto | `supported_local` | `responses.context.truncation_auto` | Drops earliest stored chain/input-array items until the estimated prompt fits the local model context window. |
| `request.unsupported_fields` | user and unsupported include values | `unsupported` | `responses.unsupported_field` | Unsupported request fields return explicit OpenAI-shaped errors. |
| `request.include_input_image_url` | include=message.input_image.image_url | `supported_local` | `responses.include.file_artifacts` | Respawn preserves input image URL/source metadata and expands a safe local artifact descriptor when requested. |
| `request.include_output_text_logprobs` | include=message.output_text.logprobs and top_logprobs | `supported_backend_capability` | `responses.include.unsupported_logprobs` | Returned only when the configured backend/model is marked with the logprobs capability and actually returns token logprobs. Ollama/gpt-oss defaults return unsupported_model_capability. |
| `request.include_hosted_tool_expansions` | file_search/web_search results/code_interpreter/computer include expansions | `unsupported` | `responses.include.hosted_tool_unsupported` | These include values are valid OpenAI values but require hosted tool execution. `web_search_call.action.sources` is supported for Respawn-local web search. |
| `request.future_unsupported_fields` | future-only fields without a local Respawn equivalent | `unsupported` | `responses.shape.unsupported_user_field` | Known unsupported fields such as the deprecated user field fail with unsupported_parameter; unknown extra fields fail schema validation. |

### Input And Output Items

| Feature ID | Surface | Status | Benchmark | Notes |
| --- | --- | --- | --- | --- |
| `io.text_messages` | message items with input_text/output_text | `supported_text_only` | `responses.input_message_list` |  |
| `io.function_call_items` | function_call output items and function_call_output input items | `supported` | `responses.tools.client_output_followup` | Protocol items are stored and replayed without local execution. |
| `io.legacy_tool_result_unsupported` | legacy tool_result items | `unsupported` | `responses.tools.unsupported_builtin_tools` | Only current Responses function_call/function_call_output items are accepted. |
| `io.reasoning_items` | reasoning input/output items | `supported_local` | `responses.reasoning` |  |
| `io.reasoning_encrypted_content` | reasoning.encrypted_content include and reasoning item encrypted_content | `supported_local` | `responses.reasoning.encrypted_roundtrip` | Respawn emits local opaque encrypted-content envelopes when include contains reasoning.encrypted_content and preserves client-supplied reasoning encrypted_content input items. |
| `io.compaction_items` | compaction input/output items with encrypted_content | `supported_local` | `responses.context.compaction` | Compaction items are opaque to clients. Respawn can decode locally generated items for subsequent local context replay. |
| `io.input_image` | input_image content parts from URL or data URL/base64 | `supported_backend_capability` | `responses.multimodal.input_image_vision` | Mapped to Ollama native images arrays when the selected model is configured with the vision capability. |
| `io.input_image_capability_errors` | input_image with a text-only model | `supported` | `responses.multimodal.input_image_unsupported_model` | Valid image request shapes fail with unsupported_model_capability when the selected model lacks vision. |
| `io.input_file` | input_file content parts from URL or data URL/base64 with text/PDF extraction | `supported_local` | `responses.multimodal.input_file_text` | Respawn extracts text before the backend call for text, Markdown, JSON, CSV/TSV, code files, and PDFs. |
| `io.input_file_file_id` | input_file.file_id referencing local Files API uploads | `supported_local` | `responses.input_file.file_id` | Uploaded user_data/assistants files are resolved through the tenant-scoped local Files API and extracted before backend generation. |
| `io.input_file_limits` | input_file MIME/extension, size, and download errors | `supported` | `responses.multimodal.file_limits` |  |
| `io.output_text_file_annotations` | output_text annotations for local input_file citations | `supported_local` | `responses.include.annotations` | When a response uses local input_file extraction, output_text carries OpenAI-shaped file_citation annotations pointing at local response artifact IDs. |
| `io.web_search_call_items` | web_search_call output items | `supported_local` | `responses.web_search.basic` | When local web search runs, Respawn emits an OpenAI-shaped `web_search_call` item before the final assistant message. |
| `io.image_generation_call_items` | image_generation_call output items with base64 image result | `supported_local` | `responses.image_generation.basic` | When local image generation runs, Respawn emits an OpenAI-shaped `image_generation_call` item containing the generated PNG as base64. |
| `io.input_audio_unsupported` | input_audio | `unsupported` | `responses.multimodal.input_audio_unsupported` | Audio remains a deliberate local exclusion until a dedicated audio/realtime/transcription phase exists. |
| `io.built_in_tool_items` | built-in tool call output items | `unsupported` | `responses.tools.unsupported_builtin_tools` | Built-in/internal tool execution remains out of scope even after function-tool protocol support. |

### Response Object

| Feature ID | Surface | Status | Benchmark | Notes |
| --- | --- | --- | --- | --- |
| `response.core_shape` | id, object, created_at, status, model, output, output_text, usage, metadata, error, incomplete_details | `supported` | `responses.blocking` |  |
| `response.background_shape` | background flag, queued/in_progress/completed/failed/cancelled lifecycle states | `supported` | `responses.background.retrieve_terminal` |  |
| `response.request_settings` | input, instructions, max_output_tokens, previous_response_id, service_tier, store, temperature, text, top_p, truncation, tools, tool_choice, max_tool_calls, parallel_tool_calls | `supported` | `responses.shape.blocking_text` |  |
| `response.output_content_shape` | output text content includes annotations and logprobs arrays | `supported` | `responses.shape.blocking_text` |  |
| `response.output_text_logprobs` | output_text.logprobs populated from backend token logprobs | `supported_backend_capability` | `responses.include.unsupported_logprobs` | Respawn stores and retrieves backend-provided logprobs, and fails explicitly when the configured backend/model cannot provide them. |
| `response.web_search_citations` | url_citation annotations from local web search | `supported_local` | `responses.web_search.citations` | Respawn maps model source markers when present and otherwise attaches bounded URL citations to the generated text. |
| `response.incomplete_status` | status=incomplete with incomplete_details.reason when max output exhaustion is detectable | `supported_local` | `responses.shape.max_output_incomplete` |  |
| `response.compaction_object` | response.compaction object from POST /v1/responses/compact | `supported_local` | `responses.compact` |  |
| `response.cached_tokens` | usage.input_tokens_details.cached_tokens | `supported_local` | `responses.prompt_cache.in_memory` |  |
| `response.reasoning_tokens` | usage.output_tokens_details.reasoning_tokens | `supported_local` | `responses.reasoning` |  |
| `response.reasoning_summary` | reasoning output item summary array | `supported_local` | `responses.reasoning.summary` | Summary text is generated by a deterministic local provider and does not expose raw backend thinking text. |

### State And Persistence

| Feature ID | Surface | Status | Benchmark | Notes |
| --- | --- | --- | --- | --- |
| `state.input_item_storage` | stored response input items with stable item IDs and original order | `supported` | `responses.items.input_storage` |  |
| `state.input_item_pagination` | input item pagination with after, before, limit, order, first_id, last_id, has_more | `supported` | `responses.items.pagination_after` |  |
| `state.files_pagination` | Files API pagination with after, limit, order, first_id, last_id, has_more | `supported_local` | `sdk.responses.create_retrieve_delete` |  |
| `state.input_items_store_false_hidden` | store=false responses do not expose input item lists | `supported` | `responses.items.store_false_hidden` |  |
| `state.input_items_tenant_scope` | stored input item lists are tenant-scoped | `supported` | `responses.items.tenant_scope` |  |
| `state.background_polling` | background response create, retrieve polling, and terminal output persistence | `supported` | `responses.background.create_poll_complete` |  |
| `state.background_terminal_retrieve` | retrieving terminal background responses after completion or cancellation | `supported` | `responses.background.retrieve_terminal` |  |
| `state.background_cancellation` | queued and in-flight background response cancellation | `supported` | `responses.background.cancel` | Cancellation is best-effort for an already running Ollama request and terminal state is persisted locally. |
| `state.background_timeout` | background job timeout to failed response status | `supported` | `responses.background.timeout` | Deterministically exercised by the mock benchmark/test suite; the real Ollama benchmark validates the HTTP path without forcing hardware-dependent slowness. |
| `state.context_compaction_records` | stored context compaction event records with source item provenance | `supported_local` | `responses.context.compaction` |  |
| `state.context_truncation_records` | stored context truncation event records with before/after token estimates | `supported_local` | `responses.context.truncation_auto` |  |
| `state.response_artifacts` | stored local response artifacts for input_image/input_file parts | `supported_local` | `responses.include.file_artifacts` | Artifact records store tenant-scoped metadata and extracted text references for multimodal inputs without duplicating large binary payloads in include expansions. |
| `state.response_artifact_pagination` | response artifact metadata pagination with after, before, limit, order, first_id, last_id, has_more | `supported_local` | `responses.include.annotations` |  |
| `state.platform_objects_tenant_scope` | local platform files are tenant-scoped and unavailable across tenants | `supported_local` | `platform_objects.tenant_scope` | File retrieval and Responses file_id resolution both require the owning tenant. |
| `state.compaction_followup_memory` | follow-up Responses requests can use local compaction items as compressed context | `supported_local` | `responses.compact.followup_memory` | Respawn can decode compaction items it created locally and injects their summary into the backend prompt. |
| `state.function_tool_item_storage` | stored function_call and function_call_output response items | `supported` | `responses.tools.retrieve_function_call` |  |
| `state.function_tool_previous_response_replay` | previous_response_id replay for function_call/function_call_output turns | `supported` | `responses.tools.previous_response_replay` |  |
| `state.function_tool_input_item_listing` | GET /v1/responses/{id}/input_items for function_call_output items | `supported` | `responses.tools.input_items_function_output` |  |
| `state.reasoning_previous_response_carryover` | stored reasoning items through previous_response_id chains | `supported_local` | `responses.reasoning.previous_response_carryover` |  |
| `state.web_search_item_storage` | stored and retrieved web_search_call items and annotations | `supported_local` | `responses.web_search.retrieve` | Stored Responses preserve `web_search_call` output items, URL annotations, and source metadata for `include=web_search_call.action.sources`. |

### Streaming Events

| Feature ID | Surface | Status | Benchmark | Notes |
| --- | --- | --- | --- | --- |
| `streaming.text_lifecycle` | response.created, response.in_progress, response.output_item.added, response.content_part.added, response.output_text.delta, response.output_text.done, response.content_part.done, response.output_item.done, response.completed or response.incomplete | `supported` | `responses.stream.lifecycle_text` |  |
| `streaming.event_sequence_and_sse_id` | SSE id plus per-event sequence_number | `supported_local` | `responses.stream.lifecycle_text` |  |
| `streaming.reasoning_summary` | response.reasoning_summary_part.added, response.reasoning_summary_text.delta, response.reasoning_summary_text.done, response.reasoning_summary_part.done | `supported_local` | `responses.stream.reasoning` |  |
| `streaming.incomplete` | response.incomplete terminal event when max-output exhaustion is detectable | `supported_local` | `responses.stream.incomplete` |  |
| `streaming.failure` | response.failed, error | `supported` | `responses.stream.failure` |  |
| `streaming.function_call_arguments` | response.function_call_arguments.delta, response.function_call_arguments.done | `supported` | `responses.tools.stream_arguments` |  |
| `streaming.web_search_call_events` | response.output_item.added/done for web_search_call before assistant text | `supported_local` | `responses.web_search.stream` |  |
| `streaming.image_generation_call_events` | response.output_item.added/done for image_generation_call | `supported_local` | `responses.image_generation.stream` |  |

### SDK Contract

| Feature ID | Surface | Status | Benchmark | Notes |
| --- | --- | --- | --- | --- |
| `streaming.sdk_parse` | official OpenAI Python SDK stream parser | `supported` | `responses.stream.sdk_parse` |  |
| `sdk.python_create_retrieve_delete` | official OpenAI Python SDK Responses create, retrieve, input_items list, delete, and Files create/list/content/delete | `supported` | `sdk.responses.create_retrieve_delete` |  |
| `sdk.request_id_headers` | x-request-id response headers and SDK _request_id parsing | `supported` | `sdk.responses.create_retrieve_delete` |  |
| `sdk.python_stream` | official OpenAI Python SDK Responses stream helper | `supported` | `sdk.responses.stream` |  |
| `sdk.python_background` | official OpenAI Python SDK background create, retrieve, and cancel | `supported` | `sdk.responses.background` |  |
| `sdk.python_errors` | official OpenAI Python SDK exception classes for 400, 404, 409, 422, and backend/server errors | `supported` | `sdk.errors` |  |

### Observability

| Feature ID | Surface | Status | Benchmark | Notes |
| --- | --- | --- | --- | --- |
| `observability.metrics` | /metrics gateway, model, token, and Ollama signals | `supported` | `metrics` |  |
| `observability.background_metrics` | background job counters, running gauge, and latency histogram | `supported` | `metrics.background_jobs` |  |
| `observability.function_tool_metrics` | function tool protocol request/call/output/unsupported/capability counters | `supported` | `metrics.function_tools` |  |
| `observability.web_search_metrics` | web search request, latency, result, error, and filtered-result metrics | `supported_local` | `metrics.web_search` |  |
| `observability.image_generation_metrics` | image generation request, latency, error, and pixel metrics | `supported_local` | `metrics.image_generation` |  |
| `observability.reasoning_metrics` | reasoning request, token, and heavy-request counters | `supported_local` | `metrics.reasoning` |  |
| `observability.context_metrics` | context compaction, truncation, overflow, token before/after, and compression ratio metrics | `supported_local` | `metrics.context_management` |  |
| `observability.include_metrics` | include expansion counters, byte counters, and capability-error counters | `supported_local` | `metrics.include_expansions` |  |
| `observability.prompt_cache_metrics` | prompt template operation counters and local prompt-cache hit ratio/token counters | `supported_local` | `metrics.prompt_cache` |  |
| `observability.full_surface_metrics` | endpoint, feature-family, status, token-kind, backend-model, job-status, cache, storage, readiness, idempotency, and operational failure metrics | `supported_local` | `metrics.full_surface` |  |

### Operations

| Feature ID | Surface | Status | Benchmark | Notes |
| --- | --- | --- | --- | --- |
| `ops.readiness_checks` | GET /readyz checks database, Ollama/backend, worker, cache, storage, optional web_search, and optional image_generation readiness | `supported_local` | `readyz` |  |
| `ops.ollama_unavailable` | Ollama/backend outage visibility through OpenAI-shaped 5xx errors, backend metrics, and operational failure metrics | `supported_local` | `ops.ollama_unavailable` |  |
| `ops.concurrent_streaming` | single-instance concurrent Responses streaming isolation | `supported_local` | `ops.concurrent_streaming` |  |
| `ops.concurrent_background` | single-instance concurrent background job isolation | `supported_local` | `ops.concurrent_background` |  |
| `ops.release_certification` | compatibility release certification through manifest coverage gate and release checklist | `supported_local` | `compatibility.coverage` |  |
| `benchmark.history_compare` | benchmark report comparison against a previous JSON report | `supported_local` | `benchmark.history_compare` |  |

## Release Gate

Compatibility changes are release-gated by the benchmark manifest coverage check. A feature whose status starts with `supported` must have real benchmark coverage, and unsupported features should fail explicitly with OpenAI-shaped errors instead of being silently ignored.

Run the real stack gate with:

```bash
cd infra/docker
make benchmark
```
