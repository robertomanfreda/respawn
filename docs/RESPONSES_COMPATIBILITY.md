# Responses Compatibility Matrix

Respawn aims to be a local, OpenAI-shaped Responses API gateway. This matrix is
the source of truth for the supported surface. Keep it in sync with code, tests,
and `infra/docker/benchmark/respawn_benchmark.py`.

The machine-readable compatibility manifest is exposed at
`GET /compatibility/responses` and mirrored in
`apps/gateway/src/services/compatibility_manifest.py`. Supported manifest rows
must stay tied to benchmark cases so compatibility drift fails during benchmark
runs.

References:

- OpenAI Responses API reference: https://developers.openai.com/api/reference/resources/responses/methods/create
- OpenAI Responses migration guide: https://developers.openai.com/api/docs/guides/migrate-to-responses
- OpenAI production best practices: https://developers.openai.com/api/docs/guides/production-best-practices
- OpenAI deployment checklist: https://developers.openai.com/api/docs/guides/deployment-checklist
- OpenAI file inputs guide: https://developers.openai.com/api/docs/guides/file-inputs
- OpenAI Files API reference: https://developers.openai.com/api/reference/resources/files
- OpenAI SDK libraries guide: https://developers.openai.com/api/docs/libraries
- OpenAI API error guide: https://developers.openai.com/api/docs/guides/error-codes/api-errors
- OpenAI function calling guide: https://developers.openai.com/api/docs/guides/function-calling
- OpenAI tools guide: https://developers.openai.com/api/docs/guides/tools
- OpenAI prompt caching guide: https://developers.openai.com/api/docs/guides/prompt-caching
- OpenAI reasoning guide: https://developers.openai.com/api/docs/guides/reasoning
- Ollama thinking guide: https://docs.ollama.com/capabilities/thinking
- Ollama tool calling guide: https://docs.ollama.com/capabilities/tool-calling

## Endpoint Matrix

| Surface | Status | Notes |
| --- | --- | --- |
| `POST /v1/responses` | Supported for text and function-tool protocol | Blocking, streaming, and background text generation plus client-driven function tool loops. Background mode requires `store=true`. |
| `GET /v1/responses/{response_id}` | Supported | Stored responses only. |
| `DELETE /v1/responses/{response_id}` | Supported | Soft delete. |
| `GET /v1/responses/{response_id}/input_items` | Supported | Reads first-class stored input items for new responses, including `function_call_output` items, with fallback for pre-migration records. Supports `after`, `before`, `limit`, and `order`. |
| `GET /v1/responses/{response_id}/artifacts` | Supported locally | Lists tenant-scoped local response artifact metadata. Supports `after`, `before`, `limit`, and `order`. |
| `GET /v1/responses/{response_id}/artifacts/{artifact_id}/content` | Supported locally | Downloads tenant-scoped text content for local response artifacts created from `input_file` normalization. |
| `POST /v1/responses/input_tokens` | Supported, estimated | Deterministic model-window-aware local estimate using the context planner, including current local prompt-cache hit details. Backend-reported usage remains authoritative after generation. |
| `POST /v1/responses/{response_id}/cancel` | Supported for background responses | Cancels queued or in-flight background jobs in the current single Respawn instance. Non-background responses return an OpenAI-shaped invalid request. |
| `POST /v1/responses/compact` | Supported locally | Stateless input-window compaction. Returns a `response.compaction` object with retained tail items plus an opaque local `compaction` item. Stored response-chain compaction is handled during `POST /v1/responses` through `context_management`. |
| `POST /v1/responses/prompts` | Supported locally | Respawn-local API-managed prompt template creation. Templates store `instructions` and/or `input` with `{{variable}}` placeholders and tenant scope. |
| `GET /v1/responses/prompts`, `GET /v1/responses/prompts/{id}` | Supported locally | Lists or retrieves local prompt templates. `version` query selects an explicit version; omitted version resolves to the latest active local version. |
| `DELETE /v1/responses/prompts/{id}` | Supported locally | Soft-deletes one local prompt template version, or the latest active version when `version` is omitted. |
| `DELETE /v1/responses/prompt_cache` | Supported locally | Clears the in-process prompt cache for the current Respawn process. Optional `prompt_cache_key` limits invalidation to one cache key. |
| `POST /v1/files` | Supported locally | Local Files API subset for uploaded platform files. Accepts multipart uploads with `file`, `purpose`, and optional OpenAI-shaped `expires_after`. |
| `GET /v1/files`, `GET /v1/files/{file_id}` | Supported locally | Lists and retrieves active tenant-scoped local files. Supports `purpose`, `after`, `limit`, and `order` for list, with `first_id`, `last_id`, and `has_more` in the response. |
| `GET /v1/files/{file_id}/content` | Supported locally | Downloads uploaded file bytes from the configured local storage backend. |
| `DELETE /v1/files/{file_id}` | Supported locally | Soft-deletes the local file object and removes stored bytes from the configured backend. |
| `POST /v1/chat/completions` | Supported | Compatibility wrapper around the configured backend. |
| `GET /v1/models` | Supported | Proxies configured backend model list. |

## Deliberate Local Exclusions

Respawn does not implement the OpenAI Conversations API. The
`/v1/conversations` endpoints, local Conversation objects, and the Responses
`conversation` request field are outside the product surface. They are not
planned as compatibility phases for the current 100% Responses target.

Local continuity is modeled only through stored Responses and
`previous_response_id`.

Respawn also does not target multi-deployment orchestration. The current product
model is one Respawn instance connected to one configured backend. Dynamic
backend routing, distributed prompt caches, shared workers across instances, and
multi-replica consistency semantics are outside the current compatibility
target.

Respawn supports the Responses function tool calling protocol, but it does not
execute tools itself. Function tools are protocol data: the model may emit
`function_call`, the client executes the function, and the client submits
`function_call_output` back to Respawn. Hosted tools, local workspace tools,
MCP hosting, shell, filesystem, git, `apply_patch`, browser, web/file/code/
computer/image tools, and other Respawn-owned tool execution remain outside the
product surface.

## Request Fields

| Field | Status | Notes |
| --- | --- | --- |
| `model` | Supported | Defaults to `DEFAULT_MODEL`. |
| `input` string | Supported | Treated as a user message. |
| `input` text message list | Supported | `message`, roles, `input_text`, and `output_text`. |
| `instructions` | Supported | Mapped to a system message. |
| `previous_response_id` | Supported | Loads stored local response chain. |
| `store` | Supported | `false` responses are not retrievable. |
| `stream` | Supported | SSE text lifecycle events. |
| `stream_options.include_obfuscation` | Supported locally | Validated for streaming requests. Respawn includes an `obfuscation` string on text delta events by default and suppresses it when `include_obfuscation=false`. |
| `background` | Supported | Creates a pollable background response with persisted job state. Requires `store=true`; `background=true` plus `stream=true` returns explicit `unsupported_parameter` until streaming replay/resume is implemented. |
| `temperature`, `top_p` | Supported | Forwarded to backend when applicable. |
| `max_output_tokens` | Supported | Mapped to backend max generation tokens. |
| `tools` | Supported for function tools | Accepts `type=function` definitions, validates names and JSON Schema parameters, maps to Ollama when possible, and never executes functions locally. Built-in/internal tool categories return explicit unsupported errors. |
| `max_tool_calls` | Supported locally for function tools | Validated before generation and enforced against backend output; unsupported backend/model behavior returns an explicit capability error. |
| `tool_choice` | Supported locally for function tools | Supports `auto`, `none`, `required`, forced function choice, and allowed function tool choices where backend-capable. Required/forced choices return a capability error if the model does not emit the requested `function_call`. |
| `parallel_tool_calls` | Supported locally for function tools | Accepted and enforced around backend output. `parallel_tool_calls=false` fails explicitly if the backend emits multiple calls. |
| `response_format`, `text.format` | Supported | `text.format` is preferred. Legacy `response_format` remains accepted and normalized into the response `text` shape. |
| `metadata` | Supported | Stored with responses. Limited to 16 string key-value pairs. |
| `include` | Partially supported | Supports `reasoning.encrypted_content`, `message.input_image.image_url`, and backend-capable `message.output_text.logprobs`. Hosted-tool include values such as `file_search_call.results`, `web_search_call.results`, and `code_interpreter_call.outputs` return explicit `unsupported_parameter`. |
| `context_management` | Supported locally for compaction | Accepts list entries with `type=compaction` and optional `compact_threshold >= 1000`. Unsupported context-management entry types or fields return explicit `unsupported_parameter`. |
| `prompt` | Supported locally | Supports the current `prompt={id, version, variables}` request shape against Respawn-local API-managed templates. Templates render before multimodal preparation, context planning, storage, and backend calls. |
| `prompt_cache_key`, `prompt_cache_retention` | Supported locally | In-process prefix cache accounting with `in_memory` or `24h` retention. Does not reuse backend KV tensors. Cache entries expire by TTL, are lost on process restart, and can be cleared with `DELETE /v1/responses/prompt_cache`. |
| `reasoning` | Supported for local text reasoning | Supports `effort` values `none`, `minimal`, `low`, `medium`, `high`; `xhigh` is accepted only for models configured with `reasoning-effort-xhigh` and maps to the closest backend setting when needed. Maps to Ollama `think` when using the Ollama backend. Supports `summary` values `auto`, `concise`, `detailed`. |
| `truncation=disabled` | Supported | Default. |
| `truncation=auto` | Supported locally | Drops earliest stored response-chain items, then earliest input list items when needed, before calling the backend. Truncation events are persisted for stored responses. |
| `service_tier` | Supported locally | Accepted and round-tripped as local metadata. Does not change local scheduling. |
| `safety_identifier` | Supported locally | Accepted and round-tripped for SDK shape compatibility. |
| `top_logprobs` | Backend capability dependent | Accepted only when the configured model is marked with `logprobs` and the backend returns token logprobs. Default Ollama/gpt-oss configuration returns `unsupported_model_capability`. |
| `user` | Not supported | Explicit `unsupported_parameter`; use `safety_identifier` or `prompt_cache_key`. |

Unknown request fields are rejected instead of being silently ignored.

## SDK Contract, Headers, And Errors

Respawn is tested against the official OpenAI Python SDK for Responses create,
retrieve, delete, stream, input item listing, background create/cancel/retrieve,
function-call follow-ups, and the local Files API subset. The repository does
not currently include Node test tooling, so Node SDK contract tests are not part
of the automated gate yet.

Every HTTP response includes `x-request-id`. If the client supplies
`x-request-id`, Respawn echoes it; otherwise it generates a local `req_...`
value. The Python SDK exposes this through raw response headers and parsed
object `_request_id` fields where the SDK supports it.

`Idempotency-Key` is supported for POST requests with a single-instance
in-memory replay cache. Reusing the same key with the same body returns the
original response body; reusing the key with a different body returns a
`409` OpenAI-shaped error with `code=idempotency_conflict`. This cache is local
process state and is not durable across Respawn restarts.

Errors use the OpenAI-shaped envelope:

```json
{"error":{"message":"...","type":"invalid_request_error","param":"...","code":"..."}}
```

Validation errors return `422`, unsupported parameters return `400`, missing
objects return `404`, idempotency conflicts return `409`, backend/server
failures return `5xx`, and local backend timeouts return `504`. The Python SDK
contract tests verify the expected exception classes for these families,
including `BadRequestError`, `NotFoundError`, `ConflictError`,
`UnprocessableEntityError`, and `InternalServerError`.

## Local Prompt Templates And Cache

Respawn supports the Responses `prompt` request field through local templates
managed by `POST /v1/responses/prompts`. This is not OpenAI-hosted prompt
management: templates live in the configured Respawn database, are scoped by the
same tenant as responses, and render to local `instructions` and/or `input`
before backend calls. The supported prompt reference shape is `id`, optional
`version`, and optional `variables`. Template placeholders use `{{name}}`.
Missing templates return `not_found`; missing variables return
`missing_prompt_variable`.

Prompt cache accounting is local single-process state. `in_memory` entries use
`PROMPT_CACHE_IN_MEMORY_TTL_SECONDS`, `24h` entries use
`PROMPT_CACHE_EXTENDED_TTL_SECONDS`, and both are bounded by
`PROMPT_CACHE_MAX_ENTRIES`. Cache entries are not persisted, are empty after a
Respawn restart, and report usage details only through Respawn's deterministic
prefix accounting. They do not skip or reuse backend KV tensors. Use
`DELETE /v1/responses/prompt_cache` to clear all entries, or pass
`prompt_cache_key` to clear one key.

Context windows are configured locally with `CONTEXT_WINDOW_DEFAULT_TOKENS`,
`MODEL_CONTEXT_WINDOWS`, and `CONTEXT_TOKEN_MARGIN`. Token counts use Respawn's
deterministic local tokenizer because the configured Ollama backend does not
currently expose exact per-request tokenizer or KV-cache telemetry. That means
`input_tokens` and context planning are stable and model-window-aware, but still
documented estimates.

## Local Files And Artifact Storage

Respawn implements the local Files API subset needed by Responses `file_id`
inputs. Uploaded files are stored as tenant-scoped `file` objects with
OpenAI-shaped fields such as `id`, `object`, `bytes`, `created_at`,
`filename`, `purpose`, and optional `expires_at`.

The storage backend is controlled by `FILE_STORAGE_BACKEND`. The default
`database` backend stores blobs in the Respawn database. The `filesystem`
backend stores bytes under `FILE_STORAGE_PATH`, defaulting to `./data/files`
relative to the gateway process. The root `.gitignore` excludes
`apps/gateway/data/` so local file bytes are not committed accidentally.

Uploads are bounded by `FILE_UPLOAD_MAX_BYTES` and per-tenant active storage is
bounded by `FILE_STORAGE_QUOTA_BYTES`. `FILE_DEFAULT_TTL_SECONDS` can expire
new uploads automatically, and the background cleanup task runs every
`FILE_CLEANUP_INTERVAL_SECONDS`. The local malware hook rejects the EICAR test
signature when `FILE_MALWARE_SCAN_ENABLED=true`; broader scanning is a
deployment integration point, not a hosted Respawn service.

Response artifact records remain separate from platform file objects. They keep
safe tenant-scoped metadata for normalized `input_image` and `input_file` parts,
while `GET /v1/responses/{response_id}/artifacts/{artifact_id}/content`
downloads only the text content Respawn extracted for local file citations.

## Input And Output Types

| Type | Status | Notes |
| --- | --- | --- |
| Text input | Supported | String or text content parts. |
| `input_text` | Supported | Text-only compatibility. |
| `input_image` | Supported when model-capable | URL and data URL/base64 content parts are normalized and mapped to Ollama `images` arrays for models configured with `vision`. Text-only models return `unsupported_model_capability`. |
| `input_file` | Supported locally | URL, data URL/base64, and local Files API `file_id` parts are resolved, size checked, text-extracted, stored, and replayed as local text context. Supports text, Markdown, JSON, CSV/TSV, code files, and PDF text extraction. |
| `input_audio` | Not supported | Deliberate local exclusion for now. Audio is not part of Phase 8 image/file work and should remain an explicit error unless a future audio/realtime/transcription phase is added. |
| Function call protocol items | Supported | Respawn emits `function_call` output items, accepts client-supplied `function_call_output`, validates matching `call_id`, stores items canonically, and replays them through `previous_response_id`. Legacy `tool_result` remains unsupported. |
| Reasoning input item | Supported | Round-tripped as an input item, including opaque `encrypted_content` when supplied, and ignored when rebuilding chat messages. |
| Compaction input item | Supported locally | Accepted as an opaque `compaction` item with `encrypted_content`. Respawn can decode local envelopes and replay them as compacted system context; invalid or foreign envelopes remain opaque. |
| Assistant message output | Supported | `message` with `output_text`. |
| Output text file annotations | Supported locally | Responses that use local `input_file` extraction include OpenAI-shaped `file_citation` annotations pointing at tenant-scoped local artifact IDs. |
| Function call output item | Supported | Output uses Responses `function_call` items. Client execution results come back as input-side `function_call_output` items. |
| Reasoning output item | Supported | Returned when `reasoning` is requested. Summary text is high-level local metadata, not raw chain-of-thought. `encrypted_content` is emitted only when requested via `include`. |
| Compaction output item | Supported locally | `POST /v1/responses/compact` and `context_management` can emit `type=compaction` items with opaque `encrypted_content`. |
| Built-in tool call items | Not supported | Web/file/code/computer/image/MCP/shell/`apply_patch`/workspace tools remain outside Respawn-hosted execution. |

## Response Object

| Field | Status | Notes |
| --- | --- | --- |
| `id`, `object`, `created_at`, `status`, `model` | Supported | OpenAI-shaped core fields. |
| `response.compaction` object | Supported locally | `POST /v1/responses/compact` returns `object=response.compaction`, an `output` item list, and local token usage. |
| `input`, `instructions`, `previous_response_id` | Supported | Created and retrieved responses expose persisted request context where available. |
| `background`, `max_output_tokens`, `service_tier`, `store`, `temperature`, `text`, `top_p`, `truncation` | Supported | Request settings are included in create and retrieve responses. |
| `tools`, `tool_choice`, `max_tool_calls`, `parallel_tool_calls` | Supported for function-tool protocol | Round-tripped in create/retrieve where applicable. Function tools are protocol data only, not executable Respawn capabilities. |
| `output` | Supported | Message, reasoning, function-call, and compaction item list. New stored responses retrieve output from first-class response item storage. |
| Output text content `annotations`, `logprobs` | Supported / capability dependent | `annotations` are populated for local file citations. `logprobs` are populated only for backend-capable non-streaming requests with `include=["message.output_text.logprobs"]`; otherwise the array is empty or the request fails with `unsupported_model_capability`. |
| `output_text` | Supported | Convenience concatenation of output text parts. |
| `usage.input_tokens`, `output_tokens`, `total_tokens` | Supported | Backend-reported after generation. |
| `usage.input_tokens_details.cached_tokens` | Supported | Local prefix-cache hit count, clamped to reported input tokens. |
| `usage.output_tokens_details.reasoning_tokens` | Supported | Backend-reported value when present, otherwise local estimate from backend reasoning text. |
| `metadata` | Supported | Echoes stored metadata. |
| `error` | Supported | OpenAI-shaped errors. |
| `incomplete_details` | Supported locally | Set when the backend/gateway can detect max-output exhaustion; otherwise remains `null`. |
| Background statuses | Supported | `queued`, `in_progress`, `completed`, `failed`, `cancelled`, and `incomplete` are persisted for background jobs where applicable. |

## State And Item Storage

| Surface | Status | Notes |
| --- | --- | --- |
| Input item records | Supported | Stored at response create time with stable item IDs and original input order for text, reasoning, and function-tool protocol items. |
| Output item records | Supported | Stored as first-class response items. Streaming stores `in_progress` items and updates them to terminal status. |
| Input item pagination | Supported | Supports `after`, `before`, `limit`, `order`, `first_id`, `last_id`, and `has_more`. |
| Files pagination | Supported locally | `GET /v1/files` supports SDK cursor pagination with `after`, `limit`, `order`, `first_id`, `last_id`, and `has_more`. |
| `store=false` item visibility | Supported | `store=false` responses are not retrievable and do not expose input item lists. |
| Input item tenant isolation | Supported | Stored input item lists are filtered by the owning response tenant. |
| Function tool protocol item storage | Supported | New `function_call` output items and `function_call_output` input items are stored as first-class response item records. |
| Function tool protocol replay | Supported | Stored `function_call` and `function_call_output` turns are replayed through `previous_response_id` as backend tool-call/tool-output messages. |
| Reasoning item storage and carryover | Supported locally | Reasoning summary and opaque encrypted-content fields are stored as first-class response items. Stored chains preserve reasoning items for retrieve/manual continuation, while backend replay still hides raw thinking from chat messages. |
| Pre-migration backfill/fallback | Supported | Migration `0002_response_item_state` backfills text/reasoning items where possible; records still missing item rows can fall back to legacy JSON columns. Legacy tool-call items are not converted. |
| Background job records | Supported | Stores one job per background response with attempts, timeout, heartbeat, cancellation request time, completion time, and error payload. Jobs are local to one Respawn instance and one configured backend. |
| Background polling | Supported | `GET /v1/responses/{response_id}` returns queued/in-progress/terminal state for stored background responses. |
| Background cancellation | Supported locally | `POST /v1/responses/{response_id}/cancel` marks queued jobs cancelled immediately and cancels in-flight local tasks best-effort. |
| Background timeout | Supported locally | Jobs exceeding `BACKGROUND_JOB_TIMEOUT_SECONDS` are marked failed with `code=background_timeout`. |
| Context event records | Supported locally | Compaction and truncation events store source response ID, strategy, source item IDs, token estimates before/after, and compacted item ID when present. |
| Compaction provenance | Supported locally | Local compaction envelopes include summary text and source item IDs. Envelopes are authenticated with the local reasoning encryption key and stored as opaque `encrypted_content` items. |
| Response artifact records | Supported locally | Stores tenant-scoped metadata for normalized `input_image` and `input_file` parts. Include expansions expose safe metadata and redact data URLs/base64 instead of duplicating large binary payloads. |
| Response artifact pagination | Supported locally | `GET /v1/responses/{response_id}/artifacts` lists safe artifact metadata with `after`, `before`, `limit`, `order`, `first_id`, `last_id`, and `has_more`. |
| Platform file records | Supported locally | Stores tenant-scoped local file metadata and bytes for the Files API subset. Active file objects are listable/retrievable by the owning tenant only; deleted or expired files cannot be used by `input_file.file_id`. |

## Streaming Events

| Event | Status |
| --- | --- |
| SSE `id` and event `sequence_number` | Supported locally |
| `response.created` | Supported |
| `response.in_progress` | Supported |
| `response.output_item.added` | Supported for text message, reasoning, function-call, and compaction output |
| `response.content_part.added` | Supported |
| `response.output_text.delta` | Supported, with optional `obfuscation` |
| `response.output_text.done` | Supported |
| `response.content_part.done` | Supported |
| `response.output_item.done` | Supported |
| `response.reasoning_summary_part.added` | Supported when `reasoning.summary` is requested |
| `response.reasoning_summary_text.delta` | Supported when `reasoning.summary` is requested |
| `response.reasoning_summary_text.done` | Supported when `reasoning.summary` is requested |
| `response.reasoning_summary_part.done` | Supported when `reasoning.summary` is requested |
| `response.function_call_arguments.delta` | Supported for streamed function-call arguments |
| `response.function_call_arguments.done` | Supported for streamed function-call arguments |
| `response.completed` | Supported |
| `response.incomplete` | Supported locally when max-output exhaustion is detectable |
| `response.failed`, `error` | Supported for failures during streaming |
| `response.cancelled` | Not part of the current Responses streaming event reference; client disconnects are persisted as `cancelled` best-effort for stored streams but cannot deliver a terminal event after disconnect. |
| Built-in/internal tool events | Not supported |
| Background streaming replay/resume / `starting_after` | Not supported; no current `starting_after` create parameter is documented for Responses streaming. |

## Background Jobs

Respawn background mode is intentionally single-instance: one Respawn process owns
the in-process worker tasks and talks to one configured backend. Job records are
durable enough for local restart/resume inside that instance, but Respawn does
not currently promise distributed workers, cross-instance leasing, or shared job
coordination.

`background=true` creates a stored response in `queued` state and returns
quickly. Clients poll `GET /v1/responses/{response_id}` until the response is
terminal. `store=false` is rejected for background requests because there would
be no pollable response object. Cancellation is best-effort once an Ollama
request is already in flight: Respawn persists `cancelled` locally and cancels
the gateway task, but it cannot guarantee that the backend process stops work
immediately.

Background metrics are exposed as:

- `gateway_background_jobs_total`
- `gateway_background_jobs_running`
- `gateway_background_job_latency_seconds`

Reasoning metrics are exposed as:

- `gateway_reasoning_requests_total`
- `gateway_reasoning_tokens_total`
- `gateway_reasoning_heavy_requests_total`

Context-management metrics are exposed as:

- `gateway_context_compactions_total`
- `gateway_context_compaction_tokens_total`
- `gateway_context_compaction_ratio`
- `gateway_context_truncations_total`
- `gateway_context_overflows_total`

Include-expansion metrics are exposed as:

- `gateway_response_include_expansions_total`
- `gateway_response_include_expansion_bytes_total`
- `gateway_response_include_capability_errors_total`

## Operations And Certification

Respawn exposes `GET /readyz` as a dependency readiness check for database,
Ollama/backend model availability, background worker state, prompt cache state,
and local file storage. The endpoint returns `503` when any component is not
ready and emits `gateway_readiness_check` plus
`gateway_readiness_check_latency_seconds`.

HTTP completion logs are structured JSON and include request ID, response ID
when available, tenant, feature family, backend, latency, status, and error
code. Logs avoid API keys and full payloads.

Additional operational metrics are exposed as:

- `gateway_endpoint_requests_total`
- `gateway_feature_requests_total`
- `gateway_idempotency_requests_total`
- `gateway_streaming_responses_running`
- `gateway_backend_model_requests_total`
- `gateway_backend_model_info`
- `gateway_backend_eval_tokens_total`
- `gateway_backend_eval_duration_seconds_total`
- `gateway_backend_eval_tokens_per_second`
- `gateway_storage_operations_total`
- `gateway_operational_failures_total`

The Compose Grafana dashboard groups panels into collapsible overview, model
API, backend, reliability, runtime, and feature-subsystem rows. It has
`$llm_backend` and `$model` variables backed by `gateway_backend_model_info`, so
normal model panels select advertised backend models instead of synthetic
benchmark error labels such as `respawn-missing-model-*`. Ollama-specific
`gateway_ollama_eval_*` metrics are kept as adapter-native debug telemetry,
while the primary throughput panels use the backend-generic
`gateway_backend_eval_*` metrics so future backends can share the same panels.

Release certification uses the benchmark manifest coverage gate plus the Phase
15 ops cases: `metrics.full_surface`, `ops.ollama_unavailable`,
`ops.concurrent_streaming`, `ops.concurrent_background`, and
`benchmark.history_compare`. Benchmark reports can compare against a previous
JSON report through `RESPAWN_BENCHMARK_COMPARE_TO`.

## Compatibility Rule

When a feature is not supported, Respawn should return an explicit OpenAI-shaped
error with `code=unsupported_parameter` instead of silently ignoring the field.
New features must update this matrix, the automated tests, and the Docker
benchmark suite.
