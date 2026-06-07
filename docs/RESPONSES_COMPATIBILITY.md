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
| `POST /v1/responses/input_tokens` | Supported, estimated | Deterministic local prompt-surface token estimate, including current local prompt-cache hit details. Backend-reported usage remains authoritative after generation. |
| `POST /v1/responses/{response_id}/cancel` | Supported for background responses | Cancels queued or in-flight background jobs in the current single Respawn instance. Non-background responses return an OpenAI-shaped invalid request. |
| `POST /v1/responses/compact` | Not supported | Needs compaction over stored response chains first. |
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
| `include` | Not supported | Explicit `unsupported_parameter`. |
| `context_management` | Not supported | Explicit `unsupported_parameter`. |
| `prompt` | Not supported | Hosted prompt templates are not local yet. |
| `prompt_cache_key`, `prompt_cache_retention` | Supported locally | In-process prefix cache accounting with `in_memory` or `24h` retention. Does not reuse backend KV tensors. |
| `reasoning` | Supported for local text reasoning | Supports `effort` values `none`, `minimal`, `low`, `medium`, `high`; maps to Ollama `think` when using the Ollama backend. Supports `summary` values `auto`, `concise`, `detailed`. |
| `truncation=disabled` | Supported | Default. |
| `truncation=auto` | Not supported | Explicit `unsupported_parameter`. |
| `service_tier` | Supported locally | Accepted and round-tripped as local metadata. Does not change local scheduling. |
| `safety_identifier` | Supported locally | Accepted and round-tripped for SDK shape compatibility. |
| `top_logprobs`, `user` | Not supported | Explicit `unsupported_parameter`. |

Unknown request fields are rejected instead of being silently ignored.

## Input And Output Types

| Type | Status | Notes |
| --- | --- | --- |
| Text input | Supported | String or text content parts. |
| `input_text` | Supported | Text-only compatibility. |
| `input_image` | Supported when model-capable | URL and data URL/base64 content parts are normalized and mapped to Ollama `images` arrays for models configured with `vision`. Text-only models return `unsupported_model_capability`. |
| `input_file` | Supported locally | URL and data URL/base64 file parts are downloaded/decoded, size checked, text-extracted, stored, and replayed as local text context. Supports text, Markdown, JSON, CSV/TSV, code files, and PDF text extraction. `file_id` waits for the local Files API phase. |
| `input_audio` | Not supported | Deliberate local exclusion for now. Audio is not part of Phase 8 image/file work and should remain an explicit error unless a future audio/realtime/transcription phase is added. |
| Function call protocol items | Supported | Respawn emits `function_call` output items, accepts client-supplied `function_call_output`, validates matching `call_id`, stores items canonically, and replays them through `previous_response_id`. Legacy `tool_result` remains unsupported. |
| Reasoning input item | Supported | Round-tripped as an input item and ignored when rebuilding chat messages. |
| Assistant message output | Supported | `message` with `output_text`. |
| Function call output item | Supported | Output uses Responses `function_call` items. Client execution results come back as input-side `function_call_output` items. |
| Reasoning output item | Supported | Returned when `reasoning` is requested. Summary text is high-level local metadata, not raw chain-of-thought. |
| Built-in tool call items | Not supported | Web/file/code/computer/image/MCP/shell/`apply_patch`/workspace tools remain outside Respawn-hosted execution. |

## Response Object

| Field | Status | Notes |
| --- | --- | --- |
| `id`, `object`, `created_at`, `status`, `model` | Supported | OpenAI-shaped core fields. |
| `input`, `instructions`, `previous_response_id` | Supported | Created and retrieved responses expose persisted request context where available. |
| `background`, `max_output_tokens`, `service_tier`, `store`, `temperature`, `text`, `top_p`, `truncation` | Supported | Request settings are included in create and retrieve responses. |
| `tools`, `tool_choice`, `max_tool_calls`, `parallel_tool_calls` | Supported for function-tool protocol | Round-tripped in create/retrieve where applicable. Function tools are protocol data only, not executable Respawn capabilities. |
| `output` | Supported | Message and reasoning item list. New stored responses retrieve output from first-class response item storage. |
| Output text content `annotations`, `logprobs` | Supported | Present as empty arrays when no backend details exist. |
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
| `store=false` item visibility | Supported | `store=false` responses are not retrievable and do not expose input item lists. |
| Input item tenant isolation | Supported | Stored input item lists are filtered by the owning response tenant. |
| Function tool protocol item storage | Supported | New `function_call` output items and `function_call_output` input items are stored as first-class response item records. |
| Function tool protocol replay | Supported | Stored `function_call` and `function_call_output` turns are replayed through `previous_response_id` as backend tool-call/tool-output messages. |
| Pre-migration backfill/fallback | Supported | Migration `0002_response_item_state` backfills text/reasoning items where possible; records still missing item rows can fall back to legacy JSON columns. Legacy tool-call items are not converted. |
| Background job records | Supported | Stores one job per background response with attempts, timeout, heartbeat, cancellation request time, completion time, and error payload. Jobs are local to one Respawn instance and one configured backend. |
| Background polling | Supported | `GET /v1/responses/{response_id}` returns queued/in-progress/terminal state for stored background responses. |
| Background cancellation | Supported locally | `POST /v1/responses/{response_id}/cancel` marks queued jobs cancelled immediately and cancels in-flight local tasks best-effort. |
| Background timeout | Supported locally | Jobs exceeding `BACKGROUND_JOB_TIMEOUT_SECONDS` are marked failed with `code=background_timeout`. |

## Streaming Events

| Event | Status |
| --- | --- |
| SSE `id` and event `sequence_number` | Supported locally |
| `response.created` | Supported |
| `response.in_progress` | Supported |
| `response.output_item.added` | Supported for text message, reasoning, and function-call output |
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

## Compatibility Rule

When a feature is not supported, Respawn should return an explicit OpenAI-shaped
error with `code=unsupported_parameter` instead of silently ignoring the field.
New features must update this matrix, the automated tests, and the Docker
benchmark suite.
