# Responses Compatibility Matrix

Respawn aims to be a local, OpenAI-shaped Responses API gateway. This matrix is
the source of truth for the supported surface. Keep it in sync with code, tests,
and `infra/docker/benchmark/respawn_benchmark.py`.

References:

- OpenAI Responses API reference: https://developers.openai.com/api/reference/resources/responses/methods/create
- OpenAI Responses migration guide: https://developers.openai.com/api/docs/guides/migrate-to-responses
- OpenAI tools guide: https://developers.openai.com/api/docs/guides/tools
- OpenAI prompt caching guide: https://developers.openai.com/api/docs/guides/prompt-caching
- OpenAI reasoning guide: https://developers.openai.com/api/docs/guides/reasoning
- Ollama thinking guide: https://docs.ollama.com/capabilities/thinking

## Endpoint Matrix

| Surface | Status | Notes |
| --- | --- | --- |
| `POST /v1/responses` | Supported, text-only | Blocking and streaming text generation. |
| `GET /v1/responses/{response_id}` | Supported | Stored responses only. |
| `DELETE /v1/responses/{response_id}` | Supported | Soft delete. |
| `GET /v1/responses/{response_id}/input_items` | Supported, text-only | Reconstructs stored request input items. |
| `POST /v1/responses/input_tokens` | Supported, estimated | Deterministic local prompt-surface token estimate, including current local prompt-cache hit details. Backend-reported usage remains authoritative after generation. |
| `POST /v1/responses/{response_id}/cancel` | Not supported | Needs background jobs first. |
| `POST /v1/responses/compact` | Not supported | Needs compaction/conversation model first. |
| `POST /v1/chat/completions` | Supported | Compatibility wrapper around the configured backend. |
| `GET /v1/models` | Supported | Proxies configured backend model list. |

## Request Fields

| Field | Status | Notes |
| --- | --- | --- |
| `model` | Supported | Defaults to `DEFAULT_MODEL`. |
| `input` string | Supported | Treated as a user message. |
| `input` text message list | Supported | `message`, roles, `input_text`, `output_text`, and function call/result items. |
| `instructions` | Supported | Mapped to a system message. |
| `previous_response_id` | Supported | Loads stored local response chain. |
| `store` | Supported | `false` responses are not retrievable. |
| `stream` | Supported | SSE text lifecycle events. |
| `temperature`, `top_p` | Supported | Forwarded to backend when applicable. |
| `max_output_tokens` | Supported | Mapped to backend max generation tokens. |
| `tools` with `type=function` | Supported | Local function-tool loop. |
| `max_tool_calls` | Supported | Per-request cap for local tool iterations. |
| `tool_choice=auto` | Supported | Other values return an explicit unsupported error. |
| `response_format`, `text.format` | Supported | Text/JSON object/JSON schema structured output. |
| `metadata` | Supported | Stored with responses. |
| `background` | Not supported | Explicit `unsupported_parameter`. |
| `conversation` | Not supported | Use `previous_response_id` for now. |
| `include` | Not supported | Explicit `unsupported_parameter`. |
| `context_management` | Not supported | Explicit `unsupported_parameter`. |
| `parallel_tool_calls` | Not supported | Explicit `unsupported_parameter`. |
| `prompt` | Not supported | Hosted prompt templates are not local yet. |
| `prompt_cache_key`, `prompt_cache_retention` | Supported locally | In-process prefix cache accounting with `in_memory` or `24h` retention. Does not reuse backend KV tensors. |
| `reasoning` | Supported for local text reasoning | Supports `effort` values `none`, `minimal`, `low`, `medium`, `high`; maps to Ollama `think` when using the Ollama backend. Supports `summary` values `auto`, `concise`, `detailed`. |
| `truncation=disabled` | Supported | Default. |
| `truncation=auto` | Not supported | Explicit `unsupported_parameter`. |
| `service_tier`, `top_logprobs`, `user` | Not supported | Explicit `unsupported_parameter`. |

Unknown request fields are rejected instead of being silently ignored.

## Input And Output Types

| Type | Status | Notes |
| --- | --- | --- |
| Text input | Supported | String or text content parts. |
| `input_text` | Supported | Text-only compatibility. |
| `input_image` | Not supported | Explicit text-only error. |
| `input_file` | Not supported | Explicit text-only error. |
| `input_audio` | Not supported | Explicit text-only error. |
| Function call output | Supported | For local tool loop and manual continuation. |
| Reasoning input item | Supported | Round-tripped as an input item and ignored when rebuilding chat messages. |
| Assistant message output | Supported | `message` with `output_text`. |
| Function call output item | Supported | Local function calls are returned as `function_call`. |
| Reasoning output item | Supported | Returned when `reasoning` is requested. Summary text is high-level local metadata, not raw chain-of-thought. |
| Built-in tool call items | Not supported | Web/file/code/computer/image tools are future work. |

## Response Object

| Field | Status | Notes |
| --- | --- | --- |
| `id`, `object`, `created_at`, `status`, `model` | Supported | OpenAI-shaped core fields. |
| `output` | Supported | Message/function-call item list. |
| `output_text` | Supported | Convenience concatenation of output text parts. |
| `usage.input_tokens`, `output_tokens`, `total_tokens` | Supported | Backend-reported after generation. |
| `usage.input_tokens_details.cached_tokens` | Supported | Local prefix-cache hit count, clamped to reported input tokens. |
| `usage.output_tokens_details.reasoning_tokens` | Supported | Backend-reported value when present, otherwise local estimate from backend reasoning text. |
| `metadata` | Supported | Echoes stored metadata. |
| `error` | Supported | OpenAI-shaped errors. |
| `incomplete_details` | Present, currently `null` | Incomplete status is not implemented yet. |

## Streaming Events

| Event | Status |
| --- | --- |
| `response.created` | Supported |
| `response.in_progress` | Supported |
| `response.output_item.added` | Supported for text message and reasoning output |
| `response.content_part.added` | Supported |
| `response.output_text.delta` | Supported |
| `response.output_text.done` | Supported |
| `response.content_part.done` | Supported |
| `response.output_item.done` | Supported |
| `response.reasoning_summary_text.done` | Supported when `reasoning.summary` is requested |
| `response.completed` | Supported |
| `response.failed`, `error` | Supported for failures during streaming |
| Reasoning raw delta/tool/built-in tool events | Not supported yet |

## Compatibility Rule

When a feature is not supported, Respawn should return an explicit OpenAI-shaped
error with `code=unsupported_parameter` instead of silently ignoring the field.
New features must update this matrix, the automated tests, and the Docker
benchmark suite.
