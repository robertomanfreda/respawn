# Ollama Gap Probe

This note captures an exploratory HTTP probe against the local Docker stack. It
is not a replacement for `make benchmark`; it is a narrative evidence check for
why Respawn exists on top of Ollama.

Probe context:

- Date: 2026-06-05
- Model: `gpt-oss:120b`
- Ollama direct: `http://127.0.0.1:11434`
- Respawn: `http://127.0.0.1:8080`
- Stack: `infra/docker/docker-compose.yml`

## Finding

Current Ollama already exposes a basic `POST /v1/responses` compatibility
surface. Respawn is therefore not useful because Ollama has no Responses
endpoint at all. Respawn is useful because it turns that shallow model-facing
surface into an application gateway with state, retrieval, client-driven
function tool protocol handling, explicit unsupported-tool errors, contract
checks, usage details, benchmark coverage, and observability.

## Probe Results

| Capability | Ollama Direct | Respawn | Why It Matters |
| --- | --- | --- | --- |
| `POST /v1/responses` | Works | Works | Both can generate a Responses-shaped object. |
| `GET /v1/responses/{id}` | `404` | Works | Respawn stores and retrieves Responses state. |
| `GET /v1/responses/{id}/input_items` | `404` | Works | Respawn exposes inspectable input history. |
| `POST /v1/responses/input_tokens` | `404` | Works | Respawn provides preflight token accounting. |
| `previous_response_id` | Accepted | Works with stored chain | Respawn makes the chain inspectable and tenant/storage aware. |
| Structured output | Works | Works plus validation/repair | Respawn validates output against JSON Schema and repairs once. |
| Function tools | May return a `function_call` | Supports function-tool protocol, storage, replay, streaming, and client-submitted outputs | Respawn preserves protocol semantics and client-executed tool loops without running tools locally. |
| Reasoning | Returns reasoning item | Returns reasoning item and usage | Respawn summarizes locally and does not expose raw reasoning content. |
| Prompt cache details | `cached_tokens=0` on repeated prefix | Reports local cached prefix tokens | Respawn exposes cache accounting for repeated prompt prefixes. |
| `store=false` retrieval | Not a durable object model | Returns `404` as expected | Respawn makes storage semantics explicit. |
| Metrics/dashboard | Ollama-native only | Gateway + model + token metrics | Respawn can be operated and benchmarked as infrastructure. |

## Concrete Observations

Direct Ollama probe:

- `POST /v1/responses` returned `200`.
- `GET /v1/responses/{id}` returned `404`.
- `GET /v1/responses/{id}/input_items` returned `404`.
- `POST /v1/responses/input_tokens` returned `404`.
- Repeating a long prompt with `prompt_cache_key` kept
  `usage.input_tokens_details.cached_tokens` at `0`.
- A function-tool request returned output item types `reasoning` and
  `function_call`, but did not execute the tool. Phase 6 uses this as a
  protocol-compatibility target while keeping tool execution on the client side.
- A simple Responses call returned `output_text: null`; all generation budget was
  consumed by reasoning in the observed run, leaving an empty message text.
- The reasoning object included raw backend reasoning-like content in fields such
  as `summary`/`encrypted_content`.

Respawn probe:

- Stored response creation, retrieval, input item listing, and input token count
  all returned `200`.
- Structured output returned parseable JSON:

```json
{"status": "ok", "feature": "respawn"}
```

- Reasoning request returned output item types `reasoning` and `message`, with
  `usage.output_tokens_details.reasoning_tokens=6`.
- Prompt-cache probe returned `cached_tokens=0` on the cold call and
  `cached_tokens=3200` on the warm call.
- `store=false` responses were not retrievable.
- `/metrics` included gateway token usage, cached input, reasoning, Ollama eval
  throughput, and Responses request counters.

## Why Respawn Exists

Respawn is the control plane around a local model runtime:

- It provides an OpenAI-shaped API surface for clients that expect Responses
  state, response IDs, input items, token counts, and usage details.
- It supports client-driven function tool protocol loops while rejecting
  hosted/internal tool execution explicitly.
- It hardens compatibility by returning explicit OpenAI-shaped errors for
  unsupported fields.
- It adds observability with Prometheus/VictoriaMetrics/Grafana signals around
  request latency, token usage, backend throughput, and feature behavior.
- It gives the project a regression contract through `make benchmark`.

## Caveats

- Ollama's Responses compatibility may improve; rerun this probe when upgrading
  Ollama.
- Respawn prompt-cache accounting is local gateway accounting. It does not skip
  backend prefill or reuse Ollama KV tensors.
- Respawn still depends on the configured backend for model quality and native
  reasoning behavior.
