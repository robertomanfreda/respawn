# Responses Compatibility Roadmap

This document is the execution roadmap for moving Respawn from a practical
text-first Responses subset to a full OpenAI-shaped Responses gateway.
It is intentionally limited to the Responses API: OpenAI Conversations API and
Respawn-hosted tool execution are deliberate non-goals. Responses function tool
calling protocol compatibility is now in scope. Every phase must leave behind
working code, docs, automated tests, and real HTTP validation against Respawn
backed by Ollama.

Keep this document in sync with:

- [`RESPONSES_COMPATIBILITY.md`](RESPONSES_COMPATIBILITY.md), the current-state
  support matrix.
- [`FUTURE_WORK.md`](FUTURE_WORK.md), the short backlog summary.
- `infra/docker/benchmark/respawn_benchmark.py`, the real HTTP feature suite.
- `apps/gateway/tests`, the fast mock/contract test suite.

## Source References

Use the current official OpenAI documentation as the compatibility target:

- Responses API reference: https://developers.openai.com/api/reference/responses/overview
- Create response reference: https://developers.openai.com/api/reference/resources/responses/methods/create
- Responses streaming events reference: https://developers.openai.com/api/reference/resources/responses/streaming-events
- Migration guide: https://developers.openai.com/api/docs/guides/migrate-to-responses
- File inputs: https://developers.openai.com/api/docs/guides/file-inputs
- Function calling: https://developers.openai.com/api/docs/guides/function-calling
- Tools overview: https://developers.openai.com/api/docs/guides/tools
- Streaming: https://developers.openai.com/api/docs/guides/streaming
- Prompt caching: https://developers.openai.com/api/docs/guides/prompt-caching
- Reasoning: https://developers.openai.com/api/docs/guides/reasoning
- Ollama thinking: https://docs.ollama.com/capabilities/thinking
- Ollama tool calling: https://docs.ollama.com/capabilities/tool-calling
- Ollama OpenAI compatibility: https://docs.ollama.com/api/openai-compatibility

Re-check these links at the start of any phase. If OpenAI changes the surface,
update this roadmap before changing code.

## Definition Of 100%

Respawn is "100% Responses compatible" when:

- Every OpenAI Responses endpoint is implemented or deliberately mapped to a
  local equivalent with the same public shape.
- Every documented request field is accepted, validated, stored, forwarded,
  executed, or rejected with an OpenAI-shaped error for a documented reason.
- Every supported output item type, lifecycle state, usage detail, pagination
  field, and streaming event has deterministic tests.
- The official OpenAI SDK can use Respawn for create, retrieve, delete, stream,
  input item listing, input token counting, background jobs, and function tool
  call round-trips without client-side shims.
- `make benchmark` exercises the implemented surface over HTTP against the
  Docker stack with `MODEL_BACKEND=ollama`.
- The benchmark fails on behavioral regressions, not only on transport errors.
- Unsupported backend limitations are explicit in the compatibility matrix and
  never silently no-op.

Deliberate local exclusions:

- OpenAI Conversations API endpoints (`/v1/conversations/...`) are not part of
  the Respawn target surface, now or in any phase of this roadmap.
- The Responses `conversation` request field is not modeled or stored. Requests
  should continue to fail validation instead of creating local conversation
  objects.
- Respawn must not add local Conversation records, Conversation list/retrieve
  endpoints, or compatibility shims that make `/v1/conversations` appear
  partially supported.
- Stored local state uses Responses-native `previous_response_id` only.
- Respawn supports the OpenAI Responses function tool calling protocol, but it
  does not execute tools itself. Tool execution remains the responsibility of
  the client application that calls Respawn.
- Respawn must not add hosted filesystem, shell, git, `apply_patch`, workspace,
  MCP-hosting, browser, code-interpreter, web-search, file-search,
  computer-use, image-generation, or other local built-in tool execution in this
  roadmap.
- Multi-deployment Respawn is not a target. The mental model is one Respawn
  instance connected to one configured backend. Do not add dynamic backend
  routing, distributed prompt caches, shared worker pools, or multi-replica
  consistency semantics in this roadmap.

This roadmap targets API and behavior compatibility. It does not promise that a
local Ollama model will match OpenAI model quality, vision accuracy, latency,
hosted platform infrastructure, or tokenizer accounting perfectly unless the
underlying backend exposes those capabilities.

## Global Rules For Every Phase

- [ ] Re-read the official docs linked above and update the phase scope if the
  OpenAI surface changed.
- [ ] Keep new behavior behind explicit validation. Unknown fields stay rejected
  unless intentionally supported.
- [ ] Preserve tenant isolation for every stored object and every retrieval path.
- [ ] Update schemas, storage migrations, service logic, unit tests, integration
  tests, contract tests, docs, benchmark cases, and metrics together.
- [ ] Add real-Ollama benchmark coverage for every user-visible feature.
- [ ] Prefer structural assertions in real-Ollama tests; use semantic assertions
  only where the feature cannot be validated structurally.
- [ ] Do not mark a feature supported in `RESPONSES_COMPATIBILITY.md` until its
  benchmark case passes against Ollama.
- [ ] Keep benchmark prompts short, bounded with `max_output_tokens`, and cheap
  enough for repeated local runs.
- [ ] Run the fast suite before benchmark:

```bash
cd apps/gateway
.venv/bin/python -m pytest
```

- [ ] Run the real stack feature suite before merging:

```bash
cd infra/docker
make up-build
make benchmark
```

- [ ] For targeted local validation, prefer:

```bash
cd infra/docker
RESPAWN_BENCHMARK_RUNS=1 RESPAWN_BENCHMARK_MAX_OUTPUT_TOKENS=32 make benchmark
```

## Phase 0 - Compatibility Harness And Baseline Lock

Goal: make compatibility work measurable before broadening the API. This phase
should not add large user-facing features; it should make failures obvious.

Status: completed on 2026-06-07. The real-Ollama exit gate was validated with
Respawn served locally and connected to the host Ollama instance on
`127.0.0.1:11434`, because that host port was already owned by the local Ollama
daemon. The exercised benchmark suite is the same HTTP suite used by
`make benchmark`.

### Scope

- [x] Add a machine-readable compatibility manifest generated from or aligned
  with `RESPONSES_COMPATIBILITY.md`.
- [x] Add a benchmark report section that lists supported, unsupported, and
  skipped surfaces.
- [x] Add per-case tags to `respawn_benchmark.py`: `core`, `state`,
  `streaming`, `tools`, `multimodal`, `background`, `reasoning`,
  `observability`.
- [x] Add environment switches for benchmark subsets, for example
  `RESPAWN_BENCHMARK_INCLUDE_TAGS` and `RESPAWN_BENCHMARK_EXCLUDE_TAGS`.
- [x] Add a benchmark mode that fails if a feature marked supported in the
  manifest has no real-Ollama benchmark case.
- [x] Add explicit benchmark metadata: Respawn version, git SHA when available,
  backend, Ollama base URL, Ollama model, database driver, run count, timeout.
- [x] Add a doc section describing how to reproduce benchmark failures.
- [x] Add CI-friendly mock smoke mode for shape checks, but keep Ollama mode as
  the phase exit gate.

### Implementation Checklist

- [x] Create `apps/gateway/src/services/compatibility_manifest.py` or a docs
  parser if keeping the Markdown table as source of truth.
- [x] Add `/healthz` or `/readyz` metadata fields only if they do not leak
  secrets.
- [x] Extend `CaseResult` with tags, expected feature id, and optional skip
  reason.
- [x] Write benchmark JSON with all cases, tags, latency, pass/fail, and
  response snippets safe for logs.
- [x] Add a small script or benchmark function that checks matrix coverage.
- [x] Add tests for benchmark helper logic without requiring Docker.

### Real Ollama Validation

- [x] `make benchmark` still passes with the existing feature set.
- [x] The benchmark output JSON includes case tags and environment metadata.
- [x] The coverage gate fails locally when a supported manifest feature has no
  benchmark case.
- [x] The coverage gate passes again after adding the missing case.
- [x] Real requests still hit Respawn HTTP endpoints, never internal Python
  services or Ollama directly.

### DoD

- [x] Fast tests pass.
- [x] Real Ollama benchmark passes.
- [x] `RESPONSES_COMPATIBILITY.md` and benchmark coverage cannot drift silently.
- [x] Every future phase can add a benchmark case and tie it to a manifest row.

## Phase 1 - Response Object Fidelity And Request Surface Parity

Goal: make `POST /v1/responses` and the stored `response` object match the
documented OpenAI shape as closely as possible for text generation before
building more platform features.

Status: completed on 2026-06-07. The real-Ollama exit gate was validated through
the Compose stack with `RESPAWN_BENCHMARK_RUNS=0` and
`RESPAWN_BENCHMARK_MAX_OUTPUT_TOKENS=32`; the run passed all cases and covered
all manifest features marked supported for Phase 1.

Note: the Phase 1 tool-calling rejection was a temporary baseline safety rule.
It is superseded by Phase 6, which adds protocol compatibility for
client-executed function tools.

### Scope

- [x] Audit all current request fields against the official create response
  reference.
- [x] Add all top-level response fields that OpenAI returns and Respawn can
  represent locally.
- [x] Decide exact behavior for fields that are local no-ops, unsupported, or
  dependent on the configured backend.
- [x] Add stricter validation for field types, enum values, metadata limits,
  list limits, and nested shape.
- [x] Add response object fields that are currently missing but safe to expose,
  such as `parallel_tool_calls`, `previous_response_id`, `text`, `reasoning`,
  `service_tier`, `store`, `temperature`, `top_p`, and `truncation`, if they are
  part of the current OpenAI object shape.
- [x] Add `incomplete` status support for generation budget exhaustion when the
  backend or gateway can detect it.
- [x] Preserve `failed`, `completed`, and `in_progress` semantics.
- [x] Normalize output content parts to include `annotations` and `logprobs`
  fields when present or empty arrays when shape-compatible.
- [x] Make `response_format` legacy support explicit and prefer `text.format`
  as the Responses-native field.
- [x] Temporarily classify Responses tool calling as unsupported, including
  `tools`, `max_tool_calls`, explicit `tool_choice`, `parallel_tool_calls`, and
  tool-call input/output items, until Phase 6 implements protocol support.
- [x] Add explicit `unsupported_parameter` validation for every not-yet-supported
  tool-calling field instead of accepting local partial behavior.
- [x] Add `service_tier` handling as accepted local metadata if no local
  scheduling difference exists, or keep unsupported with a documented reason.
- [x] Add `truncation=auto` strategy only after Phase 10 context management, or
  keep it unsupported until then.

### Implementation Checklist

- [x] Update `ResponseRequest` schema.
- [x] Update `ResponseObject` schema.
- [x] Add response serialization tests using the official Python SDK object
  parser.
- [x] Add validation tests for every newly accepted field.
- [x] Add explicit unsupported tests for fields deferred to later phases.
- [x] Update `response_output_text` to ignore non-text content safely.
- [x] Persist enough request settings to retrieve the same response shape later.
- [x] Update OpenAPI/FastAPI response models.
- [x] Update docs examples in README if any visible shape changes.

### Real Ollama Validation

- [x] Blocking response with `text.format={"type":"text"}` returns a completed
  `response` object with stable core fields.
- [x] Blocking response with `metadata` round-trips through retrieve.
- [x] Response with `max_output_tokens=1` either completes with bounded output
  or returns `status="incomplete"` with `incomplete_details.reason` when
  detectable.
- [x] Before Phase 6, requests with `tools`, `tool_choice`,
  `parallel_tool_calls`, or `max_tool_calls` return explicit
  `unsupported_parameter` errors.
- [x] Before Phase 6, requests with `function_call`, `function_call_output`, or
  `tool_result` input items return explicit `unsupported_parameter` errors.
- [x] Request with unsupported future-only fields still returns
  `unsupported_parameter`.
- [x] Official OpenAI Python SDK can create and retrieve the new shape without
  client-side patches.

Suggested benchmark cases:

- [x] `responses.shape.blocking_text`
- [x] `responses.shape.metadata_retrieve`
- [x] `responses.shape.max_output_incomplete`
- [x] `responses.unsupported_tool_calling` as the pre-Phase-6 baseline.
- [x] `responses.unsupported_tool_items` as the pre-Phase-6 baseline.
- [x] `responses.shape.unsupported_future_field`

### DoD

- [x] Fast tests pass.
- [x] Real Ollama benchmark passes.
- [x] Retrieved responses match created response shape for persisted fields.
- [x] Every create-response field is classified in
  `RESPONSES_COMPATIBILITY.md`.
- [x] No newly accepted parameter is silently ignored.

## Phase 2 - First-Class State And Item Model

Goal: replace reconstructed request/output history with first-class input and
output items that can power listing, replay, streaming resume, and compaction.

Status: completed on 2026-06-07. The fast gateway suite passed, the mock HTTP
benchmark passed, and the real-Ollama benchmark passed with
`RESPAWN_BENCHMARK_RUNS=0` and `RESPAWN_BENCHMARK_MAX_OUTPUT_TOKENS=32`.
The benchmark stack runs with auth enabled and validates tenant isolation for
stored input item lists.

### Scope

- [x] Store input items as records at create time.
- [x] Store output items as records as they are added or completed.
- [x] Add stable item IDs for all input item types.
- [x] Preserve original order and output index.
- [x] Persist item status transitions: `in_progress`, `completed`,
  `incomplete`, `failed`.
- [x] Persist item content exactly enough to retrieve/list/replay without
  reconstructing from `request_json`.
- [x] Add item pagination that matches OpenAI list semantics: `after`, `before`
  if supported, `limit`, `order`, `first_id`, `last_id`, `has_more`.
- [x] Add migration from existing response rows where possible.
- [x] Keep old `request_json` as audit/debug context, not as canonical item
  storage.

### Implementation Checklist

- [x] Add/extend `response_items` columns: `response_id`,
  `type`, `role`, `status`, `input_index`, `output_index`, `content_json`,
  `call_id`, `name`, `arguments_json`, `output_json`, `summary_json`,
  `created_at`, `completed_at`.
- [x] Add unique indexes for response item order and call IDs.
- [x] Add repository APIs: create input item, add output item, update item
  status, list items by response.
- [x] Update `input_items_from_request` callers to use storage.
- [x] Keep compatibility fallback for pre-migration records if needed.
- [x] Add tenant filters to all item queries.
- [x] Add tests for order, pagination, stable IDs, deleted response behavior,
  and tenant isolation.

### Real Ollama Validation

- [x] Create a stored response with multi-item input and retrieve
  `/v1/responses/{id}/input_items?order=asc`.
- [x] Verify item IDs remain stable across repeated list calls.
- [x] Verify `after` pagination returns the correct next page.
- [x] Verify `store=false` still does not expose input items.
- [x] Before Phase 6, verify unsupported tool-call input/output items remain
  rejected and are not accidentally persisted.
- [x] Verify retrieved response output comes from item storage, not only
  `output_json`.

Suggested benchmark cases:

- [x] `responses.items.input_storage`
- [x] `responses.items.pagination_after`
- [x] `responses.unsupported_tool_items` as the pre-Phase-6 baseline.
- [x] `responses.items.store_false_hidden`
- [x] `responses.items.tenant_scope`

### DoD

- [x] Fast tests pass.
- [x] Real Ollama benchmark passes.
- [x] Item storage is canonical for new responses.
- [x] Existing supported APIs keep their behavior.
- [x] `RESPONSES_COMPATIBILITY.md` marks item listing as first-class, not
  reconstructed.

## Phase 4 - Background Mode, Jobs, Cancellation, And Polling

Goal: support long-running Responses requests without holding the client
connection open, and enable cancellation semantics.

Status: completed on 2026-06-07. The fast gateway suite passed, the mock HTTP
benchmark passed with deterministic timeout coverage, and the real-Ollama
benchmark passed with `RESPAWN_BENCHMARK_RUNS=0` and
`RESPAWN_BENCHMARK_MAX_OUTPUT_TOKENS=32`. Timeout failure is forced in the mock
benchmark/test suite because forcing a real Ollama model to exceed a tiny
timeout is hardware-dependent; the real-Ollama benchmark still exercises the
background timeout guard path over HTTP and validates terminal state/metrics.

### Scope

- [x] Accept `background=true` in create response requests.
- [x] Return quickly with a persisted `response` in a queued or in-progress
  state according to the current OpenAI shape.
- [x] Add durable background job records.
- [x] Add a single-instance worker execution loop in the FastAPI process.
- [x] Persist job attempts, start time, heartbeat, timeout, cancellation
  request, completed time, and error.
- [x] Implement `GET /v1/responses/{id}` polling for background responses.
- [x] Implement `POST /v1/responses/{id}/cancel`.
- [x] Define cancellation behavior when Ollama request is already in flight.
- [x] Add status transitions for `queued`, `in_progress`, `completed`,
  `failed`, `cancelled`, and `incomplete` where applicable.
- [x] Add observability metrics for queued jobs, running jobs, cancellations,
  timeout, and latency.

### Implementation Checklist

- [x] Add `BackgroundJobRecord`.
- [x] Add job repository APIs.
- [x] Add worker lifecycle to FastAPI startup/shutdown.
- [x] Make backend calls cancellable at the gateway level where possible.
- [x] Add timeout handling that marks failed/incomplete deterministically.
- [x] Ensure response retrieval never exposes partial inconsistent JSON.
- [x] Add idempotency rules for cancel after complete, cancel after failed, and
  cancel of unknown response.
- [x] Update benchmark timeout defaults for background cases.

### Real Ollama Validation

- [x] Submit `background=true` request with a prompt that takes long enough to
  poll.
- [x] Immediately retrieve and observe non-terminal status.
- [x] Poll until `completed` and verify output/usage are present.
- [x] Submit another background request and cancel it before completion.
- [x] Verify cancelled response is terminal and does not later become completed.
- [x] Verify `store=false` plus `background=true` behavior is explicitly defined
  and tested.
- [x] Verify metrics include queued/running/cancelled counts.

Suggested benchmark cases:

- [x] `responses.background.create_poll_complete`
- [x] `responses.background.cancel`
- [x] `responses.background.retrieve_terminal`
- [x] `responses.background.timeout`
- [x] `metrics.background_jobs`

### DoD

- [x] Fast tests pass.
- [x] Real Ollama benchmark passes.
- [x] `background` and `cancel` move to supported in the matrix.
- [x] Cancellation is best-effort with clear status semantics.
- [x] No background job can cross tenant boundaries.

## Phase 5 - Streaming Completeness And Replay Readiness

Goal: make streaming events cover the full item lifecycle, reasoning, failures,
incomplete states, and future resume semantics.

Status: completed on 2026-06-07. The fast gateway suite passed, the real HTTP
Docker benchmark passed against Ollama with `RESPAWN_BENCHMARK_RUNS=0` and
`RESPAWN_BENCHMARK_MAX_OUTPUT_TOKENS=32`, and the official OpenAI Python SDK
successfully parsed a live stream from the Compose Respawn instance backed by
Ollama. Current OpenAI docs do not expose a `starting_after` create parameter
for Responses streaming and do not list a `response.cancelled` stream event, so
Respawn keeps stream resume/background replay unsupported and records client
disconnect cancellation best-effort for stored streams.

### Scope

- [x] Audit current streaming events against the official streaming events
  reference.
- [x] Add event objects for every implemented output item transition.
- [x] Add `response.incomplete`, `response.cancelled`, or the current official
  terminal events if documented.
- [x] Add `response.failed` consistency for errors before and after response id
  creation.
- [x] Add sequence numbers and event IDs suitable for future resume.
- [x] Evaluate and implement `starting_after` if the current OpenAI API supports
  stream resume for the relevant endpoint.
- [x] Evaluate stream obfuscation if still part of the official event surface.

### Implementation Checklist

- [x] Create a streaming event compatibility table in docs or tests.
- [x] Add event builder tests for every event type.
- [x] Persist emitted event summaries if needed for replay/resume.
- [x] Add streaming tests for text, reasoning, failure, incomplete,
  cancellation, and background status.
- [x] Add SDK streaming contract tests.
- [x] Ensure SSE formatting stays compliant: `event:`, `data:`, final newline,
  no malformed JSON.

### Real Ollama Validation

- [x] Stream a normal text response and assert lifecycle ordering.
- [x] Stream a reasoning response and assert reasoning item events.
- [x] Stream a request that fails validation after response creation and assert
  `response.failed` plus `error`.
- [x] Stream a request that hits max output behavior and assert incomplete
  lifecycle when detectable.
- [x] Stream with official Python SDK and assert parsed event types.

Suggested benchmark cases:

- [x] `responses.stream.lifecycle_text`
- [x] `responses.stream.reasoning`
- [x] `responses.stream.failure`
- [x] `responses.stream.incomplete`
- [x] `responses.stream.sdk_parse`

### DoD

- [x] Fast tests pass.
- [x] Real Ollama benchmark passes.
- [x] Streaming matrix is complete for every feature implemented so far.
- [x] Event order is deterministic and documented.
- [x] Streamed final response matches retrieve response for stored requests.

## Phase 6 - Function Tool Calling Protocol Compatibility

Goal: support the OpenAI Responses function tool calling protocol while keeping
tool execution outside Respawn. Respawn should let models emit
`function_call` output items, let clients send matching `function_call_output`
input items, preserve and replay those items through stored response state, and
map the protocol to the configured Ollama backend when the backend/model can
produce tool calls. Respawn must not execute filesystem, shell, git,
`apply_patch`, workspace, MCP, browser, code-interpreter, web-search,
file-search, computer-use, image-generation, or any other hosted tool itself.

Status: completed on 2026-06-07. The fast gateway suite passed with 101 tests,
and the real-Ollama benchmark passed with
`RESPAWN_BENCHMARK_RUNS=0 RESPAWN_BENCHMARK_MAX_OUTPUT_TOKENS=32`. The benchmark
validated function-call emission, client-submitted outputs, manual replay,
`previous_response_id` replay, retrieve/listing, streaming argument events,
forced/limited tool choices, unsupported built-in/internal tool categories, and
function-tool metrics against Respawn HTTP backed by Ollama.

### Compatibility Contract

- [x] Treat function tools as protocol data, not executable local capabilities.
- [x] Accept `tools` entries with `type="function"` and validate `name`,
  `description`, `parameters`, and `strict` according to the current OpenAI
  create-response schema.
- [x] Accept `tool_choice` values that apply to function tools, including
  automatic selection, no-tool selection, required tool selection, and explicit
  function selection when currently documented.
- [x] Accept `parallel_tool_calls` and `max_tool_calls` when the backend can
  support the requested behavior, or return an explicit backend capability error
  when it cannot.
- [x] Produce Responses output items with `type="function_call"`, stable `id`,
  stable `call_id`, `name`, JSON string `arguments`, and item `status`.
- [x] Accept input items with `type="function_call_output"`, matching
  `call_id`, `output`, optional `id`, and item `status`.
- [x] Continue to reject legacy or non-Responses aliases such as `tool_result`
  unless the current OpenAI schema explicitly includes them.
- [x] Reject OpenAI built-in tools, MCP tools, custom free-form tools, shell,
  `apply_patch`, web/file/code/computer/image tools, and any Respawn-hosted tool
  execution with OpenAI-shaped `unsupported_parameter` or capability errors.
- [x] Do not register local tools, do not call user code, and do not add a local
  tool loop that executes functions inside Respawn.
- [x] Keep the client-driven loop: create response, client executes the
  requested function(s), client sends `function_call_output`, Respawn asks the
  backend for the next model response.

### Scope

- [x] Add request schema support for function tool definitions.
- [x] Add input item schema support for `function_call_output`.
- [x] Add output item schema support for `function_call`.
- [x] Add strict validation for tool names, duplicate tool names, JSON Schema
  parameter objects, `strict`, `tool_choice`, `parallel_tool_calls`, and
  `max_tool_calls`.
- [x] Add backend capability detection for function calling per backend/model.
- [x] Map Responses function tools to the Ollama request surface when possible.
- [x] Map Ollama/OpenAI-compatible backend tool calls back to Responses
  `function_call` items.
- [x] Map `function_call_output` items from stored Responses state into the
  backend message/tool-result shape needed for the next model call.
- [x] Preserve reasoning items returned before or alongside tool calls, because
  reasoning models may require those items to be passed back with tool outputs.
- [x] Store model `function_call` output items and client
  `function_call_output` input items as canonical item records, not only in
  `request_json`.
- [x] Store function-call protocol item statuses and transitions:
  `in_progress`, `completed`, and `incomplete` where applicable.
- [x] Preserve item order across messages, reasoning items, function calls, and
  function outputs.
- [x] Support retrieve and `GET /v1/responses/{id}/input_items` for stored tool
  protocol items.
- [x] Support `previous_response_id` replay through tool-call turns without
  silently dropping `function_call` or `function_call_output` items.
- [x] Support streaming events for function-call lifecycle and argument deltas.
- [x] Support background responses that terminate with function-call output
  items and can be polled by the client.
- [x] Add metrics for function-tool requests, emitted calls, client-submitted
  outputs, backend capability errors, and unsupported tool categories.

### Implementation Checklist

- [x] Update `ResponseRequest` validation to accept function tools and reject
  non-function tool categories.
- [x] Add `FunctionTool`, `FunctionCallOutputInputItem`, and
  `FunctionCallOutputItem` schema helpers if keeping the top-level Pydantic
  models generic is no longer enough.
- [x] Add a tool schema validator that enforces OpenAI name/length limits,
  duplicate detection, JSON Schema object shape, and `strict` behavior.
- [x] Add a tool-choice validator and serializer for `auto`, `none`,
  `required`, explicit function selection, and any current official allowed
  tools shape.
- [x] Add repository APIs or extend existing item APIs for function call fields:
  `call_id`, `name`, `arguments_json`, `output_json`, status, created time, and
  completed time.
- [x] Add database indexes/constraints for `call_id` lookup within a response
  or response chain where needed for safe lookup.
- [x] Update `response_history_builder` so stored tool protocol items replay to
  the backend in the correct order.
- [x] Update `ResponseService._output_items` to serialize backend tool calls as
  `function_call` items.
- [x] Update non-streaming create so a response can complete with function-call
  items and empty `output_text`.
- [x] Update streaming event builders for:
  `response.output_item.added`,
  `response.function_call_arguments.delta`,
  `response.function_call_arguments.done`, and
  `response.output_item.done`.
- [x] Update background worker completion and retrieval paths so function-call
  responses are stored consistently.
- [x] Add SDK parser tests for function-call responses and follow-up calls with
  `function_call_output`.
- [x] Add negative tests proving Respawn never executes local tools and rejects
  unsupported built-in/MCP/internal tool categories.
- [x] Update the compatibility manifest, README, `RESPONSES_COMPATIBILITY.md`,
  and `FUTURE_WORK.md`.
- [x] Replace unsupported-tool benchmark cases with protocol-success cases plus
  targeted unsupported-category cases.

### Backend Mapping Notes

- [x] Prefer the configured Ollama OpenAI-compatible surface when it can accept
  `tools`, `tool_choice`, `parallel_tool_calls`, and function-call outputs
  without lossy conversion.
- [x] If Respawn continues using an Ollama chat-completions adapter internally,
  map Responses function tools to chat-completions `tools` and map backend
  `tool_calls` back to Responses `function_call` items.
- [x] Use deterministic IDs generated by Respawn when the backend does not
  provide OpenAI-shaped `fc_...` or `call_...` identifiers.
- [x] Preserve backend-provided call IDs when they are stable and shape-valid.
- [x] Treat backend inability to call tools as a capability error, not as a
  silent text-only fallback.
- [x] Treat malformed backend tool-call arguments as `failed` or `incomplete`
  according to the current official response/item semantics, with an
  OpenAI-shaped error.
- [x] Document model-specific limitations because Ollama support depends on the
  selected model and backend compatibility layer.

### Real Ollama Validation

- [x] Function-tool request returns a `response` whose `output` includes at
  least one `function_call` item when the prompt and `tool_choice` require it.
- [x] The returned function call has stable `id`, `call_id`, `name`,
  JSON-string `arguments`, and terminal item `status`.
- [x] Client sends a follow-up request containing the prior response output plus
  matching `function_call_output`; Respawn forwards enough context for the model
  to produce a final assistant message.
- [x] Same follow-up flow works using `previous_response_id` instead of manually
  resending all prior output items.
- [x] Retrieve returns stored `function_call` output items exactly enough for an
  SDK client to execute the tool and submit output.
- [x] `GET /v1/responses/{id}/input_items` returns stored
  `function_call_output` input items with stable IDs and pagination.
- [x] Streaming function call emits argument delta events and a final
  `response.function_call_arguments.done` event that reconstructs the same
  arguments as retrieve.
- [x] Parallel function call request either produces multiple function calls or
  returns an explicit backend capability error.
- [x] `tool_choice` forced-function request calls the requested function or
  returns an explicit backend/model capability error.
- [x] Built-in tool requests such as web search, file search, code interpreter,
  computer use, image generation, shell, `apply_patch`, and MCP return explicit
  unsupported errors.
- [x] A prompt that asks the model to use a local filesystem/shell/git/workspace
  tool cannot cause Respawn to execute anything locally.
- [x] Metrics include function-tool request, call, output, unsupported-category,
  and capability-error counters.

Suggested benchmark cases:

- [x] `responses.tools.function_call`
- [x] `responses.tools.client_output_followup`
- [x] `responses.tools.previous_response_replay`
- [x] `responses.tools.retrieve_function_call`
- [x] `responses.tools.input_items_function_output`
- [x] `responses.tools.stream_arguments`
- [x] `responses.tools.tool_choice_forced_function`
- [x] `responses.tools.parallel_or_capability_error`
- [x] `responses.tools.unsupported_builtin_tools`
- [x] `responses.tools.no_internal_execution`
- [x] `metrics.function_tools`

### DoD

- [x] Fast tests pass.
- [x] Real Ollama benchmark passes.
- [x] Function tools move to supported or backend-dependent in the compatibility
  matrix.
- [x] Built-in, MCP, custom free-form, and internal Respawn-hosted tools remain
  explicitly unsupported unless a later product decision changes the roadmap.
- [x] No Respawn code path executes client-defined functions or local workspace
  tools.
- [x] Tool-call protocol items are first-class stored response items.
- [x] `previous_response_id` replay preserves function-call turns.
- [x] Official OpenAI Python SDK can parse function-call responses and submit
  `function_call_output` follow-ups without client-side shims.
- [x] Streaming function-call events are SDK-parseable and match retrieved
  response state.
- [x] Unsupported backend/model tool capability is explicit and never silently
  falls back to text-only behavior.

## Phase 8 - Multimodal Inputs And Files

Goal: support Responses input modalities that the configured local backend can
process for images and files, with explicit fallback behavior where
Ollama/model support is absent. Audio input is deliberately excluded from this
phase; it should stay an explicit unsupported input type unless a later
dedicated audio/realtime/transcription phase is added.

Status: completed on 2026-06-07. The fast gateway suite passed, the mock HTTP
benchmark passed, and the real-Ollama benchmark passed with
`RESPAWN_BENCHMARK_RUNS=0` and `RESPAWN_BENCHMARK_MAX_OUTPUT_TOKENS=32`.
The real stack used `gpt-oss:120b` for text/file flows and
`moondream:latest` for image smoke validation inside the same Ollama backend.

### Scope

- [x] Support `input_image` content parts from URL.
- [x] Support `input_image` content parts from data URL/base64.
- [x] Support `input_file` from external URL.
- [x] Support `input_file` from data URL/base64 file data.
- [x] Keep `input_file`/`input_image` local file IDs explicitly unsupported
  until a local Files API or compatible storage exists.
- [x] Support direct text extraction for `.txt`, `.md`, `.json`, `.csv`, and
  code files.
- [x] Support PDF text extraction.
- [x] Decide whether PDF page images are sent to vision-capable Ollama models or
  text-only extracted. Phase 8 uses text-only PDF extraction.
- [x] Keep `input_audio` explicitly unsupported with a clear local exclusion
  reason.
- [x] Add model capability detection: text-only, vision, and file-text.
- [x] Return explicit model capability errors when request shape is valid but
  the configured Ollama model cannot process the modality.
- [x] Add file size, MIME type, extension, download timeout, and content safety
  limits.

### Implementation Checklist

- [x] Add content part validation and canonical storage handling.
- [x] Add file download/extraction service.
- [x] Add image normalization service for Ollama vision payloads.
- [x] Add capability detection for the configured backend/model.
- [x] Use `VISION_MODEL=moondream:latest` as the small default image benchmark
  model, while keeping `DEFAULT_MODEL=gpt-oss:120b` for text flows.
- [x] Keep `MODEL_CAPABILITIES` explicit, including `file-text` for models that
  may receive gateway-extracted file text.
- [x] Use `OLLAMA_PRELOAD_MODELS` to keep the text and vision models available
  inside the same single Ollama backend.
- [x] Use the local benchmark asset server at
  `RESPAWN_BENCHMARK_ASSET_BASE_URL` for deterministic file/image cases instead
  of depending on public internet URLs.
- [x] Add fixtures under tests for tiny image, tiny text file, tiny PDF, tiny CSV.
- [x] Add benchmark assets under `infra/docker/benchmark/assets`.
- [x] Add docs for required Ollama vision model if not using default
  `moondream:latest`.

### Real Ollama Validation

- [x] Text file input: model receives extracted text and answers a fact from it.
- [x] CSV input: model receives enough parsed text to answer a simple aggregate
  or header question.
- [x] PDF input: model receives extracted text and answers a fact from it.
- [x] Image input with a configured vision-capable Ollama model answers a simple
  visual question.
- [x] Image input with a text-only Ollama model returns a clear capability error.
- [x] Oversized file returns deterministic validation error.
- [x] External URL timeout or connection failure returns deterministic error.

Suggested benchmark cases:

- [x] `responses.multimodal.input_file_text`
- [x] `responses.multimodal.input_file_csv`
- [x] `responses.multimodal.input_file_pdf`
- [x] `responses.multimodal.input_image_vision`
- [x] `responses.multimodal.input_image_unsupported_model`
- [x] `responses.multimodal.input_audio_unsupported`
- [x] `responses.multimodal.file_limits`

### DoD

- [x] Fast tests pass.
- [x] Real Ollama benchmark passes for text/file cases using the default model.
- [x] Vision benchmark passes when a vision-capable Ollama model is configured
  or is explicitly skipped with a capability reason.
- [x] Multimodal fields move out of text-only unsupported status as appropriate.
- [x] File/image limits are enforced before backend calls.
- [x] `input_audio` remains documented as deliberately unsupported.

## Phase 9 - Reasoning Parity, Encrypted Reasoning, And Context Carryover

Goal: make reasoning items, summaries, effort settings, token accounting, and
state carryover match the Responses mental model as closely as a local backend
allows.

Status: implemented on 2026-06-08. The implementation rechecked the current
OpenAI reasoning and Responses docs plus Ollama thinking docs, added
capability-aware effort validation including `xhigh`, deterministic local
summaries, local opaque encrypted reasoning envelopes, carryover/storage, token
metrics, and benchmark cases. Fast tests passed, the mock HTTP benchmark passed,
and the real Ollama reasoning-tag benchmark passed with the targeted command
shown in the validation notes for this phase.

### Scope

- [x] Add `reasoning.effort="xhigh"` validation when current docs include it,
  mapped to the closest configured backend setting or accepted only when that
  backend supports it.
- [x] Preserve `none`, `minimal`, `low`, `medium`, `high`.
- [x] Add semantic reasoning summaries that do not expose raw chain-of-thought.
- [x] Add encrypted reasoning content as opaque local blobs if Respawn has key
  management.
- [x] Round-trip encrypted reasoning items through `store=false` workflows when
  supported.
- [x] Ensure reasoning items are preserved in manual continuation and stored
  response chains.
- [x] Improve `reasoning_tokens` accounting when Ollama reports thinking tokens
  or thinking text.
- [x] Add configured-backend capability checks for reasoning behavior.
- [x] Add metrics for effort distribution, reasoning token counts, and
  reasoning-heavy requests.

### Implementation Checklist

- [x] Add key management design before encrypted reasoning implementation.
- [x] Add summary provider abstraction with deterministic local fallback.
- [x] Add validation tests for all effort and summary values.
- [x] Add item storage for encrypted content and summary parts.
- [x] Add redaction tests proving raw reasoning is never exposed when summaries
  are requested.
- [x] Add benchmark prompts that trigger Ollama thinking where the model
  supports it.

### Real Ollama Validation

- [x] Request `reasoning={"effort":"low","summary":"auto"}` returns reasoning
  item plus final message.
- [x] Request every supported effort value and assert accepted values do not
  crash.
- [x] Request unsupported effort for the configured backend returns explicit
  error.
- [x] Reasoning plus `previous_response_id` preserves reasoning items in the
  next request.
- [x] `output_tokens_details.reasoning_tokens` is present and non-negative.
- [x] Raw backend thinking text is not exposed in response output unless the
  public contract explicitly allows it.

Validated with:

```bash
cd infra/docker
RESPAWN_BENCHMARK_INCLUDE_TAGS=reasoning RESPAWN_BENCHMARK_RUNS=0 RESPAWN_BENCHMARK_MAX_OUTPUT_TOKENS=32 make benchmark
```

Suggested benchmark cases:

- [x] `responses.reasoning.effort_matrix`
- [x] `responses.reasoning.summary`
- [x] `responses.reasoning.previous_response_carryover`
- [x] `responses.reasoning.encrypted_roundtrip`
- [x] `metrics.reasoning`

### DoD

- [x] Fast tests pass.
- [x] Real Ollama benchmark passes.
- [x] Reasoning support is configured-backend capability aware.
- [x] Reasoning items survive stored state and manual continuations.
- [x] Raw chain-of-thought exposure policy is documented and tested.

## Phase 10 - Context Management, Compaction, Truncation, And Token Counting

Goal: support long stored response chains safely with local context limits,
count tokens more accurately, and expose compaction endpoints.

Status: completed on 2026-06-08. The implementation rechecked the current
OpenAI Responses docs for `context_management`, `truncation`, and
`/v1/responses/compact`, then shipped a deterministic local context planner,
compaction endpoint, context event storage, metrics, manifest coverage, fast
tests, mock HTTP benchmark coverage, and real Ollama validation for the new
context surfaces.

### Scope

- [x] Implement `POST /v1/responses/compact`.
- [x] Implement `context_management` request handling.
- [x] Implement `truncation=auto`.
- [x] Keep `truncation=disabled` strict and fail clearly when context is too
  large.
- [x] Add exact or model-aware tokenizer-backed counting where possible.
- [x] Track prompt-cache hit accounting and document backend telemetry limits.
- [x] Add compaction records and link compacted state to source items.
- [x] Add response-chain compaction for stored `previous_response_id` state.
- [x] Add tests for preserving important facts across compaction.
- [x] Add tests for reasoning item handling during compaction.

### Implementation Checklist

- [x] Add tokenizer abstraction.
- [x] Add Ollama model context window/capability discovery or configuration.
- [x] Add context planner service.
- [x] Add deterministic local compaction summaries and structured compaction
  output.
- [x] Add storage for compaction summaries and provenance.
- [x] Add deterministic mock compactor for fast tests.
- [x] Add metrics: compact calls, input tokens before/after, compression ratio,
  truncation count, context overflow errors.

### Real Ollama Validation

- [x] `/v1/responses/input_tokens` returns model-aware count or documented local
  estimate for a known prompt.
- [x] Long stored response chain with `truncation=disabled` fails before backend
  overflow when above configured local limit.
- [x] Long stored response chain with `truncation=auto` completes and records
  truncation or compaction details.
- [x] `responses/compact` returns a compacted response object with usable
  context.
- [x] Follow-up after compaction answers a fact preserved from earlier turns.
- [x] Metrics report compaction and truncation.

Validated with:

```bash
cd apps/gateway
.venv/bin/python -m pytest -q

cd infra/docker
RESPAWN_BASE_URL=http://127.0.0.1:18080 \
RESPAWN_BENCHMARK_ASSET_BASE_URL=http://127.0.0.1:18081 \
RESPAWN_BENCHMARK_MODEL_BACKEND=mock \
RESPAWN_BENCHMARK_RUNS=0 \
RESPAWN_BENCHMARK_MAX_OUTPUT_TOKENS=32 \
../../apps/gateway/.venv/bin/python benchmark/respawn_benchmark.py

RESPAWN_BASE_URL=http://127.0.0.1:18082 \
RESPAWN_BENCHMARK_ASSET_BASE_URL=http://127.0.0.1:18081 \
RESPAWN_BENCHMARK_MODEL_BACKEND=ollama \
RESPAWN_BENCHMARK_INCLUDE_TAGS=context \
RESPAWN_BENCHMARK_RUNS=0 \
RESPAWN_BENCHMARK_MAX_OUTPUT_TOKENS=32 \
RESPAWN_BENCHMARK_COMPLETION_OUTPUT_TOKENS=32 \
RESPAWN_BENCHMARK_MODEL='gpt-oss:120b' \
RESPAWN_BENCHMARK_TEXT_MODEL='gpt-oss:120b' \
RESPAWN_BENCHMARK_VISION_MODEL=moondream:latest \
../../apps/gateway/.venv/bin/python benchmark/respawn_benchmark.py
```

The Compose `make benchmark-mock` wrapper was attempted as well, but Docker
failed before the suite started because the benchmark network disappeared while
containers were being attached. The equivalent local HTTP mock benchmark passed
with full compatibility coverage.

Suggested benchmark cases:

- [x] `responses.input_tokens.model_aware`
- [x] `responses.context.truncation_disabled_overflow`
- [x] `responses.context.truncation_auto`
- [x] `responses.context.compaction`
- [x] `responses.compact`
- [x] `responses.compact.followup_memory`
- [x] `metrics.context_management`

### DoD

- [x] Fast tests pass.
- [x] Real Ollama context benchmark passes.
- [x] `compact`, `context_management`, and `truncation=auto` are classified
  accurately in the matrix.
- [x] Context behavior is deterministic enough for operations.
- [x] Token counting limitations are documented for the configured backend/model.

## Phase 11 - Include Expansions, Logprobs, Annotations, And Artifacts

Goal: complete response enrichment surfaces that expose extra details from
logprobs, images, files, and reasoning.

Status: completed on 2026-06-08. The implementation rechecked the current
OpenAI SDK/OpenAPI-derived include surface, added a local include registry,
tenant-scoped response artifact records, safe `message.input_image.image_url`
expansions, local `input_file` citation annotations, backend-capable
`message.output_text.logprobs`, explicit capability errors for Ollama/default
logprobs, retrieve-time include expansion, size limits, metrics, tests, docs,
and benchmark coverage. Real-Ollama HTTP validation passed for the Phase 11
`include` tag subset against local `gpt-oss:120b` and `moondream:latest`.

### Scope

- [x] Implement `include` validation and routing.
- [x] Support output text logprobs when backend can provide them.
- [x] Support `top_logprobs` when backend can provide them.
- [x] Support file and artifact includes for non-tool platform objects if
  implemented.
- [x] Support reasoning encrypted content include if Phase 9 implements it.
- [x] Add annotations to output text content where a non-tool feature produces
  them.
- [x] Add artifact storage for generated or uploaded files if a non-tool feature
  requires it.
- [x] Return explicit backend capability errors for unavailable includes.

### Implementation Checklist

- [x] Add include registry with feature ids and backend capability checks.
- [x] Add artifact model and repository APIs.
- [x] Add response serializer expansion path.
- [x] Add tests for include combinations and invalid include strings.
- [x] Add size limits for expanded payloads.
- [x] Add metrics for include expansion cost/size.

### Real Ollama Validation

- [x] Request `include` for an implemented file or artifact expansion returns
  expanded details.
- [x] Request `top_logprobs` against a backend that cannot provide it returns a
  clear capability error or unsupported error.
- [x] Request annotations from an implemented non-tool feature returns output
  text annotations.
- [x] Retrieve stored response with include returns the same expansion shape.

Validated with:

```bash
cd apps/gateway
.venv/bin/python -m pytest -q

cd apps/gateway
RESPAWN_BASE_URL=http://127.0.0.1:18080 \
RESPAWN_BENCHMARK_ASSET_BASE_URL=http://127.0.0.1:18081 \
RESPAWN_BENCHMARK_MODEL_BACKEND=mock \
RESPAWN_BENCHMARK_INCLUDE_TAGS=include \
RESPAWN_BENCHMARK_RUNS=0 \
RESPAWN_BENCHMARK_MAX_OUTPUT_TOKENS=32 \
RESPAWN_BENCHMARK_COMPLETION_OUTPUT_TOKENS=32 \
RESPAWN_BENCHMARK_EXPECT_OLLAMA_METRICS=false \
.venv/bin/python ../../infra/docker/benchmark/respawn_benchmark.py

cd apps/gateway
RESPAWN_BASE_URL=http://127.0.0.1:18082 \
RESPAWN_BENCHMARK_ASSET_BASE_URL=http://127.0.0.1:18081 \
RESPAWN_BENCHMARK_MODEL_BACKEND=ollama \
RESPAWN_BENCHMARK_INCLUDE_TAGS=include \
RESPAWN_BENCHMARK_RUNS=0 \
RESPAWN_BENCHMARK_MAX_OUTPUT_TOKENS=32 \
RESPAWN_BENCHMARK_COMPLETION_OUTPUT_TOKENS=32 \
RESPAWN_BENCHMARK_MODEL='gpt-oss:120b' \
RESPAWN_BENCHMARK_TEXT_MODEL='gpt-oss:120b' \
RESPAWN_BENCHMARK_VISION_MODEL=moondream:latest \
.venv/bin/python ../../infra/docker/benchmark/respawn_benchmark.py
```

Suggested benchmark cases:

- [x] `responses.include.file_artifacts`
- [x] `responses.include.annotations`
- [x] `responses.include.unsupported_logprobs`
- [x] `responses.retrieve.include`
- [x] `metrics.include_expansions`

### DoD

- [x] Fast tests pass.
- [x] Real Ollama benchmark passes.
- [x] `include` is no longer globally unsupported.
- [x] Each include is independently documented as supported, unsupported, or
  dependent on the configured backend.
- [x] Expanded payloads obey size and tenant boundaries.

## Phase 12 - Hosted Prompt Templates And Local Prompt Cache

Goal: support `prompt` inputs and make prompt-cache behavior explicit for the
single Respawn instance model.

### Scope

- [x] Define local hosted prompt template storage.
- [x] Support `prompt` request field according to the current OpenAI shape.
- [x] Support prompt variables and versioning.
- [x] Decide whether templates are API-managed, file-managed, or both.
- [x] Keep prompt cache scoped to the local Respawn process.
- [x] Add cache key retention semantics.
- [x] Add cache invalidation and TTL behavior.
- [x] Add observability for prompt template usage and cache hit ratio.

### Implementation Checklist

- [x] Add `PromptTemplateRecord`.
- [x] Add prompt rendering service with validation.
- [x] Add migration and repository APIs.
- [x] Add tests for template variables, missing variables, versions, and tenant
  isolation.
- [x] Add benchmark cases for prompt rendering and cache hit behavior.

### Real Ollama Validation

- [x] Create or load a prompt template.
- [x] Call `/v1/responses` with `prompt` and variables; output reflects rendered
  prompt.
- [x] Re-run with the same `prompt_cache_key` and observe cached token details.
- [x] Restart Respawn and verify local in-memory cache behavior is documented
  and deterministic.
- [x] Missing template or variable returns OpenAI-shaped error.

Suggested benchmark cases:

- [x] `responses.prompt.template_render`
- [x] `responses.prompt.template_missing`
- [x] `responses.prompt_cache.in_memory`
- [x] `metrics.prompt_cache`

### DoD

- [x] Fast tests pass.
- [x] Real Ollama benchmark passes.
- [x] `prompt` is supported or explicitly scoped if only local templates exist.
- [x] Prompt cache behavior is documented as single-instance local state.

Validation completed on 2026-06-08:

```bash
cd apps/gateway
.venv/bin/pytest -q
# 135 passed, 4 warnings

PYTHONPATH=.:../../infra/docker/benchmark .venv/bin/python - <<'PY'
from src.services.compatibility_manifest import compatibility_manifest
import respawn_benchmark as benchmark
coverage = benchmark.manifest_coverage(compatibility_manifest(), benchmark.benchmark_cases())
print(f"covered={len(coverage['covered_supported_features'])} missing={len(coverage['missing_supported_features'])}")
assert not coverage["missing_supported_features"]
PY
# covered=87 missing=0

RESPAWN_BASE_URL=http://127.0.0.1:18083 \
RESPAWN_BENCHMARK_MODEL_BACKEND=mock \
RESPAWN_BENCHMARK_MODEL=gpt-oss-120b \
RESPAWN_BENCHMARK_TEXT_MODEL=gpt-oss-120b \
RESPAWN_BENCHMARK_INCLUDE_TAGS=prompt \
RESPAWN_BENCHMARK_RUNS=0 \
RESPAWN_BENCHMARK_MAX_OUTPUT_TOKENS=32 \
RESPAWN_BENCHMARK_COMPLETION_OUTPUT_TOKENS=32 \
RESPAWN_BENCHMARK_EXPECT_OLLAMA_METRICS=false \
.venv/bin/python ../../infra/docker/benchmark/respawn_benchmark.py
# Feature cases: passed=4 failed=0 skipped=73

RESPAWN_BASE_URL=http://127.0.0.1:18085 \
RESPAWN_BENCHMARK_MODEL_BACKEND=ollama \
RESPAWN_BENCHMARK_MODEL='gpt-oss:120b' \
RESPAWN_BENCHMARK_TEXT_MODEL='gpt-oss:120b' \
RESPAWN_BENCHMARK_INCLUDE_TAGS=prompt \
RESPAWN_BENCHMARK_RUNS=0 \
RESPAWN_BENCHMARK_MAX_OUTPUT_TOKENS=32 \
RESPAWN_BENCHMARK_COMPLETION_OUTPUT_TOKENS=32 \
RESPAWN_BENCHMARK_EXPECT_OLLAMA_METRICS=true \
RESPAWN_BENCHMARK_TIMEOUT_SECONDS=240 \
.venv/bin/python ../../infra/docker/benchmark/respawn_benchmark.py
# Feature cases: passed=4 failed=0 skipped=73
```

## Phase 13 - Files And Local Platform Dependencies

Goal: implement enough local platform APIs to support full Responses behavior
for file inputs and non-tool artifact storage.

Status: completed on 2026-06-08. The exit gate was validated with Respawn
served locally against the host Ollama daemon on `127.0.0.1:11434` using
`gpt-oss:120b`. The phase added the local Files API subset, tenant-scoped
platform file storage, `input_file.file_id` resolution, artifact content
download, TTL cleanup, quota/size validation, and benchmark coverage for the
new platform surfaces.

### Scope

- [x] Add local Files API subset if required by `input_file` file IDs.
- [x] Add artifact download endpoints.
- [x] Add lifecycle management: create, retrieve, list, delete, TTL.
- [x] Add storage configuration for local disk or database blobs.
- [x] Add tenant isolation and quota controls.
- [x] Add malware/content-type validation hooks where appropriate.

### Implementation Checklist

- [x] Design platform object model.
- [x] Add migrations.
- [x] Add routers and schemas.
- [x] Add storage abstraction.
- [x] Add cleanup job for TTL/deleted artifacts.
- [x] Add benchmark assets and cleanup behavior.
- [x] Add docs for local storage paths and ignored files.

### Real Ollama Validation

- [x] Upload/create a file, reference it in `input_file`, and get a correct
  answer.
- [x] Delete platform objects and verify they are no longer usable.
- [x] Quota/size violations return deterministic errors.

Suggested benchmark cases:

- [x] `files.create_retrieve_delete`
- [x] `responses.input_file.file_id`
- [x] `platform_objects.tenant_scope`

### DoD

- [x] Fast tests pass.
- [x] Real Ollama benchmark passes.
- [x] Platform object APIs are sufficient for implemented Responses features.
- [x] Storage cleanup is tested.
- [x] Secrets and local files remain out of git.

Validation:

```bash
cd apps/gateway
.venv/bin/python -m pytest
# 140 passed, 4 warnings

PYTHONPATH=.:../../infra/docker/benchmark .venv/bin/python - <<'PY'
from src.services.compatibility_manifest import compatibility_manifest
import respawn_benchmark as benchmark
coverage = benchmark.manifest_coverage(compatibility_manifest(), benchmark.benchmark_cases())
print(f"covered={len(coverage['covered_supported_features'])} missing={len(coverage['missing_supported_features'])}")
if coverage["missing_supported_features"]:
    raise SystemExit(1)
PY
# covered=91 missing=0
```

Real Ollama targeted gate:

```bash
cd apps/gateway
DATABASE_URL=sqlite+aiosqlite:////tmp/respawn-phase13-ollama.db \
MODEL_BACKEND=ollama \
OLLAMA_BASE_URL=http://127.0.0.1:11434 \
AUTH_DISABLED=false \
LOCAL_OPENAI_API_KEYS=local-dev-key:tenant-local,respawn-other-key:tenant-other \
DEFAULT_MODEL='gpt-oss:120b' \
PROMPT_CACHE_MIN_TOKENS=8 \
BACKEND_TIMEOUT_SECONDS=180 \
.venv/bin/uvicorn src.main:app --host 127.0.0.1 --port 18087

RESPAWN_BASE_URL=http://127.0.0.1:18087 \
RESPAWN_BENCHMARK_MODEL_BACKEND=ollama \
RESPAWN_BENCHMARK_MODEL='gpt-oss:120b' \
RESPAWN_BENCHMARK_TEXT_MODEL='gpt-oss:120b' \
RESPAWN_BENCHMARK_INCLUDE_TAGS=files \
RESPAWN_BENCHMARK_RUNS=0 \
RESPAWN_BENCHMARK_MAX_OUTPUT_TOKENS=32 \
RESPAWN_BENCHMARK_COMPLETION_OUTPUT_TOKENS=32 \
RESPAWN_BENCHMARK_EXPECT_OLLAMA_METRICS=false \
RESPAWN_BENCHMARK_TIMEOUT_SECONDS=240 \
.venv/bin/python ../../infra/docker/benchmark/respawn_benchmark.py
# Feature cases: passed=3 failed=0 skipped=77
```

## Phase 14 - SDK Contract, Error Parity, And Backwards Compatibility

Goal: harden Respawn as a drop-in OpenAI-compatible target for client SDKs and
existing integrations.

Status: completed on 2026-06-08. The phase expanded the official OpenAI Python
SDK contract gate, added single-instance `Idempotency-Key` replay support,
normalized request-id/error behavior, added Files and artifact pagination
coverage, and introduced real HTTP `sdk.*` benchmark cases. Node SDK contract
tests were evaluated and left as future work because the repo does not
currently include Node test tooling.

### Scope

- [x] Expand official OpenAI Python SDK contract tests for every supported
  endpoint.
- [x] Add Node SDK contract tests if the repo adopts Node test tooling.
  No Node test tooling exists in the repo, so this remains documented future
  work rather than a partial SDK gate.
- [x] Verify streaming helpers parse all emitted events.
- [x] Normalize error shapes: validation, unsupported parameter, not found,
  conflict, rate/limit, backend unavailable, and timeout.
- [x] Add request ID headers if compatible with OpenAI SDK expectations.
- [x] Add idempotency-key support if required by current OpenAI behavior.
- [x] Add pagination behavior parity for all list endpoints.
- [x] Add backwards compatibility tests for existing Respawn-supported subset.

### Implementation Checklist

- [x] Add `tests/contract/test_responses_full_sdk.py`.
- [x] Add error schema tests.
- [x] Add header tests.
- [x] Add list pagination tests for responses input items, files, and artifacts.
- [x] Add compatibility snapshots for representative JSON shapes.
- [x] Add docs for known local differences.

### Real Ollama Validation

- [x] Official Python SDK create/retrieve/delete/list/stream works with Respawn.
- [x] SDK background create/poll/cancel works.
- [x] SDK function tool helpers parse `function_call` items and submit
  `function_call_output` follow-ups; unsupported built-in/internal tool
  categories receive documented errors.
- [x] SDK errors map to expected exception classes for 400, 404, 409, 422, 500.

Suggested benchmark cases:

- [x] `sdk.responses.create_retrieve_delete`
- [x] `sdk.responses.stream`
- [x] `sdk.responses.background`
- [x] `sdk.errors`

### DoD

- [x] Fast tests pass.
- [x] Real Ollama benchmark passes.
- [x] SDK compatibility is part of the release gate.
- [x] Known local differences are documented, not surprising.

Validation:

```bash
cd apps/gateway
.venv/bin/python -m pytest
# 148 passed, 4 warnings

PYTHONPATH=.:../../infra/docker/benchmark .venv/bin/python - <<'PY'
from src.services.compatibility_manifest import compatibility_manifest
import respawn_benchmark as benchmark
coverage = benchmark.manifest_coverage(compatibility_manifest(), benchmark.benchmark_cases())
print(f"covered={len(coverage['covered_supported_features'])} missing={len(coverage['missing_supported_features'])}")
if coverage["missing_supported_features"]:
    raise SystemExit(1)
PY
# covered=100 missing=0
```

Real Ollama targeted gate:

```bash
cd apps/gateway
DATABASE_URL=sqlite+aiosqlite:////tmp/respawn-phase14-ollama.db \
MODEL_BACKEND=ollama \
OLLAMA_BASE_URL=http://127.0.0.1:11434 \
AUTH_DISABLED=false \
LOCAL_OPENAI_API_KEYS=local-dev-key:tenant-local,respawn-other-key:tenant-other \
DEFAULT_MODEL='gpt-oss:120b' \
PROMPT_CACHE_MIN_TOKENS=8 \
BACKEND_TIMEOUT_SECONDS=240 \
.venv/bin/uvicorn src.main:app --host 127.0.0.1 --port 18089

RESPAWN_BASE_URL=http://127.0.0.1:18089 \
RESPAWN_BENCHMARK_MODEL_BACKEND=ollama \
RESPAWN_BENCHMARK_MODEL='gpt-oss:120b' \
RESPAWN_BENCHMARK_TEXT_MODEL='gpt-oss:120b' \
RESPAWN_BENCHMARK_INCLUDE_TAGS=sdk \
RESPAWN_BENCHMARK_RUNS=0 \
RESPAWN_BENCHMARK_MAX_OUTPUT_TOKENS=32 \
RESPAWN_BENCHMARK_COMPLETION_OUTPUT_TOKENS=32 \
RESPAWN_BENCHMARK_EXPECT_OLLAMA_METRICS=false \
RESPAWN_BENCHMARK_TIMEOUT_SECONDS=300 \
.venv/bin/python ../../infra/docker/benchmark/respawn_benchmark.py
# Feature cases: passed=4 failed=0 skipped=80
```

## Phase 15 - Observability, Operations, And Release Certification

Goal: make the full Responses surface operable in the local single-instance
deployment model.

### Scope

- [x] Add metrics for every endpoint, feature family, status, backend model,
  token kind, job status, and error code.
- [x] Add Grafana panels for responses lifecycle, streaming, background jobs,
  context management, files, and cache.
- [x] Add structured logs with request id, response id, tenant, feature,
  backend, latency, status, and error code.
- [x] Add health/ready checks for database, Ollama, worker, cache, and storage.
- [x] Add benchmark historical comparison.
- [x] Add release checklist for compatibility certification.
- [x] Add single-instance load/concurrency benchmark for background jobs and
  streaming.
- [x] Add failure injection tests for Ollama outage, database outage, cache
  outage, and storage outage.

### Implementation Checklist

- [x] Update `observability/metrics.py`.
- [x] Update VictoriaMetrics scrape config only if needed.
- [x] Update Grafana dashboard JSON.
- [x] Add benchmark report comparison mode.
- [x] Add ops docs in README or a dedicated observability doc.
- [x] Keep Compose profiles scoped to the local single-instance stack.

### Real Ollama Validation

- [x] Run full `make benchmark` and verify metrics endpoint contains all new
  counters/histograms.
- [x] Open Grafana and verify panels populate after benchmark.
- [x] Stop Ollama during a request and verify backend unavailable error plus
  metrics/logs.
- [x] Run concurrent streaming/background requests within one Respawn instance
  and verify no cross-response item leakage.
- [x] Run benchmark twice and verify historical comparison output.

Suggested benchmark cases:

- [x] `metrics.full_surface`
- [x] `ops.ollama_unavailable`
- [x] `ops.concurrent_streaming`
- [x] `ops.concurrent_background`
- [x] `benchmark.history_compare`

### DoD

- [x] Fast tests pass.
- [x] Real Ollama benchmark passes.
- [x] Dashboard reflects all major feature families.
- [x] Release checklist can certify a version as Responses-compatible.
- [x] Operational failure modes are observable and documented.

## Final 100% Certification Checklist

Do not claim 100% compatibility until every item below is complete.

- [ ] `RESPONSES_COMPATIBILITY.md` has no "Not supported" row for any current
  OpenAI Responses endpoint or field unless the row documents a deliberate local
  incompatibility accepted by project maintainers.
- [ ] Every supported row has fast tests and at least one real-Ollama benchmark
  case.
- [ ] `POST /v1/responses` supports blocking, streaming, background, text,
  multimodal where backend-capable, structured output, reasoning, context
  management, prompt templates, and supported includes.
- [ ] `GET /v1/responses/{id}` returns complete stored state for every terminal
  status.
- [ ] `DELETE /v1/responses/{id}` is tenant-safe and idempotency behavior is
  documented.
- [ ] `GET /v1/responses/{id}/input_items` is item-store backed and paginated.
- [ ] `POST /v1/responses/input_tokens` is model-aware or clearly documented as
  estimated for each backend.
- [ ] `POST /v1/responses/{id}/cancel` works for background/in-flight jobs.
- [ ] `POST /v1/responses/compact` works for stored response-chain state.
- [ ] Function tool calling protocol works end to end: `tools`,
  `tool_choice`, `max_tool_calls`, `parallel_tool_calls`, `function_call`
  output items, and client-supplied `function_call_output` input items are
  validated, stored, replayed, retrieved, streamed, and benchmarked where the
  configured backend/model is capable.
- [ ] Respawn still does not execute tools itself; built-in, MCP, custom
  free-form, filesystem, shell, git, `apply_patch`, workspace, browser,
  code-interpreter, web-search, file-search, computer-use, image-generation,
  and other hosted tool categories return stable OpenAI-shaped errors unless a
  later roadmap explicitly adds them.
- [ ] Multimodal input has capability-aware validation and real tests.
- [ ] Reasoning items, summaries, encrypted content if supported, and usage
  details are persisted and redacted correctly.
- [ ] Streaming event coverage matches every implemented output item lifecycle.
- [ ] Error shapes are OpenAI-compatible and parsed by the official SDK.
- [ ] `make benchmark` passes against Ollama from a clean Compose stack.
- [ ] Benchmark report is archived for the release.
- [ ] README, compatibility matrix, roadmap, and operational docs agree.

## Suggested Phase Order

The recommended order is:

1. Phase 0: harness.
2. Phase 1: response object and request parity.
3. Phase 2: first-class item model.
4. Phase 4: background and cancel.
5. Phase 5: streaming completeness.
6. Phase 6: function tool protocol compatibility.
7. Phase 10: context management and compaction.
8. Phase 8: multimodal files and images.
9. Phase 9: reasoning parity.
10. Phase 11: includes and artifacts.
11. Phase 12: hosted prompts and local prompt cache.
12. Phase 13: platform dependencies.
13. Phase 14: SDK/error hardening.
14. Phase 15: observability and certification.

This order front-loads storage and lifecycle primitives because most later
features depend on item-native state, background execution, and reliable
streaming.
