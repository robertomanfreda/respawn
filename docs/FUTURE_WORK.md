# Future Work

This backlog tracks the main gaps between Respawn and the full OpenAI Responses
API. It deliberately does not track OpenAI Conversations API support: no
`/v1/conversations` endpoints, local Conversation objects, or Responses
`conversation` request-field support are planned for the current target. Keep
[`RESPONSES_COMPATIBILITY.md`](RESPONSES_COMPATIBILITY.md) as the current-state matrix and this file
as the short forward-looking backlog. Use
[`RESPONSES_100_ROADMAP.md`](RESPONSES_100_ROADMAP.md) as the detailed phased
execution plan with checklists, real Ollama validation, and phase DoD.

Respawn is also scoped as a single-instance gateway: one Respawn process talks
to one configured backend. Multi-deployment orchestration, dynamic backend
routing, distributed prompt caches, shared workers across Respawn instances, and
multi-replica consistency semantics are out of scope for now.

## 1. Full Responses Contract

- Implement `include` expansions such as output logprobs, file/artifact
  details, image URLs, and reasoning encrypted content.
- Implement `context_management`, hosted prompt templates, service-tier
  scheduling semantics, and richer truncation behavior.
- Keep prompt caching as local single-instance prefix accounting with clear TTL
  semantics.
- Add the `compact` endpoint.

## 2. Multimodal Inputs

- Keep improving file input extraction beyond the Phase 8 baseline if Respawn
  adopts more local parsers.
- Add local Files API support for `input_file.file_id` and `input_image.file_id`
  in the platform dependency phase.
- Keep `input_audio` explicitly unsupported for now. Audio belongs in a future
  dedicated audio/realtime/transcription decision, not in Phase 8 image/file
  work.
- Keep `MODEL_CAPABILITIES` current as new local models are validated for
  `text`, `file-text`, and `vision`.

## 3. Function Tool Calling Protocol

- Keep the implemented OpenAI Responses function tool protocol covered in the
  compatibility manifest and real-Ollama benchmark suite.
- Track model-specific Ollama limitations for function calling and return
  explicit capability errors when the selected backend/model cannot satisfy a
  required or forced tool choice.
- Keep tool execution outside Respawn. The client that calls Respawn is
  responsible for executing functions and sending outputs back.
- Keep local `web_search`, `file_search`, `code_interpreter`, `computer_use`,
  `image_generation`, MCP hosting, filesystem, shell, git, `apply_patch`,
  workspace tools, and skills execution out of scope.
- Add any future tool protocol refinements only when OpenAI documents new
  Responses function-tool fields or events.

## 4. Reasoning Support

- Add semantic reasoning summaries that do not expose raw chain-of-thought.
- Add encrypted reasoning content if Respawn gets a local encryption/key
  management story that can safely round-trip opaque reasoning items.
- Keep configured-backend reasoning mappings explicit, starting from Ollama
  `think`/`message.thinking`.
- Add dashboard panels for reasoning token rate, reasoning-heavy requests, and
  effort distribution.

## 5. State And Item Model

- Broaden first-class item storage beyond the current text/reasoning subset as
  new non-tool item types are supported.
- Persist item status transitions in a way that can power richer
  streaming/replay.
- Use canonical item storage as the substrate for future compaction and stream
  resume semantics.

## 6. Streaming Completeness

- Keep function-call streaming events covered:
  `response.function_call_arguments.delta`,
  `response.function_call_arguments.done`, and matching output item lifecycle
  events.
- Revisit stream replay/resume if the Responses API exposes a create-time
  `starting_after` or equivalent resume parameter.
- Keep background streaming replay out of scope until Respawn has an explicit
  replay/resume contract; polling remains the supported background path.
- Add richer semantic reasoning-summary streaming only if Phase 9 introduces a
  summary provider that can stream summary content incrementally.

## 7. Response Object Fidelity

- Add richer `incomplete_details` and incomplete status handling.
- Add annotations/citations to output text when web/file search exists.
- Add logprobs and `top_logprobs` support if the configured backend can provide
  them.
- Add exact tokenizer-backed input token counting per model.
- Add model-specific cached-token accounting when the configured backend exposes
  exact cache telemetry.

## 8. Benchmark And CI

- Keep `make benchmark` as both a latency benchmark and a feature regression
  suite.
- Add benchmark scenarios whenever a new Respawn feature lands.
- Add optional CI smoke mode using the mock backend so the benchmark can run
  without a local GPU/model.
- Add historical benchmark comparison once the project has enough stable runs.
