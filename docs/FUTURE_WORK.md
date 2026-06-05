# Future Work

This backlog tracks the main gaps between Respawn and the full OpenAI Responses
API. Keep [`RESPONSES_COMPATIBILITY.md`](RESPONSES_COMPATIBILITY.md) as the current-state matrix and this file
as the forward-looking roadmap.

## 1. Full Responses Contract

- Add background response mode with durable jobs, polling, cancellation, and
  timeout/TTL handling.
- Add conversation objects and conversation endpoints instead of relying only on
  `previous_response_id`.
- Implement `include` expansions such as output logprobs, file-search results,
  code-interpreter outputs, image URLs, and reasoning encrypted content.
- Implement `context_management`, hosted prompt templates, service tiers, and
  richer truncation behavior.
- Move prompt caching from in-process prefix accounting toward a durable,
  distributed cache if Respawn runs with multiple replicas.
- Add `cancel` and `compact` endpoints.

## 2. Multimodal Inputs

- Support `input_image` from URL, data URL, and future local file IDs.
- Support `input_file` with file URLs, uploaded file references, filenames, and
  text extraction where appropriate.
- Decide whether audio input belongs in Respawn or a separate local realtime/audio
  layer.
- Add validation and benchmark coverage for every supported input type.

## 3. Built-In Tool Platform

- Implement local `web_search` with explicit citations/sources.
- Implement local `file_search` with vector store management and result includes.
- Implement local `code_interpreter` with sandboxed execution and artifact
  outputs.
- Evaluate local `computer_use`, `image_generation`, remote MCP, tool search, and
  shell/skills support as separate capabilities.
- Add dashboard panels for built-in tool calls, duration, errors, and result size.

## 4. Reasoning Support

- Add semantic reasoning summaries that do not expose raw chain-of-thought.
- Add encrypted reasoning content if Respawn gets a local encryption/key
  management story that can safely round-trip opaque reasoning items.
- Broaden backend mappings beyond Ollama `think`/`message.thinking`.
- Add dashboard panels for reasoning token rate, reasoning-heavy requests, and
  effort distribution.

## 5. State And Item Model

- Store input items and output items as first-class records instead of
  reconstructing input items from `request_json`.
- Add stable item IDs for all request input items.
- Add item pagination that matches OpenAI behavior more closely.
- Persist tool call outputs and item status transitions in a way that can power
  richer streaming/replay.

## 6. Streaming Completeness

- Add richer reasoning summary delta events if semantic summaries become
  streamable.
- Add function-call argument delta events.
- Add built-in tool call and output events.
- Add incomplete/cancelled/background lifecycle events.
- Evaluate `starting_after` and stream resume semantics.
- Evaluate stream obfuscation support.

## 7. Response Object Fidelity

- Add richer `incomplete_details` and incomplete status handling.
- Add annotations/citations to output text when web/file search exists.
- Add logprobs and `top_logprobs` support if a backend can provide them.
- Add exact tokenizer-backed input token counting per model.
- Add model-specific cached-token accounting when backends expose exact cache
  telemetry.

## 8. Benchmark And CI

- Keep `make benchmark` as both a latency benchmark and a feature regression
  suite.
- Add benchmark scenarios whenever a new Respawn feature lands.
- Add optional CI smoke mode using the mock backend so the benchmark can run
  without a local GPU/model.
- Add historical benchmark comparison once the project has enough stable runs.
