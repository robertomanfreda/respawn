# Respawn Future Work

This document tracks only candidate work that is not implemented today. The
source of truth for current support remains `docs/COMPATIBILITY.md`.

Implemented features should not stay here as roadmap residue. Move future work
into the compatibility matrix only after it has implementation, tests or
benchmark coverage, and OpenAI-shaped failure behavior for unsupported paths.

## Current Non-Goals

The unsupported areas below are deliberately excluded until they have a clear
execution boundary, sandbox model, artifact policy, and operator controls.

## Priorities

| Priority | Workstream | Why it matters |
| --- | --- | --- |
| P1 | Hosted tool result expansions | Required before OpenAI-hosted tool include values can be represented locally. |
| P2 | Code interpreter | Powerful but requires sandboxed execution and artifact lifecycle management. |
| P2 | Computer/browser automation | Requires a controlled UI environment, action loop, approvals, and isolation. |
| P2 | Hosted MCP and local executors | Useful for agents, but unsafe without explicit ownership and sandbox boundaries. |
| P3 | Audio and realtime-style inputs | Separate modality with different backend and streaming requirements. |
| P3 | Image editing and partial-image streaming | Requires edit inputs, intermediate artifacts, and progressive output events. |

## Hosted Tool Result Expansions

Unsupported today:

- `include=file_search_call.results`
- hosted `web_search` result expansions
- `include=code_interpreter_call.outputs`
- `include=computer_call.output`

Requirements:

- Define local item shapes for each hosted-tool output Respawn chooses to
  support.
- Store, retrieve, stream, and replay those items through `previous_response_id`.
- Keep unsupported hosted include values explicit instead of silently dropping
  them.
- Add benchmark coverage for every include value that becomes supported.

## Code Interpreter

Code interpreter would require Respawn to host sandboxed execution rather than
only proxy model output.

Required design work:

- Isolated runtime container per request, tenant, or session.
- CPU, memory, disk, network, and wall-clock limits.
- Uploaded-file mounting and generated-artifact persistence.
- Cleanup policy for temporary files, kernels, and outputs.
- Explicit operator configuration and metrics for execution activity.
- Tests proving code cannot escape the sandbox or access unauthorized files.

Until that exists, code/file execution tool types should remain explicit
unsupported errors.

## Computer And Browser Automation

Required design work:

- Controlled browser or desktop environment.
- Screenshot capture, action application, and observe/click/type loops.
- Network allowlists and timeout limits.
- Human-approval hooks for sensitive actions.
- Artifact storage for screenshots and traces.
- Tests proving actions cannot escape the configured browser or desktop
  sandbox.

Unsupported browser/computer paths should continue to fail explicitly.

## Hosted MCP And Local Executors

Future executor support would need:

- A clear distinction between client-executed protocol data and Respawn-hosted
  execution.
- Per-tool operator enablement.
- Strict sandboxing for shell, filesystem, git, workspace, and `apply_patch`
  actions.
- Audit logs and metrics for every executed action.
- Tenant isolation and denial-by-default behavior.

Without those controls, shell/filesystem/git/workspace/apply-patch/MCP-hosting
tool categories should stay unsupported.

## Audio And Realtime Inputs

Unsupported today:

- `input_audio`
- realtime audio loops
- transcription/audio generation surfaces

Required design work:

- Backend selection for transcription, audio understanding, or generation.
- Streaming request and response handling.
- File/artifact handling for audio inputs and outputs.
- Capability errors for models that cannot process audio.
- Benchmarks for success and unsupported-model paths.

## Image Editing And Partial Images

Required design work:

- Request validation for edit/image-to-image fields.
- Backend adapter support for masks, init images, and strength settings.
- Streaming event support for partial images.
- Storage and retrieval of intermediate image artifacts when exposed.
- Benchmark coverage for generated final images and partial-image event order.

## Graduation Checklist

Before moving any item from this document into supported compatibility:

- Add implementation behind a documented configuration path.
- Add unit or integration tests for success and failure cases.
- Add benchmark coverage for every supported public behavior.
- Add metrics or logs when behavior affects operations.
- Update `docs/COMPATIBILITY.md` and the manifest source together.
- Update `docs/OPERATIONS.md` only when operators need runtime guidance.
