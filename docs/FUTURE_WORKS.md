# Respawn Future Work

This document tracks candidate work that is intentionally outside the current
compatibility claim. The source of truth for what Respawn supports today remains
`docs/COMPATIBILITY.md`.

Future work should graduate into the compatibility matrix only after it has
implementation, explicit tests or benchmark coverage, and OpenAI-shaped failure
behavior for unsupported paths.

## Priorities

| Priority | Workstream | Why it matters |
| --- | --- | --- |
| P0 | Codex compatibility probe | Confirms whether Codex can use Respawn through the native Responses wire protocol today. |
| P1 | Agent tool protocol expansion | Removes the main blocker for richer agent clients when they use native Responses tool types instead of plain function tools. |
| P2 | Browser/web tool support | Useful for up-to-date agent workflows, but needs a clear local or hosted execution model. |
| P2 | Code interpreter and computer tool support | Powerful but higher-risk, because they require sandboxing, artifact handling, and operator controls. |

## Codex Compatibility Probe

Goal: verify and document whether Codex CLI or Codex for VS Code can point at
Respawn as a custom Responses-compatible provider.

Known starting point:

```toml
model_provider = "respawn"
model = "gpt-oss:120b"

[model_providers.respawn]
name = "Respawn"
base_url = "http://localhost:8080/v1"
env_key = "OPENAI_API_KEY"
wire_api = "responses"
```

Candidate tasks:

- Run a minimal Codex request through Respawn with `wire_api = "responses"`.
- Capture the exact request shape Codex sends for simple chat, shell work,
  file edits, and patch application.
- Determine whether Codex exposes its local tools as ordinary `function` tools
  or native Responses built-in tools.
- Add a small reproducible probe under the benchmark or tooling tree.
- Document a supported local Codex configuration once the behavior is proven.

Acceptance criteria:

- A Codex smoke test can complete through Respawn with a local model.
- Respawn logs and metrics identify the request as `backend=ollama`,
  `model=<selected model>`, and the relevant Responses feature family.
- Unsupported Codex tool shapes fail with explicit OpenAI-shaped errors instead
  of schema surprises.

## Agent Tool Protocol Expansion

Goal: support the tool protocol shapes needed by agent clients without turning
Respawn into an unsafe local tool executor by accident.

Current boundary:

- Respawn supports `function` tool protocol data.
- Respawn does not execute tools locally.
- Built-in, hosted, MCP, shell, apply-patch, browser, code, computer, image, and
  internal tool categories are currently rejected.

Candidate order:

1. Discover the real tool schemas emitted by Codex against a custom Responses
   provider.
2. Add protocol-only acceptance for low-risk native tool items where the client
   remains responsible for execution.
3. Add explicit capability errors for tool types that require Respawn-hosted
   execution.
4. Consider local execution only behind clear configuration, sandboxing, and
   audit logging.

Likely first candidates:

- `apply_patch`: useful for coding agents if the client or harness applies the
  patch and Respawn only stores/streams the item shape.
- Local shell/function bridge: useful only if represented as client-executed
  function calls or gated behind a strict executor.
- Web search/browser: useful for current information, but should start as a
  clearly documented execution boundary.

Acceptance criteria:

- Supported tool shapes are stored, streamed, replayed through
  `previous_response_id`, and covered by benchmark cases.
- Tool execution ownership is obvious from docs and error messages.
- No local filesystem, shell, browser, or GUI action can run unless explicitly
  enabled by operator configuration.

## Browser And Web Support

Goal: support agent workflows that need current web information or browser
interaction.

Two separate products should not be conflated:

- Web search: query-and-citation style retrieval for up-to-date information.
- Browser/computer automation: screenshot, click, type, and observe loops over
  a real UI.

Candidate tasks:

- Decide whether web search is hosted, local, or client-executed.
- Add request validation and include expansion behavior for the selected model.
- Add citation and source metadata storage if web results are returned through
  Respawn.
- For browser automation, require sandbox boundaries, network allowlists,
  timeout limits, and human-approval hooks before any implementation.

Acceptance criteria:

- Web results or browser actions are auditable through logs and metrics.
- Unsupported browser/computer paths fail explicitly.
- The compatibility matrix distinguishes web search from computer-use flows.

## Code Interpreter And Computer Use

Goal: evaluate whether Respawn should ever host sandboxed execution tools.

Code interpreter would require:

- A sandboxed Python/runtime container.
- Uploaded and generated file handling.
- Resource limits for CPU, memory, disk, network, and wall-clock time.
- Artifact persistence and cleanup policies.
- Clear tenant isolation.

Computer use would require:

- A controlled desktop/browser environment.
- Screenshot capture and action application loops.
- Strong allowlists, timeouts, and operator approvals.
- Tests that verify actions cannot escape the intended sandbox.

Recommendation:

- Keep these as P2 until Codex compatibility and the local tool execution
  boundary are proven.
- Prefer client-executed tools first, because Codex already owns many local
  development actions in its harness.

## Graduation Checklist

Before moving any item from this document into supported compatibility:

- Add implementation behind a documented config path.
- Add unit or integration tests for success and failure cases.
- Add benchmark coverage for any supported public behavior.
- Add metrics or logs when the behavior affects operations.
- Update `docs/COMPATIBILITY.md` and the manifest source together.
- Update `docs/OPERATIONS.md` only when operators need new runtime guidance.
