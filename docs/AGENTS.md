# AGENTS.md

This file is the durable working guide for coding agents in this repository. Keep
it short, practical, and current.

## Project Shape

Respawn is a local OpenAI-compatible gateway focused on the Responses API. The
runtime default is Ollama, while the deterministic mock backend is used for fast
tests and smoke checks. The current architecture is single-instance and
single-backend: one Respawn process is expected to talk to one configured model
backend.

Main areas:

- `apps/gateway`: FastAPI gateway, adapters, schemas, storage, and tests.
- `infra/docker`: local Compose stack for Respawn, Postgres, Ollama,
  VictoriaMetrics, Grafana, and benchmark runners.

## Working Rules

- Read the surrounding code before changing behavior. Prefer existing patterns
  over new abstractions.
- Keep changes scoped. Do not silently refactor unrelated code.
- Treat `infra/docker/env.example` as publishable defaults. Treat
  `infra/docker/env` as ignored local overrides only.
- Do not commit secrets, model blobs, databases, benchmark result files, or local
  environment overrides.
- For OpenAI API compatibility work, check the current official OpenAI docs and
  make unsupported behavior explicit instead of adding silent no-ops.
- When changing public API behavior, update schemas, tests, README examples, and
  the benchmark suite together.
- When changing Responses compatibility, update
  [`RESPONSES_COMPATIBILITY.md`](RESPONSES_COMPATIBILITY.md); when documenting
  not-yet-supported work, update [`FUTURE_WORK.md`](FUTURE_WORK.md).
- When adding or renaming metrics, update VictoriaMetrics/Grafana provisioning
  and the dashboard JSON in the same change.

## Verification

Gateway tests:

```bash
cd apps/gateway
.venv/bin/python -m pytest
```

Docker stack:

```bash
cd infra/docker
make up
make ps
make logs
```

Benchmark and feature suite:

```bash
cd infra/docker
make benchmark
```

The benchmark must always call Respawn HTTP endpoints, not Ollama or internal
Python services directly. It is both a timing benchmark and a feature regression
suite. Whenever Respawn gains, removes, or changes a user-visible feature, update
`infra/docker/benchmark/respawn_benchmark.py` so the feature is covered.

## Responses Compatibility

Respawn currently supports a practical subset of the OpenAI Responses API:
blocking and streaming text responses, response retrieval/deletion, input item
listing, input token counting, Responses-native `previous_response_id` state,
structured outputs, local prompt-cache accounting, local reasoning items,
Responses function tool calling protocol compatibility without local tool
execution, chat completions, models, auth, metrics, and persistence. It does not
target the OpenAI Conversations API.

When expanding Responses compatibility:

- Preserve the OpenAI-shaped request and response surface where possible.
- Keep the implementation scoped to one Respawn instance and one configured
  backend; do not add multi-deployment, distributed cache, backend routing, or
  external-worker assumptions.
- Prefer explicit `400`/`422` style errors for unsupported fields over accepting
  fields that do nothing.
- Treat function tools as protocol data only. Do not add filesystem, shell, git,
  `apply_patch`, workspace, MCP-hosting, or other local tool execution inside
  Respawn.
- Keep the support matrix in [`RESPONSES_COMPATIBILITY.md`](RESPONSES_COMPATIBILITY.md) accurate.
- Keep major gaps and roadmap notes in [`FUTURE_WORK.md`](FUTURE_WORK.md).
- Add focused tests under `apps/gateway/tests`.
- Add benchmark coverage under `infra/docker/benchmark`.
- Keep Ollama-specific translation inside `apps/gateway/src/adapters`.

## Benchmark Expectations

The benchmark should report latency for core request paths and fail with a
non-zero exit code when a covered feature breaks. Keep prompts short and
deterministic. For model-sensitive checks, assert structural behavior first, then
minimal semantic output only when the feature requires it.

Benchmark output files belong in `infra/docker/benchmark-results/`, which is
ignored by Git.
