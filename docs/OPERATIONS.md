# Respawn Operations

Respawn is operated as one gateway instance connected to one configured model
backend. The operational goal is to make local failures visible enough to
certify a Responses-compatible release without pretending Respawn is a
multi-region hosted control plane.

References:

- OpenAI production best practices: https://developers.openai.com/api/docs/guides/production-best-practices
- OpenAI deployment checklist: https://developers.openai.com/api/docs/guides/deployment-checklist
- OpenAI streaming guide: https://developers.openai.com/api/docs/guides/streaming-responses
- OpenAI SDK libraries guide: https://developers.openai.com/api/docs/libraries

## Health And Readiness

`GET /healthz` reports process liveness, Respawn version, and compatibility
manifest version.

`GET /readyz` returns `200` only when these checks are ready:

- `database`: executes a lightweight `select 1`.
- `ollama`: lists backend models and verifies `DEFAULT_MODEL` is present.
- `worker`: verifies the background task registry and file cleanup task.
- `cache`: verifies the in-process prompt cache exists.
- `storage`: validates the configured file storage backend and filesystem path
  when `FILE_STORAGE_BACKEND=filesystem`.

Readiness emits `gateway_readiness_check` and
`gateway_readiness_check_latency_seconds` for each check. A failed check returns
`503` with the failed component and increments
`gateway_operational_failures_total`.

## Logs

HTTP completion logs are structured JSON. Each request log includes:

- `request_id`
- `response_id` when the response body contains one
- `tenant` when auth is enabled
- `feature`
- `backend`
- `latency_ms`
- `status`
- `error_code` for error responses

Logs intentionally avoid API keys and full request or response payloads.

## Metrics

The `/metrics` endpoint is Prometheus-compatible and is scraped by
VictoriaMetrics in the Compose stack. Core operational families include:

- HTTP: `gateway_requests_total`, `gateway_endpoint_requests_total`,
  `gateway_feature_requests_total`, `gateway_request_latency_seconds`
- Errors and failures: `gateway_errors_total`,
  `gateway_operational_failures_total`
- Responses lifecycle: `gateway_responses_total`,
  `gateway_response_latency_seconds`, `gateway_inflight_responses`,
  `gateway_streaming_responses_running`
- Background jobs: `gateway_background_jobs_total`,
  `gateway_background_jobs_running`, `gateway_background_job_latency_seconds`
- Tokens and backend: `gateway_model_token_usage_total`,
  `gateway_token_usage_total`, `gateway_backend_requests_total`,
  `gateway_backend_model_requests_total`,
  `gateway_backend_model_info`, `gateway_model_backend_latency_seconds`
- Backend-native throughput: `gateway_backend_eval_tokens_total`,
  `gateway_backend_eval_duration_seconds_total`,
  `gateway_backend_eval_tokens_per_second`
- Context, include, prompt cache, and files:
  `gateway_context_*`, `gateway_response_include_*`,
  `gateway_prompt_cache_*`, `gateway_storage_operations_total`
- Ollama native throughput/debug detail: `gateway_ollama_eval_*`

The provisioned `Respawn Model Gateway` Grafana dashboard groups panels into
collapsible rows for overview signals, model APIs, LLM backend behavior,
traffic/reliability, runtime jobs, and feature subsystems. It exposes
`$llm_backend` and `$model` variables so operators can filter the same panels
across Ollama today and future backends such as vLLM. The model variable is
backed by `gateway_backend_model_info`, so synthetic benchmark labels such as
`respawn-missing-model-*` remain visible in raw error metrics but are excluded
from the normal model performance views.

## Failure Injection

Use the benchmark ops tag for deterministic operational checks:

```bash
cd infra/docker
RESPAWN_BENCHMARK_INCLUDE_TAGS=ops RESPAWN_BENCHMARK_RUNS=1 make benchmark
```

The Ollama outage case uses a missing model by default so the benchmark can
exercise the same backend failure path without manually stopping containers. To
perform the manual outage drill:

1. Start the stack with `make up-build`.
2. Begin a streaming or blocking `/v1/responses` request.
3. Stop Ollama with `docker compose --env-file env stop ollama`.
4. Verify the request returns an OpenAI-shaped `5xx` error with
   `code=backend_error` or `code=backend_timeout`.
5. Verify `/metrics` includes updated `gateway_backend_requests_total`,
   `gateway_errors_total`, and `gateway_operational_failures_total`.
6. Restart Ollama and confirm `/readyz` returns `ready`.

Fast integration tests inject database, cache, storage, and worker readiness
failures by replacing the corresponding app-state dependency and verifying
`/readyz` returns `503` with the failed check named in the response. For manual
drills, stop Postgres or make `FILE_STORAGE_PATH` unwritable, then verify the
same `/readyz` and `gateway_operational_failures_total` behavior before
restoring the component.

## Benchmark History

Benchmark reports are written under `infra/docker/benchmark-results/`. Compare a
new report to a previous run by setting `RESPAWN_BENCHMARK_COMPARE_TO`:

```bash
cd infra/docker
RESPAWN_BENCHMARK_COMPARE_TO=/results/respawn-benchmark-previous.json make benchmark
```

The report adds a `comparison` section with passed/failed/skipped deltas, new
and removed cases, latency p50 deltas, slowest regressions, and largest
improvements.

## Release Certification Checklist

Before tagging a Responses-compatible release:

- Run the fast gateway test suite.
- Run `make benchmark` against the real Ollama stack with coverage gate enabled.
- Confirm `compatibility.coverage` reports zero missing supported features.
- Run or inspect `metrics.full_surface`, `ops.concurrent_streaming`,
  `ops.concurrent_background`, `ops.ollama_unavailable`, and
  `benchmark.history_compare`.
- Open Grafana after the benchmark and verify the major panels are populated.
- Compare the benchmark report to the previous release candidate.
- Review `docs/RESPONSES_COMPATIBILITY.md` and
  `docs/RESPONSES_100_ROADMAP.md` for agreement with the manifest version.
- Record any deliberate incompatibility as an explicit unsupported feature, not
  as silent behavior.

Compose keeps the benchmark services behind the `benchmark` profile. The normal
stack remains Respawn, Postgres, Ollama, VictoriaMetrics, and Grafana.
