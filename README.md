# Respawn

Respawn is a local, OpenAI Responses API compatible gateway for self-hosted LLMs.

It gives OpenAI SDKs a familiar `/v1` API while keeping model serving in your own
stack. Respawn stores response state, reconstructs `previous_response_id`
conversation history, normalizes streaming events, and forwards generation to an
OpenAI-compatible Ollama endpoint.

Respawn is intentionally an API gateway. It does not implement GPU inference,
tokenizers, batching, KV cache management, quantization, or model loading. Those
jobs belong to Ollama and the model runtime underneath it.

## Highlights

- `POST /v1/responses` with create, stream, retrieve, and delete support.
- `POST /v1/chat/completions` passthrough-style compatibility.
- OpenAI Python SDK contract tests for common Responses flows.
- Postgres persistence for stored Responses state.
- Stateful `previous_response_id` reconstruction in the gateway.
- SSE streaming with Responses-style event payloads.
- Function tool calling with a small built-in local registry.
- Structured-output JSON Schema validation with one repair attempt.
- API-key authentication with tenant-scoped stored responses.
- OpenAI-shaped errors.
- Prometheus metrics and structured JSON logging.
- Docker Compose stack for Respawn, Postgres, and Ollama.
- Deterministic mock backend for local development and CI.

## Repository Layout

```text
apps/gateway/
  src/
    api/                 FastAPI routes
    adapters/            Mock and Ollama model backends
    observability/       Logging and Prometheus metrics
    schemas/             API request, response, error, and streaming models
    security/            API-key auth and tenant resolution
    services/            Response orchestration, tools, state, structured output
    storage/             SQLAlchemy models, repository, and Alembic migrations
    streaming/           SSE event rendering
    tools/               Built-in tool registry
  tests/                 Unit, integration, streaming, and SDK contract tests
  Dockerfile
  alembic.ini
  pyproject.toml

infra/docker/
  docker-compose.yml     Respawn + Postgres + Ollama
  env.example            Safe defaults for the Compose stack
```

## Quick Start

The fastest local path uses the mock backend and SQLite. It does not require
Docker or a running model server.

```bash
cd apps/gateway
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[test]"
uvicorn src.main:app --host 0.0.0.0 --port 8080
```

Then point the official OpenAI Python SDK at Respawn:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="local-dev-key",
)

response = client.responses.create(
    model="gpt-oss-120b",
    input="Ciao, spiegami Kubernetes in una frase.",
)

print(response.output_text)
```

By default, the local Python process uses:

- `MODEL_BACKEND=mock`
- `AUTH_DISABLED=true`
- `DATABASE_URL=sqlite+aiosqlite:///./gateway.db`
- `AUTO_CREATE_TABLES=true`

## Docker And Ollama

The Compose stack starts:

- `respawn`, the FastAPI gateway
- `postgres`, the persistence layer
- `ollama`, the local model server

Run it from the Compose directory:

```bash
cd infra/docker
cp env.example env
docker compose --env-file env up --build
```

Useful checks:

```bash
curl -s http://localhost:8080/healthz
curl -s http://localhost:8080/readyz
curl -s http://localhost:8080/v1/models
curl -s http://localhost:8080/metrics
```

If you already have Ollama models on the host, set `OLLAMA_MODELS_PATH` in
`infra/docker/env` before starting Compose:

```text
OLLAMA_MODELS_PATH=/usr/share/ollama/.ollama
```

If you prefer Docker to own the model storage, leave:

```text
OLLAMA_MODELS_PATH=ollama-data
```

and pull the model inside the Ollama container:

```bash
docker compose --env-file env exec ollama ollama pull gpt-oss:120b
```

The local `infra/docker/env` file is ignored by Git. Keep publishable defaults in
`infra/docker/env.example`.

## Configuration

Respawn reads configuration from environment variables. These are the important
defaults:

| Variable | Default | Notes |
| --- | --- | --- |
| `APP_HOST` | `0.0.0.0` | Uvicorn bind host. |
| `APP_PORT` | `8080` | Uvicorn bind port. |
| `DATABASE_URL` | `sqlite+aiosqlite:///./gateway.db` | Use Postgres for Compose and production-like runs. |
| `MODEL_BACKEND` | `mock` | Supported values: `mock`, `ollama`. |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama OpenAI-compatible base URL. |
| `DEFAULT_MODEL` | `gpt-oss-120b` | Used when requests omit `model`. |
| `AUTH_DISABLED` | `true` | Disable bearer-token checks for local development. |
| `LOCAL_OPENAI_API_KEYS` | `local-dev-key:tenant-local` | Comma-separated `key:tenant` mappings. |
| `STORE_DEFAULT` | `true` | Default for Responses `store` when omitted. |
| `MAX_CHAIN_DEPTH` | `50` | Maximum `previous_response_id` chain depth. |
| `MAX_TOOL_ITERATIONS` | `8` | Maximum model-tool loop turns. |
| `TOOL_TIMEOUT_SECONDS` | `15` | Timeout for built-in tool execution. |
| `BACKEND_TIMEOUT_SECONDS` | `120` | Timeout for model backend HTTP calls. |
| `STREAM_HEARTBEAT_SECONDS` | `15` | Reserved heartbeat cadence for streaming paths. |
| `MAX_OUTPUT_TOKENS_DEFAULT` | `2048` | Default chat-completions `max_tokens`. |
| `AUTO_CREATE_TABLES` | `true` | Local shortcut. Use Alembic in container and production-like paths. |

The Docker stack overrides a few values so the containers talk to each other:

```text
DATABASE_URL=postgresql+asyncpg://respawn:respawn@postgres:5432/respawn
MODEL_BACKEND=ollama
OLLAMA_BASE_URL=http://ollama:11434/v1
DEFAULT_MODEL=gpt-oss:120b
AUTO_CREATE_TABLES=false
```

## API Surface

Respawn currently exposes:

- `POST /v1/responses`
- `GET /v1/responses/{response_id}`
- `DELETE /v1/responses/{response_id}`
- `POST /v1/chat/completions`
- `GET /v1/models`
- `GET /models`
- `GET /healthz`
- `GET /readyz`
- `GET /metrics`

`POST /v1/responses` accepts the practical subset used by the official SDKs:

```json
{
  "model": "gpt-oss-120b",
  "input": "string or list of input items",
  "instructions": "optional system/developer instructions",
  "previous_response_id": "resp_...",
  "store": true,
  "stream": false,
  "temperature": 0.7,
  "top_p": 1,
  "max_output_tokens": 1024,
  "tools": [],
  "tool_choice": "auto",
  "response_format": {},
  "text": {"format": {}},
  "metadata": {}
}
```

Responses use OpenAI-style object shapes and IDs:

- Responses: `resp_...`
- Messages: `msg_...`
- Tool calls: `call_...`
- Usage records: `usage_...`

## Stateful Responses

Respawn implements `previous_response_id` in the gateway, not in Ollama.

When a request includes `previous_response_id`, Respawn:

1. Loads the stored response chain from Postgres or SQLite.
2. Enforces `MAX_CHAIN_DEPTH`.
3. Rejects missing, deleted, inaccessible, or cross-tenant parents.
4. Reconstructs the full chat history.
5. Appends the new input.
6. Sends the complete conversation to the backend.
7. Stores the new response when `store=true`.

Deleted responses are soft-deleted and cannot be retrieved or reused as future
conversation parents.

## Tools

Function tools are passed to the backend as OpenAI-compatible chat-completions
tools. When the model emits tool calls, Respawn validates arguments, executes
registered local tools, appends tool results to the conversation, and calls the
model again until it gets final assistant output or reaches
`MAX_TOOL_ITERATIONS`.

Built-in tools:

- `echo`
- `get_current_time`
- `calculator`

The calculator uses a restricted arithmetic AST evaluator. Respawn does not
enable arbitrary code execution by default.

## Structured Outputs

Respawn supports JSON Schema structured output through either:

- `response_format`
- SDK-style `text={"format": ...}`

The gateway asks the backend for structured output where possible, validates the
final text as JSON against the schema, retries once with a repair instruction,
and returns an OpenAI-shaped `structured_output_validation_failed` error if the
repair also fails.

## Authentication And Tenants

Set `AUTH_DISABLED=true` for local development.

When auth is enabled:

- Bearer tokens are read from the `Authorization` header.
- `LOCAL_OPENAI_API_KEYS` maps API keys to tenant IDs.
- Stored responses are tenant-scoped.
- Cross-tenant retrieve, delete, and `previous_response_id` requests return not
  found.

Example:

```text
AUTH_DISABLED=false
LOCAL_OPENAI_API_KEYS=local-dev-key:tenant-local,other-key:tenant-other
```

## Observability

Respawn includes:

- Structured JSON logging.
- Request IDs through `x-request-id`.
- Request latency metrics.
- Backend latency metrics.
- Token usage metrics.
- Tool execution metrics.
- Error counters.
- `/metrics` Prometheus endpoint.

Logs avoid API keys and full request payloads.

## Development

Install test dependencies and run the suite:

```bash
cd apps/gateway
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[test]"
python -m pytest
```

Run migrations explicitly:

```bash
cd apps/gateway
DATABASE_URL=sqlite+aiosqlite:///./gateway.db alembic upgrade head
```

Validate the Compose file:

```bash
docker compose -f infra/docker/docker-compose.yml config
```

Automated coverage includes request validation, ID prefixes, input normalization,
stateful response chains, soft delete behavior, OpenAI-shaped errors, auth and
tenant isolation, tool execution, structured-output repair, streaming event
formatting, Ollama adapter behavior, and OpenAI Python SDK contract checks.

The GitHub Actions workflow in `.github/workflows/ci.yml` runs the gateway test
suite on Python 3.12 for pushes and pull requests.

## Limitations

- Respawn is a local compatibility gateway, not the hosted OpenAI service.
- It does not reproduce hosted OpenAI tools unless local equivalents are added.
- It does not implement GPU inference or model serving.
- Model quality and tool-calling quality depend on the local model.
- Multimodal support depends on the backend model and adapter support.
- Streaming and SDK parity may need updates as SDKs evolve.
- A public release should include an explicit license chosen by the maintainer.
