ENV_FILE ?= $(firstword $(wildcard env) env.example)
COMPOSE = docker compose --env-file $(ENV_FILE)
BASE_FILES = -f docker-compose.yml
BENCHMARK_FILES = -f docker-compose.yml -f docker-compose.benchmark.yml
BENCHMARK_MOCK_FILES = -f docker-compose.benchmark.mock.yml
REAL_BENCHMARK_OUTPUT ?= /results/respawn-benchmark-real.json
REAL_BENCHMARK_RUNS ?= 3
REAL_BENCHMARK_TIMEOUT_SECONDS ?= 300

.PHONY: help env config up up-build down stop restart ps logs logs-respawn logs-ollama build pull preload models health ready metrics grafana benchmark benchmark-real benchmark-mock benchmark-up benchmark-clean

help:
	@printf "Respawn Docker helpers\n\n"
	@printf "Usage:\n"
	@printf "  make up              Start the stack\n"
	@printf "  make up-build        Build and start the stack\n"
	@printf "  make down            Stop and remove stack containers\n"
	@printf "  make ps              Show container status\n"
	@printf "  make logs            Follow all logs\n"
	@printf "  make logs-respawn    Follow Respawn logs\n"
	@printf "  make logs-ollama     Follow Ollama logs\n"
	@printf "  make benchmark       Run the benchmark/feature suite\n"
	@printf "  make benchmark-mock  Run benchmark smoke mode with mock backend\n"
	@printf "  make benchmark-up    Start stack, then run benchmark\n"
	@printf "  make models          List Ollama models in the container\n"
	@printf "  make metrics         Print Respawn metrics\n"
	@printf "  make env             Create a local override env if missing\n\n"
	@printf "Release gates:\n"
	@printf "  make benchmark-real  Run full real Ollama benchmark with no tag filters\n\n"
	@printf "Using env file: $(ENV_FILE)\n"

env:
	@test -f env || printf "# Local overrides for this machine.\n# Example: OLLAMA_MODELS_PATH=/usr/share/ollama/.ollama\n" > env
	@printf "env ready: %s\n" "$$(pwd)/env"

config:
	$(COMPOSE) $(BASE_FILES) config

up:
	$(COMPOSE) $(BASE_FILES) up -d

up-build:
	$(COMPOSE) $(BASE_FILES) up -d --build

down:
	$(COMPOSE) $(BASE_FILES) down

stop:
	$(COMPOSE) $(BASE_FILES) stop

restart:
	$(COMPOSE) $(BASE_FILES) restart respawn

ps:
	$(COMPOSE) $(BASE_FILES) ps -a

logs:
	$(COMPOSE) $(BASE_FILES) logs -f

logs-respawn:
	$(COMPOSE) $(BASE_FILES) logs -f respawn

logs-ollama:
	$(COMPOSE) $(BASE_FILES) logs -f ollama ollama-preload

build:
	$(COMPOSE) $(BASE_FILES) build respawn

pull:
	$(COMPOSE) $(BASE_FILES) pull

preload:
	$(COMPOSE) $(BASE_FILES) up --force-recreate ollama-preload

models:
	$(COMPOSE) $(BASE_FILES) exec ollama ollama list

health:
	@curl -fsS http://localhost:8080/healthz && printf "\n"

ready:
	@curl -fsS http://localhost:8080/readyz && printf "\n"

metrics:
	@curl -fsS http://localhost:8080/metrics

grafana:
	@printf "Grafana: http://localhost:3000\n"
	@printf "Default login: admin / respawn\n"

benchmark:
	@mkdir -p benchmark-results
	RESPAWN_BENCHMARK_UID=$$(id -u) RESPAWN_BENCHMARK_GID=$$(id -g) $(COMPOSE) $(BENCHMARK_FILES) up -d --build respawn
	RESPAWN_BENCHMARK_UID=$$(id -u) RESPAWN_BENCHMARK_GID=$$(id -g) RESPAWN_BENCHMARK_GIT_SHA=$$(git -C ../.. rev-parse --short HEAD 2>/dev/null || true) $(COMPOSE) $(BENCHMARK_FILES) run --rm respawn-benchmark

benchmark-real:
	@mkdir -p benchmark-results
	MODEL_BACKEND=ollama \
	RESPAWN_BENCHMARK_UID=$$(id -u) \
	RESPAWN_BENCHMARK_GID=$$(id -g) \
	RESPAWN_BENCHMARK_INCLUDE_TAGS= \
	RESPAWN_BENCHMARK_EXCLUDE_TAGS= \
	RESPAWN_BENCHMARK_COMPARE_TO= \
	RESPAWN_BENCHMARK_COVERAGE_GATE=true \
	RESPAWN_BENCHMARK_EXPECT_OLLAMA_METRICS=true \
	RESPAWN_BENCHMARK_RUNS=$(REAL_BENCHMARK_RUNS) \
	RESPAWN_BENCHMARK_TIMEOUT_SECONDS=$(REAL_BENCHMARK_TIMEOUT_SECONDS) \
	RESPAWN_BENCHMARK_OUTPUT=$(REAL_BENCHMARK_OUTPUT) \
	$(COMPOSE) $(BENCHMARK_FILES) up -d --build respawn
	MODEL_BACKEND=ollama \
	RESPAWN_BENCHMARK_UID=$$(id -u) \
	RESPAWN_BENCHMARK_GID=$$(id -g) \
	RESPAWN_BENCHMARK_GIT_SHA=$$(git -C ../.. rev-parse --short HEAD 2>/dev/null || true) \
	RESPAWN_BENCHMARK_INCLUDE_TAGS= \
	RESPAWN_BENCHMARK_EXCLUDE_TAGS= \
	RESPAWN_BENCHMARK_COMPARE_TO= \
	RESPAWN_BENCHMARK_COVERAGE_GATE=true \
	RESPAWN_BENCHMARK_EXPECT_OLLAMA_METRICS=true \
	RESPAWN_BENCHMARK_RUNS=$(REAL_BENCHMARK_RUNS) \
	RESPAWN_BENCHMARK_TIMEOUT_SECONDS=$(REAL_BENCHMARK_TIMEOUT_SECONDS) \
	RESPAWN_BENCHMARK_OUTPUT=$(REAL_BENCHMARK_OUTPUT) \
	$(COMPOSE) $(BENCHMARK_FILES) run --rm respawn-benchmark

benchmark-mock:
	@mkdir -p benchmark-results
	$(COMPOSE) -p respawn_benchmark_mock $(BENCHMARK_MOCK_FILES) up -d --build respawn
	RESPAWN_BENCHMARK_UID=$$(id -u) RESPAWN_BENCHMARK_GID=$$(id -g) RESPAWN_BENCHMARK_GIT_SHA=$$(git -C ../.. rev-parse --short HEAD 2>/dev/null || true) $(COMPOSE) -p respawn_benchmark_mock $(BENCHMARK_MOCK_FILES) run --rm respawn-benchmark

benchmark-up: up-build benchmark

benchmark-clean:
	rm -rf benchmark-results
