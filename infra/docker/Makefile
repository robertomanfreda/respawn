ENV_FILE ?= $(firstword $(wildcard env) env.example)
COMPOSE = docker compose --env-file $(ENV_FILE)
BASE_FILES = -f docker-compose.yml
BENCHMARK_FILES = -f docker-compose.yml -f docker-compose.benchmark.yml

.PHONY: help env config up up-build down stop restart ps logs logs-respawn logs-ollama build pull preload models health ready metrics grafana benchmark benchmark-up benchmark-clean

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
	@printf "  make benchmark-up    Start stack, then run benchmark\n"
	@printf "  make models          List Ollama models in the container\n"
	@printf "  make metrics         Print Respawn metrics\n"
	@printf "  make env             Create a local override env if missing\n\n"
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
	RESPAWN_BENCHMARK_UID=$$(id -u) RESPAWN_BENCHMARK_GID=$$(id -g) $(COMPOSE) $(BENCHMARK_FILES) run --rm respawn-benchmark

benchmark-up: up-build benchmark

benchmark-clean:
	rm -rf benchmark-results
