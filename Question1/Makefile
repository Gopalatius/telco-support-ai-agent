COMPOSE ?= docker-compose

.DEFAULT_GOAL := help

.PHONY: help install test coverage format lint typecheck run ingest smoke eval-retrieval eval-chat docker-build docker-up docker-ingest docker-smoke docker-down docker-logs

help:
	@printf "%s\n" \
	"install        Install local dependencies with uv" \
	"test           Run pytest" \
	"coverage       Run pytest with coverage for application code" \
	"format         Run ruff formatter" \
	"lint           Run ruff" \
	"typecheck      Run ty" \
	"run            Start the API locally with Granian" \
	"ingest         Ingest the knowledge base locally" \
	"smoke          Run the live provider smoke test locally" \
	"eval-retrieval Run the labeled dense/hybrid/rerank retrieval benchmark" \
	"eval-chat      Run chat-quality and latency evaluation with multi-turn and escalation cases" \
	"docker-build   Build the app image" \
	"docker-up      Start Qdrant and the API container" \
	"docker-ingest  Ingest the knowledge base through Docker" \
	"docker-smoke   Verify the Dockerized API" \
	"docker-logs    Tail API container logs" \
	"docker-down    Stop the Docker stack"

install:
	rtk uv sync --all-extras

test:
	rtk uv run pytest

coverage:
	rtk uv run pytest --cov=src/telco_agent --cov-report=term-missing

format:
	rtk uv run ruff format .

lint:
	rtk uv run ruff check .

typecheck:
	rtk uv run ty check src

run:
	rtk uv run granian --interface asgi --reload telco_agent.api.main:app

ingest:
	rtk uv run python scripts/ingest.py

smoke:
	rtk uv run python scripts/smoke.py

eval-retrieval:
	rtk uv run python scripts/eval_retrieval.py

eval-chat:
	rtk uv run python scripts/eval_chat.py

docker-build:
	@printf "==> Building app image\n"
	rtk $(COMPOSE) build app

docker-up:
	@printf "==> Starting Qdrant and API\n"
	rtk $(COMPOSE) up -d qdrant app

docker-ingest:
	@printf "==> Ingesting knowledge base through Docker\n"
	rtk $(COMPOSE) run --rm ingest

docker-smoke:
	@printf "==> Checking health endpoint\n"
	rtk proxy curl -sSf http://localhost:8000/health
	@printf "\n==> Sending sample chat request\n"
	rtk proxy curl -sS -X POST http://localhost:8000/chat -H 'Content-Type: application/json' -d '{"message":"How much is the late payment fee?","history":[]}'
	@printf "\n"

docker-logs:
	rtk $(COMPOSE) logs -f app

docker-down:
	@printf "==> Stopping Docker stack\n"
	rtk $(COMPOSE) down
