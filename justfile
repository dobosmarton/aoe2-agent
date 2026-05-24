# AoE2 LLM Arena — Monorepo Commands
#
# After the workspace migration, every recipe runs through `uv run --package
# <name>` so the right editable install resolves. `uv sync` installs every
# workspace member from packages/* into .venv/ in one shot.

set dotenv-load

# ── Install ────────────────────────────────────────────────────────────────

# Editable install of every workspace member + dev tools.
install:
    uv sync

# Editable install without dev tools (slim).
install-prod:
    uv sync --no-dev

# Refresh the lock after pyproject changes.
lock:
    uv lock

# ── Run entry points ───────────────────────────────────────────────────────

# Gameplay agent (Windows VM).
agent *ARGS:
    uv run --package gameplay-agent aoe2-agent {{ARGS}}

# Detection server (macOS host).
server *ARGS:
    uv run --package detection-server aoe2-server {{ARGS}}

# Autoresearch prompt-optimization loop.
autoresearch *ARGS:
    uv run --package autoresearch aoe2-autoresearch {{ARGS}}

# Arena CLI subcommands.
arena-race profile="packages/arena/src/profiles/v1.yaml":
    uv run --package arena aoe2-arena race {{profile}}

arena-smoke:
    uv run --package arena aoe2-arena smoke

arena-rank profile="packages/arena/src/profiles/ranking-v1.yaml":
    uv run --package arena aoe2-arena rank {{profile}}

# Arena replay/inspect web server.
arena-web-dev port="8000":
    uv run --package arena-web aoe2-arena-web --port {{port}}

# ── Quality gate ───────────────────────────────────────────────────────────

lint:
    uv run ruff check .

lint-fix:
    uv run ruff check . --fix

format:
    uv run ruff format .

typecheck:
    uv run basedpyright

test *ARGS:
    uv run pytest {{ARGS}}

coverage:
    uv run pytest --cov --cov-report=term-missing

check: lint typecheck test

# ── Scenarios + evaluation ────────────────────────────────────────────────

eval *ARGS:
    uv run --package gameplay-agent python -m gameplay_agent.scenario_runner {{ARGS}}

eval-all:
    uv run --package gameplay-agent python -m gameplay_agent.scenario_runner --all

# ── Detection (training / inference) ──────────────────────────────────────

export-onnx model="detection/inference/models/aoe2_yolo_v5.pt":
    uv run --package detection python -m detection.training.export_onnx --model {{model}}

export-coreml model="detection/inference/models/aoe2_yolo_v5.pt":
    uv run --package detection python -m detection.training.export_coreml --model {{model}}

train model="yolo11n.pt" *ARGS:
    uv run --package detection python -m detection.training.train_yolo --model {{model}} {{ARGS}}

sync-classes:
    cp packages/detection/src/training/config/classes.yaml packages/detection-server/src/classes.yaml

# ── Synthetic arena infrastructure ────────────────────────────────────────

arena-infra-up:
    docker compose up -d --wait

arena-infra-down:
    docker compose down

arena-infra-nuke:
    docker compose down -v

arena-infra-logs:
    docker compose logs -f

arena-infra-status:
    docker compose ps
    @echo ""
    @echo "Health summary:"
    @docker compose ps --format '{{{{.Service}}}}\t{{{{.Status}}}}' | column -t

arena-up: arena-infra-up

# Print the env-var exports needed to point producers at the compose-stack
# Redis broker. `REDIS_PASSWORD` is in .env from the compose setup; when in
# scope, `make_broker()` auto-builds the AUTH'd URL — no need to construct
# REDIS_URL by hand. Use:
#     set -a; . ./.env; set +a
#     export ARENA_BROKER_BACKEND=redis
#     just arena-smoke
arena-broker-redis-env:
    @set -a; . ./.env; set +a; \
        echo "export ARENA_BROKER_BACKEND=redis"; \
        echo "# REDIS_PASSWORD is already in scope from .env; make_broker()"; \
        echo "# auto-builds redis://:\$REDIS_PASSWORD@localhost:6379/0"

# ── UI (Vite + React + Tailwind) ──────────────────────────────────────────

arena-ui-install:
    cd ui && bun install

arena-ui-dev: arena-ui-install
    cd ui && bun run dev

arena-web-build: arena-ui-install
    cd ui && bun run build
