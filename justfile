# AoE2 LLM Arena — Monorepo Commands
#
# Layout: apps/* are deployable units (services, CLIs, frontends);
# packages/* are reusable libraries imported by apps. Every Python recipe runs
# through `uv run --package <name>`, which resolves by package name regardless
# of which subdirectory the member lives in. `uv sync` installs every member
# (apps + packages) into .venv/ in one shot.

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
arena-race profile="apps/arena/src/profiles/v1.yaml":
    uv run --package arena aoe2-arena race {{profile}}

arena-smoke:
    uv run --package arena aoe2-arena smoke

arena-rank profile="apps/arena/src/profiles/ranking-v1.yaml":
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

# ── Experiments (ledger: experiments/results.tsv) ──────────────────────────
#
# Every behavior-affecting change needs at least one recorded game before it
# merges (IMPROVEMENT-PLAN.md P0.1). Games run on the Windows VM with the game
# open; the gate runs anywhere.

# Run one full game and append a row to the ledger.
experiment description="manual game run" *ARGS:
    uv run --package autoresearch python -m autoresearch.game_runner --description "{{description}}" {{ARGS}}

# Record N baseline games in a row (P0.1 calls for 3-5).
experiment-baseline n="3" *ARGS:
    for i in $(seq {{n}}); do \
        uv run --package autoresearch python -m autoresearch.game_runner \
            --description "baseline run $i/{{n}}" {{ARGS}}; \
    done

# Merge gate: fails unless the ledger has a row recorded at HEAD.
# Use `--any` to only require a non-empty ledger (bootstrap mode).
experiment-gate *ARGS:
    uv run --package autoresearch python -m autoresearch.experiment_gate {{ARGS}}

# ── Detection (training / inference) ──────────────────────────────────────

export-onnx model="detection/inference/models/aoe2_yolo_v5.pt":
    uv run --package detection python -m detection.training.export_onnx --model {{model}}

export-coreml model="detection/inference/models/aoe2_yolo_v5.pt":
    uv run --package detection python -m detection.training.export_coreml --model {{model}}

train model="yolo11n.pt" *ARGS:
    uv run --package detection python -m detection.training.train_yolo --model {{model}} {{ARGS}}

sync-classes:
    cp packages/detection/src/training/config/classes.yaml apps/detection-server/src/classes.yaml

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

# ── Live dashboard (redis broker shared by arena run + web backend) ────────
#
# To watch a run live in the dashboard, both the producer (arena) and the
# consumer (web backend) must share the compose-stack Redis. `set dotenv-load`
# already puts REDIS_PASSWORD in scope, so make_broker() builds the AUTH'd URL.
# Workflow:
#     just arena-infra-up          # 1. Redis (+ stack) up
#     just arena-web-dev-redis      # 2. web backend on :8000 (redis mode)
#     just arena-ui-dev             # 3. dashboard on :5173
#     just arena-rank-redis         # 4. a run — appears live in the UI

arena-web-dev-redis port="8000":
    ARENA_BROKER_BACKEND=redis uv run --package arena-web aoe2-arena-web --port {{port}}

arena-rank-redis profile="apps/arena/src/profiles/ranking-v1.yaml":
    ARENA_BROKER_BACKEND=redis uv run --package arena aoe2-arena rank {{profile}}

arena-race-redis profile="apps/arena/src/profiles/v1.yaml":
    ARENA_BROKER_BACKEND=redis uv run --package arena aoe2-arena race {{profile}}

# ── Dashboard (apps/dashboard — Vite + React + Tailwind) ──────────────────
#
# JS install is workspace-wide from repo root: `bun install` populates the
# hoisted node_modules/ for both apps/dashboard and apps/landing in one shot.

arena-ui-install:
    bun install

arena-ui-dev: arena-ui-install
    cd apps/dashboard && bun run dev

arena-web-build: arena-ui-install
    cd apps/dashboard && bun run build

# ── Landing (apps/landing — Astro docs site) ──────────────────────────────

landing-dev: arena-ui-install
    cd apps/landing && bun run dev

landing-build: arena-ui-install
    cd apps/landing && bun run build

# Drive the dashboard with Playwright and capture the four arena UI panel
# screenshots embedded in the landing's "See it in action" section. Reads
# real events from DuckDB fixtures under logs/arena/, so no Anthropic API
# key is needed. Auto-starts api (:8000) + dashboard (:5173) if not running.
capture-screenshots: arena-ui-install
    cd apps/landing && bun run capture:screenshots
