# AoE2 LLM Agent

An AI agent that plays Age of Empires 2: Definitive Edition using a two-tier LLM architecture: a Sonnet strategist reads the resource bar via local OCR and sets goals, a Sonnet executor reads YOLO entity detections and executes actions. Both LLM tiers are text-only — no image is ever sent to Claude.

## Architecture

```
Screenshot → YOLO Detection → Entity List (text)
Screenshot → Local OCR (RapidOCR) → Resource Readings (text)
                                    ↓
Resource Readings → Strategist (Sonnet, text) → Goals
                                    ↓
Entity List + Goals + Resources → Executor (Sonnet, text) → Actions
                                                             ↓
                                                       Mouse/Keyboard
```

**Two-model design:**

| Role | Model | Input | Output | Frequency |
|------|-------|-------|--------|-----------|
| Strategist | `claude-sonnet-4-6` | Text (resources via local OCR) + game state | Goals + resource readings | Every 10 turns, or on alarm |
| Executor | `claude-sonnet-4-6` | Text only (entities, goals, resources) | Mouse/keyboard actions | Every turn |

The executor runs Sonnet (moved from Haiku for more reliable instruction-following) with a per-call `effort` knob (default `low`) for speed. Routine turns take a single-shot structured call; combat/housing turns take an agentic tool loop.

The executor never sees screenshots. All visual information comes from YOLO entity detection (text list of class/position/confidence) and the strategist's cached resource readings.

## The Game Loop

Each iteration (~3-5 seconds):

1. **Capture** — Screenshot the game window via `mss`
2. **Detect** — Run YOLO v9 (single-pass @1280) on screenshot → list of entities with IDs, classes, positions
3. **Classify ownership** — Color-based blue-dominance check on military units (own vs enemy)
4. **Alarm check** — Scan for enemy military → inject emergency defense goals if found
5. **Strategist** (periodic) — reads resources from the bar via local OCR (RapidOCR), then Sonnet creates/updates goals from that text
6. **Reactive tier** — a deterministic, no-LLM rule layer handles routine upkeep first: villager queuing to the age's order target (30 Dark / 35 Feudal), Feudal prep (mill + lumber camp), the mining camp in Feudal, house building at low headroom, and the age-up press. Many turns need no LLM call at all.
7. **Build context** — Assemble text: entities + goals + resources + memory + game knowledge
8. **Execute** — Sonnet reads text context, returns structured actions (Pydantic-validated)
9. **Act** — Execute mouse clicks / keyboard presses via pyautogui
10. **Remember** — Update memory, evaluate goal progress, compute rewards

## Requirements

- Windows 10/11 with AoE2:DE installed
- Python 3.11+ (x64, not ARM64)
- Anthropic API key

## Installation

The repo is a polyglot monorepo:

- **Python (`uv workspace`)** — 10 workspace members. Deployable units live in `apps/`
  (agent, api, arena, autoresearch, detection-server, training-api); reusable libraries
  live in `packages/` (core, data, detection, evaluation). One `uv sync` installs every
  member editable into a shared `.venv/`, plus the dev dependency-group (ruff,
  basedpyright, pytest, fakeredis, hypothesis, pre-commit).
- **TypeScript (`bun workspace`)** — two frontend apps (`apps/dashboard` Vite/React,
  `apps/landing` Astro) share a single root `bun.lock` and hoisted `node_modules/`.
  One `bun install` from the repo root resolves both apps.

```bash
# Install uv: https://docs.astral.sh/uv/getting-started/installation/
uv sync                        # everything (members + dev)
uv sync --no-dev               # slim install, no dev tools
# Or run a specific package's entry point directly:
uv run --package gameplay-agent aoe2-agent
uv run --package arena aoe2-arena race
uv run --package detection-server aoe2-server --model <path>
```

Per-package optional extras (e.g. `.mlpackage` CoreML builds need CoreML,
detection training needs ultralytics) are declared on each package's own
`pyproject.toml`. Reach them via `uv sync --all-extras`.

## Configuration

The project reads configuration from environment variables. For local development we keep
them in a gitignored `.env` file (loaded by `docker compose` automatically, and by the
agent when launched via `just agent`). A documented template lives at `env.example`.

### Quick start

```bash
cp env.example .env        # then edit .env and fill in the values below
```

At minimum, set `ANTHROPIC_API_KEY` — that's all the **gameplay agent** needs. Every other
variable in `env.example` is for the **Synthetic Arena infrastructure** (Langfuse + MinIO +
ClickHouse + Redis + Postgres) and is only consumed by `just arena-infra-up`. If you're
not running the arena stack yet, leaving those blank is fine.

### Gameplay agent (real-game tier)

```bash
# Windows VM
set ANTHROPIC_API_KEY=your-key-here

# macOS / Linux
export ANTHROPIC_API_KEY=your-key-here
```

| Env Var | Default | Purpose |
|---------|---------|---------|
| `ANTHROPIC_API_KEY` | — | Claude API authentication (required) |
| `AOE2_MODEL` | `claude-sonnet-4-6` | Executor model |
| `AOE2_EXECUTOR_EFFORT` | `low` | Executor effort (`low`/`medium`/`high`) |
| `AOE2_STRATEGIST_MODEL` | `claude-sonnet-4-6` | Strategist model |
| `AOE2_STRATEGIST_INTERVAL` | `10` | Run strategist every N turns |
| `AOE2_LOOP_DELAY` | `0.3` | Seconds between iterations |
| `AOE2_SAVE_SCREENSHOTS` | `true` | Save screenshots to logs/ |
| `AOE2_OCR_BACKEND` | `rapidocr` | Resource-bar OCR backend (`rapidocr`/`template`/`tesseract`) |
| `AOE2_DETECTION_HOST` | — | Remote detection server URL (e.g., `http://192.168.64.1:8420`) |
| `AOE2_TEMPERATURE` | `0.0` | Anthropic Messages API temperature (lowest variance) |
| `AOE2_SEED` | — | Local-RNG seed (build-retry jitter); unset = OS entropy |

### Event broker backend (Phase C)

The arena runs every event through an `EventBroker` (see
`docs/design/event-broker-architecture.md`). The default in-process broker has
zero external dependencies and is fine for single-machine work — local
development, CI, and `just arena-smoke` all use it implicitly.

For cross-process replay (a producer in one process feeding consumers in
another — e.g. the FastAPI web server live-tailing a CLI race), switch to the
Redis backend:

| Env Var | Default | Purpose |
|---|---|---|
| `ARENA_BROKER_BACKEND` | `inprocess` | `inprocess` or `redis`. Read once at `make_broker()`; any other value raises `ValueError`. |
| `REDIS_URL` | see below | Explicit connection URL when backend is `redis`. Takes priority over the smart default. |
| `REDIS_PASSWORD` | unset | If set (and `REDIS_URL` is *not* set), the default becomes `redis://:${REDIS_PASSWORD}@localhost:6379/0` — the compose-stack path. Otherwise the default is bare `redis://localhost:6379/0`. |

```bash
# Bring up the compose stack (provides Redis with REDIS_PASSWORD auth):
just arena-infra-up

# Point the agent at it. REDIS_PASSWORD is already in .env from the
# compose setup; the broker will auto-build the URL with it:
export ARENA_BROKER_BACKEND=redis
just arena-smoke   # or any other arena CLI

# Or set REDIS_URL explicitly for non-compose Redis (managed cloud, remote
# host, etc.):
export REDIS_URL="redis://:secret@redis.example.com:6380/0"
```

Install the Redis client when using this backend (`fakeredis` covers CI tests;
real-Redis local work needs the `broker-redis` extra):

```bash
pip install -e ".[broker-redis]"
```

### Reproducibility (Phase 3)

Determinism is asymptotic — per [arxiv 2408.04667](https://arxiv.org/html/2408.04667v5), expect ~5–12% per-decision variance even with `temperature=0`. Promise *statistical* replay over N trials, not byte-identical traces.

Three knobs to make runs as reproducible as Anthropic and the Python stack allow:

- **`AOE2_TEMPERATURE=0.0`** (default) is Anthropic's lowest-variance temperature. Raise it (e.g. `0.7`) for output diversity at the cost of reproducibility.
- **`AOE2_SEED=<int>`** seeds the local RNG used in `executor.py`'s build-retry jitter and Phase 1's `world_sim.render()` default fallback. Two runs with the same seed produce the same RNG sequence. Leave unset to get today's stochastic behavior (OS entropy). **Not passed to the Anthropic API** — `messages.create()` doesn't accept `seed=` as of late 2025; this is purely for the local code paths.
- **Pin model snapshots.** Set `AOE2_MODEL` and `AOE2_STRATEGIST_MODEL` to a dated snapshot (e.g. `claude-sonnet-4-6-2026-XX-XX`) rather than the floating family alias. Floating tags can move under you between runs.

### Synthetic Arena infrastructure (optional)

Required only when bringing up the Docker stack (`just arena-infra-up`). All seven
variables below must be set to non-empty values — Langfuse refuses to boot with empty
secrets. **Never commit the populated `.env`** (it's gitignored).

**Prerequisites:**
- A running Docker daemon (Docker Desktop, OrbStack, or compatible). The current `docker-compose.yml` is tested against OrbStack on macOS; if OrbStack is installed but not running, start it first with `orb start` — `just arena-infra-up` will otherwise fail with `dial unix .../docker.sock: no such file or directory`.
- **At least ~10 GiB of free disk** before the first pull. The full stack (langfuse, postgres, clickhouse, minio, redis, otel-collector) is ~7 GB of images plus a few GB of volumes. Pulls that run out of space mid-extraction leave Docker's layer database in an inconsistent state (`failed to register layer: file exists`), which then poisons all subsequent pulls — see the troubleshooting section below.

| Env Var | How to generate | Notes |
|---|---|---|
| `LANGFUSE_SALT` | `openssl rand -base64 32` | Password-hashing salt inside Langfuse |
| `LANGFUSE_NEXTAUTH_SECRET` | `openssl rand -base64 32` | Session-cookie signing secret |
| `LANGFUSE_ENCRYPTION_KEY` | `openssl rand -hex 32` | **Must be 32-byte hex.** Encrypts API keys at rest |
| `LANGFUSE_DB_PASSWORD` | `openssl rand -base64 24 \| tr -d '=+/'` | Postgres password (no special chars; ends up in a `DATABASE_URL`) |
| `CLICKHOUSE_PASSWORD` | `openssl rand -base64 24 \| tr -d '=+/'` | ClickHouse `default` user password |
| `REDIS_PASSWORD` | `openssl rand -base64 24 \| tr -d '=+/'` | Redis AUTH password |
| `MINIO_ROOT_USER` | Pick a username (default in `env.example`: `arena`) | MinIO admin user |
| `MINIO_ROOT_PASSWORD` | `openssl rand -base64 24 \| tr -d '=+/'` | Min 8 chars; MinIO rejects shorter |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Leave as `http://localhost:4318` | Where the native agent sends OTLP spans |

**One-shot generator** — paste this once after copying `env.example` to `.env`, then fill the
output back into `.env` (or pipe directly with a script of your choosing):

```bash
{
  echo "LANGFUSE_SALT=$(openssl rand -base64 32)"
  echo "LANGFUSE_NEXTAUTH_SECRET=$(openssl rand -base64 32)"
  echo "LANGFUSE_ENCRYPTION_KEY=$(openssl rand -hex 32)"
  echo "LANGFUSE_DB_PASSWORD=$(openssl rand -base64 24 | tr -d '=+/')"
  echo "CLICKHOUSE_PASSWORD=$(openssl rand -base64 24 | tr -d '=+/')"
  echo "REDIS_PASSWORD=$(openssl rand -base64 24 | tr -d '=+/')"
  echo "MINIO_ROOT_PASSWORD=$(openssl rand -base64 24 | tr -d '=+/')"
}
```

Verify the stack accepts the values:

```bash
just arena-infra-up        # docker compose up -d --wait
just arena-infra-status    # every service should be "healthy"
```

Langfuse UI lands at <http://localhost:3000>; MinIO console at <http://localhost:9001>.

#### Notes on the compose file

- `langfuse-web` and `langfuse-worker` healthchecks target `http://$(hostname):PORT/...` rather than `http://localhost:...`. The Langfuse v3 image starts Next.js with `-H $(hostname)`, which binds Next.js to the container's external interface only — `localhost` returns `Connection refused`. Use the `CMD-SHELL` form (with `$$(hostname)` to escape compose interpolation) if you adjust these.
- `otel-collector` runs `healthcheck: disable: true` because the upstream image (`otel/opentelemetry-collector-contrib`) is distroless: no shell, no `wget`, no `busybox` — any in-container probe fails with `OCI runtime exec failed: ... no such file or directory`. The collector logs `"Everything is ready"` itself once started, and nothing in the stack `depends_on` its health.

#### Troubleshooting

| Symptom (from `just arena-infra-up`) | Likely cause | Fix |
|---|---|---|
| `dial unix .../docker.sock: no such file or directory` | Docker daemon not running | `orb start` (OrbStack) or launch Docker Desktop |
| `failed to register layer: rename .../tmp/write-set-N .../sha256/<hex>: file exists` | Orphan layers in `layerdb` from a previously-killed pull (usually caused by disk filling up mid-extraction) | First free disk: `docker image prune -a -f`. Then `orb restart docker` and retry. If errors persist on different SHAs, the daemon has multiple orphan chain-ids whose `diff` digest isn't referenced by any image manifest — `prune` won't remove them because they're unreachable from any image. The reliable fix is to compute the set of reachable chain-ids from all manifests in `imagedb/content/sha256/` (fold `rootfs.diff_ids` as `chain[i] = sha256("sha256:<chain[i-1]> sha256:<diff[i]>")`) and `rm -rf` the unreachable entries from `layerdb/sha256/` plus their `cache-id` overlay2 backings. Last-resort: `orb delete -f docker` (nukes all Docker data) |
| `unexpected EOF` mid-pull | Either flaky network *or* the Docker daemon crashed (often disk pressure) | Check `df -h` and `orb status` — if OrbStack went to `Stopped`, the daemon died; bring it back with `orb start` and free disk before retrying |
| `dependency failed to start: container ... is unhealthy` | A service started but its healthcheck never goes green | `docker logs <container>` to see if the app is actually up. If yes, the healthcheck itself is wrong (wrong port, wrong host, missing tooling in image) — inspect with `docker inspect <container> --format '{{json .State.Health}}'` |

## Usage

```bash
# Run the agent
just agent

# Run N iterations
just agent --iterations 20

# Single test iteration (no action execution)
just agent --test

# Run the detection server (macOS host)
just server --model packages/detection/src/inference/models/aoe2_yolo_v9.onnx

# Frontend dev servers (workspace-wide install happens once at repo root)
bun install
just arena-ui-dev          # apps/dashboard (Vite + React, dashboard SPA)
just landing-dev           # apps/landing (Astro docs site)

# Detection training tracker (dataset coverage + prelabel review, :8100)
just training-seed         # populate logs/training/tracker.db from disk (idempotent)
just training-api-dev      # then browse http://localhost:5173/training

# Autoresearch: timed experiment with metrics
uv run --package autoresearch python -m autoresearch.game_runner --time-budget 600 --description "test run"
```

## Project Structure

```
agent/                                     # monorepo root (uv + bun workspaces)
├── pyproject.toml                         # uv workspace declaration + shared tool config
├── package.json                           # bun workspace declaration (apps/dashboard, apps/landing)
├── bun.lock                               # single root lockfile for the JS workspace
├── justfile                               # cross-language task runner
├── docs/                                  # Architecture chapters, ADRs, runbooks
├── tests/                                 # Cross-package integration tests
├── apps/                                  # Deployable units (services, CLIs, frontends)
│   ├── agent/                             # Python — real-game loop + providers + scenarios (was packages/gameplay-agent)
│   ├── api/                               # Python — FastAPI + SSE backend for replay/fork (was packages/arena-web)
│   ├── arena/                             # Python — synthetic arena CLI (race / smoke / rank)
│   ├── autoresearch/                      # Python — prompt-optimization loop
│   ├── dashboard/                         # TypeScript — Vite + React arena replay UI + training tracker (was ui/)
│   ├── detection-server/                  # Python — macOS-hosted YOLO inference endpoint
│   ├── landing/                           # TypeScript — Astro docs site (was web/)
│   └── training-api/                      # Python — FastAPI + SQLite detection dataset tracker
├── packages/                              # Reusable libraries (imported by apps)
│   ├── core/                              # Pure types: Event, Payload, WorldState, DetectedEntity
│   ├── data/                              # AoE2 game-knowledge SQLite DB
│   ├── detection/                         # YOLO inference, training, labeling, SLD extraction
│   └── evaluation/                        # Event broker, DuckDB persister, world sim, fork
└── logs/                                  # Runtime: screenshots, goal logs, DuckDB event files
```

Each Python workspace member has its own `pyproject.toml` declaring deps + optional
extras. The dependency graph is one-way (`apps/` → `packages/`, never the reverse),
enforced by uv at install time via `[tool.uv.sources]` in the workspace root. uv
resolves members **by package name**, not by directory path — so internal
dependencies like `dependencies = ["core", "detection"]` work regardless of which
subdirectory the member lives in.

For the full chapter-level walkthrough see [`docs/index.md`](docs/index.md).

## Key Systems

### Goal Management (`gameplay_agent/goals.py`)

The strategist creates prioritized goals (e.g., "Reach 10 population", "Advance to Feudal Age"). The executor receives these as context and follows them in priority order. Goals have:
- **Type**: local (complete quickly) or global (long-term)
- **Metric**: population, food, wood, gold, stone, age
- **Priority**: 1-10 (10 = most urgent)
- **Progress**: 0.0-1.0, auto-computed from game state

### Alarm System (`gameplay_agent/goals.py`)

Scans YOLO detections for 21 enemy military classes. Uses color-based ownership detection (`detection/inference/ownership.py`) to distinguish own units (blue, Player 1) from enemy units. On enemy detection:
- Injects priority-10 "Defend base" goal
- Triggers early strategist wake-up

### Entity Detection (`detection/`)

60-class YOLO v9 model served single-pass @1280 (real micro-F1 ≈ 0.67; the older 92.2% mAP50 was a synthetic v5-lineage figure). Entities persist across frames via IoU tracking (e.g., `sheep_0` stays `sheep_0`). The executor supports 9 base action types (click, right_click, press, build, queue_villager, drag, wait, scroll, detect) and can target entities by class (`target_class: "sheep"`) or by ID (`target_id: "sheep_0"`).

### Remote Detection Server (`server/`)

Offloads YOLO inference to the macOS host's Neural Engine via CoreML (~15ms per tile vs ~1.2s on VM CPU). The agent talks to it over HTTP with automatic fallback to local ONNX.

### Action Feedback (`gameplay_agent/game_loop.py`)

Action success/failure is tracked via `ActionResult` objects returned by the executor. Failed actions (e.g., unresolved target_id) are recorded in memory and fed back to the LLM as context for the next turn.

### Autoresearch (`autoresearch/`)

Automated experiment framework. Runs timed games, collects metrics (peak population, food gathered, survival time, action success rate), and scores performance for prompt optimization.

### Synthetic Perception (`evaluation/world_sim.py`)

Projects an in-memory `WorldState` to the same `DetectedEntity` schema the real YOLO detector emits, so the agent's perception layer can be exercised against a fully deterministic, in-process world — no game, no screenshots, no model weights. This is the first step of the synthetic-arena buildout (Langfuse + the perception projection live in the same tier).

Two API surfaces:

- `evaluation.world_sim.render(state, width, height, seed=None) -> list[DetectedEntity]` — pure projection. One `town_center` always, `state.population` villagers, one entity per `state.buildings` entry placed on a stable index-based grid. Villagers in `villager_queue` are deliberately excluded (queued for production, not yet on the map). Confidence is `1.0` (ground truth). Same `state` + same dims + same seed ⇒ identical output.
- `detection.inference.mock.mock_detect_from_world(screenshot, id_factory, world_state) -> list[DetectedEntity]` — sibling of `mock_detect()` that delegates to `render()`. Use this where the real-game tier would call `mock_detect()`.

Example:

```python
from evaluation.world_sim import WorldState, render

state = WorldState(
    food=200.0, wood=200.0, gold=0.0, stone=0.0,
    population=5, pop_cap=25, age="Dark Age",
    buildings=["mill"], villager_queue=[], age_up_ticks_remaining=0, turn=0,
)
entities = render(state, 1920, 1080)  # list[DetectedEntity], confidence=1.0
```

**Schema lock.** `tests/test_detector.py::TestSyntheticRenderSchemaContract` is parametrized over both `mock_detect` (legacy) and `mock_detect_from_world` (new), asserting 10 invariants on each (id non-empty, class_name in canonical YOLO list, bbox well-ordered and within dims, center inside bbox, confidence ∈ [0,1], area > 0, `to_dict()` keys, sort order, id uniqueness) plus a state-sensitivity test (`population=15` yields 7 more villager entities than `population=8`). Any future drift between the two perception surfaces fails CI.

**Real-game tier impact: zero.** Nothing in `gameplay_agent/` was modified; the existing `mock_detect()` keeps its frozen Dark-Age fixture behavior. Synthetic-arena callers (Phase 2 and beyond) reach for the new function explicitly.

## Documentation

See [docs/index.md](docs/index.md) for the full table of contents (8 parts, 23 chapters).

Most useful entry points:

- **Real-game agent**: Parts 1–4 ([game loop](docs/part1-architecture/02-game-loop-pipeline.md), [LLM integration](docs/part2-llm-integration/04-provider-pattern.md), [detection](docs/part3-entity-detection/07-detector-architecture.md), [game knowledge](docs/part4-game-knowledge/10-knowledge-database.md)).
- **Synthetic arena**: [Part 6](docs/part6-evaluation-arena/14-arena-overview.md) (CLI / broker / ranking / world sim).
- **Arena web UI**: [Part 7](docs/part7-arena-web/19-web-architecture.md) (SSE backend + Vite/React frontend).
- **Autoresearch prompt loop**: [Part 8](docs/part8-autoresearch/22-autoresearch-overview.md).
- **Why decisions were made**: [Architecture Decision Records](docs/adr/) (broker-first, Redis Streams, basedpyright, Bradley-Terry, Vite/React).
- **Operational checklists**: [Runbooks](docs/runbooks/) (Redis ops, switching broker backend, debug a stuck fork, Windows VM bring-up).
- **Deployment**: [deployment-guide.md](docs/deployment-guide.md) (Mac + Windows VM first-time setup).
- **Detection internals**: [detection/README.md](detection/README.md).

## License

MIT
