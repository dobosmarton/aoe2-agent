# AoE2 LLM Agent

An AI agent that plays Age of Empires 2: Definitive Edition using a two-tier LLM architecture: a Sonnet strategist reads screenshots and sets goals, a Haiku executor reads YOLO entity detections and executes actions.

## Architecture

```
Screenshot → YOLO Detection → Entity List (text)
                                    ↓
Screenshot → Strategist (Sonnet) → Goals + Resource Readings
                                    ↓
Entity List + Goals + Resources → Executor (Haiku) → Actions
                                                       ↓
                                                 Mouse/Keyboard
```

**Two-model design:**

| Role | Model | Input | Output | Frequency |
|------|-------|-------|--------|-----------|
| Strategist | `claude-sonnet-4-6` | Screenshot (vision) + game state | Goals + resource readings | Every 10 turns, or on alarm |
| Executor | `claude-haiku-4-5` | Text only (entities, goals, resources) | Mouse/keyboard actions | Every turn (~1s) |

The executor never sees screenshots. All visual information comes from YOLO entity detection (text list of class/position/confidence) and the strategist's cached resource readings.

## The Game Loop

Each iteration (~3-5 seconds):

1. **Capture** — Screenshot the game window via `mss`
2. **Detect** — Run YOLO v5 on screenshot → list of entities with IDs, classes, positions
3. **Classify ownership** — Color-based blue-dominance check on military units (own vs enemy)
4. **Alarm check** — Scan for enemy military → inject emergency defense goals if found
5. **Strategist** (periodic) — Sonnet reads screenshot, extracts resources, creates/updates goals
6. **Build context** — Assemble text: entities + goals + resources + memory + game knowledge
7. **Execute** — Haiku reads text context, returns structured actions (Pydantic-validated)
8. **Act** — Execute mouse clicks / keyboard presses via pyautogui
9. **Remember** — Update memory, evaluate goal progress, compute rewards

## Requirements

- Windows 10/11 with AoE2:DE installed
- Python 3.11+ (x64, not ARM64)
- Anthropic API key

## Installation

```bash
python -m venv venv
venv\Scripts\activate          # Windows; on macOS/Linux: source venv/bin/activate
pip install -e .               # core agent
pip install -e ".[dev,server]" # add dev tooling + detection server
```

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
| `AOE2_MODEL` | `claude-haiku-4-5` | Executor model |
| `AOE2_STRATEGIST_MODEL` | `claude-sonnet-4-6` | Strategist model |
| `AOE2_STRATEGIST_INTERVAL` | `10` | Run strategist every N turns |
| `AOE2_LOOP_DELAY` | `1.0` | Seconds between iterations |
| `AOE2_SAVE_SCREENSHOTS` | `true` | Save screenshots to logs/ |
| `AOE2_DETECTION_HOST` | — | Remote detection server URL (e.g., `http://192.168.64.1:8420`) |

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
just server --model detection/inference/models/aoe2_yolo_v5.onnx

# Autoresearch: timed experiment with metrics
python -m autoresearch.game_runner --time-budget 600 --description "test run"
```

## Project Structure

```
agent/
├── gameplay_agent/                # Gameplay agent (Windows VM)
│   ├── main.py                    # CLI entry point
│   ├── config.py                  # Pydantic config with env var overrides
│   ├── game_loop.py               # Main capture→detect→think→act loop
│   ├── executor.py                # Mouse/keyboard action execution (dispatch pattern)
│   ├── models.py                  # Pydantic models (7 action types, LLMResponse)
│   ├── entity_utils.py            # Entity attribute extraction and summary formatting
│   ├── memory.py                  # Working memory and game state tracking
│   ├── goals.py                   # Goal management, alarm system, rewards
│   ├── screen.py                  # Screenshot capture via mss
│   ├── window.py                  # AoE2 window detection and focus
│   └── providers/                 # LLM providers (Claude executor + strategist)
├── server/                        # Detection API server (macOS host)
│   ├── app.py                     # FastAPI + CoreML/ONNX inference
│   └── classes.yaml               # Bundled class definitions
├── pyproject.toml                 # Project + tool config; single source of truth for deps
├── detection/                     # YOLO entity detection (shared)
│   ├── inference/
│   │   ├── detector.py            # EntityDetector, 60 classes, tracking
│   │   ├── remote_detector.py     # HTTP client for detection server
│   │   ├── ownership.py           # Blue-dominance ownership classifier
│   │   ├── thresholds.py          # Per-class confidence thresholds
│   │   ├── frame_diff.py          # Frame differencing for rescan optimization
│   │   └── models/                # YOLO model weights
│   ├── training/                  # Synthetic data gen + YOLO training
│   ├── labeling/                  # CVAT/COCO labeling tools
│   └── docs/                      # Detection documentation
├── data/                          # Game knowledge database
├── prompts/                       # System prompts (executor + strategist)
├── autoresearch/                  # Automated experiment framework
├── justfile                       # Monorepo commands
└── logs/                          # Screenshots and goal logs
```

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

60-class YOLO v5 model with 92.2% mAP50 accuracy. Entities persist across frames via IoU tracking (e.g., `sheep_0` stays `sheep_0`). The executor supports 7 action types (click, right_click, press, drag, wait, scroll, detect) and can target entities by class (`target_class: "sheep"`) or by ID (`target_id: "sheep_0"`).

### Remote Detection Server (`server/`)

Offloads YOLO inference to the macOS host's Neural Engine via CoreML (~15ms per tile vs ~1.2s on VM CPU). The agent talks to it over HTTP with automatic fallback to local ONNX.

### Action Feedback (`gameplay_agent/game_loop.py`)

Action success/failure is tracked via `ActionResult` objects returned by the executor. Failed actions (e.g., unresolved target_id) are recorded in memory and fed back to the LLM as context for the next turn.

### Autoresearch (`autoresearch/`)

Automated experiment framework. Runs timed games, collects metrics (peak population, food gathered, survival time, action success rate), and scores performance for prompt optimization.

## Documentation

See [docs/index.md](docs/index.md) for detailed architecture documentation.

See [detection/README.md](detection/README.md) for the entity detection system.

## License

MIT
