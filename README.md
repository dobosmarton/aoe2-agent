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

```bash
set ANTHROPIC_API_KEY=your-key-here
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
