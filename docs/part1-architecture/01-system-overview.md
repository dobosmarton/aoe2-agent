# Chapter 1: System Overview

The AoE2 LLM Arena agent plays Age of Empires II autonomously using a two-tier LLM architecture. A Sonnet strategist reads screenshots and sets goals; a Sonnet executor reads YOLO-detected entities as text and executes mouse/keyboard actions. No game API, no OCR, no memory-mapped data.

<aside class="prereqs">

Python 3 and the `async`/`await` mental model. If `asyncio` is new, see [Glossary §asyncio](../glossary.md#a).

</aside>

## 1.1 Two-Tier Architecture

The agent splits decision-making into two models:

**Strategist (Sonnet)** — Runs every 10 turns (or on alarm). Receives the full screenshot as a vision input. Reads resource values, population, and age from the game UI. Creates 3-5 prioritized goals and caches resource readings for the executor.

**Executor (Sonnet)** — Runs every turn. Receives only text: YOLO entity list, cached resource readings, active goals, memory context, and game knowledge. Returns structured actions (clicks, key presses) validated as Pydantic models. Routine turns take a fast single-shot call; combat/housing turns take an agentic tool loop (see [Chapter 4 §4.3](../part2-llm-integration/04-provider-pattern.md)).

The split separates concerns: the strategist owns slow, vision-based planning; the executor owns rapid, text-only tactics. Both run `claude-sonnet-4-6` — the executor was moved from Haiku to Sonnet for more reliable instruction-following — and a per-call `effort` knob (default `low`) keeps the executor fast.

## 1.2 Component Map

```
agent/
├── gameplay_agent/                       # Core agent runtime
│   ├── main.py                # CLI entry point, provider creation
│   ├── config.py              # Pydantic configuration with env var overrides
│   ├── game_loop.py           # Main capture→detect→alarm→strategist→execute cycle
│   ├── memory.py              # Working memory and game state tracking
│   ├── goals.py               # Goal management, alarm system, reward computation
│   ├── goal_logger.py         # Goal progress and completion logging
│   ├── executor.py            # Action execution via pyautogui (dispatch pattern)
│   ├── models.py              # Pydantic action/response validation (7 action types)
│   ├── entity_utils.py        # Entity attribute extraction and summary formatting
│   ├── screen.py              # Screenshot capture via mss
│   ├── window.py              # Game window detection and focus management
│   └── providers/
│       ├── base.py            # Abstract LLM provider interface
│       ├── claude.py          # Sonnet executor (text-only, no images)
│       └── strategist.py      # Sonnet strategist (vision + goal generation)
├── detection/                 # YOLO entity detection (optional)
│   ├── inference/
│   │   ├── detector.py        # EntityDetector, 60 classes, IoU tracking
│   │   ├── remote_detector.py # HTTP client for detection server
│   │   ├── ownership.py       # Blue-dominance ownership classifier
│   │   ├── thresholds.py      # Per-class confidence thresholds
│   │   ├── frame_diff.py      # Frame differencing for rescan optimization
│   │   └── models/            # YOLO v5 model weights (.pt)
│   ├── training/              # Synthetic data gen + YOLO training
│   ├── labeling/              # CVAT integration + class remapping
│   └── extraction/            # SLD sprite extraction from game files
├── data/                      # Game knowledge (optional)
│   ├── game_knowledge.py      # SQLite database wrapper
│   └── knowledge_base/        # Static game data files
├── prompts/
│   ├── system.md              # Executor system prompt
│   └── strategist.md          # Strategist system prompt
├── autoresearch/              # Automated experiment framework
│   ├── game_runner.py         # Timed experiments with metrics
│   ├── orchestrator.py        # Prompt mutation loop
│   ├── metrics.py             # Scoring and analysis
│   └── json_utils.py          # Robust JSON extraction from LLM output
└── logs/                      # Screenshots, goal logs
```

## 1.3 Graceful Degradation

The agent won't crash without optional subsystems, but YOLO detection is practically required for meaningful gameplay.

**Detection** — imported inside a try/except at module level in `apps/agent/src/game_loop.py`:

```python
try:
    from detection.inference.detector import EntityDetector, get_detector
    DETECTION_AVAILABLE = True
except ImportError:
    DETECTION_AVAILABLE = False
```

Without detection, the executor has no entity list — it cannot target units, buildings, or resources by class or ID. The strategist can still read screenshots and set goals, but the executor is limited to hotkeys and hardcoded coordinates. In practice, this makes the agent nearly non-functional: it can't gather resources, train units, or build at specific locations. Detection is technically optional (the agent starts and runs) but practically required for any useful gameplay.

**Game Knowledge** — imported inside a try/except in `apps/agent/src/providers/claude.py`:

```python
try:
    from data.game_knowledge import GameKnowledge, get_db
    GAME_KNOWLEDGE_AVAILABLE = True
except ImportError:
    GAME_KNOWLEDGE_AVAILABLE = False
```

Without the knowledge database, no dynamic context injection occurs. The executor still receives the system prompt and memory context. This is a minor degradation — the agent plays reasonably without it.

**Window Management** — pygetwindow is optional at `apps/agent/src/window.py`. When unavailable, functions return `True` by default — the agent assumes the game is running and focused. Screenshot capture falls back to the full primary monitor.

> **Key Insight**: Detection is the critical optional dependency. Without YOLO, the executor is essentially blind — the experience is very poor. Game knowledge and window management are truly additive enhancements that degrade gracefully.

## 1.4 Configuration

Configuration uses a Pydantic `BaseModel` with environment variable overrides (`apps/agent/src/config.py`):

| Setting | Env Var | Default | Purpose |
|---------|---------|---------|---------|
| `anthropic_api_key` | `ANTHROPIC_API_KEY` | `""` | Claude API authentication |
| `model` | `AOE2_MODEL` | `claude-sonnet-4-6` | Executor model (instruction-following) |
| `executor_effort` | `AOE2_EXECUTOR_EFFORT` | `low` | Executor `output_config` effort (`low`/`medium`/`high`) |
| `strategist_model` | `AOE2_STRATEGIST_MODEL` | `claude-sonnet-4-6` | Strategist model (vision, deeper reasoning) |
| `strategist_interval` | `AOE2_STRATEGIST_INTERVAL` | `10` | Run strategist every N turns |
| `max_tokens` | — | `1536` | Max response tokens per executor call |
| `max_tool_iterations` | — | `7` | Max tool roundtrips per turn (tool-loop path) |
| `detection_imgsz` | — | `1280` | YOLO inference resolution |
| `screenshot_quality` | — | `85` | JPEG quality (1-100) |
| `loop_delay` | `AOE2_LOOP_DELAY` | `0.3` | Seconds between iterations |
| `action_delay` | — | `0.05` | Seconds between individual actions |
| `pipeline_commit_max` | `AOE2_PIPELINE_COMMIT_MAX` | `2` | Actions committed per pipelined (routine) turn; the tail is discarded |
| `save_screenshots` | `AOE2_SAVE_SCREENSHOTS` | `true` | Log screenshots to disk |
| `log_dir` | — | `logs` | Screenshot and log output directory |

A global singleton `config = Config.from_env()` is created at module load time and imported throughout the codebase.

## 1.5 Async-First Architecture

The entire agent runs on asyncio:

- **Entry point**: `asyncio.run(main_async(args))` in `apps/agent/src/main.py`
- **API clients**: `anthropic.AsyncAnthropic` for both executor and strategist
- **Game loop**: `game_loop()` in `apps/agent/src/game_loop.py`
- **Action execution**: `execute_actions()` in `apps/agent/src/executor.py`
- **Delays**: `asyncio.sleep()` for non-blocking waits

pyautogui calls are synchronous but fast (sub-millisecond per click), so they don't block meaningfully.

<aside class="concept" data-title="Async-first architecture (why one event loop, not threads)">

The agent is built on a single asyncio event loop, not threads or processes. The reason is **structural concurrency without locks**: every coroutine runs to the next `await` before yielding, so two coroutines reading and writing the same in-memory data (the entity cache, the goal manager, the memory deque) can never interleave mid-statement. No `threading.Lock`, no race conditions on shared state, no `asyncio.to_thread` for the hot path.

Two patterns recur:

- **`await foo()` for sequential work** — the most common shape. You're waiting on the result before continuing.
- **`asyncio.create_task(foo())` for fire-and-forget parallelism** — you want the work to run *while* something else proceeds, and you'll await the result later (or never). The strategist call in §2.1 Step 7 is the canonical example: the loop dispatches the strategist into the background and continues with the reactive tier + executor on the main path, then awaits the strategist result at cleanup. Routine turns lean on the same pattern for RTC pipelining — the next turn's executor plan computes in the background while the current committed head executes.

The single-loop invariant breaks the moment you call into blocking code (the synchronous `pyautogui.click()` is fast enough that we accept the block; a sync database driver wouldn't be). For genuine background CPU work you'd use `loop.run_in_executor` to dispatch to a thread pool; the broker uses `loop.call_soon_threadsafe` to marshal CLI cross-thread publishes back onto the main loop — see [Appendix B §B.6](../appendix/02-event-brokers-and-redis-streams.md).

</aside>

## 1.6 Logging

Structured logging via structlog with colored console output, configured in `apps/agent/src/main.py`.

Key log events: `iteration_start`, `screenshot_captured`, `detection_complete`, `strategist_response`, `strategist_goals_updated`, `llm_response`, `actions_executed`, `routine_executed`, `pipeline_head_committed`, `action_verification`, `alarm_triggered`, `turn_reward`.

---

## Summary

- Two-tier architecture: Sonnet strategist (vision, goals) + Sonnet executor (text-only, actions; single-shot routine turns + tool loop for combat)
- A deterministic reactive tier handles routine villager upkeep every turn with no LLM call; routine turns pipeline (RTC) and entity-affecting actions are verified by re-detection
- Detection is practically required for useful gameplay; game knowledge and window management are truly optional
- Pydantic for config and validation, structlog for observability, asyncio for concurrency
- Goal-driven gameplay with alarm system for emergency defense

## Related Topics

- [Chapter 2: Game Loop Pipeline](./02-game-loop-pipeline.md) — the iteration cycle in detail
- [Chapter 4: Provider Pattern](../part2-llm-integration/04-provider-pattern.md) — how LLM providers are abstracted
- [Chapter 7: Detector Architecture](../part3-entity-detection/07-detector-architecture.md) — the optional YOLO system
