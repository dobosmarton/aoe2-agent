# Chapter 1: System Overview

The AoE2 LLM Arena agent plays Age of Empires II autonomously using three AI roles. Local OCR and detection publish immutable observations. A strategist sets goals and allocation; TypeSafe System One selects bounded routine economy, age-up, and basic military actions; an executor handles combat and bounded recovery. No game API or memory-mapped data is used, and no screenshot is sent to a model.

<aside class="prereqs">

Python 3 and the `async`/`await` mental model. If `asyncio` is new, see [Glossary §asyncio](../glossary.md#a).

</aside>

## 1.1 Decision Architecture

The agent runs three concurrent loops—perception, actor-with-policy, and deliberate work—over one action catalog and per-game purchase ledger:

**Strategist** — Periodically proposes goals and resource allocation from locally observed state. It cannot modify observed resources, population, age, or idle status.

**TypeSafe policy** — Once per ordinary frame, chooses a named action and allocation focus from a feasible catalog. The actor awaits advice for up to two seconds, then revalidates it against the latest observation under the input lock. A newer frame alone does not invalidate a useful choice. Invalid, low-confidence, timed-out, or failed advice uses a deterministic catalog fallback.

**Deliberate executor** — Runs only for combat alarms, tactical handoffs, or recovery after three failed actions/settlements or a 30-second food stall. Recovery has a three-tool limit and only catalog-guarded economic tools. Each purchase reserves resources before the spending input; later observations settle it. Camera-moving input requires a fresh frame before any dependent click.

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
│   ├── models.py              # Pydantic action/response validation (8 action types)
│   ├── entity_utils.py        # Entity attribute extraction and summary formatting
│   ├── screen.py              # Screenshot capture via mss
│   ├── window.py              # Game window detection and focus management
│   └── providers/
│       ├── base.py            # Abstract LLM provider interface
│       ├── executor_provider.py          # Sonnet executor (text-only, no images)
│       └── strategist.py      # Sonnet strategist (local OCR + goal generation)
├── detection/                 # YOLO entity detection (optional)
│   ├── inference/
│   │   ├── detector.py        # EntityDetector, 60 classes, IoU tracking
│   │   ├── remote_detector.py # HTTP client for detection server
│   │   ├── ownership.py       # Blue-dominance ownership classifier
│   │   ├── thresholds.py      # Per-class confidence thresholds
│   │   ├── frame_diff.py      # Frame differencing for rescan optimization
│   │   └── models/            # YOLO26 (v9) model weights (.pt/.onnx)
│   ├── training/              # Synthetic data gen + YOLO training
│   ├── labeling/              # CVAT integration + class definitions
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

Without detection, the executor has no entity list — it cannot target units, buildings, or resources by class or ID. The strategist can still read the resource bar (local OCR) and set goals, but the executor is limited to hotkeys and hardcoded coordinates. In practice, this makes the agent nearly non-functional: it can't gather resources, train units, or build at specific locations. Detection is technically optional (the agent starts and runs) but practically required for any useful gameplay.

**Game Knowledge** — imported inside a try/except in `apps/agent/src/providers/executor_provider.py`:

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
| `llm_api_key` | `AOE2_LLM_API_KEY` | `""` | Model API authentication |
| `typesafe_api_key` | `TYPESAFE_API_KEY` | `""` | Required routine policy authentication |
| `typesafe_model` | `AOE2_TYPESAFE_MODEL` | `jev-1.13.0` | System One policy model |
| `policy_timeout` | `AOE2_POLICY_TIMEOUT` | `2.0` | Bounded policy wait before latest-frame revalidation |
| `llm_wire` | `AOE2_LLM_WIRE` | `openai` | Adapter: `openai`, `zen` or `anthropic` |
| `llm_base_url` | `AOE2_LLM_BASE_URL` | `""` | Endpoint override; empty uses the adapter's own |
| `model` | `AOE2_MODEL` | `gpt-5.6-luna` | Executor model (fast; runs every turn) |
| `executor_effort` | `AOE2_EXECUTOR_EFFORT` | `low` | Executor `output_config` effort (`low`/`medium`/`high`) |
| `strategist_model` | `AOE2_STRATEGIST_MODEL` | `gpt-5.6-terra` | Strategist model (strong; deeper reasoning) |
| `strategist_interval` | `AOE2_STRATEGIST_INTERVAL` | `10` | Run strategist every N turns |
| `max_tokens` | — | `1536` | Max response tokens per executor call |
| `max_tool_iterations` | — | `7` | Max tool roundtrips per turn (tool-loop path) |
| `detection_imgsz` | — | `1280` | YOLO inference resolution (matches v9's training resolution) |
| `screenshot_quality` | — | `85` | JPEG quality (1-100) |
| `ocr_backend` | `AOE2_OCR_BACKEND` | `rapidocr` | Resource-bar OCR backend (`rapidocr`/`template`/`tesseract`) |
| `perceive_interval` | `AOE2_PERCEIVE_INTERVAL` | `0.5` | Seconds between perception frames |
| `action_delay` | — | `0.05` | Seconds between individual actions |
| `save_screenshots` | `AOE2_SAVE_SCREENSHOTS` | `true` | Log screenshots to disk |
| `log_dir` | — | `logs` | Screenshot and log output directory |

The 3 model defaults follow `AOE2_LLM_WIRE`, because a model name belongs to its vendor: the `anthropic` wire serves `claude-haiku-4-5` to the executor and `claude-sonnet-5` to the strategist. `config._MODELS_BY_WIRE` holds the table. An env override still wins per role.

`AOE2_LLM_WIRE` tolerates case and surrounding whitespace, so `ZEN` and `" zen "` both resolve. An unrecognised name raises at startup rather than falling back — `config._parse_wire` is the only validator, and a silent fallback would play a whole game on a vendor nobody chose:

```
ValueError: unknown AOE2_LLM_WIRE='zzz'; expected one of 'anthropic', 'openai', 'zen'
```

The valid set is the `WireName` Literal in `config.py`, and the `--wire` CLI choices derive from it. See [Provider Pattern §4.2](../part2-llm-integration/04-provider-pattern.md) for how the factories turn a name into a client.

A global singleton `config = Config.from_env()` is created at module load time and imported throughout the codebase.

## 1.5 Async-First Architecture

The entire agent runs on asyncio:

- **Entry point**: `asyncio.run(main_async(args))` in `apps/agent/src/main.py`
- **API clients**: `openai.AsyncOpenAI` by default for both executor and strategist, or `anthropic.AsyncAnthropic` on the `anthropic` adapter — each `ChatWire` owns its own client
- **Game loop**: `game_loop()` in `apps/agent/src/game_loop.py`
- **Action execution**: `execute_actions()` in `apps/agent/src/executor.py`
- **Delays**: `asyncio.sleep()` for non-blocking waits

pyautogui calls are synchronous but fast (sub-millisecond per click), so they don't block meaningfully.

<aside class="concept" data-title="Async-first architecture (why one event loop, not threads)">

The agent runs three tasks on one asyncio event loop: perception publishes immutable frames; the actor awaits one bounded TypeSafe choice and revalidates it against the latest frame; deliberate work handles combat, tactical handoff, and recovery. The actor and deliberate task share an `asyncio.Lock` around game input. Perception can still run while input is in progress, so purchase records retain both observation and input revisions. Screen capture is dispatched with `asyncio.to_thread`.

Two patterns recur:

- **`await foo()` for sequential work** — the most common shape. You're waiting on the result before continuing.
- **`asyncio.create_task(foo())` for owned background work** — the strategist updates goals while perception and routine choices continue. Shutdown cancels and awaits this task and closes the policy and executor providers.

The single-loop invariant breaks the moment you call into blocking code (the synchronous `pyautogui.click()` is fast enough that we accept the block; a sync database driver wouldn't be). For genuine background CPU work you'd use `loop.run_in_executor` to dispatch to a thread pool; the broker uses `loop.call_soon_threadsafe` to marshal CLI cross-thread publishes back onto the main loop — see [Appendix B §B.6](../appendix/02-event-brokers-and-redis-streams.md).

</aside>

## 1.6 Logging

Structured logging via structlog with colored console output, configured in `apps/agent/src/main.py`.

Key log events: `strategist_goals_updated`, `policy_advice_outcome`, `act_decided`, `act_execution_outcome`, `action_outcome`, `frame_refresh_timed_out`, and `alarm_triggered`. Actor logs include source and execution ticks, revalidation reason, input revision, action ID, reservations, and final outcome.

---

## Summary

- Three-loop architecture: perception owns observed facts, TypeSafe selects routine actions, and deliberate work handles tactics and bounded recovery while the strategist updates goals.
- TypeSafe selects routine actions from one catalog; deterministic fallback and deliberate economic tools share its feasibility rules. The ledger records pending purchases and observed outcomes, while re-detection and HUD changes provide confirmation.
- Detection is practically required for useful gameplay; game knowledge and window management are truly optional
- Pydantic for config and validation, structlog for observability, asyncio for concurrency
- Goal-driven gameplay with alarm system for emergency defense

## Related Topics

- [Chapter 2: Game Loop Pipeline](./02-game-loop-pipeline.md) — the iteration cycle in detail
- [Chapter 4: Provider Pattern](../part2-llm-integration/04-provider-pattern.md) — how LLM providers are abstracted
- [Chapter 7: Detector Architecture](../part3-entity-detection/07-detector-architecture.md) — the optional YOLO system
