# Chapter 1: System Overview

The AoE2 LLM Arena agent plays Age of Empires II autonomously using a pure vision approach: no game API, no OCR, no memory-mapped data. Every decision is derived from a screenshot analyzed by Claude Vision.

## 1.1 The Pure Vision Thesis

The agent treats AoE2 as a visual problem. Each iteration captures a screenshot of the game window and sends it to Claude as a base64-encoded JPEG image. Claude returns structured JSON containing strategic reasoning, game state observations, and a list of actions to execute.

This approach was chosen over alternatives (game API hooking, OCR pipelines, replay parsing) for three reasons:

1. **Generality** -- the same architecture works for any visually-driven game without reverse-engineering proprietary protocols.
2. **Alignment with LLM strengths** -- modern vision-language models excel at interpreting complex visual scenes and reasoning about spatial relationships.
3. **Simplicity** -- no binary patching, no DLL injection, no version-specific offsets to maintain.

The tradeoff is latency (each API call takes 1-3 seconds) and cost (vision tokens are expensive). The 2-second loop delay in `src/config.py:20` reflects this constraint.

## 1.2 Component Map

```
agent/
├── src/                    # Core agent runtime
│   ├── main.py             # CLI entry point, provider creation
│   ├── config.py           # Pydantic configuration with env var overrides
│   ├── game_loop.py        # Main capture→detect→think→act→remember cycle
│   ├── memory.py           # Working memory and game state tracking
│   ├── executor.py         # Action execution via pyautogui
│   ├── models.py           # Pydantic action/response validation
│   ├── screen.py           # Screenshot capture via mss
│   ├── window.py           # Game window detection and focus management
│   └── providers/
│       ├── base.py         # Abstract LLM provider interface
│       └── claude.py       # Anthropic Claude implementation
├── detection/              # YOLO entity detection (optional)
│   ├── inference/          # Runtime detector + model weights
│   ├── training/           # Synthetic data gen + YOLO training
│   ├── labeling/           # CVAT integration + class remapping
│   └── extraction/         # SLD sprite extraction from game files
├── data/                   # Game knowledge (optional)
│   ├── game_knowledge.py   # SQLite database wrapper
│   ├── fetch_aoe2_data.py  # API data fetcher
│   └── knowledge_base/     # Static JSON files (units, buildings, civs, techs)
└── prompts/
    └── system.md           # System prompt for Claude
```

## 1.3 Graceful Degradation

Every subsystem beyond the core loop is optional. The agent starts and runs with only an Anthropic API key.

**Detection** -- imported inside a try/except at `src/game_loop.py:20-25`:

```python
try:
    from detection.inference.detector import EntityDetector, get_detector
    DETECTION_AVAILABLE = True
except ImportError:
    DETECTION_AVAILABLE = False
```

When detection is unavailable, the agent relies entirely on Claude's vision to interpret the screenshot. It cannot use `target_id` references but can still click on coordinates.

**Game Knowledge** -- imported inside a try/except at `src/providers/claude.py:24-29`:

```python
try:
    from data.game_knowledge import GameKnowledge, get_db
    GAME_KNOWLEDGE_AVAILABLE = True
except ImportError:
    GAME_KNOWLEDGE_AVAILABLE = False
```

Without the knowledge database, no dynamic context injection occurs. Claude still receives the system prompt and memory context but without affordable-unit suggestions.

**Window Management** -- pygetwindow is optional at `src/window.py:10-16`:

```python
try:
    import pygetwindow as gw
    PYGETWINDOW_AVAILABLE = True
except ImportError:
    PYGETWINDOW_AVAILABLE = False
```

When unavailable, functions like `is_game_running()` and `ensure_game_focused()` return `True` by default -- the agent assumes the game is running and focused. Screenshot capture falls back to the full primary monitor.

> **Key Insight**: The agent runs entirely without YOLO detection or game knowledge -- both are additive enhancements. A developer can test the core loop with just an Anthropic API key and a running game window.

## 1.4 Configuration

Configuration uses a Pydantic `BaseModel` with environment variable overrides (`src/config.py:8-39`):

| Setting | Env Var | Default | Purpose |
|---------|---------|---------|---------|
| `anthropic_api_key` | `ANTHROPIC_API_KEY` | `""` | Claude API authentication |
| `model` | `AOE2_MODEL` | `claude-sonnet-4-5-20250929` | Which Claude model to use |
| `max_tokens` | -- | `1024` | Max response tokens per API call |
| `screenshot_quality` | -- | `85` | JPEG quality (1-100) |
| `loop_delay` | `AOE2_LOOP_DELAY` | `2.0` | Seconds between iterations |
| `action_delay` | -- | `0.05` | Seconds between individual actions |
| `save_screenshots` | `AOE2_SAVE_SCREENSHOTS` | `true` | Log screenshots to disk |
| `log_dir` | -- | `logs` | Screenshot and log output directory |

A global singleton `config = Config.from_env()` is created at module load time (`src/config.py:39`) and imported throughout the codebase.

## 1.5 Async-First Architecture

The entire agent runs on asyncio:

- **Entry point**: `asyncio.run(main_async(args))` at `src/main.py:101`
- **API client**: `anthropic.AsyncAnthropic` at `src/providers/claude.py:52`
- **Game loop**: `async def game_loop()` at `src/game_loop.py:28`
- **Action execution**: `async def execute_actions()` at `src/executor.py:172`
- **Delays**: `asyncio.sleep()` for non-blocking waits between iterations and actions

The async design keeps the event loop responsive during API calls and action execution. pyautogui calls are synchronous but fast (sub-millisecond per click), so they don't block meaningfully.

## 1.6 Logging

Structured logging via structlog with colored console output, configured at `src/main.py:14-26`:

```python
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.dev.ConsoleRenderer(colors=True),
    ],
)
```

Log events include: `iteration_start`, `screenshot_captured`, `detection_complete`, `llm_response`, `actions_executed`, and error events with structured context.

---

## Summary

- Pure vision approach: screenshot in, JSON actions out
- Three optional subsystems (detection, knowledge, window management) each degrade gracefully
- Pydantic for config and validation, structlog for observability, asyncio for concurrency
- Single entry point (`main.py`) with provider pattern for LLM flexibility

## Related Topics

- [Chapter 2: Game Loop Pipeline](./02-game-loop-pipeline.md) -- the iteration cycle in detail
- [Chapter 4: Provider Pattern](../part2-llm-integration/04-provider-pattern.md) -- how LLM providers are abstracted
- [Chapter 7: Detector Architecture](../part3-entity-detection/07-detector-architecture.md) -- the optional YOLO system
