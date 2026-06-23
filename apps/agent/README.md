# `gameplay-agent/` — Real-Game Agent + Scenario Runner

The Windows VM tier: the loop that screenshots AoE2:DE, sends frames to a
detection server, asks Claude what to do, and clicks. Also home to the
scenario runner and its test helpers (the same Pydantic action model and
providers are exercised by both real-game and synth-tier scenario tests).

## What's here

```
packages/gameplay-agent/src/
├── main.py                # CLI entry: `aoe2-agent`
├── game_loop.py           # Real-game capture → detect → think → act cycle
├── synth_game_loop.py     # Stripped-down loop arena uses (talks to WorldState, no pyautogui)
├── executor.py            # Action dispatch + execution (pyautogui)
├── models.py              # Pydantic action types (click, drag, press, etc.)
├── memory.py              # AgentMemory + working_memory + metrics snapshot
├── memory_chain.py        # Cross-game persistent memory (loads + saves notes-to-self)
├── goals.py               # Goal manager + alarm system + reward tracking
├── entity_utils.py        # DetectedEntity formatting helpers
├── detection_phase.py     # YOLO call + ownership classification per loop iteration
├── strategist_phase.py    # Periodic Sonnet text call (resources via local OCR) → goal updates
├── turn_phases.py         # Glue between detection / strategist / executor per turn
├── screen.py              # mss-based screenshot capture
├── window.py              # AoE2 window detect + focus (pygetwindow optional)
├── overlay.py             # Tkinter live overlay (optional)
├── goal_logger.py         # Per-game goal/score TSV writer
├── config.py              # Pydantic config from env vars
├── providers/             # ClaudeProvider (executor) + StrategistProvider (text + local OCR)
├── prompts/               # System prompts (core.md, hotkeys.md, strategist.md, ages/*.md)
├── scenario_runner.py     # `python -m gameplay_agent.scenario_runner ...` (multi-turn harness)
├── assertions.py          # Assertion DSL used by scenario fixtures
├── context_builder.py     # Builds executor context from a scenario fixture
├── test_isolation.py      # _isolate_memories_dir, _mock_executor, _seed_detected_entities
├── fixture_builder.py     # Real-game log → scenario fixture (`python -m gameplay_agent.fixture_builder`)
├── log_to_scenario.py     # structlog game.txt → scenario YAML (`python -m gameplay_agent.log_to_scenario`)
├── strategist_eval.py     # Standalone strategist evaluation harness
└── scenarios/             # *.yaml fixtures consumed by scenario_runner
```

## Common entry points

```bash
just agent                                       # Real game on Windows VM
just agent --iterations 50
just agent --test                                # One iteration, no clicks

just eval-all                                    # Run every scenario fixture
uv run --package gameplay-agent \
    python -m gameplay_agent.scenario_runner \
    packages/gameplay-agent/src/scenarios/age_up_gate_fires.yaml
```

## Where to read more

- [Part 1 — Real-game Architecture](../../docs/part1-architecture/01-system-overview.md) — system overview, game loop, action model.
- [Part 2 — LLM Integration](../../docs/part2-llm-integration/04-provider-pattern.md) — providers, prompts, context injection.
- [Chapter 22 — Autoresearch](../../docs/part8-autoresearch/22-autoresearch-overview.md) — uses this package's `game_loop` + `memory_chain` to drive prompt-mutation experiments.

## Conventions

- **Pure leaves only inside `core`.** Anything here can import from `core`, `detection`, `evaluation`, `data` — never the reverse.
- **Prompt files ship inside the wheel** via `package-data = ["prompts/**/*.md"]`. `PROMPTS_DIR = Path(__file__).parent.parent / "prompts"` resolves correctly in both editable and installed modes.
- **Scenario fixtures live with their consumer.** `scenarios/*.yaml` ship inside this package; the scenario runner finds them via `Path(__file__).resolve().parent / "scenarios"`.
