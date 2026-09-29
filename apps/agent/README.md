# `gameplay-agent/` — Real-Game Agent + Scenario Runner

The Windows VM tier captures AoE2:DE screenshots, reads the HUD locally,
detects entities, and issues mouse/keyboard input. TypeSafe chooses routine
named actions; the actor enforces basic-economy deadlines and shared catalog
feasibility. Strategic and bounded tactical calls use the configured provider.
Automated replays test logic, not live Windows game effects.

## What's here

```
apps/agent/src/            # importable as `gameplay_agent`
├── main.py                # CLI entry: `aoe2-agent`
├── game_loop.py           # Real-game capture → detect → think → act cycle
├── policy/                # Shared action catalog, fallback, and economy obligations
├── villager_roles.py      # Villager job inference (what each detected villager is gathering)
├── synth_game_loop.py     # Stripped-down loop arena uses (talks to WorldState, no pyautogui)
├── executor.py            # Action dispatch + execution (pyautogui)
├── models.py              # Pydantic action types (click, press, build, queue_villager, etc.)
├── memory.py              # AgentMemory + working_memory + metrics snapshot
├── memory_chain.py        # Cross-game persistent memory (loads + saves notes-to-self)
├── goals.py               # Goal manager + alarm system + reward tracking
├── entity_utils.py        # DetectedEntity formatting helpers
├── resource_ocr.py        # Local resource-bar + idle-badge OCR (RapidOCR / template backend; replaced Claude vision)
├── game_profile.py        # Validated roster and fixed 4v4 setup contract
├── preflight.py           # Read-only live-window/HUD/selection qualification checks
├── detection_phase.py     # YOLO call + ownership classification per loop iteration
├── strategist_phase.py    # Periodic Sonnet text call (resources via local OCR) → goal updates
├── turn_phases.py         # Glue between detection / strategist / executor per turn
├── screen.py              # mss-based screenshot capture
├── window.py              # AoE2 window detect + focus (pygetwindow optional)
├── overlay.py             # Tkinter live overlay (optional)
├── goal_logger.py         # Per-game goal/score TSV writer
├── config.py              # Pydantic config from env vars
├── providers/             # ExecutorProvider (executor) + StrategistProvider (text + local OCR)
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

# Copy the Arabia or Highland example, fill the actual eight-player lobby
# roster and exported hotkey/game-build details, then validate Windows.
uv run --no-sync python -m gameplay_agent.preflight --profile my-4v4-profile.json
AOE2_GAME_PROFILE=my-4v4-profile.json just agent

just eval-all                                    # Run every scenario fixture
uv run --package gameplay-agent \
    python -m gameplay_agent.scenario_runner \
    apps/agent/src/scenarios/age_up_gate_fires.yaml
```

The [Arabia](profiles/magyars_arabia_4v4.example.json) and
[Highland](profiles/magyars_highland_4v4.example.json) examples are deliberately
unqualified. The declared map is recorded for comparison but does not change
gameplay or require verification; the screen reader cannot recognize a map name.
Recorded experiments with no profile or unverified roster, hotkeys, or ownership
colors are invalid for scoring. Preflight exits nonzero until the window, HUD,
selection, roster, hotkey, and ownership-color gates are satisfied. It
checks geometry and screen readings without sending input; a human must verify
hotkey effects and labeled team colors in the game before setting those flags.
Without qualified ownership colors, non-blue units remain `unknown` and cannot
trigger team alarms. The September 27 run was
a four-player game, not the target 4v4 profile.

Input-dependent captures use a lightweight HUD read without age text OCR. A
resource assignment requires a fresh detector view after the idle-villager
camera jump; old screen coordinates cannot authorize the click. If that view
has no safe target, the specific assignment is deferred so another eligible
resource or infrastructure action can proceed.

The screen-controlled 4v4 profile is not yet qualified as a playable game
agent. Current tests cover HUD replay, selection guards, catalog feasibility,
and scripted economic/age actions. They do not validate hover feedback,
production queues, construction previews, minimap navigation, tactical control
groups, or live binding effects. The Windows opening, sustained-play, and
three-match gates remain required before claiming successful team play. Highland
has water-separated terrain, but this profile does not yet qualify cross-river
navigation or naval play; its first gate is the same land-opening economy.

## Where to read more

- [Part 1 — Real-game Architecture](../../docs/part1-architecture/01-system-overview.md) — system overview, game loop, action model.
- [Part 2 — LLM Integration](../../docs/part2-llm-integration/04-provider-pattern.md) — providers, prompts, context injection.
- [Chapter 22 — Autoresearch](../../docs/part8-autoresearch/22-autoresearch-overview.md) — uses this package's `game_loop` + `memory_chain` to drive prompt-mutation experiments.

## Conventions

- **Pure leaves only inside `core`.** Anything here can import from `core`, `detection`, `evaluation`, `data` — never the reverse.
- **Prompt files ship inside the wheel** via `package-data = ["prompts/**/*.md"]`. `PROMPTS_DIR = Path(__file__).parent.parent / "prompts"` resolves correctly in both editable and installed modes.
- **Scenario fixtures live with their consumer.** `scenarios/*.yaml` ship inside this package; the scenario runner finds them via `Path(__file__).resolve().parent / "scenarios"`.
