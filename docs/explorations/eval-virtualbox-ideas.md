# Virtual Box for AoE2 Agent Evaluation — Analysis & Idea Pool

> **Note (2026-06):** Written before the strategist replaced Claude vision with local OCR. References below to a "vision-LLM" strategist and "vision-pipeline regression" describe the pre-OCR design; the strategist is now text-only and reads the resource bar via OCR (`resource_ocr.py`). The `vision_fixtures/` directory is now used for OCR regression. Current state: [System Overview](../part1-architecture/01-system-overview.md).

## Context

Running real AoE2 games against the agent is **slow and expensive**:
- Mac VM (Parallels/VMware) takes ~2–3 min to boot AoE2 DE
- Each real-time turn takes 5–15 s (LLM 3–8 s + perception 0.2–0.5 s + game-world settling 0.5–2 s + enforced loop_delay)
- A meaningful 30-turn game costs ~$1–3 in API spend and ~5–10 minutes wall-clock
- The Mac fan spins, RAM/CPU are saturated, and you can't run more than one in parallel

The eval framework added in `evaluation/runner.py` is a great start — it runs the **real `ExecutorProvider.get_actions()`** against YAML-fixture state with `execute_action` mocked — but it only covers single-turn, hand-authored scenarios. Most real bugs (stuck loops, memory bleeding, age-transition regressions, model drift between Sonnet/Haiku/Opus) emerge across multiple turns or under perceptual conditions the fixtures don't cover.

**Goal of this doc:** survey the design space for a "virtual box" — a fast, cheap test environment for the agent that runs without AoE2 DE and without the Mac VM — and present an **idea pool** ranked by leverage-per-effort. This is exploration, not implementation.

---

## What we already have

| Layer | Status |
|---|---|
| **Single-turn LLM evaluation** | ✅ `evaluation/runner.py` + `assertions.py` + 13 YAML fixtures + 51-test pytest harness (gated `--runlive`). Mocks `execute_action`; everything else is real. |
| **Memory system isolation** | ✅ `_isolate_memories_dir` backs up real memories, plants fixture memories, restores on exit. |
| **Variant matrices** | ✅ Same scenario × multiple memory states already supported (`variants:` in YAML). |
| **Memory extraction & loading** | ✅ `autoresearch/memory_chain.py` writes `memories/*.md`; loaded into the cached system block. |
| **Real-game telemetry** | ✅ `logs/YYYY_MM_DD/game.txt` — 100–250 KB structlog streams per game (~30–60 turns each), full prompts, tool calls, costs. 2 days of real-game data on disk. |
| **Vision training corpus** | ✅ `training_data_v4.tar.gz` (1.4 GB) and `v5.tar.gz` (3.5 GB) — annotated AoE2 screenshots for YOLO. *(Historical — pre-v6 dataset refs; the current detector is v9 @1280, deployed, trained on a cleaned real+synthetic set.)* |
| **Perception is already remote** | ✅ Detector runs at `http://172.16.216.1:8420` — agent process is already decoupled from the YOLO infra. |

## What we don't have

| Gap | Implication |
|---|---|
| **Multi-turn evaluation** | Can't catch stuck loops, memory accumulation regressions, goal-completion timing, or age-transition behavior. |
| **Strategist coverage** | The strategist (separate vision-LLM call, every 3–10 turns) is **completely absent** from eval — fixtures hardcode resources/goals. We can't test strategist drift across model versions. |
| **Recorded game replay** | Real games produce structlog text but **no JSON snapshot per turn**. Replaying a real game state through a different model requires log parsing first. |
| **World-state simulator** | Nothing approximates AoE2 economy (villager production, resource gather rates, building completion, age-up timing) — so even if we had multi-turn fixtures, state can't evolve realistically across turns. |
| **Vision-input fidelity tests** | Strategist takes a real screenshot. We have annotated screenshots in the training tarballs but no harness that pairs `screenshot → expected resource readings → expected goal output`. |
| **Cross-model bench** | No standardized way to A/B Sonnet vs. Haiku vs. Opus on identical inputs. |

---

## Mental model: what a virtual box has to provide

A real per-turn cycle (traced in `gameplay_agent/game_loop.py:490–660`) consists of:

```
  capture screenshot ──┐
                       ├──► strategist (every 3–10 turns, vision-LLM)
                       │      └─► resource readings + goals
  YOLO detection ──────┤
                       ├──► executor (per-turn agentic LLM with tool loop)
  goal/state context ──┘      └─► reasoning + actions
                                    │
                                    ▼
                       pyautogui ──► real game world ──► next screenshot
```

Each box can be: **real**, **mocked with fixed value**, **simulated**, or **replayed from disk**. The trade is **fidelity vs. cost**:

| Component | Real cost / turn | Cheapest viable substitute |
|---|---|---|
| Screen capture + AoE2 DE | seconds + VM | static image from training tarball, or no image at all |
| YOLO detection | ~200 ms remote | fixture entity list (text), or replay a recorded list |
| Strategist call | ~$0.01, ~3 s | fixture `resource_readings + goals` block |
| Executor call (LLM agentic loop) | ~$0.05–0.10, ~5 s | **keep real** — this is what we're testing |
| Action execution | ~50–200 ms | mock (already done) |
| Game-world physics (state evolution) | seconds of real time | tiny Python state machine OR replay from logs |

The current eval mocks the bottom three rows. The big leverage is in the top three.

---

## Idea pool

Ranked by **leverage-per-effort**, with deliberate over-coverage so the user can pick and mix.

### Tier 1 — High leverage, low/medium effort

#### 1. Multi-turn scenario harness with a tiny world simulator
**What:** Extend YAML fixtures with `turns:` list. The runner runs N turns sequentially: each turn the executor LLM picks actions, a small Python "world simulator" applies action effects to the state (apply `build q` → +1 house counter, `queue_villager` → pop+1 after T turns, `press z` → Feudal age-up timer), then the next turn runs with the updated state. No real game; world is a state machine driven by ~10 effect handlers.

**Fidelity:** Low — it's AoE2-lite, not AoE2 DE. Resource gather rates, building costs, and age-up timings are approximated.
**Cost:** Just LLM cost (~$0.10–0.50 per N-turn scenario, depending on N).
**Unlocks:** Stuck-loop detection, memory accumulation under load, goal-completion timing, age-transition behavior, "did the agent eventually win/lose?" assertions.
**Effort:** Medium. A primitive simulator (8–10 state vars, 5 effect types, no economy curves) is ~300–500 LOC and unlocks ~80% of multi-turn bug categories. A more accurate one (proper villager production rates, building queue times, resource decay) is ~1000–2000 LOC.
**Files touched:** `evaluation/runner.py`, new `evaluation/world_sim.py`, fixture YAML schema extension.
**Reuses:** Existing `_mock_executor`, `_isolate_memories_dir`, assertion DSL in `evaluation/assertions.py`.

#### 2. Log → scenario synthesis tool (real-game derived fixtures)
**What:** Parse `logs/YYYY_MM_DD/game.txt` structlog streams into per-turn JSON snapshots, then auto-generate scenario YAMLs from selected interesting turns. "Turn N where the agent failed to build a house at pop_cap-2 in this real game → freeze that state into a regression fixture."

**Fidelity:** High — real-game state, real entity lists, real goal/strategist outputs.
**Cost:** Free (one-shot parse).
**Unlocks:** Auto-grow the regression corpus from every real game played. Permanent test coverage for any bug seen once.
**Effort:** Low. ~200–400 LOC: a structlog parser + a YAML emitter. The structlog format is regular (`timestamp [level] event_name field=value field=value`), so regex extraction is straightforward.
**Files touched:** new `evaluation/log_to_scenario.py`, no changes to runner or assertions.
**Reuses:** Existing fixture format and assertion DSL — output is just a YAML file.
**Cross-cutting benefit:** Also enables idea #3 (snapshot replay) since structured snapshots are exactly what's needed.

#### 3. Snapshot/record-replay framework for cross-model A/B
**What:** When a real game runs, dump turn-by-turn JSON snapshots (entities, resources, goals, prompt context, turn iteration). Then offline, replay each snapshot through the LLM provider with a *different* model — without re-running the game. Compare action outputs side-by-side.

**Fidelity:** Maximum (real game inputs, just a different model).
**Cost:** LLM cost only, scaled by `turns × models_under_test`. ~$5–15 to fully A/B Sonnet vs. Haiku vs. Opus on a 30-turn game.
**Unlocks:** Deterministic model bench-marking. "Does Haiku follow the inhibitory memory rules as well as Sonnet?" answered directly. Also enables eval drift detection: re-run the same captured game next month against the same model and check if results regressed.
**Effort:** Low–medium. Two pieces:
  1. Add a `--record` flag to `gameplay_agent/main.py` that writes `logs/<game>/snapshots/turn_NNN.json` during normal play. ~100 LOC instrumentation.
  2. New `evaluation/replay.py` that loads a snapshot dir, hot-swaps the model, runs `ExecutorProvider.get_actions()` per snapshot, and reports diffs. ~200 LOC.
**Files touched:** `gameplay_agent/game_loop.py` (instrumentation), new `evaluation/replay.py`, new `evaluation/diff_report.py`.
**Reuses:** `ExecutorProvider`, `AgentMemory`, the existing context-builder.

#### 4. Strategist-in-the-loop in eval
**What:** Today, fixtures hardcode `resources`, `goals`, and `current_age`. Add an optional `strategist_response:` block to fixtures so the runner can inject synthetic strategist output. Wire `_isolate_strategist` similarly to `_isolate_memories_dir`. Optionally, **also** allow running the real strategist against a fixture screenshot (for vision-pipeline regression).

**Fidelity:** Medium (synthetic) or high (real-strategist against captured frames).
**Cost:** Free (synthetic) or ~$0.01/turn (real strategist).
**Unlocks:** Test "what happens when the strategist gives the executor a *bad* goal?" Test strategist degradation across model versions on captured screenshots. Test executor-strategist contracts (e.g., does the executor still age up if the strategist forgets to set the goal?).
**Effort:** Low for synthetic mode (~100 LOC), medium for real-vision mode (need to wire screenshot path through the runner, ~300 LOC).
**Files touched:** `evaluation/runner.py`, fixture YAML schema, optionally `gameplay_agent/providers/strategist.py` (for cleaner DI).

### Tier 2 — Medium leverage, medium effort

#### 5. Vision-input fidelity tests (strategist regression)
**What:** Pair real screenshots from `training_data_v5.tar.gz` with hand-labeled "expected resource readings" + "expected goals." The strategist's vision-LLM is run against each screenshot; assertions check it produces approximately-correct OCR + goal output. Lets you verify strategist accuracy across model versions or prompt edits without running games.
**Fidelity:** High for the strategist, ignores the executor.
**Cost:** ~$0.01 per screenshot, ~$0.50–1 for a 50-screenshot panel.
**Unlocks:** Catch "Sonnet 4.7 misreads the population indicator" or "new strategist prompt loses goal-priority calibration."
**Effort:** Medium. Hand-labeling 30–50 screenshots is the slow part. Once labeled, the harness is ~150 LOC.
**Files touched:** new `evaluation/strategist_eval.py`, new `evaluation/vision_fixtures/`.

#### 6. Memory-system ablation matrix
**What:** Existing `variants:` is single-axis. Generalize to N-axis matrices: scenario × {memory states} × {models} × {prompt versions}. Generate a comparison report (which combinations produce the desired action). Useful when tuning the memory chain.
**Fidelity:** Inherits from underlying scenario.
**Cost:** Multiplies LLM cost by matrix size — `O(scenarios × memory_variants × models)`. Easy to bankroll wrong; needs cost guardrails.
**Unlocks:** "Which model + memory combination is most reliable for inhibitory rules?" answered with a heatmap.
**Effort:** Low–medium for the matrix runner (~250 LOC). Most of the cost is API spend, not engineering.
**Files touched:** `evaluation/runner.py` matrix mode, new `evaluation/matrix_report.py`.

#### 7. Adversarial scenario generator (LLM-driven)
**What:** Use a stronger model (Opus) to write new scenario YAMLs that probe specific failure modes. Prompt: "Generate 10 fixtures where the agent should not queue villagers but probably will, given Dark Age default behavior." Hand-curate, accept the good ones into the corpus.
**Fidelity:** Variable — generated fixtures need human review.
**Cost:** ~$0.50–2 per generation batch.
**Unlocks:** Cheap corpus growth without hand-authoring. Especially useful for finding edge cases the engineer didn't think of.
**Effort:** Low (~150 LOC, mostly a prompt + YAML schema validation pass).
**Files touched:** new `evaluation/scenario_gen.py`.

### Tier 3 — Speculative or higher effort

#### 8. Snapshot diff-trace UI
**What:** A small TUI/HTML viewer that loads a recorded game's snapshots and shows a side-by-side: model A's chosen action vs. model B's, with reasoning highlighted. Makes idea #3's output reviewable.
**Effort:** Medium (~500 LOC if HTML; less if textual-only diff). Pure infrastructure / quality-of-life.

#### 9. Pixel-replay (full screenshot sequence + recorded actions)
**What:** Save full screenshot sequence + action transcripts during real games; offline, feed screenshots into the strategist + a synthesized entity list into the executor. The closest thing to "deterministic real game replay" without re-running AoE2.
**Fidelity:** Very high.
**Cost:** Storage (each game ~50–200 MB of JPEG); LLM cost.
**Effort:** High. Requires capturing every screenshot during real games, plus a replay-runner that reconstructs the per-turn state. Probably overkill given idea #3 (JSON snapshot replay) covers ~90% of the same use cases at <1% the storage.

#### 10. Headless AoE2 alternative (OpenAge / clone)
**What:** Replace AoE2 DE with an open-source RTS engine (OpenAge, OpenRA, or similar) running headless on Linux. No VM, no real-time wait, deterministic.
**Fidelity:** Different game — agent strategies don't transfer 1:1.
**Effort:** Multi-month. Probably not worth it unless full RL training becomes a goal.

#### 11. Gameplay video → state extraction
**What:** Record gameplay videos, run YOLO + strategist over each frame, derive state. Then test the agent against video-derived state.
**Effort:** High. Mostly redundant with idea #2 (log parsing) since real games already produce structured logs. Only justifies itself if you want to test against *human* gameplay videos as adversarial input.

---

## Recommended starting trio

If picking three to combine for the biggest immediate win, I'd pair:

1. **Idea #2 — Log → scenario synthesis** *(foundation; one-time effort, then auto-growing corpus)*
2. **Idea #3 — Snapshot/record-replay** *(unlocks model A/B; the JSON snapshot format from #2 is reused here)*
3. **Idea #1 — Multi-turn harness with tiny world simulator** *(unlocks the bugs that single-turn fixtures structurally cannot catch)*

Together these would let you:
- Test scenarios cheaply (multi-turn synthetic worlds, ~$0.10–0.50 per scenario)
- Compare models on identical real-game states (record-replay)
- Auto-grow the regression corpus from every real game played
- All without booting the VM or AoE2 DE

> **Why this trio fits the user's framing** — The user mentioned text/images/videos as possible modalities. Notice this trio is **all text-based**: structlog parsing, JSON snapshots, fixture YAML. The image modality (screenshots) is only needed for *strategist* evaluation (idea #5), and gameplay videos are mostly redundant with #2. The cheapest, fastest wins live in the text-only lane because the agent's actual decision input is text-shaped already (entity list + resources + goals); only the strategist's input is visual. That's a useful constraint to plan around.

---

## Open questions that should shape sequencing

These don't block writing the plan, but answers will inform which idea to start first:

1. **Simulator vs. replay flavor:** Do you want the virtual box to be primarily a *synthesized* environment (build our own AoE2-lite world, idea #1) or primarily a *replayed* environment (capture real games, replay state through different models, idea #3)? Each is good; they unlock different things.
2. **Model comparison priority:** Is "test the agent across Sonnet/Haiku/Opus" a near-term goal? If yes, idea #3 jumps to first place.
3. **Strategist coverage urgency:** Are you actively iterating on the strategist prompt or vision behavior? If yes, idea #5 becomes near-term; if no, defer it.
4. **Cost ceiling:** Is `$5–10/day` of LLM eval spend acceptable, or do we need to engineer for `$1/day`? Affects matrix size and how aggressively we cache/batch.

---

## Verification: how to know each idea is working

| Idea | Verification |
|---|---|
| #1 Multi-turn harness | A 20-turn fixture catches a deliberately-introduced "agent loops on `press h` forever" regression. Pop counter advances correctly when `queue_villager` is called. |
| #2 Log → scenario | Run on `logs/2026_04_25/game.txt`; confirm the parser produces ≥1 scenario per "interesting turn" (verification failure, age transition, alarm). Run the synthesized scenario; it should pass against the model that played it. |
| #3 Snapshot replay | Record a real 10-turn game. Replay snapshot-3 through Sonnet vs. Haiku; confirm both produce sensible (though differing) actions. Diff report renders. |
| #4 Strategist injection | Inject a strategist response that says "ignore housing" while pop is at cap; confirm executor still builds a house (precedence rule from `core.md` overrides). |
| #5 Vision regression | 30 hand-labeled screenshots; current Sonnet hits ≥90% accuracy on resource OCR. Re-run with Haiku; expect a measurable drop. |
| #6 Matrix | Run scenario × {3 memory states} × {2 models}; output is a 3×2 pass/fail grid in <5 min. |

---

## Critical files to be modified (when implementation begins)

- `evaluation/runner.py` — extend for multi-turn (#1), strategist injection (#4), matrix mode (#6)
- `evaluation/assertions.py` — add multi-turn assertions (e.g., `eventually_includes`, `state_evolves_to`)
- `gameplay_agent/game_loop.py:490-660` — add `--record` snapshot instrumentation (#3)
- `gameplay_agent/providers/strategist.py` — minor DI cleanup so the strategist can be replayed (#4, #5)
- New: `evaluation/world_sim.py` (#1), `evaluation/log_to_scenario.py` (#2), `evaluation/replay.py` (#3), `evaluation/strategist_eval.py` (#5), `evaluation/scenario_gen.py` (#7)
- Fixture YAML schema extensions (multi-turn, strategist responses, matrix)

## What this plan deliberately does not do

- Pick a single idea to implement. The user asked for an idea pool; that's what this is.
- Specify exact API surfaces or data formats. Those should be designed once an idea is selected.
- Estimate spend or effort to two decimal places. Numbers given are order-of-magnitude.
- Address Pattern A (assertion shape too narrow) or Pattern B (model ignores inhibitory memories). Those are existing eval-quality issues separate from the virtual box question.
