# Chapter 14: Seven-Round Run Map

A complete timing breakdown of the first 7 iterations (rounds) of a game run, showing every step the agent executes and its estimated wall-clock cost. Includes analysis of two realistic optimizations: async strategist and loop delay reduction.

<aside class="prereqs">

[Chapter 2 — Game Loop Pipeline](./02-game-loop-pipeline.md). This chapter is the per-step timing table behind chapter 2's prose description.

</aside>

> **Note (since this analysis was written).** Two of the optimizations below have shipped: the strategist runs asynchronously (Optimization A) and `loop_delay` now defaults to **0.3 s** (Optimization B). The executor also gained a **single-shot path** for routine turns (one roundtrip instead of the agentic tool loop — see [Chapter 4 §4.3](../part2-llm-integration/04-provider-pattern.md)). The tables below keep the original `loop_delay = 1.0 s` / always-tool-loop baseline so the optimization math in §14.5 stays self-consistent; treat them as the methodology, not today's wall-clock.

## 14.1 Per-Step Timing Reference

Every iteration executes the same pipeline. Steps marked **conditional** only run on specific turns.

| # | Step | Typical Time | Source | Condition |
|---|------|-------------|--------|-----------|
| 1 | Game running check | ~5 ms | `window.py:is_game_running()` | Every turn |
| 2 | Ensure game focus | ~50 ms | `window.py:ensure_game_focused()` | Every turn |
| 3 | Screenshot capture | ~20 ms | `screen.py:capture_screenshot()` via mss | Every turn |
| 4 | YOLO detection (single-pass @1280) | one forward pass | `detector.detect_fast()` (`adaptive_sahi=False`) | Every turn — the deployed path |
| 5 | Entity ownership classification | ~5 ms | `packages/detection/src/inference/ownership.py` | Every turn (if entities detected) |
| 6 | Alarm check | ~10 ms | `goals.py:check_alarm()` | Every turn (if entities detected) |
| 7 | Strategist API call (Sonnet, text — resources via local OCR) | ~5000 ms | `providers/strategist.py` | Turn 1, every 10th turn, on alarm (3-turn cooldown) |
| 8 | Build LLM context | ~10 ms | `game_loop.py:_build_llm_context()` | Every turn |
| 9 | Executor agentic loop (1–7 tool calls) | ~2000 ms | `providers/executor_provider.py` | Every turn |
| 10 | Process response + memory update | ~50 ms | `game_loop.py:_process_response()` | Every turn |
| 11 | Ground commands (zoom, scout) | ~250 ms | `game_loop.py:_get_ground_commands()` | Turn 1 only |
| 12 | Action execution (3–5 actions) | ~250 ms | `executor.py` at 50 ms/action | Every turn (or fallback) |
| 13 | Loop delay (sleep) | 1000 ms | `config.loop_delay = 1.0` | Every turn |

**Config defaults** (from `config.py`): `loop_delay=0.3` (the tables below use the pre-optimization `1.0` baseline — see the note above), `strategist_interval=10`, `detection_imgsz=1280`, `adaptive_sahi=False`, `full_sahi_interval=5` (only consulted when `adaptive_sahi=True`), `action_delay=0.05`, `max_tool_iterations=7`, `executor_effort="low"`.

> **Detection mode (v9).** The agent now runs a **single forward pass at `imgsz=1280`** on every turn (`adaptive_sahi=False`) — the deployed v9 model is trained at 1280, and SAHI lowers real F1 at retina resolution (see [Chapter 7 §7.4](../part3-entity-detection/07-detector-architecture.md)). The per-round "full/adaptive SAHI" distinctions and the millisecond detection figures in the timelines below are **illustrative/historical** from the pre-v6 design; treat detection as one constant single-pass cost per turn regardless of round (the 1280 pass is somewhat heavier than the old 640 figures shown).

---

## 14.2 Round-by-Round Overview

| Round | Strategist? | Detection Mode | Ground Cmds? | Notes |
|-------|-------------|----------------|--------------|-------|
| 1 | Yes | Single-pass @1280 | Yes | Heaviest round — first strategist + zoom/scout |
| 2 | No | Single-pass @1280 | No | Normal |
| 3 | No | Single-pass @1280 | No | Normal |
| 4 | No | Single-pass @1280 | No | Normal |
| 5 | No | Single-pass @1280 | No | Normal (no SAHI; `full_sahi_interval` only applies when `adaptive_sahi=True`) |
| 6 | No | Single-pass @1280 | No | Normal |
| 7 | No | Single-pass @1280 | No | Normal |

---

## 14.3 Detailed Timeline

### Round 1 — First Iteration (Heaviest)

| # | Step | Time | Cumulative |
|---|------|------|-----------|
| 1 | Game running check | 5 ms | 5 ms |
| 2 | Ensure game focus | 50 ms | 55 ms |
| 3 | Screenshot capture | 20 ms | 75 ms |
| 4 | YOLO detection (single-pass @1280) | 150 ms | 225 ms |
| 5 | Entity ownership classification | 5 ms | 230 ms |
| 6 | Alarm check | 10 ms | 240 ms |
| 7 | **Strategist API call** (Sonnet, text — resources via local OCR) | 5000 ms | 5240 ms |
| 8 | Build LLM context | 10 ms | 5250 ms |
| 9 | Executor agentic loop | 2000 ms | 7250 ms |
| 10 | Process response + memory update | 50 ms | 7300 ms |
| 11 | **Ground commands** (scroll ×5, select scout, auto-scout) | 250 ms | 7550 ms |
| 12 | Action execution (~4 actions) | 200 ms | 7750 ms |
| 13 | Loop delay | 1000 ms | 8750 ms |

**Round 1 total: ~8.75 s**

### Rounds 2–4, 6–7 — Normal Iterations

| # | Step | Time | Cumulative |
|---|------|------|-----------|
| 1 | Game running check | 5 ms | 5 ms |
| 2 | Ensure game focus | 50 ms | 55 ms |
| 3 | Screenshot capture | 20 ms | 75 ms |
| 4 | YOLO detection (single-pass @1280) | 150 ms | 225 ms |
| 5 | Entity ownership classification | 5 ms | 230 ms |
| 6 | Alarm check | 10 ms | 240 ms |
| 7 | Strategist — *skipped* | 0 ms | 240 ms |
| 8 | Build LLM context | 10 ms | 250 ms |
| 9 | Executor agentic loop | 2000 ms | 2250 ms |
| 10 | Process response + memory update | 50 ms | 2300 ms |
| 11 | Ground commands — *skipped* | 0 ms | 2300 ms |
| 12 | Action execution (~4 actions) | 200 ms | 2500 ms |
| 13 | Loop delay | 1000 ms | 3500 ms |

**Normal round total: ~3.5 s**

### Round 5 — Normal Iteration

> Pre-v6 this round forced a full SAHI scan (`iteration % 5 == 0`). With `adaptive_sahi=False` there is no forced full scan, so Round 5 is now an ordinary single-pass turn — identical in shape to Rounds 2–4.

| # | Step | Time | Cumulative |
|---|------|------|-----------|
| 1 | Game running check | 5 ms | 5 ms |
| 2 | Ensure game focus | 50 ms | 55 ms |
| 3 | Screenshot capture | 20 ms | 75 ms |
| 4 | YOLO detection (single-pass @1280) | 150 ms | 225 ms |
| 5 | Entity ownership classification | 5 ms | 230 ms |
| 6 | Alarm check | 10 ms | 240 ms |
| 7 | Strategist — *skipped* | 0 ms | 240 ms |
| 8 | Build LLM context | 10 ms | 250 ms |
| 9 | Executor agentic loop | 2000 ms | 2250 ms |
| 10 | Process response + memory update | 50 ms | 2300 ms |
| 11 | Ground commands — *skipped* | 0 ms | 2300 ms |
| 12 | Action execution (~4 actions) | 200 ms | 2500 ms |
| 13 | Loop delay | 1000 ms | 3500 ms |

**Round 5 total: ~3.5 s**

---

## 14.4 Cumulative 7-Round Timeline

| Round | Type | Round Time | Cumulative Elapsed |
|-------|------|-----------|-------------------|
| 1 | Strategist + Ground Cmds | 8.75 s | 8.75 s |
| 2 | Normal | 3.50 s | 12.25 s |
| 3 | Normal | 3.50 s | 15.75 s |
| 4 | Normal | 3.50 s | 19.25 s |
| 5 | Normal | 3.50 s | 22.75 s |
| 6 | Normal | 3.50 s | 26.25 s |
| 7 | Normal | 3.50 s | 29.75 s |

**Total 7-round run: ~29.8 s**

Time distribution across the full run:

| Component | Total Time | % of Run |
|-----------|-----------|----------|
| Executor agentic loop (API) | 14.00 s | 46.8% |
| Loop delay (sleep) | 7.00 s | 23.4% |
| Strategist API call | 5.00 s | 16.7% |
| Action execution + ground cmds | 1.45 s | 4.9% |
| YOLO detection | 1.37 s | 4.6% |
| Other (focus, screenshot, context, memory) | 1.09 s | 3.6% |

The executor API calls dominate, followed by the sleep delay and the single strategist call on round 1.

---

## 14.5 Optimization Analysis

### Optimization A: Async Strategist

**Problem**: The strategist call on round 1 blocks the loop for ~5 s (3–8 s range). This is the single most expensive step in a 7-round run at 16.7% of total time.

**Proposal**: Fire the strategist as a background `asyncio.create_task()`. The executor continues immediately with default/previous goals. When the strategist response arrives, goals update asynchronously.

```
Current (blocking):
  Round 1: ... → [Strategist 5000ms] → [Executor 2000ms] → ...
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                  Total: 7000ms sequential

Proposed (async):
  Round 1: ... → [Executor 2000ms with default goals] → ...
                  [Strategist 5000ms in background..........]
                  ^^^^^^^^^^^^^^^^
                  Total: 2000ms (strategist completes during round 2)
```

**Impact on round 1**:

| Step | Current | With Async Strategist |
|------|---------|----------------------|
| Strategist API call | 5000 ms | 0 ms (background) |
| Round 1 total | 8834 ms | **3834 ms** |

**Tradeoffs**:

| Aspect | Detail |
|--------|--------|
| Round 1 goals | Uses default base goals ("Queue villagers", "Gather food", "Advance to Feudal Age") from `strategist.py:190-198` until real goals arrive |
| Resource readings | Empty on round 1; executor works without resource context for 1–2 turns |
| Goal staleness | Goals update when the async task completes — typically during round 2. Acceptable since goals are high-level strategic directives, not per-action commands |
| Thread safety | No issue — asyncio is single-threaded cooperative. Goal updates happen between awaits |
| Implementation | ~15 lines changed in `game_loop.py`: replace `await _run_strategist()` with `asyncio.create_task()` + pre-seed default goals in `GoalManager.__init__()` |

**Verdict**: Realistic. Saves **5.0 s** on a 7-round run (16.7% improvement). The default goals are reasonable for early Dark Age play and closely match what the strategist would generate anyway.

### Optimization B: Loop Delay Reduction

**Problem**: The 1000 ms `loop_delay` sleep accounts for 7.0 s across 7 rounds (23.4% of total time). It runs every single iteration.

**Why it exists**: Prevents rapid-fire inputs to the game and allows the screen to update between iterations.

**Analysis**: The pipeline already introduces ~1.5–3.5 s of natural latency per iteration:
- Detection: 150–234 ms
- Executor API loop: 1000–3500 ms
- Action execution: 200+ ms with built-in delays (`action_delay=0.05s`, `BUILD_SETTLE_DELAY=0.15s`)

The game renders at 60 fps (16 ms/frame). The screen fully updates within 50–100 ms of any action. The existing action-level delays already pace inputs.

**Three scenarios**:

| Scenario | `loop_delay` | Per-round savings | 7-round savings | Risk |
|----------|-------------|-------------------|-----------------|------|
| Current | 1.0 s | — | — | None |
| Conservative | 0.3 s | 0.7 s | 4.9 s | Minimal — 300 ms is still 18 frames of render time |
| Aggressive | 0.0 s | 1.0 s | 7.0 s | Screenshots may capture mid-animation; rapid API calls |

**Verdict**: Reducing to 0.3 s is safe and realistic. Saves **4.9 s** on a 7-round run (16.4% improvement). Eliminating entirely (0.0 s) is viable but carries minor risk of stale screenshots.

### Combined Impact

| Scenario | Round 1 | Rounds 2–4,6–7 | Round 5 | 7-Round Total | Savings |
|----------|---------|-----------------|---------|---------------|---------|
| **Current** | 8.83 s | 3.50 s | 3.58 s | **29.91 s** | — |
| Async strategist only | 3.83 s | 3.50 s | 3.58 s | **24.91 s** | 5.0 s (16.7%) |
| Loop delay 0.3 s only | 8.13 s | 2.80 s | 2.88 s | **24.81 s** | 5.1 s (17.0%) |
| **Both combined** | 3.13 s | 2.80 s | 2.88 s | **19.81 s** | **10.1 s (33.8%)** |

With both optimizations, a 7-round run drops from **~30 s to ~20 s** — a one-third reduction in wall-clock time with no loss of gameplay quality.

---

## 14.6 Variability and Edge Cases

The timings above assume a clean run with no alarms. Real runs may vary:

| Event | Effect on Timing |
|-------|-----------------|
| **Alarm triggered** (enemy detected) | Strategist runs on alarm turn (+5 s). Alarm goal injected at priority 10. (Pre-v6 an alarm also forced a full SAHI scan; with `adaptive_sahi=False` detection stays single-pass.) |
| **Rescan during executor loop** | A `press` action with `rescan: true` triggers mid-turn screenshot + detection. Adds ~50–300 ms per rescan depending on frame differ result. |
| **Strategist retry** (API error) | SDK retries up to 2× with exponential backoff. Could add 5–15 s on failure turns. Falls back to default goals on total failure. |
| **Executor max iterations** | If the executor uses all 7 tool call iterations, the agentic loop may take 3.5 s+ instead of the typical 2 s. |
| **Game not focused** | `ensure_game_focused()` fails → 1 s sleep + skip iteration entirely. Round time becomes ~1 s of wasted wall-clock. |
| **Remote detection server down** | Falls back to local ONNX inference. Detection time jumps from ~150 ms to ~1200 ms per turn. |

---

## 14.7 Visual Timeline (7 Rounds)

```
Time (s)  0         5         10        15        20        25        30
          |---------|---------|---------|---------|---------|---------|
Round 1   [===STRATEGIST====][EXEC][ACT][SLEEP]
Round 2                                          [DET][EXEC][ACT][SLP]
Round 3                                                               [DET][EXEC][ACT][SLP]
Round 4                                                                                    [DET][EXEC][ACT][SLP]
Round 5                                                                                                         [DET][EXEC][ACT][SLP]
Round 6                                                                                                                                [DET][EXEC][ACT][SLP]
Round 7                                                                                                                                                     [DET][EXEC][ACT][SLP]

Legend: DET = detection (single-pass @1280, every turn)  EXEC = executor API  ACT = action execution  SLP = sleep
        STRATEGIST = Sonnet API call (text — resources via local OCR; only round 1 in a 7-round run)
```
