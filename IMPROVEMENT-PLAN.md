# Agent Precision & Detection Improvement Plan

**Date:** 2026-07-09
**Scope:** How to get from "agent that survives" to "agent that comfortably plays AoE2 DE" — decides strategy, sees the map, guides villagers by need — while keeping cost low and latency fast. Hard constraint (already honored by the current design, keep it): **no screenshots to foundation models**; all perception stays local (YOLO + OCR + classical CV), the LLM only ever sees text.

---

## Where we are today (baseline facts)

| Area | Current state | Evidence |
|---|---|---|
| Detection model | YOLO26n, 60 classes, single-pass @1280, real-frame micro-F1 ≈ **0.67** (P 0.676 / R 0.665) | `docs/part3-entity-detection/07`, `08` |
| Detection blind spots | Military units near-zero recall on real frames (v7 eval: `knight_line` 0/45, `cavalry_archer` 0/67, `militia_line` 0/46, `sheep` 0/11 TP); 11 classes with **zero real labels** in v8 | `training_data_v7/eval_real_summary.json`, `training_data_v8/merge_summary.json` |
| Real training data | Only **187 real train / 32 real val images** (v8), vs 2400 synthetic | `training_data_v8/merge_summary.json` |
| Map awareness | Viewport-only. **No minimap parsing, no off-screen memory** — an entity scrolled out of view ceases to exist | `detection_phase.py`, docs survey |
| Villager guidance | Single undifferentiated `villager` class; job inferred by **140px proximity heuristic**; idle count read only as a **badge-presence boolean** | `villager_roles.py`, `resource_ocr.py`, `reactive.py` |
| Decision layer | Two `claude-sonnet-4-6` calls: executor **every turn** (routine ~2–4s, combat tool-loop up to 20–30s), strategist every 3–10 turns. Text-only, well-cached | `providers/executor_provider.py`, `strategist.py` |
| Cost | ~$1–3 per 30-turn real game (both roles on Sonnet) | `docs/explorations/eval-virtualbox-ideas.md` |
| Evaluation | Scenario fixtures + OCR regression exist, but `experiments/results.tsv` is **empty** — no experiment has ever been recorded against the composite score | `experiments/results.tsv` |
| Hygiene | Version drift: README says served model is v7, `get_detector()` defaults to v6, config/docs say v9. Uncommitted working tree (16 modified files) | detection `README.md` vs `detector.py:852` vs `apps/agent/src/config.py` |

The architecture itself is sound and already matches the stated goals (local perception, text-only LLM, cached prompts, reactive layer for reflexes). The gaps are **recall on units, global map awareness, villager-level precision, per-turn LLM cost, and the missing measurement loop**. The plan below attacks those in priority order.

> **Note (2026-07-19):** the table above is the 2026-07-09 snapshot. Several rows have since moved: the idle badge is now read as a **digit** (not just presence), the version drift is largely closed (v9 everywhere; `get_detector()` picks the newest bundled model), and the reactive tier now handles most routine turns with **no LLM call** (villager orders, Feudal/Castle prep, houses, age-up) — so "executor every turn" no longer holds. See the per-item **Status** callouts below and the run-review doc for the T-5xx work.

---

## P0 — Fix the measurement loop first

> You cannot improve precision you don't measure. Every item below this section will produce a number; without a baseline those numbers are noise.

### P0.1 Record a baseline in `experiments/results.tsv` and make it mandatory

**What:** Run 3–5 full games with the current stack (v9 detector, Sonnet/Sonnet) and record composite score, survival, population, age reached, action-success rate. Add a `just experiment` recipe that refuses to merge a change without a row in the TSV.

**Why:** The harness (`scenario_runner.py`, `assertions.py`, `strategist_eval.py`) is built, but the results table has only a header. Several past regressions (exp_0011 age hallucination, exp_0013 town-bell collapse) were caught by ad-hoc observation and are now baked into thresholds — a recorded baseline catches the next one systematically instead of by luck.

**Improves:** Every subsequent item becomes falsifiable. This is also the prerequisite for autoresearch Phases 2–5 (which are designed around exactly this feedback signal).

### P0.2 Grow the real-frame detection eval set from 33 to ≥200 images

**What:** The "metric of record" (real micro-F1) currently rests on ~33 real labeled frames. Capture frames across ages, biomes, and army compositions during the P0.1 baseline games; label in CVAT (bootstrap with `prelabel.py --open-vocab`); keep them as a frozen eval split that never enters training.

**Why:** With 33 images, a per-class recall number like `sheep 3/6` swings ±17 percentage points on a single sprite. You cannot tune per-class thresholds (`thresholds.py`, `--conf-sweep`) or judge a retrain against that variance. 200 frames is roughly a weekend of labeling with open-vocab prelabeling and pays for itself on the first retrain decision.

**Improves:** Trustworthy per-class P/R, meaningful threshold sweeps, and a stable go/no-go gate for every future model version.

### P0.3 Resolve model-version drift (30 minutes, do it now)

> **Status (2026-07-19): largely resolved.** `config.py` is the source of truth (v9); `get_detector()`/`resolve_model_path()` now pick the **newest bundled** `aoe2_yolo_v*` (not v6); `packages/detection/README.md` and `TRAINING_GUIDE.md` read v9. The root `README.md` + detection-server/VM-bringup residuals were the last stragglers and are fixed in the 2026-07-19 docs refresh.

**What:** Make `apps/agent/src/config.py` the single source of truth for the served model; update `packages/detection/README.md` (says v7), `get_detector()` default (resolves v6), and `TRAINING_GUIDE.md` (quotes v5 metrics, says "no v6 yet"). Also commit or discard the 16 modified files in the working tree.

**Why:** Three documents name three different production models. The next debugging session that runs the *local* detector fallback will silently evaluate v6 (F1 0.42) while believing it's v9 (F1 0.67) — that exact scale-mismatch class of bug already cost a debugging cycle once (v6 @1280 dropping F1 to 0.21).

**Improves:** Eliminates a whole category of "why did detection get worse" false alarms.

---

## P1 — Detection: close the recall gap where it costs games

The model is genuinely good at static scenery (trees, TCs, houses) and genuinely blind to the things that decide games: **moving military units and huntables**. Fixing that is not "train a bigger model" — it's a data problem.

### P1.1 Gameplay-driven active learning (autoresearch Phase 4, promoted)

**What:** Instrument the game loop to auto-harvest hard frames during real games:
- Tracker/detector disagreement (Kalman predicts an entity, detector loses it → save frame).
- LLM-reported failed actions on a `target_id` (executor clicked, nothing changed → the box was wrong → save frame).
- Alarm turns (combat frames are exactly where the rare classes appear).

Dump to a `hard_frames/` inbox, prelabel with open-vocab (`prelabel.py`), human-review the disagreements in CVAT, merge into the next dataset version.

**Why:** The rare-class tail (`knight_line` 0/45 recall, etc.) exists because those units barely appear in the 187 real images — and synthetic composites don't capture real motion blur, unit clumping, and combat clutter. Every real game the agent plays generates free, *perfectly targeted* training data: the frames it fails on are by definition the frames the model needs. This converts gameplay hours into model improvement automatically — the flywheel the autoresearch plan already sketched but never built.

**Improves:** Recall on exactly the classes that matter for combat and hunting, at near-zero marginal labeling cost (only disagreements get human review).

### P1.2 Scenario-farmed data for the zero-real-label classes

**What:** Use the AoE2 scenario editor to place the 11 zero-real-label classes (trebuchet, scorpion, king, krepost, trade carts, unique units…) in varied terrain, then screenshot-sweep with the existing capture tooling. 30 minutes of editor work per class yields hundreds of real-engine frames with known ground truth (you placed them, so labels are semi-automatic).

**Why:** These classes will *never* appear organically in early-game self-play, so P1.1 can't reach them. Scenario farming is the cheapest source of real-renderer data (real lighting, terrain, UI overlays — everything synthetic compositing approximates).

**Improves:** Removes the "model has literally never seen a real trebuchet" failure mode before Castle/Imperial play makes it fatal.

### P1.3 Split the `villager` class by job sprite

**What:** AoE2 DE renders villagers with distinct sprites per task: carrying wood, mining, farming, building (hammer), hunting (bow), fishing, idle-standing. Split `villager` into ~6 subclasses (`villager_lumber`, `villager_miner`, `villager_farmer`, `villager_builder`, `villager_hunter`, `villager_idle`) in `classes.yaml`, relabel (sprite differences make CVAT labeling fast, and open-vocab prelabeling handles most of it), retrain.

**Why:** This directly replaces the weakest link in villager guidance: `villager_roles.py` currently guesses a villager's job from whether it stands within 140px of a resource — which mislabels walkers, builders passing a mine, and anyone on a farm next to the TC. The sprite *is* the ground truth and the detector is already looking at it. This is the single highest-leverage change for the "guide villagers according to current needs" goal.

**Improves:** Accurate per-job villager counts in the LLM context ("6 on food, 4 on wood, 2 idle" becomes reliable), correct `reassign_villager` targeting, and it makes the reactive idle-dispatch smarter without any LLM involvement.

### P1.4 Consider one model-size step up, benchmarked, quantized

**What:** Train YOLO26**s** (small, ~3× nano FLOPs) on the same v9-era data, export static ONNX @1280, benchmark on the actual serving path (CoreML/ANE on the Mac detection server, DirectML on the VM fallback). Adopt only if real-F1 gain ≥ +0.05 and single-pass latency stays under ~80ms on the server path.

**Why:** Nano is the right call for a 1.2s-CPU VM, but the deployed path is a remote CoreML server at ~15ms/tile — there is latency headroom being left on the table. Small-object recall at 3024px-wide frames is partly a capacity problem that data alone won't fix. The gate matters: adopt on measured F1-per-ms, not on vibes.

**Improves:** Recall on ~20px sheep/berries/deer and crowded combat scenes, staying within the "fast and local" budget. Skip if P1.1–P1.3 data work already lifts real-F1 past ~0.80 — data beats capacity until it doesn't.

### P1.5 Per-class threshold tuning as a standing artifact

**What:** After each retrain, run `evaluate_real.py --conf-sweep` on the frozen ≥200-frame eval set and commit the recommended per-class thresholds to `thresholds.py` via the existing `sync_thresholds` mechanism. Treat the mill 0.55 floor as the template: FP-prone buildings get floors, recall-critical small objects get lowered thresholds — but derived from the sweep, not from incident response.

**Why:** Current thresholds are a mix of auto-generated values and post-incident patches (mill↔house). A single uniform confidence is provably wrong for a 60-class head with this much class imbalance; the sweep infrastructure already exists and just needs to run against a trustworthy eval set (hence P0.2 first).

**Improves:** Converts recall/precision trade-offs from firefighting into a repeatable, data-driven step of every release.

---

## P2 — Map awareness: give the agent a memory of the world

This is the biggest *architectural* gap. Detection sees the viewport; the game happens on the whole map. Both items here are classical CV + bookkeeping — **zero ML training, zero LLM tokens, ~ms latency**.

### P2.1 Minimap parser (classical CV, no ML)

**What:** Crop the minimap (fixed HUD region, same runtime-calibration approach `resource_ocr.py` already uses), and parse three things per tick:
1. **Player-colored blobs** → approximate positions of own/ally/enemy presence map-wide (fixed palette → simple color masks).
2. **Explored vs black fog** → exploration frontier for scouting decisions.
3. **The white viewport rectangle** → the camera's map coordinates (this is the anchor P2.2 needs).

Emit a compact text summary: `"minimap: enemy red blob NE (~grid F7), 62% explored, camera at D4"`.

**Why:** Strategy is impossible without global awareness — "where is the enemy," "is my second TC area safe," "what's unexplored" are unanswerable from a viewport. The minimap is a *rendered, palette-stable, fixed-position* miniature of exactly the state we need; parsing it is a solved classical-CV problem (color masks + connected components), infinitely cheaper than any learned approach, and fully fog-of-war-fair since it only shows what the player has explored. The original implementation plan even specced a `minimap_parser.py` — it was dropped in the pivot, not rejected.

**Improves:** Strategist can reason about map control, attack direction, and scouting; alarm system gains directionality ("attack from the north" instead of "3 enemy units visible"); enables the executor to navigate ("click minimap at F7") instead of blind camera-scrolling with H/arrow keys.

### P2.2 Persistent world model (entity memory anchored to map coordinates)

**What:** A `WorldModel` layer between detection and context-building:
- Use the minimap viewport rectangle (P2.1) to convert each viewport detection into **map-anchored coordinates**.
- **Static entities** (buildings, trees, gold/stone tiles, berries): persist forever once seen; mark stale-but-remembered when off-screen; delete only when a fresh look at their tile shows them gone.
- **Dynamic entities** (units): keep with a decaying confidence/TTL.
- Feed the LLM a two-part context: "visible now" (current YOLO frame, as today) + "known world" (compact summary: "TC at D4, gold at E6 (seen t-40), enemy barracks seen at F7 t-120").

**Why:** Today the agent has amnesia — scrolling to check the woodline erases its knowledge of the TC area, which is why the codebase fights stale-coordinate bugs (`_re_resolve_from_intent`, the core.md warnings about raw x/y after camera keys). Those are symptoms of missing state, patched at the action layer. A world model fixes the disease: the LLM plans against stable map knowledge, and camera movement becomes a navigation detail instead of a knowledge reset. The Kalman tracker already maintains per-entity identity within the viewport; this extends the same idea across camera moves.

**Improves:** Strategic planning quality (knows where its resources and the enemy are without looking), fewer wasted "scroll around to re-find things" turns (each one costs an LLM call ≈ seconds + tokens), and it structurally reduces the stale-coordinate failure class rather than patching it per-action.

### P2.3 Read idle-villager *count*, not just presence

> **Status (2026-07-19): DONE.** `resource_ocr.read_idle_count` reads the badge digit via a template-NCC bank; `GameState.idle_count` / `idle_streak` exist and size the reactive dispatch (with a trust gate). Note the count still mis-reads in some states (pinned-1, and pinned-41 in Feudal) — that's the open **T-302** instrumentation item, tracked in the run-review doc, not this one.

**What:** The idle badge digit is a tiny fixed-font number at a calibrated HUD position. RapidOCR fails on it, but a 10-glyph template-match (the `template` backend already exists in `resource_ocr.py` for exactly this style of problem) will read it near-perfectly.

**Why:** `reactive.py` currently dispatches up to 3 idle villagers *when the badge exists* — it can't tell 1 idle from 9 idle, so mass-idle events (post-combat, TC rally mistakes) drain slowly over many turns.

**Improves:** Idle-time — the #1 economic KPI in AoE2 — handled fully reactively with correct urgency, no LLM tokens spent.

---

## P3 — Decision layer: cheaper, faster, without losing reliability

The perception stack is already local and free. Nearly all recurring cost is the **per-turn executor call**. The lever is not a cheaper model across the board (Haiku-as-executor was tried and reverted for reliability) — it's **calling the LLM less** and calling it with better inputs.

### P3.1 Expand the reactive layer into a "standing orders" system

**What:** Promote more routine behavior from LLM-decided to rule-executed, with the strategist setting *parameters* instead of the executor deciding *actions*:
- Strategist emits a **target villager allocation** (e.g. `{food: 6, wood: 4, gold: 0}`) as part of its goals.
- The reactive layer continuously enforces it: new villagers from TC and re-dispatched idles route to the most-understaffed resource (trivial once P1.3 gives real per-job counts).
- Standard reflexes become rules: build a house when `pop ≥ cap − 3` (all inputs already OCR'd per turn), re-queue farms, keep TC producing.
- The executor LLM is then invoked only on **exceptions**: alarm, housed, stuck-loop, strategist goal change, resource starvation, or every Nth turn as a sanity check.

**Why:** Dark-Age AoE2 economy is nearly deterministic — human build orders are literally scripts. Burning a Sonnet call every 2–4s to decide "queue a villager, send it to berries" is paying frontier-model prices for a lookup table, and it's also *slower* than a rule (milliseconds vs seconds — in an RTS, decision latency is itself a precision loss). The architecture already trusts this pattern: villager queuing and idle dispatch moved into `reactive.py` and got *more* reliable, not less. This extends a proven direction rather than betting on a new one.

**Improves:** Cost per game drops an estimated 60–80% (most turns become $0), routine-turn latency drops from seconds to milliseconds, and LLM attention concentrates on the turns where judgment actually matters — which tends to improve those decisions too (less prompt fatigue from repetitive turns, more cache-stable context).

### P3.2 Tiered executor: retry the small model with a guardrail, not a prayer

**What:** With P3.1 in place, re-test `claude-haiku-4-5` as the *exception-turn* executor with an escalation rule: if Haiku's plan fails validation (unresolvable `target_id`, rejected coordinates, stuck-loop warning active) or the turn is combat, escalate that same turn to Sonnet. Measure via scenario fixtures (`scenario_runner.py` exists precisely for this A/B) before any live adoption.

**Why:** The earlier Haiku failure was as an *every-turn* executor with weaker context. Post-P1.3/P2.x the context is more structured (real job counts, world model, minimap), and post-P3.1 the call volume is low enough that an occasional escalation is cheap. The scenario harness makes this a measurable experiment instead of a leap of faith — and if Haiku still fails the fixtures, keep Sonnet and lose nothing.

**Improves:** Further cost reduction on the remaining LLM turns; the escalation path bounds the downside.

### P3.3 Compute gather rates locally and hand them to the strategist

**What:** Per-turn resource OCR already produces a time series. Derive rates (Δfood/min, Δwood/min, net of spending — estimable since actions and their costs are logged) and include them in strategist context: `"food +142/min, wood +89/min, gold +0/min"`.

**Why:** Absolute stockpiles are a lagging indicator; *rates* are what strategy decisions actually need ("can I afford Feudal in 90s?", "is my gold income zero?"). Right now the strategist would have to infer rates by comparing snapshots across its own sparse invocations — the game loop can compute them exactly, for free, every tick.

**Improves:** Strategist decision quality (age-up timing, allocation targets for P3.1) with ~20 extra prompt tokens and zero extra calls.

---

## P4 — Learning loop: make games make the agent better

### P4.1 Actually populate the cross-game memory chain

> **Status (2026-07-19): unblocked, still to populate.** The chain was a silent no-op on the VM for ~13 runs because one cp1252 byte made every load/save raise (`utf-8` decode error). That's fixed (T-536: tolerant reads + explicit-utf-8 writes), so the mechanism now runs — but `memories/` is still empty. Remaining: clean the corrupt VM file, then wire extraction into the baseline runs.

**What:** The pipeline (post-game Haiku extraction → ranked notes → cached system block, with `[applied: ...]` attribution) is fully built, but `memories/` is **empty** — it has never run against real games. Wire it into the P0.1 baseline runs and every game thereafter; review the first few extractions by hand to tune the extraction prompt.

**Why:** The autoresearch plan's own diagnosis is that the agent "never learns from its gameplay." The cheapest fix is turning on the learning mechanism that already exists. Attribution tags then tell you which memories actually change behavior — feeding P4.2.

**Improves:** Cross-game improvement at the cost of one Haiku call per game (~fractions of a cent).

### P4.2 Multi-turn regression scenarios from real logs

**What:** Close the `log_to_scenario.py` TODO (logs currently store entity *counts* but not coordinates, so replayed scenarios can't reconstruct positions — start logging the full detection payload, it's a few KB per turn). Then convert each interesting real-game moment (age-up decision, first raid response, housing crunch) into a multi-turn fixture, growing the 18-fixture suite from live play.

**Why:** The eval exploration doc's own gap list: no multi-turn eval means stuck loops, memory-accumulation regressions, and age-transition regressions ship undetected. Real-game logs are the richest source of realistic fixtures, and the conversion tool is 90% built — it's blocked on one logging change.

**Improves:** Regression safety net for every P1–P3 change, at fixture-replay cost (cents) instead of live-game cost (dollars + VM minutes).

---

## What NOT to do (and why)

- **Don't send screenshots to foundation models** — reaffirmed. At even 1 image/turn, vision tokens would dwarf the entire current budget, add seconds of latency, and reintroduce the 30–50% coordinate-guessing accuracy the YOLO pivot escaped. The research already showed local perception wins on all three axes (cost, speed, precision).
- **Don't revive the DLL-injection route** (`aoe2-ai-module`) for the main agent. The fog-of-war fairness problem is unresolved (per-player visibility undocumented/likely absent) and it version-locks you to game internals. *One legitimate niche:* an **offline ground-truth oracle** for evaluation only — running it in eval games to score detection/OCR accuracy against true state would supercharge P0.2 without any fairness issue, since it never feeds the playing agent. Treat that as optional.
- **Don't re-enable SAHI on current models.** The scale-mismatch finding (tiles present objects at non-training scale; F1 0.42→0.21) stands. If tiling ever returns, it must be trained-for (tile-cropped training data), which the 1280 single-pass path has made unnecessary so far.
- **Don't expand the 60-class taxonomy further** (except the P1.3 villager split) until the existing tail has real recall. Every class added below ~50 real instances is a liability that dilutes the head.

---

## Suggested sequence & expected payoff

| Order | Item | Effort | Cost impact | Precision impact |
|---|---|---|---|---|
| 1 | P0.3 version-drift fix + commit tree | hours | — | prevents false regressions |
| 2 | P0.1 baseline runs + results.tsv | 1 day | — | makes everything measurable |
| 3 | P2.3 idle-count template OCR | 1 day | — | economy KPI, reactive-only |
| 4 | P0.2 eval set → 200 frames | 2–3 days | — | trustworthy metrics |
| 5 | P2.1 minimap parser | 3–5 days | — | global awareness unlocked |
| 6 | P1.3 villager job classes (+retrain) | 1 week | — | villager guidance precision, feeds P3.1 |
| 7 | P3.1 standing-orders reactive layer | 1 week | **−60–80% $/game, −latency** | frees LLM for real decisions |
| 8 | P1.1 active-learning harvest | 1 week | — | recall flywheel begins |
| 9 | P2.2 world model | 1–2 weeks | fewer wasted turns | strategic memory |
| 10 | P1.2 scenario farming rare classes | 2–3 days | — | late-game classes exist |
| 11 | P3.3 gather rates | 1 day | — | strategist quality |
| 12 | P4.1 memory chain on | hours | +~$0.01/game | cross-game learning |
| 13 | P4.2 log→scenario fixtures | 3–5 days | cheaper regression tests | safety net |
| 14 | P1.4 YOLO26s benchmark | 3 days | — | only if data work plateaus |
| 15 | P3.2 tiered Haiku executor | 3 days | further −$ | gated on fixtures |

**North-star metrics** (all measurable with existing tooling after P0):
- Real-frame detection micro-F1: **0.67 → 0.85+**, with no class the agent acts on below 0.5 recall.
- Cost per 30-turn game: **$1–3 → <$0.50** (P3.1 does most of this).
- Routine-turn decision latency: **2–4s → <100ms** for rule-handled turns.
- Idle-villager time and action-success rate: tracked per game in `results.tsv`, trending up.
