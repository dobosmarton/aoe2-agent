# Move the agent policy from Python code into a learned rule registry

## Context

The last run proves the problem. In `logs/2026_08_15_1/logs.txt:1986` the final
metrics read `llm_calls=88 llm_errors=88 llm_error_rate=1.0` and
`highest_age='Feudal Age'`. Every LLM call failed on a temperature error. The
agent still reached the Feudal Age. Line 1129 names the actor:
`press intent='Research Feudal Age (reactive)' key=z`.

The agent plays the Dark Age from hardcoded rules. The LLM adds almost nothing.
Three layers hold the strategy:

1. `apps/agent/src/reactive.py` — 472 lines. A complete Dark Age bot.
2. `apps/agent/src/executor.py` — build gates, cost tables, the order ledger.
3. `apps/agent/src/prompts/ages/dark.md` — a build order with JSON templates.

Each constant cites a run review. 36 distinct findings (`F-1` to `F-46`,
`T-530` to `T-543`, `V-4`) sit as comments across 8 files, 13 of them in
`reactive.py`. **A learning loop already runs. The learner is the user, and the
output medium is Python constants.** This plan moves that loop into the agent.

### The agent is far too slow for a live game

Median seconds per turn, measured from the run logs:

| Run | p50 | p90 | worst |
| --- | --- | --- | --- |
| `2026_07_13_1` | 5 s | 17 s | 39 s |
| `2026_08_15_1` | 17 s | 28 s | 44 s |
| `2026_07_17` | 35 s | 64 s | 84 s |

An AoE2 villager trains in 25 s. At 35 s per turn the agent acts less than once
per villager. Two independent phases dominate, and each leads in a different
run:

| Phase | `07_13_1` | `08_15_1` | `07_17` |
| --- | --- | --- | --- |
| screenshot → OCR | 0 s | **11 s** | 1 s |
| detection → LLM response | 3 s | 1 s (erroring) | **25 s** |

`build_placement_retry` alone burned 153 s across one 90-turn run.

The root cause behind both: **the agent does not measure its own latency.** Only
`build_placement_retry` and `ocr_engine_warmed` carry timers.

### Three blockers stop the agent from learning

- **The objective rewards the wrong thing.** `apps/autoresearch/src/metrics.py`
  weights survival 0.30 and population 0.25, against age 0.20. Population
  saturates at 50. An agent that optimizes this score stays in the Dark Age.
- **The offline harness measures a different agent.** Neither
  `apps/agent/src/synth_game_loop.py` nor `apps/agent/src/scenario_runner.py`
  calls `reactive.decide()`.
- **The simulator cannot see the main decision.** `world_sim.tick()` adds a flat
  `FOOD_GATHER_RATE = 20.0` whatever the villagers do, and `render()` emits no
  resources. Villager allocation has no effect.

The memory system is built but inert. `memory_chain.py` works, `memories/` is
empty, and `apps/agent/src/main.py` never calls it.

**Intended outcome.** The strategy becomes data. A fast engine reads a rule
registry every tick. The LLM never blocks that tick — it writes parameters. The
rules earn their place from measured outcomes.

**Accepted cost.** The agent will reach the Feudal Age more slowly for some
time. AoE2 openings are near-optimal scripts, and an agent that re-derives one
each game is worse than one that runs it. The gain is generalization to the
Castle Age, the Imperial Age and combat, plus a learning result that is
measurable.

---

## Phase 0 — Freeze the knowledge, and start measuring

Run this before anything deletes code. Phase 1.6 deletes `reactive.py`, and 13
of the 36 findings live only in its comments.

### 0.1 The seed corpus

Add `apps/agent/knowledge/seed/`:

| Path | Content |
| --- | --- |
| `rules/*.yaml` | The `reactive.py` rules, one per constant, with `F-` provenance. |
| `findings.md` | All 36 `F-` / `T-` / `V-` findings: tag, file, the constant it justifies, the rationale text. |
| `prompt_rules.md` | The strategy in `prompts/ages/*.md` and `core.md` — build order, gather ratios, age-up gate. |
| `archive/` | The 9 memory files in `logs/2026_04_24/memories/` and `logs/2026_04_25/memories/`. |
| `baseline.tsv` | The 14 `experiments/results.tsv` rows, stamped `score_version: 1`. |

Convert the archived memories; do not import them. They use the old prose
format, they cite turn numbers, which `memory_chain.EXTRACTION_SYSTEM` now
forbids, and `006_missing_feudal_age_target.md` is 0 bytes.

### 0.2 Seed rules are protected

A seed rule carries `provenance: seed`. It is **exempt from retirement until it
has been ablated at least once**. Without this, the first noisy sweep deletes
knowledge that took months to acquire.

### 0.3 Instrument the loop

Add a per-phase timer to `apps/agent/src/game_loop.py`: capture, OCR, detect,
deliberate, act. Log milliseconds per phase per turn. Add p50 and p90 turn
latency to `experiments/results.tsv`.

Do this first. The numbers in the Context section come from 1-second log
timestamps, which is too coarse to tune against.

---

## Phase 1 — Make the policy data, not code

Behavior must not change here. The 53 tests in `tests/test_reactive.py` prove it.

### 1.1 One state view

Add `apps/agent/src/policy/state.py` with a frozen `PolicyState`. Build it from
`memory.GameState` (real game) and `core.WorldState` (simulator).

Fields: `age`, the 4 resources, `population`, `population_cap`,
`villagers_ordered`, `buildings_seen`, `idle_present`, `idle_count`,
`idle_streak`, `villager_jobs`, `turn`, `captured_at`.

Reuse `apps/agent/src/villager_roles.py` `job_counts` for `villager_jobs`.

### 1.2 The rule registry

Add `apps/agent/src/policy/rules.py`. The registry is the agent's own rule
store. The agent appends to it; Phase 6 gives each rule a learned weight.

```yaml
- id: feudal_prep_mill
  when: "age == 'Dark Age' and population >= 12 and 'mill' not in buildings_seen"
  then: {type: build, building_key: w}
  weight: 60              # learned in 6.1; seeded by hand
  cost: {wood: 100}       # declared, so the engine can reserve
  max_state_age_ms: 3000  # see 3.4
  provenance: seed/F-41
  enabled: true
  stats: {fired: 0, ablation_delta: null, ablation_ci: null}
```

`when` is an expression over `PolicyState` fields only. Use a restricted
evaluator, never `eval`. Allow comparison, membership, `and`, `or`, `not`, and
field names. Reject anything else at load time.

`then` must pass `apps/agent/src/models.py` `validate_actions`. Seed `cost` from
`executor.py:708` `_BUILD_WOOD_COST` so one table stays canonical.

**The registry loads whole. It is never searched at play time.** The engine
tests every `when` against `PolicyState` each tick. That is exact retrieval and
costs microseconds. Semantic search would return rules whose triggers do not
match. Search belongs at write time only — see 4.2.

Cap the registry per age, as `memory_chain` caps memories at 20.

### 1.3 The engine

Add `apps/agent/src/policy/engine.py` with
`decide(entities, state, alarm) -> list[dict]`.

**A weight alone cannot order the rules. The engine must reserve resources.**
Two rules can both want the same 100 wood. Today `reactive.py` hides this in
ordered if-chains and in `_wood_bank_target`, which ranks mill above lumber camp
above mining camp above farm. A plain sort by weight loses that, and the engine
emits two builds the executor then rejects.

Selection is 3 steps:

1. Collect every rule whose `when` matches and whose state is fresh enough.
2. Sort by descending `weight`.
3. Walk the list, subtract each `cost` from a running balance, and drop a rule
   the balance can no longer pay.

Ship this before Phase 6 learns any weight.

Two behaviors read the entity list rather than the state, so they stay in
Python: idle-target resolution (reuse `entity_utils.nearest_class_of_kind` and
`first_center_of_class`) and the allocation comparison in 4.1.

### 1.4 Seed the registry

Export the Phase 0.1 rules into `apps/agent/policy/`: `dark_age.yaml`,
`feudal_age.yaml`, `castle_age.yaml`, `safety_floor.yaml`.

### 1.5 The safety floor stays in code

These are game facts, not strategy:
- `executor.py:722` `_BUILD_PREREQ_CLASS = {"a": "mill"}`. Without a mill the
  `A` key builds an Outpost. Safety critical.
- `executor.py:114` `_GAME_POP_CAP_LIMIT = 200`.
- `executor.py:643` `STALE_COORDS_DETAIL` and `CAMERA_KEYS`.
- `executor.py:708` `_BUILD_WOOD_COST`.

### 1.6 Delete `reactive.py`

Point `game_loop.py:127` at `policy.engine.decide`. Run
`pytest tests/test_reactive.py`. All 53 tests must pass with only the import
changed.

---

## Phase 2 — Fix the objective

### 2.1 Record age timestamps and victory

`memory.update_age` stamps the time for each age reached. Add
`age_times: dict[str, float]` to `AgentMemory`, and `feudal_time_s` and
`castle_time_s` to `MetricsSnapshot`.

`game_end_reason` holds no real `victory` today. Detect the game-over screen, or
prompt the operator in `apps/autoresearch/src/game_runner.py`.

### 2.2 Rewrite the composite

| Term | Now | New |
| --- | --- | --- |
| `age_progress` | 0.20 | 0.40 |
| `time_to_next_age` | — | 0.25 |
| `economy` | 0.15 | 0.20 |
| `action_success` | 0.10 | 0.10 |
| `survival` | 0.30 | 0.05 |
| `population` | 0.25 | 0.00 |

Score `time_to_next_age` against a reference of about 600 s. A victory sets the
composite to 1.0.

**Warning.** This breaks comparison. Add a `score_version` column first. The 14
existing rows stay at version 1.

---

## Phase 3 — Three clocks, so acting is never blocked

This is the answer to the latency evidence. The current loop couples perception,
deliberation and action into one tick, so the tick is as slow as the slowest
phase.

### 3.1 The act loop

Target **250 ms p95**. It reads the latest snapshot and the latest parameters,
runs `policy.engine.decide`, and executes. It **never awaits** perception or an
LLM.

### 3.2 The perception loop

Screenshot, detection, OCR. It writes an immutable snapshot carrying
`captured_at`. It runs at its own cadence and never blocks the act loop.

### 3.3 The deliberation loop

The strategist and the executor run asynchronously. Call them every iteration if
wanted. **They write parameters only — goals, allocation, rule weights, enable
flags. They never write actions.**

This is the load-bearing constraint, and the measurements force it. LLM latency
is 3 s to 25 s p50. At 25 s an answer is more than 100 act ticks old. Screen
coordinates are stale by then; an allocation target is not.

This is also why the rule registry is necessary rather than merely tidy. The
registry is the interface that lets slow thinking drive fast acting.

### 3.4 Staleness budgets, not freshness guarantees

Each rule declares `max_state_age_ms`. An age-up press needs a fresh food
reading. An idle dispatch does not. When the snapshot is older than a rule
allows, **skip that rule this tick. Never block.**

### 3.5 Take the known costs off the act path

- **OCR.** Measured at 11 s p50 in `2026_08_15_1`. Move it to the perception
  cadence. Resource values change slowly and predictably.
- **Build retries.** `build_placement_retry` burned 153 s in one run, p50 9 s per
  event. Make settlement a background check, not an inline wait.
- **Remove the pipelining exceptions.** `config.pipeline_commit_max = 2`
  discards most of each plan, and combat turns still run synchronously. Both
  exist only because actions sat on the critical path. Neither is needed once
  the LLM emits parameters.

### 3.6 Budgets

| Loop | Target |
| --- | --- |
| act | ≤ 250 ms p95 |
| perceive | ≤ 2 s p95 |
| deliberate | unbounded, non-blocking |

---

## Phase 4 — Give the strategist the pen

### 4.1 Allocation replaces the gather pattern

Extend `StrategistResponse` in `providers/strategist.py:34`:

```python
allocation: dict[str, int]   # {"food": 6, "wood": 4, "gold": 0, "stone": 0}
```

The engine routes each idle villager to the most understaffed resource by
comparing `allocation` against `PolicyState.villager_jobs`. This removes
`_IDLE_PATTERN_BY_AGE` and `_idle_pattern`.

Update `prompts/strategist.md`. Remove the phase build order in lines 16 to 29.
Add the allocation contract.

### 4.2 Rule proposals, deduplicated at write time

Add `rule_proposals: list[Rule]` to `StrategistResponse`. A proposal writes to
`apps/agent/policy/proposed/` with `enabled: false`. A disabled rule is
evaluated and counted, but not executed. That is shadow mode.

**Search the registry here, and only here.** Before a proposal is written, find
the rules whose triggers overlap it and hand those to the strategist, so it
edits one instead of adding a near-duplicate. Without this, the registry fills
with variants and every weight becomes noisy.

Detect overlap structurally. Both `when` expressions parse to predicates over
the same `PolicyState` fields. Compare the field sets and the comparison bounds.
No embeddings are needed. This replaces `memory_chain._existing_titles`.

### 4.3 Split the models

`config.py` already carries `model` and `strategist_model`, both
`gpt-5.6-luna`. Set the executor to a fast model and the strategist to a strong
one. Phase 3.3 makes executor latency far less critical, so choose on quality.

---

## Phase 5 — Make the simulator able to teach

### 5.1 Villager-aware gather

Add `villager_jobs: dict[str, int]` to `core.WorldState`. In `world_sim.tick`:

```python
food = state.food + FOOD_RATE_PER_VILLAGER * state.villager_jobs["food"]
```

Apply the same to wood, gold and stone. A new villager joins the job with the
largest shortfall against the allocation.

### 5.2 Render resources

`world_sim.render()` returns only `[town_center, *villagers, *buildings]`. Add
sheep, berry bushes, trees, gold mines and stone mines at stable positions.
Without them `nearest_class_of_kind` returns `None` and every targeting rule
breaks in the simulator.

### 5.3 Run the engine in the simulator

Call `policy.engine.decide` each turn in `synth_game_loop.py` and in
`scenario_runner.py`. This is what makes the offline harness test the agent that
plays.

### 5.4 Rank rule sets

Add `policy_set: str` to `ConfigProfile` in `apps/arena/src/config_profile.py`.
`apps/arena/src/ranking.py` then ranks registries with bootstrap confidence
intervals. No new statistics code is needed.

---

## Phase 6 — Close the loop

### 6.1 Rank by ablation, never by correlation

**Do not credit a rule for the games it fired in.** That measure inverts the
truth. Counting firings in the run that reached Feudal:

| Rule | Firings | Score produced |
| --- | --- | --- |
| `Queue villager (reactive)` | 21 | none by itself |
| `Research Feudal Age (reactive)` | 1 | the whole age score |

A correlation measure ranks the queue rule 21 times above the rule that scored.

Measure causally. For each rule: disable it, replay the arena over the 4
scenarios in `apps/arena/src/scenarios.py`, record the composite delta with a
bootstrap confidence interval, and write `stats.ablation_delta` and
`stats.ablation_ci` back. `weight` becomes a function of `ablation_delta`.

Leave-one-out is affordable at 15 to 30 rules. Re-run on a schedule, not per
game.

**Gate the promotion.** Change a weight only when the interval is clear of 0.
Never rerank `safety_floor.yaml`. Retire a rule when it never fires over N
episodes, or when `ablation_delta` stays positive, which means the agent scores
better without it. Retirement sets `enabled: false`. Respect the Phase 0.2
exemption.

### 6.2 Extraction writes rules, not prose

Change `memory_chain.EXTRACTION_SYSTEM` to emit the 1.2 schema. Keep the
first-person note as `provenance`. Wire `extract_memories` into
`apps/agent/src/main.py`; today only `game_runner.py` calls it.

### 6.3 The bootstrap experiment: cold against seeded

Run 2 arms over the same scenarios:

- **COLD** — `safety_floor.yaml` only. No strategy rules.
- **SEEDED** — the safety floor plus the Phase 0 seed corpus.

Let the strategist propose in both. Promote on shadow-mode evidence per 6.1.

Two questions get answered at once: whether the loop rediscovers food first,
then mill and lumber camp, then 500 food, then Feudal — and **what the human's
months of run reviews were actually worth**, as the gap between the arms.

Report the result whatever it is. A negative result is a finding.

---

## Sequence

| Order | Phase | Why here |
| --- | --- | --- |
| 1 | 0 | Nothing may delete `reactive.py` before the findings are frozen. |
| 2 | 1 | No behavior change. The 53 tests prove it. |
| 3 | 2 | Every later measurement needs the right objective. |
| 4 | 3 | The latency fix is independent of learning and pays off immediately. |
| 5 | 5.1 to 5.3 | The loop measures nothing until the simulator sees allocation. |
| 6 | 4 | The strategist needs a working measurement to aim at. |
| 7 | 5.4, 6.1, 6.2 | Ablation needs cheap episodes to ablate over. |
| 8 | 6.3 | The experiment. |

Three constraints cross phase boundaries:

- **0.1 before 1.6.** 13 findings live only in `reactive.py` comments.
- **1.3 before 6.1.** A learned weight on an engine that cannot reserve produces
  rejected actions, not better play.
- **4.2 before proposals at volume.** Duplicate rules make every ablation
  measurement noisy.

---

## Verification

**Phase 0.** `findings.md` contains 36 tags. Assert the count in a test, so a
later refactor cannot silently drop one.

**Phase 0.3.** Run 1 game. Confirm `results.tsv` carries p50 and p90 turn
latency, and that per-phase milliseconds appear in the log.

**Phase 1.** `pytest tests/test_reactive.py` — 53 tests pass with only the import
changed. `just eval-all` — the 18 scenario fixtures still pass.

**Phase 1.3.** Unit test: 100 wood, 2 matched build rules costing 100 each.
Assert 1 action is emitted, and that it is the higher-weight rule.

**Phase 2.** `pytest tests/test_metrics.py`. Recompute the 14 rows under version
2 and confirm the Feudal run `2026_08_15_1` now outranks the Dark Age runs. It
does not today.

**Phase 3.** The load-bearing check. Run 1 game and assert:
- act loop p95 ≤ 250 ms
- turn latency p50 improves against the 5 s / 17 s / 35 s baselines
- no `llm_response` sits on the act path — grep the trace for an awaited LLM
  call inside a tick

Add a test that stalls the LLM for 30 s and asserts the act loop keeps ticking.

**Phase 4.** Scenario fixture: 8 villagers on wood, 0 on food. Assert the
emitted allocation moves villagers to food.

**Phase 5.** `world_sim` test: 2 states differing only in `villager_jobs`.
Assert the food-heavy state banks 500 food in fewer turns. This fails today.
Run `just arena-smoke`, which is offline and needs no key.

**Phase 6.1.** The calibration check. Assert 2 known results:
- Removing `feudal_prep_mill` costs score, with an interval clear of 0.
- The ranking does **not** place `queue_villager` above `age_up_feudal`. If it
  does, the attribution is correlation-based and the implementation is wrong.

Write the second assertion before the sweep runs.

**End to end.** `just experiment "policy engine baseline"` on the VM. Compare
against `2026_08_15_1`. Then set `AOE2_LLM_API_KEY` to an invalid value and run
again — the LLM-down run shows how much the registry alone carries.

---

## Risks

- **The simulator is low fidelity by design.** A rule that wins there can lose in
  the real game. Every promoted rule needs 1 real game before it is trusted.
- **The restricted evaluator is an attack surface.** The strategist writes the
  `when` expressions. Reject unknown names at load time; never use `eval`.
- **Three loops mean shared mutable state.** The snapshot must be immutable and
  swapped by reference. A partially written snapshot read mid-tick is the
  classic bug here.
- **A stale snapshot can drive a wrong action.** `max_state_age_ms` is the
  mitigation, and skipping beats blocking. Confirm each safety-floor rule has a
  tight budget.
- **A noisy weight feeds back into play.** The confidence gate in 6.1 is the
  mitigation. The safety floor is never reranked.
- **Ablation cost grows with the registry.** Linear in rule count. Past about 60
  rules, group the rules and ablate the groups first.
- **The objective rewrite invalidates the ledger.** `score_version` is the
  mitigation. Do not delete the old rows.
