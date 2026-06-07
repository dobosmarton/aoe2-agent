# Chapter 18 — Synthetic World Sim

`packages/evaluation/src/world_sim.py` is an **AoE2-lite economy simulator**: enough state to let the agent's actions meaningfully evolve a world across N turns, without booting the real game. It is the substrate for the arena harness (`race`, `rank`, `smoke`) and for the multi-turn scenario runner (`packages/evaluation/src/runner.py`).

The fidelity is deliberately low. This is a behavioural regression harness, not a game engine. The goal is catching stuck loops, inhibitory-memory failures, and age-transition regressions — not simulating AoE2 exactly.

<aside class="prereqs">

[Chapter 14 — Arena Overview](./14-arena-overview.md) for what consumes this simulator. Familiarity with discrete-time simulation (state → step → state) — no PDE or physics background needed.

</aside>

## `WorldState`

`world_sim.py:106`. A mutable dataclass:

```python
@dataclass
class WorldState:
    food: float
    wood: float
    gold: float
    stone: float
    population: int
    pop_cap: int
    age: str                       # "Dark Age" | "Feudal Age" | "Castle Age" | "Imperial Age"
    buildings: list[str]           # may contain duplicates (multiple houses)
    villager_queue: list[int]      # countdown ticks per pending villager
    age_up_ticks_remaining: int    # 0 = not in progress
    turn: int = 0
```

The Pydantic mirror `WorldStateSnapshot` (`event_log.py:57`) is what gets embedded in `turn_start` events for forking. `from_world_state` / `to_world_state` are pure conversions; nothing else crosses that boundary.

## The economic model

Constants are all named at the top of `world_sim.py:30–70`:

- **Resource gather** is flat per turn: `+20 food`, `+15 wood` (`FOOD_GATHER_RATE`, `WOOD_GATHER_RATE`). No villager-assignment tracking — gather rates are constant regardless of how many villagers exist. This is the load-bearing simplification.
- **Villagers cost 50 food** and take 3 ticks to produce (`VILLAGER_COST_FOOD`, `VILLAGER_PRODUCTION_TICKS`). The queue is a list of countdown ints; each `tick()` decrements them and promotes any that hit zero into `population`.
- **Buildings** have a wood cost and a name keyed by the same key the executor uses in real-game prompts. `q` = house (25 wood, +5 pop_cap), `w` = mill, `r` = lumber camp, `e` = mining camp, `a` = farm, `s` = blacksmith, `t` = dock. See `BUILDING_COSTS` and `BUILDING_NAMES`.
- **Age-up costs 500 food and takes 6 ticks**, gated by four prerequisites checked in `_feudal_prereqs_met`: age must be Dark, food ≥ 500, population ≥ 22, and `{mill, lumber_camp}` both built. Press `z` (`_apply_age_up`) silently no-ops if *any* prerequisite is missing. The "silently no-ops" behaviour is the one the `strategy` prompt variant is built to compensate for — see `apps/arena/src/prompts.py:23`.

## How a turn is applied

The arena loop applies LLM-emitted actions and then ticks once:

```python
state = apply_actions(state, actions)   # zero-or-more action handlers
state = tick(state)                     # gather, complete villagers, advance age
```

`apply_action` (`world_sim.py:242`) dispatches through `_ACTION_HANDLERS` keyed on `action["type"]`. Unknown action types silently no-op — the loop accepts anything the executor emits without crashing. Click/right_click/drag/wait/scroll/detect are accepted but have no economic effect; they exist so the synth tier stays faithful to the real executor's action surface (any drift in the surface would force every test to track both).

`tick` (`world_sim.py:272`):

1. Decrements every entry in `villager_queue`. Entries hitting zero become `+1 population`.
2. Decrements `age_up_ticks_remaining` if non-zero; when it hits zero, advance via `_next_age`.
3. Adds flat gather to `food` and `wood`.
4. Bumps `turn`.

The `replace`-style state evolution (`dataclasses.replace` everywhere) keeps each function pure-ish; no in-place mutation makes it trivial to log before/after states for fork-diffing.

## Synthetic perception (`render`)

`world_sim.py:430` is the perception-projection layer. It maps a `WorldState` to a list of `DetectedEntity` — the same schema `packages/detection/src/inference/detector.py` emits from real YOLO inference:

- 1 `town_center` near screen centre, jittered ±100 px x / ±50 px y by the RNG.
- `state.population` villagers scattered around the TC within ±150/±100 px.
- One entity per `state.buildings` entry, laid out on a stable 4-column grid offset 220 px from the TC.

Key invariants:

- **Same state + same dims + same seed ⇒ identical output.** Determinism is the point — perception variance is a separate axis we can switch on independently.
- **`villager_queue` entries are deliberately excluded.** Queued villagers are not yet on the map for the agent to perceive.
- **Confidence = 1.0** everywhere (ground truth). No bbox jitter, no missed detections.
- **The RNG is local** (`random.Random(seed)`). Never mutates global `random` state — important because the real-game tier still uses module-level `random`.

This function is the substrate for `packages/detection/src/inference/mock.py:mock_detect_from_world` — the entry the arena uses where the real-game tier would call YOLO.

### Schema-lock test

`tests/test_detector.py::TestSyntheticRenderSchemaContract` is parametrized over both `mock_detect` (the original frozen Dark-Age fixture) and `mock_detect_from_world` (this projection). It asserts 10 invariants on each: id non-empty, class_name in canonical YOLO list, bbox well-ordered and within screen dims, center inside bbox, confidence ∈ [0,1], area > 0, `to_dict()` keys, sort order, id uniqueness, and `population=15` yields 7 more villager entities than `population=8`. Any drift between the two perception surfaces fails CI. This is what makes it safe to swap perception backends without touching every call site.

## Multi-turn assertions (`evaluate_end_state`)

`world_sim.py:313`. The scenario runner's end-state spec uses ≥-semantics for numeric fields and exact equality for strings:

```yaml
end_state:
  age: "Feudal Age"     # exact
  population: 15        # at least 15
```

Failures come back as plain strings (`packages/evaluation/src/runner.py` formats them with turn counts). Unknown WorldState field names produce a failure rather than silently passing — `end_state: { spaghetti: 7 }` will tell you it's a typo.

<details class="deep-dive">
<summary>Deep dive — Determinism in LLM agents (why temp=0 is not enough)</summary>

This chapter's synthetic world sim is **fully deterministic** — same state, same seed, identical output. The agent that runs against it is decidedly not. That asymmetry is worth understanding because every "is this prompt better?" experiment depends on being able to attribute outcome differences to the change you made, not to noise from the LLM.

**Sources of LLM nondeterminism, ranked by impact:**

1. **Sampling temperature > 0.** `temperature=0.5` means each token is sampled from the distribution, so re-running the same prompt yields different completions. Knob is obvious; impact is enormous.
2. **Temperature = 0 doesn't mean deterministic.** With `temperature=0` the API returns the argmax token. But the underlying inference still has *internal* nondeterminism because (a) modern accelerators do BF16/FP16 reduction in non-deterministic order across batch elements, and (b) the provider may route your request to different GPU clusters with slightly different numerics. Anthropic's docs are explicit: temperature=0 minimizes but does not eliminate variance.
3. **Tool-path variance.** Even with the same first action, a small difference downstream (a tool returning a slightly different snapshot) can cascade into completely different sequences over 60 turns. This dominates run-to-run variance in long agentic loops more than token-level sampling does.
4. **Hidden retries and load balancing.** A transient 5xx + automatic retry from tenacity can change the order of operations subtly. Provider-side load balancing can route consecutive calls to different model replicas with different cache hit rates.
5. **Prompt-cache hits vs misses.** A request that hits the cache reuses the provider's pre-computed KV state. A miss recomputes it. The completions *should* be identical at temperature=0; in practice there are reports of marginal drift.

**How to measure your nondeterminism.** Pick a fixed prompt and a deterministic environment (this chapter's synth sim). Run N times at the same temperature and measure outcome divergence. Empirically: at `temperature=0` we see ~3–5% of runs diverge by turn 20; at `temperature=0.5` it's ~30–60%. That's why the ranking chapter uses 5 rounds per (profile × scenario) — fewer rounds and the bootstrap CIs swallow the actual signal.

**The standard mitigations.**

- **Pass a `seed` parameter** where the API supports it (OpenAI does; Anthropic doesn't have a public seed parameter as of writing). At best this reduces sampling variance to zero; it does *not* eliminate hardware-level nondeterminism.
- **Run more samples.** Boring, expensive, and the most reliable. The bootstrap CIs in chapter 17 are honest about how much of the result is noise.
- **Hold non-causal variables fixed.** Same prompt cache state (warm up the cache once before the experiment), same model snapshot (Anthropic versions models by date), same temperature.
- **Test with the *deterministic* substrate.** The synth world sim is deterministic by design. If two runs against it produce different scores, all of that difference came from the LLM. That's the only experimental setup where you can cleanly attribute variance.

**The pragmatic bar:** "deterministic enough that a real effect at the 5% level is detectable in N=5 rounds with bootstrap CIs." We don't try to make the LLM perfectly reproducible — we make the *experiment* statistically powerful enough that we don't need to.

</details>

## Real-game tier impact: zero

Nothing in `apps/agent/src/` was modified by the arena buildout. The existing `mock_detect()` keeps its frozen Dark-Age fixture behaviour. Arena callers reach for `render()` / `mock_detect_from_world()` explicitly. The synth_game_loop in `apps/agent/src/synth_game_loop.py` is a separate code path from the real `game_loop.py`.

This is the line that keeps the architecture honest: arena improvements never threaten the production agent.

## Related reading

- [Chapter 14 — Arena Overview](./14-arena-overview.md) — how the world sim plugs into `race` / `rank` / `smoke`.
- [Chapter 7 — Detector Architecture](../part3-entity-detection/07-detector-architecture.md) — the canonical 60-class taxonomy `render()` projects into.
- [`docs/design/synthetic-arena-analysis.md`](../design/synthetic-arena-analysis.md) — the original "what exists, what's missing" matrix that motivated this module (frozen historical analysis).
