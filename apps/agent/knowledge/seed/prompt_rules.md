# Prompt-encoded strategy, frozen

**Frozen 2026-08-17.** `ADAPTIVE-AGENT-PLAN.md` 4.1 rewrites
`prompts/strategist.md`, and the age prompts follow. This is the strategy that
would otherwise be lost in that rewrite.

Source: `apps/agent/src/prompts/core.md`, `prompts/ages/{dark,feudal,castle,imperial}.md`,
and `packages/data/src/game_knowledge.py::get_early_game_priorities`.

This is the **third** hardcoded layer named in the plan's Context. It is
strategy the model recites rather than decides.

---

## 1. Dark Age

From `prompts/ages/dark.md`. Headline: *"Your only goal in Dark Age is to grow
your economy as fast as possible. No military, no fighting."*

| Rule | Value | Already a seed rule? |
| --- | --- | --- |
| Mill + 3 farms | by turn 5-8, **not** turn 15+ | no — timing rule, see below |
| Villager allocation | 6-8 food, 3-4 wood | yes — `allocation.yaml` Dark 3:2 |
| Villager order target | 30, then bank | yes — `queue_villager_dark` |
| Lumber camp | by turn 10-15 | partly — `feudal_prep_lumber_camp` gates on pop 12, not turn |
| Auto Scout | press `,` then `G`, **once** | no — not in `reactive.py` at all |
| Age-up prereq pair | Mill AND Lumber Camp | yes — `age_up_feudal` |
| Build menu | economic (`Q`) only in Dark Age | no |

### The proactive-farm rule has no code equivalent

> *"Plant farms BEFORE the food crisis, not after. Farms take ~60 s to start
> producing — if you wait until food < 50, the crisis is already locked in.
> Sheep deplete around turn 10-12; the farm pipeline must already be running.
> (Past games dropped below 50 food for 12 of the first 20 turns because farms
> came too late.)"*

Trigger: 100 wood AND 4+ villagers gathering food.

**`reactive.py` has no equivalent.** Its farm rule is *reactive* — it fires only
when a food slot finds nothing huntable on screen, which is after the crisis has
started. This is a genuine gap in the seed registry, and a good first candidate
for a strategist-proposed rule.

### Gathering order

sheep → berries (build Mill near berries) → farms (Mill anywhere, then 1 farm
per food villager).

### The one-Mill rule, and its counter-rule

*"One Mill is enough — decide from the Detected Entities list, never from
habit."* A second Mill wastes 100 wood; farms produce food, Mills do not.

But also: *"Sanity check the mill detection (avoid a house-as-mill trap)."* If a
`mill` appears in detections while the strategist reasoning says no mill is
visible, or the agent has never built one and food keeps starving, treat it as a
misdetection and build a real Mill.

**These two rules conflict by design.** The prompt resolves it by cost
asymmetry: a redundant Mill costs 100 wood, while farms with no drop-off never
fix food. Any learned rule pair here must preserve that asymmetry. Related:
F-36, the persistent phantom mill.

### Dark Age prohibitions

- **Never right-click a boar.** (Safety floor, F-20.)
- **Never build Towers or any defensive building** — not even under attack.
- **Never press B (town bell) or T (garrison)** — no exceptions in Dark Age.
- **Ignore the strategist's `alarm` flag** in Dark Age; continue the build order.
- If accidentally garrisoned: `H` → `V` (All Back to Work).

---

## 2. Feudal Age

From `prompts/ages/feudal.md`. Headline: 85% economy, 15% military.

### Castle age-up gate

All 4 must hold: age reads Feudal, food ≥ 800, gold ≥ 200, and **2 Feudal
buildings** exist — barracks, archery range, stable, blacksmith, or market.
Houses, mills, lumber camps and mining camps do **not** count.

**No code enforces this.** `reactive.py` has no Castle age-up rule at all — only
`castle_prep_mining_camp`. The Castle gate exists solely in this prompt, and the
qualifying-class list is not mirrored anywhere in Python.

> *"Missing Castle Age leaves you with no knights, no monks, no unique units, and
> no Castle. Against any non-trivial AI you will lose."*

### Feudal allocation and gold

| Rule | Value | Seeded? |
| --- | --- | --- |
| Villager allocation | 10-12 food, 6-8 wood, 3-4 gold | yes — `allocation.yaml` 2:2:1 |
| Order target | 35, then bank for Castle | yes — `queue_villager_feudal` |
| Mining camp | within the first 2 turns of Feudal | yes — `castle_prep_mining_camp` |
| Villagers on gold | **at least 4, permanently** | no |
| Gold emergency | if gold < 100, halt new buildings and divert 2+ villagers | partly — `bank_gold_for_castle` biases at < 200 |
| Loom | research when able (`H` → `A`, 50 gold) | no |

> *"Gold is the gating resource — defend the gold-on-gold ratio. Past games
> stalled at 50-110 gold because villagers kept being pulled back to food/wood."*

### Farm management

1 farm per food villager. Reseed expired farms immediately. Farms need a
drop-off nearby (TC or Mill).

**Farm expiry is not modelled anywhere** — not in `reactive.py`, not in
`world_sim`. A learned rule cannot discover reseeding until the simulator
represents it.

---

## 3. Castle and Imperial

`prompts/ages/castle.md` and `imperial.md` follow the same shape: allocation
ratios, an age-up gate, and build priorities. `reactive.py` covers these ages
**only** through `_IDLE_PATTERN_BY_AGE` (seeded in `allocation.yaml`).

Everything else in these two prompts is unimplemented in code. This is where the
plan's claim that the machinery should generalize gets tested — there is no
script here to beat.

---

## 4. Universal rules from `core.md`

The every-turn checklist, in order:

1. Idle villagers are auto-distributed. Do **not** use `send_all_idle` — it dumps
   everyone onto one tile.
2. Villager queuing is automatic up to the age order target.
3. **Housed (pop = pop cap)?** Build a house immediately. *"This is the #1
   game-losing mistake."*
4. Need houses soon? Build one at `population >= pop_cap - 3`. Never more than
   one per turn — a house adds 5 slots.
5. **Food emergency (food < 50):** dedicate the entire turn to farms. Use
   `reassign_villager` from wood rather than waiting for an idle villager.
6. Keep at least half the villagers on food. Never 0 food gatherers.

Note the threshold drift: the prompt says house at `pop_cap - 3`, the executor
rejects above 4 headroom, and `reactive.py` triggers at 2. Three numbers for one
rule. The registry collapses them to one — `house_when_headroom_gone`, at 2.

### Never return 0 actions

> *"NEVER return 0 actions. If you have nothing else to do, queue a villager,
> build a needed house/farm, or advance your build order."*

6-7 turns per game were lost to narrated plans with no actions (F-7, F-19).

### The town bell rule

`B` may be pressed only when **all three** hold: 3+ enemy military within ~500 px
of the TC, `under_attack: true` or the TC visibly taking damage, and age is **not**
Dark. A single spearman, scout or militia is never a reason.

Captured in `rules/safety_floor.yaml` with a note: it is prompt-enforced today,
and a rewritten prompt would silently drop it.

---

## 5. `get_early_game_priorities()`

A hardcoded Python string in `packages/data/src/game_knowledge.py:619`, injected
into every executor prompt:

1. FOOD FIRST: send all villagers to sheep.
2. Keep the TC producing: `H` then `Q`.
3. Build a house at 4/5 pop.
4. Villager cost: 50 food, 25 s train time.
5. House cost: 25 wood, +5 population.

Items 4 and 5 are game data and belong in the knowledge base. Items 1-3 are
strategy and duplicate rules already seeded — item 3 is a **fourth** housing
threshold, disagreeing with the other three.

---

## What is NOT captured in code anywhere

The honest gap list. These exist only in prompts, and the seed registry does not
reproduce them:

1. **Proactive farms before the food crisis** (Dark Age) — the highest-value gap.
2. **The Castle Age up-gate** — no Python rule exists.
3. **Auto Scout**, pressed once at game start (partly in `_get_ground_commands`).
4. **4 permanent villagers on gold** from Feudal onward.
5. **Farm reseeding** — and the simulator cannot represent it.
6. **Loom**, and every other technology. No rule mentions research at all.
7. **All Castle and Imperial strategy** beyond gather ratios.

Items 1, 2 and 4 are the strongest candidates for the first strategist-authored
rules, because each one has a concrete trigger and a documented past failure.
