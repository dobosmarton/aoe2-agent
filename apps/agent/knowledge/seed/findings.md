# Seed findings — the run-review knowledge, frozen

**Frozen 2026-08-17, before Phase 1 of `ADAPTIVE-AGENT-PLAN.md` deletes
`reactive.py`.**

Every constant in the agent carries a tag pointing back to the run review that
justified it. This file is that knowledge lifted out of the code comments, so it
survives the refactor.

48 distinct tags exist. **36 live in source files** and are lost when those files
change. 12 live only in tests, which the plan does not delete.

Read this before writing any rule. Most of these findings are not opinions — they
are failures that cost a whole game each.

- `F-n` — a finding from a numbered run review (`docs/run-reviews/`).
- `T-n` — a tracked task raised by a review.
- `V-4` — the drift guard: constants duplicated across modules, pinned by tests.

---

## How to read a tag's status

| Status | Meaning |
| --- | --- |
| **at risk** | The rationale exists only in a source comment that Phase 1 rewrites. |
| test-only | The rationale lives in a test docstring. Tests are not deleted. |

---

## 1. Economy and build order

These are the findings that become rules. Each one is a strategy decision the
agent must keep making.

### F-8 — a famine must not honour the normal gather rotation
`reactive.py` · `_FOOD_CRISIS_THRESHOLD = 60` · **at risk**

Below 60 food, routing villagers to wood and gold on the normal rotation starves
villager production. Run 1. The rotation is overridden toward food.

### F-21 — but an all-food override starves the farms that end the famine
`reactive.py` · `_FARM_WOOD_COST = 60` · **at risk**

Run 4. Pure all-food routing pinned wood at 0, which locked out the farm economy
the famine needed. The override keeps a wood slot: 2 food to 1 wood while wood
sits under a farm's cost plus margin.

**These two findings are a pair. A rule that implements one without the other
recreates the failure it was meant to fix.**

### F-23 — bank above the cost, not to the cost
`reactive.py` · `_WOOD_BANK_MARGIN = 20` · **at risk**

Run 5. Six farm attempts failed at wood 48-59. A purchase must not leave the
stock exactly at the boundary, so every bank target carries +20.

### F-34 — the bank target must be the binding goal, not the cheapest one
`reactive.py` · `_LUMBER_CAMP_WOOD_COST = 100` · **at risk**

Run 8. The wood bank targeted only the farm (60), so wood plateaued at 65 while
the lumber camp cost 100. The camp is the second Feudal prerequisite, and it was
rejected 19 times at 37-79 wood. Feudal stayed unreachable.

Priority order for the wood bank: mill, then lumber camp, then mining camp, then
farm.

### F-41 — the fast tier needs its own path to the mill
`reactive.py` · `_MILL_WOOD_COST = 100` · **at risk**

Run 12. The executor was down for 85 of 95 turns and the agent starved, because
only the LLM had ever built a mill. The mill is both a Feudal prerequisite and
the farm unlock — the entire late-Dark-Age food engine hangs off it.

**This finding is the direct ancestor of the whole plan.** It was fixed by adding
another hardcoded rule.

### F-46 — Feudal is not the finish line, and gold is the next wall
`reactive.py` · `_CASTLE_GOLD_COST = 200` · **at risk**

Run 13. 25 minutes in Feudal, 1833 wood banked idle, gold parked at 90. No tier
ever built a mining camp. Castle research needs 800 food and 200 gold; gold is
the scarce half.

### F-45 — gather income hides a purchase
`executor.py` · `_PLACEMENT_INCOME_SLACK = 20` · **at risk**

Run 13. A 30-villager economy gathered +140 wood across a 25-wood house
settlement, so the raw wood delta judged every real purchase as MISSING and the
circuit breaker locked out five building classes. Income is modelled and
deducted, not covered by fixed slack — no constant survives both a 30-villager
economy and a 4-villager opening.

### F-38 — brake on orders, never on delivered population
`executor.py` · `_VILLAGER_ORDER_TARGET_BY_AGE`, `_STARTING_VILLAGERS = 4` · **at risk**

Run 11. Orders lead the HUD population by the TC queue depth (about 25 s per
villager against a 10 s turn). A brake on delivered population pressed `q` 36
times, every one at population 15 or below, and the queue delivered 40 villagers
whose food cost WAS the Feudal bank.

### F-16 — a queue that never stops can never bank 500 food
`reactive.py` · `_FEUDAL_FOOD_COST = 500` · **at risk**

Run 3. With the villager queue firing every turn at 50 food each, 500 never
accumulates. Once the age's villager target is ordered, the queue stops and food
banks.

### F-11 / T-538 — the order target is per age
`executor.py` · `_VILLAGER_ORDER_TARGET_BY_AGE = {"Dark Age": 30, "Feudal Age": 35}` · **at risk**

Run 13 reached Feudal, and a flat Dark Age target of 30 overruled the Feudal 35
while the rejection message still taught "bank for the Feudal Age" in Feudal.
Dark Age 30 is a user directive: enough economy to bank the 500-food cost. Every
order past the target is that age's bank being spent.

### F-45 (house rule) — do not build houses at 15 headroom
`executor.py` · `_HOUSE_HEADROOM_MAX = 4` · **at risk**

The 2026-07-11 run built houses at 15+ headroom, 125 wood spent, while the first
farm starved for 60. A "raise pop cap" goal kept re-triggering because every
house succeeds.

---

## 2. The age-up gate

### F-26 — Feudal needs two qualifying buildings, and houses do not count
`reactive.py` · `_FEUDAL_PREREQ_CLASSES = frozenset({"mill", "lumber_camp"})` · **at risk**

Run 6. 14 age-up presses no-oped against a greyed button, with only the mill
built, while 767 food sat banked. Mirrored by
`evaluation.world_sim.FEUDAL_PREREQ_BUILDINGS`; a drift test pins the pair.

### F-27 / F-32 — how to press the age-up safely
`reactive.py` · `_age_up_actions` · **at risk**

Selecting the TC with `h` clears any open build menu or placement ghost by
switching selection, so `z` cannot land in the econ menu, where `Z` is Outpost.

An `escape` prefix is **wrong**: run 8 showed escape with nothing to cancel OPENS
the game menu and pauses the game.

Gate the press on the prerequisites being visibly met, so it fires once when it
can succeed rather than spamming no-ops. Each stray press was also a chance for
the F-27 UI-context leak.

---

## 3. Perception — what the agent may and may not believe

### F-36 / F-29 — a detection sighting is never proof of ownership
`executor.py` · `_SIGHTING_MIN_FRAMES = 3` · **at risk**

Run 7's flickering phantom poisoned the build gates at 1 frame. Run 9's
**persistent** phantom mill beat the 3-frame threshold too. A mill-less econ menu
then builds OUTPOSTS through the unlocked farm slot — 14 of them.

Gate evidence must be self-generated: only a wood-delta-confirmed purchase or a
visually verified placement counts as owned. Sightings are a context line only.

### F-11 — detection cannot see foundations
`executor.py` · pending-placement settlement · **at risk**

Run 2. YOLO cannot see building foundations, so a fresh rescan reports almost
every REAL placement as failed. That false negative caused a duplicate mill. The
resource bar is authoritative; the vision model is not.

### F-17 — one wood drop confirms at most one pending build
`executor.py` · `_settle_pending_placements` · **at risk**

Run 3. A single 160→8 wood drop settled two pending mills. Confirmed spend is
deducted per shared baseline before the next entry is judged.

### F-4 / T-302 — the idle-count digit under-reads
`memory.py` · `idle_streak` · **at risk**

The badge digit pinned at 1 while 8 villagers idled. Presence (the badge colour)
is the robust gate; the count is distrusted after the badge stays lit 4
consecutive turns. `None` means unreadable and must never be treated as 0.

### F-5 — a silent model substitution served v5 weights for a v9 config
`detection/inference/detector.py` · `_resolve_or_substitute` · **at risk**

Weights are gitignored, so a pull never ships them. Substitute the newest bundled
weights rather than degrading to mock, but say so **loudly**.

---

## 4. Execution safety — the rules that must stay in code

These are game facts, not strategy. `ADAPTIVE-AGENT-PLAN.md` §1.5 keeps them in
Python.

### F-33 / T-525 — coordinates go stale the moment the camera moves
`executor.py` · `CAMERA_KEYS = {h, ., ,}`, `STALE_COORDS_DETAIL` · **at risk**

Run 8. Coordinates computed before a camera move land on arbitrary terrain. The
executor refuses a raw x/y click that follows a camera key in the same batch.
`auto_placement` resolves at click time instead.

### F-32 — never send escape, F10 or F3
`models.py` · `_GAME_PAUSING_KEYS` · **at risk**

Escape with nothing to cancel opens the menu. F10 is the menu. F3 pauses. UI
state is cleared by selecting the TC instead.

### F-20 — a lone right-click on a boar is an attack, and the boar wins
`entity_utils.py` · `GATHER_CLASSES_BY_KIND` · **at risk**

Run 4 lost three villagers this way. Real boar hunting needs 3+ villagers and TC
luring. Deer are excluded too: at real F1 ≈ 0.67 a boar misread as deer is fatal,
so species labels are not trusted with villager lives.

### F-12 — a farm is never a gather target
`entity_utils.py` · `GATHER_CLASSES_BY_KIND` · **at risk**

Each farm supports exactly one villager, and clicking an occupied one does
nothing. Run 2: bare ground misdetected as a farm stranded the villager. Idle
villagers get a **fresh farm built** instead — the builder auto-farms the field it
finishes.

### T-530 / F-37 — stop paying for a build that keeps vanishing
`executor.py` · `_MISSING_STREAK_LIMIT = 3`, `_MISSING_SUPPRESS_SNAPSHOTS = 5` · **at risk**

Run 9. 32 consecutive missing farm settlements were retried blindly, each one
buying an unintended outpost. A streak means something is systematically wrong.

### The farm key builds an Outpost without a mill
`executor.py` · `_BUILD_PREREQ_CLASS = {"a": "mill"}` · **at risk**

Safety critical, and the reason several findings above compound. This gate must
never move into a learned rule.

---

## 5. Infrastructure findings

### F-40 — bounded ints in the schema blew the decoding grammar
`models.py` · `_MAX_X`, `_MAX_Y`, `_MAX_WAIT_MS` · **at risk**

Run 12. 22 numeric bounds across the Action union compiled to a constrained
decoding grammar over Anthropic's size limit, and **every** executor turn 400'd
with "compiled grammar is too large". Ranges are enforced by `field_validator`,
never `Field(ge=, le=)`, so the field stays unbounded in the schema.

### F-44 / T-536 — one bad byte zeroed the memory feature for 13 runs
`memory_chain.py` · `_read_memory_file` · **at risk**

A single cp1252 em-dash (0x97) in one file made every load raise, silently
disabling cross-game memory for 13 straight VM runs. Reads now use
`errors="replace"`.

### T-533 — a dead executor must not read as a valid experiment
`memory.py` · `llm_calls`, `llm_errors`, `llm_error_rate` · **at risk**

Run 12 logged 90 executor errors and still recorded `accepted=true`. An
`llm_error_rate` near 1.0 means the game was played by the reactive tier alone.

**This is the metric that exposed the problem the whole plan addresses.** Run
`2026_08_15_1` scored `llm_error_rate=1.0` and still reached the Feudal Age.

### F-6 — build the OCR engine off the loop
`game_loop.py` · OCR warm-up task · **at risk**

Engine construction plus first inference cost 10-15 s on the VM and were the bulk
of the post-startup freeze.

### F-1 — an unfocusable window burned 12 of 30 iterations
`game_loop.py` · `_MAX_FOCUS_FAILURES = 15` · **at risk**

The run had no end-reason label. It now aborts as `lost_focus`.

### F-7 / F-19 — a narrated plan with zero actions wastes the turn
`executor_provider.py` · `_EMPTY_ACTIONS_NUDGE` · **at risk**

6-7 turns per game were lost this way. One bounded retry converts most of them.

### V-4 — the duplicated constants are pinned by drift tests
`reactive.py` and `executor.py` · **at risk**

`reactive.py` duplicates the executor's cost table so it stays dependency-free.
Drift tests are the cross-check. **Phase 1.2 removes this duplication** by seeding
rule `cost` from `executor._BUILD_WOOD_COST`, so V-4 should retire with it.

---

## 6. Test-only tags

These 12 keep their rationale in test docstrings, which the plan does not touch:

`T-202` OCR age sampling on the template backend · `T-203` the loop OCRs the frame
once per iteration · `T-302` idle-digit geometry fix, still open · `T-506`
single-shot zero-action retry · `T-510` / `T-511` Feudal banking and age-up ·
`T-512` known-buildings line · `T-515` reactive-to-executor drift pins · `T-518` /
`T-527` goal-driven wood bank target · `T-525` stale-coordinate guard · `T-601`

---

## What this corpus is for

`ADAPTIVE-AGENT-PLAN.md` §6.3 runs two arms:

- **COLD** — the safety floor only.
- **SEEDED** — the safety floor plus these findings, as rules.

The gap between them measures what this knowledge was worth. That number does not
exist yet, and it is the honest way to answer whether the months of run reviews
beat an agent that starts from nothing.
