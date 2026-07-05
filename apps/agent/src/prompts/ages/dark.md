# Dark Age — 100% Economy

Your only goal in Dark Age is to grow your economy as fast as possible. No military, no fighting.

## Mill + 3 farms by turn 5 (PROACTIVE FOOD RULE)

**Plant farms BEFORE the food crisis, not after.** Farms take ~60 s to start producing — if you wait until food < 50, the crisis is already locked in.

**As soon as you have 100 wood AND 4+ villagers gathering food:**
1. If NO `mill` is in the Detected Entities list, build ONE Mill (`build` with `building_key="w"`) next to a berry bush. If a `mill` is already detected, skip this — one Mill is all you ever need.
2. Plant 3 farms (`build` × 3 with `building_key="a"`) adjacent to the Mill or TC.

By turn 5–8, NOT turn 15+. Sheep deplete around turn 10–12; the farm pipeline must already be running. (Past games dropped below 50 food for 12 of the first 20 turns because farms came too late.)

## Dark Age Checklist (in addition to universal checklist)

**Villager queuing:**
- **Population < 22**: Use `queue_villager` EVERY turn. If 150+ food, call it 2–3 times.
- **Population 22+**: STOP queuing villagers — save food for Feudal (500). Each queued villager delays Feudal research by 25 s.

**Villager allocation:** 6–8 on food, 3–4 on wood initially. Never have 0 food gatherers.

**Scout:** Enable Auto Scout ONCE. Press `,` then `G`. The scout will explore on its own and reveal sheep, deer, gold, stone, and the enemy base.

**Lumber Camp:** 2+ villagers on wood without one? → Build now (`build` with `building_key="r"`, 100 wood). Without it villagers waste half their time walking. **Build by turn 10–15.** Counts as one of the 2 prereq buildings for Feudal.

**Berries:** `berry_bush` detected but no Mill nearby? → Build a Mill (`build` with `building_key="w"`) and send 3–4 villagers via `send_villager target_class=berry_bush`.

**Feudal Age transition:** see the **Age-up Gate** in core.md. Two notes specific to Dark Age: (a) qualifying prereq buildings are Lumber Camp, Mill, Mining Camp, Barracks, or Dock — Mill + Lumber Camp is the easiest path; (b) wait for the TC queue to drain before pressing Z (each queued villager delays the research by 25 s).

**Mill + Farms emergency:** NO sheep AND no berry_bush in entity list AND food < 100? → P10 EMERGENCY. Drop everything else and get farms running this turn (template below). Do NOT keep sending villagers to wood when food is the bottleneck.

**One Mill is enough — decide from the Detected Entities list, never from habit:**
- **`mill` IS in the Detected Entities list** → you already have your food drop-off. Do NOT build another Mill. Build **farms only** (`building_key="a"`), one per idle food villager, placed adjacent to that Mill or the TC.
- **NO `mill` in the Detected Entities list** → build exactly ONE Mill (`building_key="w"`) this turn, then farms around it.

A second Mill wastes 100 wood and fixes nothing — farms, not Mills, are what produce food. If food stays low turn after turn, the answer is *more farms*, not another Mill.

**Sanity check the mill detection (avoid a house-as-mill trap).** Detection sometimes labels a house as `mill`. If a `mill` appears in detections BUT the strategist reasoning says *no mill visible*, OR you have **never built a Mill this game and food keeps starving**, treat the `mill` as a misdetection and **build a real Mill** (`building_key="w"`). A redundant Mill costs 100 wood; farms with no drop-off building never fix food — that's the far worse failure.

## Food Economy

**Gathering order:** sheep → berries (build Mill near berries) → farms (build Mill anywhere, then 1 farm per food villager).

**Notes:**
- **NEVER right-click a boar.** Boars fight back and kill villagers. Ignore them; use sheep → berries → farms.
- Each farm supports only 1 villager. Don't double-up.
- If `target_class: "sheep"` fails once, sheep are not detected — build Mill + farms instead, do not retry sheep.

## Emergency: Under Attack

**In Dark Age: NEVER press B (town bell) or T (garrison) — no exceptions.** Even with 3+ enemy military, the TC's auto-arrows + economy continuity beats garrisoning. The Town Bell Rule in core.md only applies in Feudal Age and later.

**NEVER build Towers or any defensive/military building in Dark Age — not even under attack.** Towers cost wood + stone, take the villager off the economy, and cannot be built from the economic (Q) menu anyway. A starving economy loses far faster than enemy harassment does; the TC's own arrows handle raiders. Under threat, **keep building economy** (houses, Mill, farms) and gathering — do not switch to defenses.

**Strategist's `alarm` flag in Dark Age: ignore it.** Continue your build order.

**Accidentally garrisoned?** Press H → V (All Back to Work) to release everyone.

## Build Menu Restriction

In Dark Age, ONLY use the Q build menu (economic: House, Mill, Mining Camp, Lumber Camp, Farm). Do NOT touch W (military) or V (advanced) menus.

## Dark Age Action Templates

**RECOMMENDED Dark Age multi-task pattern** (every turn):
```json
[
  {"type": "press", "key": "h", "rescan": true, "intent": "Go to TC"},
  {"type": "right_click", "target_class": "sheep", "intent": "Set gather point to sheep"},
  {"type": "queue_villager", "intent": "Queue villager (auto-gathers)"},
  {"type": "queue_villager", "intent": "Queue another villager"},
  {"type": "send_all_idle", "target_class": "tree", "intent": "Sweep ALL idle villagers to wood"}
]
```

**FOOD EMERGENCY — pick the template by what you detect** (when NO sheep/berry_bush AND food < 100):

**Omit `x`/`y` on every `build`** — the executor auto-places each building on **open ground near the town centre** (it picks the emptiest spot; clicking a fixed coordinate lands on the TC or a house and fails). Only pass `x`/`y` if you have a specific *detected* empty tile in mind.

**A) NO `mill` detected yet — build ONE Mill, then farms:**
```json
[
  {"type": "build", "building_key": "w", "intent": "Build Mill near TC (100 wood)"},
  {"type": "build", "building_key": "a", "intent": "Build farm near Mill (60 wood)"},
  {"type": "build", "building_key": "a", "intent": "Build another farm (60 wood)"}
]
```

**B) `mill` ALREADY in Detected Entities — farms ONLY, no second Mill:**
```json
[
  {"type": "build", "building_key": "a", "intent": "Build farm next to existing Mill (60 wood)"},
  {"type": "build", "building_key": "a", "intent": "Build farm next to existing Mill (60 wood)"},
  {"type": "build", "building_key": "a", "intent": "Build farm next to existing Mill (60 wood)"}
]
```
