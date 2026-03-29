# Dark Age — 100% Economy

Your only goal in Dark Age is to grow your economy as fast as possible. No military, no fighting.

## Dark Age Checklist (in addition to universal checklist)

**Villager queuing:**
- **Population < 22**: Use `queue_villager` EVERY turn. If you have 150+ food, call it 2-3 times.
- **Population 22+**: STOP queuing villagers. Save food for Feudal Age (500 food). Do NOT press Q — every queued villager delays Feudal research by 25 seconds because the TC production queue is sequential.

**Villager allocation:** 6-8 on food, 3-4 on wood initially. Never have 0 food gatherers.

**Scout:** Enable Auto Scout! Press `,` (select idle military) then `G` (Auto Scout). The scout will explore the map automatically — you only need to do this ONCE. The scout reveals sheep, boar, deer, gold, stone, and the enemy base.

**Lumber Camp:** Are 2+ villagers gathering wood without a Lumber Camp? → Build one NOW using `build` with `building_key="r"`.
Without a Lumber Camp, villagers waste half their time walking to TC to drop off wood. **You MUST build a Lumber Camp by turn 10-15.** It also counts as one of the 2 Dark Age buildings needed for Feudal Age.

**Berries:** Are berry_bush detected but no mill nearby? → Build a Mill next to the berries using `build` with `building_key="w"`. Then send 3-4 villagers to gather berries with `send_villager` targeting `berry_bush`.

**Feudal Age transition:** Population 22+ AND food >= 500? → Research Feudal Age! Press H (go to TC) → Z (research age up).
- **CRITICAL:** Make sure you have STOPPED queuing villagers and the TC queue is empty BEFORE pressing Z. If villagers are still queued, Feudal research goes to the BACK of the queue — each villager ahead adds 25 seconds of delay. Wait for the queue to finish, THEN press Z.
- **PREREQUISITE**: You MUST have built 2 Dark Age buildings BEFORE pressing Z. Qualifying buildings: Lumber Camp, Mill, Mining Camp, Barracks, or Dock. Houses do NOT count. The easiest path is **Mill + Lumber Camp** — you should already have both.
- **Do NOT press Z if food < 500.** 445 is NOT enough — gather more food first.

**Mill + Farms emergency:** NO sheep or berry_bush in entity list AND food < 100? → EMERGENCY!
- Step 1: Build a Mill on open ground (Q → W → click, 100 wood). Mill unlocks farms and is a food drop-off.
- Step 2: Build 3+ Farms adjacent to the Mill (Q → A → click near Mill, 60 wood each).
- If no space near the Mill, build farms adjacent to TC as fallback (TC is also a food drop-off).
- This is your #1 priority — do NOT keep sending villagers to wood when food is the bottleneck.

## Emergency: Under Attack

**In Dark Age: IGNORE all enemy units completely.** Do NOT press B (town bell), do NOT press T (garrison). Your TC has arrows that automatically shoot nearby enemies. Keep all villagers gathering — economy matters more than 1 lost villager.

**If you accidentally garrisoned villagers (TC shows garrisoned units):**
Press H (go to TC) → V (All Back to Work) → this sends all villagers back to their tasks immediately.

## Food Economy Progression

Follow this order for food gathering:
1. **Sheep** (free, near TC) — gather these first by right-clicking them
2. **Berries** — As soon as you see `berry_bush` in the entity list, build a Mill next to them (Q→W, 100 wood) and send 3-4 villagers to gather berries. Berries are your MAIN food source after sheep run out. Do NOT skip berries.
3. **Farms** — when no sheep or berry_bush appear in the entity list, build farms (Q→A, 60 wood each) near your TC. Farms provide infinite food. **Each farm supports only 1 villager.** Build 1 farm per villager you want gathering food. Do NOT send multiple villagers to the same farm — only the first one will work it.

**NEVER right-click on a boar.** Boars are aggressive — they fight back and WILL kill your villagers. Boar luring requires advanced micro that you cannot do. Ignore boars entirely. Use sheep → berries → farms instead.

**If sheep/berry_bush are NOT in the Detected Entities list, they can't be targeted.** Instead:
- Send villagers to trees for wood (trees are always detected)
- Build a Mill + farms when you have enough wood (160 wood: 100 Mill + 60 farm)
- Do NOT use `target_class: "sheep"` if sheep is not in the entity list — it will fail every time
- **If `target_class: "sheep"` failed once, do NOT try again. Build Mill + farms immediately.**

**Key signal to transition**: If the detected entity list has NO sheep and NO berry_bush, you MUST build a Mill first (if you don't have one), then build farms. Each idle food villager with nothing to gather needs a farm. **You need at least 1 Mill before you can build farms.**

**CRITICAL**: Running out of food is the #1 way to lose. If food is below 100 and dropping, this is a P10 emergency — drop everything else and get food income going immediately.
**Food gathering priority**: sheep → berries (build Mill near berries) → FARMS (build Mill anywhere to unlock farms, then build farms near Mill; use TC as fallback drop-off). If no sheep or berry_bush detected, build Mill + Farms immediately — do NOT keep sending all villagers to wood.

## Wood Economy: Build Lumber Camps

When sending villagers to gather wood, **always build a Lumber Camp near the trees first**.
Villagers must walk to a drop-off building (TC or Lumber Camp) to deposit wood. Trees far from TC = villagers spend most of their time walking, not gathering.

**Rule:** If you're sending 2+ villagers to trees, build a Lumber Camp first:
- Select villager (. rescan) → Q → R → click near trees (100 wood)
- THEN send additional villagers to those same trees

## Build Menu Restriction

**In Dark Age, ONLY use the Q build menu** (economic: House, Mill, Mining Camp, Lumber Camp, Farm). Do NOT use W (military) or V (more buildings) until Feudal Age.

## Dark Age Action Templates

**Set food gather point + queue villagers (BEST pattern when food > 50):**
```json
[
  {"type": "press", "key": "h", "rescan": true, "intent": "Go to TC — sheep/berries visible here"},
  {"type": "right_click", "target_class": "sheep", "intent": "Set TC gather point to sheep"},
  {"type": "queue_villager", "intent": "Queue villager (auto-gathers sheep)"}
]
```

**Build lumber camp near trees (1 turn):**
```json
[
  {"type": "build", "building_key": "r", "x": 1500, "y": 800, "intent": "Build lumber camp on open ground NEAR trees (NOT on them)"}
]
```

**Build Mill near berry bushes (1 turn):**
```json
[
  {"type": "build", "building_key": "w", "x": 2400, "y": 1050, "intent": "Build Mill on open ground BELOW berry bushes (NOT on them)"}
]
```

**Build a farm when food sources are gone (1 turn):**
```json
[
  {"type": "build", "building_key": "a", "x": 1500, "y": 850, "intent": "Build farm near TC"}
]
```

**Enable Auto Scout (do this ONCE early in the game):**
```json
[
  {"type": "press", "key": ",", "rescan": true, "intent": "Select idle scout"},
  {"type": "press", "key": "g", "intent": "Auto Scout — explores map automatically"}
]
```
After pressing G, the scout explores the map on its own forever. No need to manually direct it each turn. When the scout finds sheep, you can right-click them toward your TC to bring them home.

**RECOMMENDED: Set gather point + queue vils + sweep idles (do this every turn!):**
Check the entity list FIRST — only use `target_class` for food sources that are actually detected.
If sheep AND berry_bush are missing from the entity list, send villagers to wood and build farms.
```json
[
  {"type": "press", "key": "h", "rescan": true, "intent": "Go to TC — see sheep/berries"},
  {"type": "right_click", "target_class": "sheep", "intent": "Set gather point to sheep"},
  {"type": "press", "key": "q", "intent": "Queue villager (auto-gathers)"},
  {"type": "press", "key": "q", "intent": "Queue another villager"},
  {"type": "press", "key": ".", "rescan": true, "intent": "Idle vill 1"},
  {"type": "right_click", "target_class": "tree", "intent": "Send to trees"},
  {"type": "press", "key": ".", "rescan": true, "intent": "Idle vill 2"},
  {"type": "right_click", "target_class": "tree", "intent": "Send to trees"}
]
```

**FOOD EMERGENCY: Build Mill + Farms (when NO sheep/berry_bush in entity list):**
```json
[
  {"type": "queue_villager", "intent": "Queue villager before building"},
  {"type": "build", "building_key": "w", "x": 1500, "y": 850, "intent": "Build Mill near TC (100 wood)"},
  {"type": "build", "building_key": "a", "x": 1550, "y": 900, "intent": "Build farm near TC/Mill (60 wood)"},
  {"type": "build", "building_key": "a", "x": 1450, "y": 900, "intent": "Build another farm (60 wood)"}
]
```

**Find missing entities (when target_class keeps failing):**
Only use this when target_class has failed — it takes 5-10 seconds.
```json
[
  {"type": "press", "key": "h", "rescan": true, "intent": "Go to TC area"},
  {"type": "detect", "intent": "Full scan — looking for sheep/berries near TC"},
  {"type": "right_click", "target_class": "sheep", "intent": "Send to sheep (should be visible now)"}
]
```
