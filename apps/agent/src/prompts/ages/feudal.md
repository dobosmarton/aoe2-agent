# Feudal Age — 85% Economy, 15% Military

You've reached Feudal Age. Economy is still the priority, but you should start preparing for military and planning the Castle Age transition.

## Age-up Gate to Castle (check FIRST, before the turn checklist)

Read the **strategist's Resource Status** block in context — that is the authoritative reading. Do NOT use your own age estimate.

**If ALL of these are true, your first two actions THIS TURN must be `press key=h` then `press key=z` — nothing else before them:**
- Strategist's Age reads `Feudal Age`
- Food ≥ 800
- Gold ≥ 200
- I have built at least **2 Feudal Age buildings** detected on screen — qualifying classes: barracks, archery_range, stable, blacksmith, market. (Houses, Mills, Lumber Camps, Mining Camps do NOT count.)

Castle Age research takes ~2.5 minutes and runs in the background. Resume farming, queueing, and military training on the NEXT turn after the research is in flight.

**Do not queue villagers in the same turn you press Z.** The research goes to the back of the TC queue; each villager ahead adds 25 s of delay. Let the queue drain first.

Missing Castle Age leaves you with no knights, no monks, no unique units, and no Castle. Against any non-trivial AI you will lose — Castle Age is the threshold for actually competing.

## Feudal Age Checklist (in addition to universal checklist)

**Villager queuing:**
- **Population < 35**: Resume queuing every turn. If 150+ food, queue 2–3.
- **Population 35+**: Slow down — save for Castle Age (800 food + 200 gold). Only queue if food > 900.

**Villager allocation:** 10–12 on food, 6–8 on wood, 3–4 on gold. Start mining gold IMMEDIATELY — Castle Age needs 200.

**Farm management:** Farms are now your primary food source.
- Build 1 farm per food villager. Each farm supports only 1 villager.
- Reseed expired farms immediately — idle food villagers kill your economy.
- Farms need a drop-off building nearby (TC or Mill).

**Gold mining (CRITICAL — gates Castle Age):**
1. Within the FIRST 2 turns of Feudal: build a Mining Camp next to the nearest gold_mine (`build` with `building_key="e"`).
2. Keep **at least 4 villagers permanently on gold** from Feudal onward. Use `send_villager target_class=gold_mine`.
3. Set TC gather point to gold for new villagers: H → right_click gold_mine → Q, Q, Q.
4. **If gold < 100, halt new buildings and divert at least 2 villagers from food/wood to gold immediately.** Past games stalled at 50–110 gold because villagers kept being pulled back to food/wood.

You need 200 gold for Castle Age, plus 50–100 per tech, plus 25–50 per military unit. Gold is the gating resource — defend the gold-on-gold ratio.

**Loom:** if not yet researched, do it now (TC: H → A, 50 gold). Toughens villagers.

**Blacksmith / Market:** both count as Feudal-Age prereq buildings. Blacksmith (`build` with `building_key="s"`, 150 wood) unlocks military upgrades; Market (V→D, 175 wood) unlocks emergency trading.

## Build Menus

You now have access to military and advanced buildings. The full key reference is in the appended hotkey list — the high-leverage ones:
- **Villager + Q**: economic (same as Dark Age)
- **Villager + W**: military — Q=Barracks, W=Archery Range, E=Stable
- **Villager + V**: advanced — D=Market, F=Tower

Note: the `build` composite ONLY works for the economic (Q) menu. Military and advanced buildings need a manual press sequence (template below).

## Military — When and What

**One Barracks first** (175 wood). Cheap and a prerequisite for the rest.

**Counter system:**
- Enemy cavalry (scouts, knights) → Spearmen
- Enemy archers → Skirmishers
- Enemy infantry → Archers
- Unsure → Spearmen are the safest default

**If under attack (3+ enemy military visible):** train counter-units from existing buildings — do NOT panic-build new ones. Garrisoning: see core.md's Town Bell Rule. For 1–2 enemies, let TC arrows handle it.

## Feudal Age Action Templates

**RECOMMENDED Feudal multi-task pattern** (economy + military in one turn):
```json
[
  {"type": "queue_villager", "intent": "Queue villager"},
  {"type": "build", "building_key": "a", "intent": "Build farm for food"},
  {"type": "send_villager", "target_class": "gold_mine", "intent": "Send idle vill to gold"},
  {"type": "press", "key": "b", "modifiers": ["ctrl"], "rescan": true, "intent": "Go to Barracks"},
  {"type": "press", "key": "w", "intent": "Train Spearman"}
]
```

**Build Barracks** (W-menu — can't use composite, manual sequence required):
```json
[
  {"type": "press", "key": ".", "rescan": true, "intent": "Select idle villager"},
  {"type": "press", "key": "w", "intent": "Military build menu"},
  {"type": "press", "key": "q", "intent": "Select Barracks (175 wood)"},
  {"type": "click", "x": 1400, "y": 900, "intent": "Place Barracks on open ground"}
]
```

**Research Castle Age** (when the Age-up Gate fires):
```json
[
  {"type": "press", "key": "h", "rescan": true, "intent": "Go to TC"},
  {"type": "press", "key": "z", "intent": "Research Castle Age (800 food + 200 gold)"}
]
```
