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

**Villager queuing:** the reactive tier queues villagers up to the Feudal **order target of 35** automatically, then banks food + gold for Castle. You rarely need to queue by hand.
- **Fewer than 35 ordered**: queuing is fine. If 150+ food, 2–3 is fine.
- **35 ordered**: STOP — save for Castle Age (800 food + 200 gold). Only queue if food > 900.

**Villager allocation:** 10–12 on food, 6–8 on wood, 3–4 on gold. Start mining gold IMMEDIATELY — Castle Age needs 200.

**Farm management:** Farms are now your primary food source.
- Build 1 farm per food villager. Each farm supports only 1 villager.
- Reseed expired farms immediately — idle food villagers kill your economy.
- Farms need a drop-off building nearby (TC or Mill).

**Gold mining (CRITICAL — gates Castle Age):**
1. Within the FIRST 2 turns of Feudal: build a Mining Camp next to the nearest gold_mine (`build` with `building_key="e"`). The reactive tier now also auto-builds this once you reach Feudal — you're a backstop if it hasn't landed.
2. Keep **at least 4 villagers permanently on gold** from Feudal onward. Use `send_villager target_class=gold_mine`.
3. Set TC gather point to gold for new villagers: H → right_click gold_mine → Q, Q, Q.
4. **If gold < 100, halt new buildings and divert at least 2 villagers from food/wood to gold immediately.** Past games stalled at 50–110 gold because villagers kept being pulled back to food/wood.

You need 200 gold for Castle Age, plus 50–100 per tech, plus 25–50 per military unit. Gold is the gating resource — defend the gold-on-gold ratio.

**Loom:** if not yet researched, do it now (`research` with `tech="loom"`, 50 gold).
Toughens villagers. The same composite researches horse_collar, double_bit_axe and
gold_mining — each one permanently speeds up a gathering line.

**Blacksmith / Market:** both count as Feudal-Age prereq buildings. Blacksmith (`build` `menu="q"` `building_key="s"`, 150 wood) unlocks military upgrades; Market (`build` `menu="v"` `building_key="d"`, 175 wood) unlocks emergency trading.

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

**Build Barracks or Market** (both count toward the Castle Age gate):
```json
[
  {"type": "build", "menu": "w", "building_key": "q", "intent": "Barracks (175 wood) — Castle prereq 1 of 2"},
  {"type": "build", "menu": "v", "building_key": "d", "intent": "Market (175 wood) — Castle prereq 2 of 2"}
]
```

**Research Castle Age** (when the Age-up Gate fires):
```json
[{"type": "research", "tech": "castle_age", "intent": "800 food + 200 gold banked, 2 Feudal buildings up"}]
```
The HUD spend confirms it next turn. If the cost never leaves your resources the
button was greyed out — read the failure detail, fix the requirement, and do NOT
press it again in the meantime.
