# Feudal Age — 85% Economy, 15% Military

You've reached Feudal Age. Economy is still the priority, but you should start preparing for military and planning the Castle Age transition.

## Feudal Age Checklist (in addition to universal checklist)

**Villager queuing:**
- **Population < 35**: Resume queuing villagers every turn. If you have 150+ food, queue 2-3.
- **Population 35+**: Slow down queuing. Save for Castle Age research (800 food + 200 gold). Only queue if food > 900.

**Villager allocation:** 10-12 on food, 6-8 on wood, 3-4 on gold. Start mining gold NOW — you need 200 gold for Castle Age.

**Farm management:** Farms are now your primary food source. Sheep and berries are likely gone.
- Build 1 farm per food villager. Each farm supports only 1 villager.
- Reseed expired farms immediately — idle food villagers kill your economy.
- If food is dropping and you have idle villagers, build more farms (60 wood each).
- Build farms near TC or Mill (they need a drop-off building nearby).

**Gold mining:** Build a Mining Camp near gold_mine if you haven't already (`build` with `building_key="e"`). Send 3-4 villagers to gold. You need 200 gold for Castle Age.

**Castle Age transition:** Population 30+ AND food >= 800 AND gold >= 200? → Research Castle Age!
- Press H (go to TC) → Z (research age up).
- **PREREQUISITE**: You MUST have built 2 Feudal Age buildings. Qualifying: Blacksmith, Market, Stable, Archery Range. Houses and Dark Age buildings do NOT count.
- **Recommended path**: Build a **Blacksmith** (for upgrades) + **Market** (for trade/selling) or **Stable** (for scouts/knights in Castle Age).
- Make sure TC queue is empty before pressing Z.

## Build Menus — Now Expanded

You now have access to military and advanced buildings:
- **Villager + Q**: Economic build menu (same as Dark Age: House, Mill, Mining Camp, Lumber Camp, Farm)
- **Villager + W**: Military build menu — **NEW!**
  - Q = Barracks (175 wood) — trains infantry
  - W = Archery Range (175 wood) — trains archers, skirmishers
  - E = Stable (175 wood) — trains cavalry
  - R = Siege Workshop (200 wood) — NOT available until Castle Age
  - F = Monastery (175 wood) — NOT available until Castle Age
- **Villager + V**: More buildings — **NEW!**
  - D = Market (175 wood) — trade, buy/sell resources
  - F = Tower (125 wood, 25 stone) — defensive structure
  - S = Palisade Wall — cheap walls

## Military — When and What to Build

**When to build military:**
- Build a **Barracks** first (175 wood). This is cheap and a prerequisite for other military buildings.
- Only start training units when: economy is stable (pop 25+, food > 200, wood > 200) OR you are under attack.
- If you are NOT under attack, focus on economy. One Barracks is enough for now.

**If under attack (3+ enemy military units visible):**
- Train military from existing buildings. Do NOT panic-build new buildings while being attacked.
- **Spearmen** (Barracks → W, 35 food 25 wood): Counter cavalry (scouts, knights). Cheap and fast.
- **Skirmishers** (Archery Range → W, 25 food 35 wood): Counter archers. Good in groups.
- **Scouts** (Stable → Q, 80 food): Fast, good for raiding. You already have one from the start.
- Use garrison if overwhelmed: press T to garrison villagers into TC. Press V to ungarrison when safe.

**Counter system:**
- Enemy has cavalry (scouts, knights) → train Spearmen
- Enemy has archers → train Skirmishers
- Enemy has infantry (militia, men-at-arms) → train Archers
- Not sure what enemy has → Spearmen are the safest default (cheap, fast to train)

**Scouting the enemy:** Your scout should be auto-exploring. If you spot the enemy base, note what buildings they have:
- Enemy Archery Range → they'll make archers → prepare Skirmishers
- Enemy Stable → they'll make cavalry → prepare Spearmen
- Enemy Barracks only → they might rush with infantry → Archers or just wall up

## Economy Priorities

1. **Food**: Farms are the backbone. Always have enough farms for your food villagers. Build new farms as old ones expire.
2. **Wood**: Keep 6-8 villagers on wood. Wood pays for farms (60 each), buildings, and military.
3. **Gold**: Send 3-4 villagers to gold. Build Mining Camp near gold_mine. You need 200 gold for Castle Age.
4. **Research Loom** if not done yet (50 gold, at TC: H → A). Loom makes villagers tougher — useful now that enemies may attack.
5. **Build Blacksmith** (Villager → Q → S, 150 wood) for military upgrades. Also counts toward Castle Age prerequisites.
6. **Build Market** (Villager → V → D, 175 wood) for emergency resource trading. Also counts toward Castle Age prerequisites.
7. **Build additional Lumber Camps** as tree lines deplete. If villagers are walking far to trees, build a new Lumber Camp closer.

## Feudal Age Action Templates

**NOTE:** The `build` composite only works for economic buildings (Q menu). Military buildings (W menu) and advanced buildings (V menu) require manual press sequences below.

**Build Barracks (1 turn):**
```json
[
  {"type": "press", "key": ".", "rescan": true, "intent": "Select idle villager"},
  {"type": "press", "key": "w", "intent": "Military build menu"},
  {"type": "press", "key": "q", "intent": "Select Barracks (175 wood)"},
  {"type": "click", "x": 1400, "y": 900, "intent": "Place Barracks on open ground"}
]
```

**Train Spearmen (when under attack):**
```json
[
  {"type": "press", "key": "b", "modifiers": ["ctrl"], "rescan": true, "intent": "Go to Barracks"},
  {"type": "press", "key": "w", "intent": "Train Spearman (35 food, 25 wood)"},
  {"type": "press", "key": "w", "intent": "Train another Spearman"}
]
```

**Build Blacksmith (1 turn) — uses economic build menu (Q), not military (W):**
```json
[
  {"type": "build", "building_key": "s", "x": 1600, "y": 850, "intent": "Build Blacksmith on open ground (150 wood)"}
]
```

**Build Market (1 turn):**
```json
[
  {"type": "press", "key": ".", "rescan": true, "intent": "Select idle villager"},
  {"type": "press", "key": "v", "intent": "More buildings menu"},
  {"type": "press", "key": "d", "intent": "Select Market (175 wood)"},
  {"type": "click", "x": 1300, "y": 950, "intent": "Place Market on open ground"}
]
```

**Send villagers to gold (1 turn):**
```json
[
  {"type": "send_villager", "target_class": "gold_mine", "intent": "Send idle villager to gold mine"}
]
```

**Build Mining Camp near gold (1 turn):**
```json
[
  {"type": "build", "building_key": "e", "x": 1800, "y": 700, "intent": "Build Mining Camp on open ground near gold_mine"}
]
```

**Build farm (keep food income going):**
```json
[
  {"type": "build", "building_key": "a", "x": 1500, "y": 850, "intent": "Build farm near TC/Mill for food income"}
]
```

**Research Castle Age (when ready: 800F + 200G + 2 Feudal buildings):**
```json
[
  {"type": "press", "key": "h", "rescan": true, "intent": "Go to TC"},
  {"type": "press", "key": "z", "intent": "Research Castle Age (800 food + 200 gold)"}
]
```

**RECOMMENDED: Feudal multi-task pattern (economy + military in one turn):**
```json
[
  {"type": "queue_villager", "intent": "Queue villager"},
  {"type": "build", "building_key": "a", "x": 1500, "y": 850, "intent": "Build farm for food"},
  {"type": "send_villager", "target_class": "gold_mine", "intent": "Send idle vill to gold"},
  {"type": "press", "key": "b", "modifiers": ["ctrl"], "rescan": true, "intent": "Go to Barracks"},
  {"type": "press", "key": "w", "intent": "Train Spearman"}
]
```

**Emergency defense (being attacked in Feudal):**
```json
[
  {"type": "press", "key": "b", "modifiers": ["ctrl"], "rescan": true, "intent": "Go to Barracks"},
  {"type": "press", "key": "w", "intent": "Train Spearman"},
  {"type": "press", "key": "w", "intent": "Train Spearman"},
  {"type": "queue_villager", "intent": "Keep economy going even during attack"}
]
```
