# Chapter 10: Knowledge Database

The game knowledge system provides structured AoE2 data to the agent. A SQLite database stores unit stats, building costs, technologies, and counter relationships. At runtime, it answers queries like "what units can I afford with 300 food and 200 gold in Feudal Age?"

## 10.1 Architecture

Two layers:

1. **Static knowledge base** (`data/knowledge_base/`) -- JSON files fetched from external APIs, stored on disk.
2. **Runtime database** (`data/game_knowledge.py`) -- SQLite wrapper that loads the JSON data and provides query methods for dynamic context injection.

The static files are fetched once by `data/fetch_aoe2_data.py` and committed to the repo. The runtime database is populated from these files (or directly from APIs) at agent startup.

## 10.2 Data Sources

### halfon.aoe2.se (Primary)

URL: `halfon.aoe2.se/data/units_buildings_techs.de.json`

Provides detailed stats for every unit and building in the game:
- Hit points, attack, armor (melee + pierce), range
- Resource costs (food, wood, gold, stone)
- Training/build time
- Line of sight, movement speed
- Entity class codes (used for building vs unit classification)

### SiegeEngineers/aoe2techtree (Secondary)

GitHub: `SiegeEngineers/aoe2techtree/master/data/data.json` + `strings.json`

Provides civilization and technology data:
- All 50 civilizations with bonuses and help text
- Technology trees per civilization
- Unit/building availability by civ

## 10.3 Static Knowledge Base

6 files in `data/knowledge_base/`:

| File | Records | Content |
|------|---------|---------|
| `units.json` | 1,102 | Full unit stats (HP, attack, armor, costs, class) |
| `buildings.json` | 670 | Building stats (HP, costs, category) |
| `civs.json` | 50 | Civilization bonuses and descriptions |
| `techs.json` | 181 | Technology costs and research times |
| `counters.json` | ~20 | Unit counter relationships (rock-paper-scissors) |
| `summary.md` | -- | LLM-readable strategic reference |

### counters.json

Hardcoded counter relationships:

```json
{
  "archer": ["skirmisher", "mangonel", "cavalry"],
  "cavalry": ["pikeman", "camel", "monk"],
  "infantry": ["archer", "hand_cannoneer", "scorpion"],
  "siege": ["cavalry", "bombard_cannon"]
}
```

### summary.md

A concise strategic reference designed to be token-efficient for LLM consumption:
- Unit counter matrix
- Key strategic principles (economy, scouts, composition)
- Build order examples (Fast Castle, Scouts into Knights, Archer Rush)
- Age advancement costs (Feudal: 500F, Castle: 800F+200G, Imperial: 1000F+800G)

## 10.4 SQLite Runtime Database

`data/game_knowledge.py` wraps a SQLite database with four tables.

### Schema

**units** table:
```sql
CREATE TABLE units (
    id INTEGER PRIMARY KEY,
    name TEXT,
    localized_name TEXT,
    hit_points INTEGER,
    attack INTEGER,
    melee_armor INTEGER,
    pierce_armor INTEGER,
    range REAL,
    food_cost INTEGER,
    wood_cost INTEGER,
    gold_cost INTEGER,
    stone_cost INTEGER,
    train_time INTEGER,
    age TEXT,
    type TEXT,
    class INTEGER
)
```

**buildings**, **technologies**, **counters** tables follow similar patterns.

### Building Detection Heuristics

Classifying entities from halfon data as "unit" vs "building" is non-trivial because the API returns both in a single list. `game_knowledge.py:52-79` uses a multi-signal approach:

1. **Known IDs** (`KNOWN_BUILDING_IDS` at line 53) -- 20+ hardcoded building IDs (Town Center=109, Barracks=12, Castle=82, etc.)
2. **Class codes** (`BUILDING_CLASSES` at line 43) -- game entity class 3, 11, 20, 21, 30, 51
3. **Name keywords** -- contains "tower", "castle", "wall", "mill", "barracks", etc.
4. **HP heuristics** -- HP > 500 with 0 attack suggests a building

### Unit Classification

`UNIT_CLASSES` at `game_knowledge.py:20-40` maps game class codes to categories:

| Class | Type | Examples |
|-------|------|----------|
| 0 | archer | Archer, Crossbow |
| 4 | civilian | Villager |
| 6 | infantry | Militia, Man-at-Arms |
| 8 | cavalry | Knight, Scout |
| 13 | siege | Ram, Mangonel |
| 18 | monk | Monk |
| 23 | domestic | Sheep, Turkey |

### Essential Data Fallback

`game_knowledge.py` hardcodes essential units and buildings with `INSERT OR IGNORE`:

- **Units**: Villager (50F, Dark), Militia (60F+20G, Dark), Archer (25W+45G, Feudal), Scout Cavalry (80F, Dark), Spearman (35F+25W, Feudal), Sheep (resource, Dark)
- **Buildings**: House (25W), Town Center (275W+100S), Barracks (175W), Mill (100W), Lumber Camp (100W), Mining Camp (100W), Farm (60W)

> **Key Insight**: Essential game data is hardcoded as a fallback. If the halfon API is down, changes its format, or the network is unavailable, the agent still knows the basics: Villagers cost 50 food, Houses cost 25 wood, Barracks cost 175 wood. The API data enriches but can never break the baseline.

## 10.5 Runtime Queries

### get_affordable_units(resources, age, limit)

Returns units the player can currently afford, filtered by age:

```python
def get_affordable_units(self, resources, age="dark", limit=10):
    # Filters: food_cost <= resources["food"] AND wood_cost <= ... AND age <= current_age
```

### get_affordable_buildings(resources, age, limit)

Same pattern for buildings.

### get_context_for_state(age, resources, entities=None)

Generates a 200-500 token context string combining:
- Affordable units list
- Affordable buildings list
- Counter info for any enemy units detected

### get_early_game_priorities()

Returns static early-game tips:
- "Keep TC producing villagers at all times"
- "Build houses before getting housed (pop = pop cap)"
- "Scout early to find sheep, boar, deer, and enemy"
- "Focus 6 on food (sheep), then get wood for houses and buildings"

### get_counter_info(unit_name)

Returns counter relationships from hardcoded data:
```
"Archer countered by: Skirmisher, Mangonel, Cavalry"
```

## 10.6 Data Fetching Pipeline

`data/fetch_aoe2_data.py` orchestrates the full data pipeline:

1. **fetch_json(url)** -- HTTP GET with User-Agent header and 30s timeout
2. **process_halfon_data(data)** -- parses the halfon JSON into unit and building lists
3. **process_techtree_data(data, strings)** -- extracts civs and techs from aoe2techtree
4. **generate_summary()** -- creates the LLM-readable `summary.md`
5. **main()** -- runs the full pipeline, writes all 6 JSON files

Run manually when data needs refreshing:
```bash
python -m data.fetch_aoe2_data
```

---

## Summary

- SQLite database with units, buildings, technologies, counters
- Two external data sources: halfon.aoe2.se (stats) + aoe2techtree (civs/techs)
- Multi-signal building detection heuristics for classifying raw API data
- Hardcoded essential data as API-failure fallback
- Runtime queries: affordable units/buildings, counters, early game priorities
- Dynamic context injection generates 200-500 token strings tailored to current game state

## Related Topics

- [Chapter 6: Context Injection](../part2-llm-integration/06-context-injection.md) -- how the database feeds into the LLM prompt
- [Chapter 5: Prompt Engineering](../part2-llm-integration/05-prompt-engineering.md) -- the system prompt that frames the context
