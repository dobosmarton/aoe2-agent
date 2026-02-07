#!/usr/bin/env python3
"""
Fetch AoE2 game data from open APIs and build a structured knowledge base.

Data sources:
  - halfon.aoe2.se: Unit/building stats (already used by game_knowledge.py)
  - SiegeEngineers/aoe2techtree: Tech tree data, civ info, unit relationships
  - HSZemi/aoe2dat: Backup source for raw game data

Output: agent/data/knowledge_base/
  - units.json       All units with stats, costs, age, training building
  - buildings.json   All buildings with stats, costs, age
  - civs.json        All civilizations with bonuses, unique units, tech trees
  - techs.json       All technologies with effects, costs
  - counters.json    Unit counter relationships
  - summary.md       LLM-readable strategic reference

Usage:
    python -m data.fetch_aoe2_data
"""

import json
import urllib.request
import sys
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "knowledge_base"

# Data source URLs
HALFON_URL = "https://halfon.aoe2.se/data/units_buildings_techs.de.json"
TECHTREE_DATA_URL = "https://raw.githubusercontent.com/SiegeEngineers/aoe2techtree/master/data/data.json"
TECHTREE_STRINGS_URL = "https://raw.githubusercontent.com/SiegeEngineers/aoe2techtree/master/data/locales/en/strings.json"

# Unit class IDs from game data
UNIT_TYPE_MAP = {
    0: "archer", 1: "artifact", 2: "trade_boat", 3: "building",
    4: "civilian", 6: "infantry", 8: "cavalry", 11: "building",
    12: "cavalry", 13: "siege", 14: "predator", 18: "monk",
    19: "trade_unit", 22: "infantry", 23: "livestock", 30: "tower",
    36: "ship", 44: "ship", 51: "wall", 52: "resource",
    53: "resource", 54: "resource", 56: "resource", 57: "boar",
    58: "sheep",
}

# Known counter relationships (from game mechanics)
COUNTER_MAP = {
    "archer": {
        "strong_vs": ["infantry", "spearman"],
        "weak_vs": ["skirmisher", "mangonel", "cavalry", "eagle_warrior"],
        "notes": "Kite infantry; vulnerable to high pierce armor units"
    },
    "cavalry": {
        "strong_vs": ["archer", "siege", "monk", "skirmisher"],
        "weak_vs": ["spearman", "camel", "monk_large_groups"],
        "notes": "Mobile; raid economy; avoid pike/camel masses"
    },
    "infantry": {
        "strong_vs": ["building", "eagle_warrior", "trash_unit"],
        "weak_vs": ["archer", "hand_cannoneer", "cavalry"],
        "notes": "Cost-effective; good for raiding and building destruction"
    },
    "spearman": {
        "strong_vs": ["cavalry", "camel", "elephant"],
        "weak_vs": ["archer", "infantry", "mangonel"],
        "notes": "Cheap anti-cavalry; always pair with ranged units"
    },
    "skirmisher": {
        "strong_vs": ["archer", "spearman"],
        "weak_vs": ["cavalry", "infantry", "mangonel"],
        "notes": "Trash counter to archers; useless vs melee"
    },
    "knight": {
        "strong_vs": ["archer", "siege", "villager", "monk"],
        "weak_vs": ["pikeman", "camel", "halberdier"],
        "notes": "Power unit; high HP and attack; expensive"
    },
    "siege": {
        "strong_vs": ["infantry", "archer", "building"],
        "weak_vs": ["cavalry", "bombard_cannon"],
        "notes": "Area damage; needs protection; slow"
    },
    "monk": {
        "strong_vs": ["knight", "elephant", "expensive_units"],
        "weak_vs": ["light_cavalry", "eagle_warrior", "archer_mass"],
        "notes": "Convert high-value targets; fragile; need micro"
    },
    "camel": {
        "strong_vs": ["cavalry", "knight"],
        "weak_vs": ["archer", "infantry", "monk"],
        "notes": "Anti-cavalry specialist; no building bonus"
    },
    "eagle_warrior": {
        "strong_vs": ["monk", "siege", "archer"],
        "weak_vs": ["infantry", "cavalry"],
        "notes": "Meso-American cavalry replacement; fast; good raid unit"
    },
}


def fetch_json(url: str) -> dict:
    """Fetch JSON from URL with timeout."""
    print(f"  Fetching {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "aoe2-llm-arena/1.0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def process_halfon_data(data: dict) -> tuple[list[dict], list[dict]]:
    """Process halfon.aoe2.se data into units and buildings lists."""
    units = []
    buildings = []

    for id_str, entity in data.get("units_buildings", {}).items():
        try:
            entity_id = int(id_str)
        except ValueError:
            continue

        cost = entity.get("cost", {})
        entity_class = entity.get("class", -1)
        entity_type = entity.get("type", 0)
        name = entity.get("localised_name") or entity.get("name", "")
        internal_name = entity.get("name", "")

        # Skip junk entries
        if not name or name.startswith("OLD-") or entity.get("hit_points", 0) <= 0:
            continue

        record = {
            "id": entity_id,
            "name": name,
            "internal_name": internal_name,
            "hit_points": entity.get("hit_points", 0),
            "attack": entity.get("attack", 0),
            "melee_armor": entity.get("melee_armor", 0),
            "pierce_armor": entity.get("pierce_armor", 0),
            "line_of_sight": entity.get("line_of_sight", 0),
            "garrison_capacity": entity.get("garrison_capacity", 0),
            "cost": {
                "food": cost.get("food", 0),
                "wood": cost.get("wood", 0),
                "gold": cost.get("gold", 0),
                "stone": cost.get("stone", 0),
            },
        }

        # Classify as building or unit
        is_building = (
            entity_type == 80
            or entity_class in (3, 11, 20, 21, 30, 51)
            or _is_building_by_name(name)
        )

        if is_building:
            record["category"] = _categorize_building(name)
            buildings.append(record)
        else:
            record["class"] = UNIT_TYPE_MAP.get(entity_class, "unknown")
            record["total_cost"] = sum(cost.values())
            units.append(record)

    return units, buildings


def _is_building_by_name(name: str) -> bool:
    """Check if entity is a building by name."""
    name_lower = name.lower()
    keywords = [
        "tower", "castle", "wall", "gate", "house", "mill", "camp",
        "dock", "range", "stable", "barracks", "center", "monastery",
        "market", "university", "blacksmith", "wonder", "outpost",
        "workshop", "trap", "farm", "krepost", "donjon", "harbor",
        "folwark", "mule cart", "caravanserai", "feitoria",
    ]
    return any(kw in name_lower for kw in keywords)


def _categorize_building(name: str) -> str:
    """Categorize a building by its name."""
    name_lower = name.lower()
    military = ["barracks", "range", "stable", "workshop", "castle", "krepost", "donjon"]
    defensive = ["tower", "wall", "gate", "outpost"]
    economic = ["house", "mill", "camp", "dock", "market", "farm", "center",
                "blacksmith", "monastery", "university", "feitoria", "folwark"]

    if any(k in name_lower for k in military):
        return "military"
    elif any(k in name_lower for k in defensive):
        return "defensive"
    elif any(k in name_lower for k in economic):
        return "economic"
    return "other"


def process_techtree_data(data: dict, strings: dict) -> tuple[list[dict], list[dict]]:
    """Process aoe2techtree data for civilizations and technologies."""
    civs = []
    techs = []

    # Extract civilizations
    civ_names = data.get("civ_names", {})
    civ_helptexts = data.get("civ_helptexts", {})

    for civ_key, name_id in civ_names.items():
        civ_name = strings.get(str(name_id), civ_key)
        help_id = civ_helptexts.get(civ_key, "")
        help_text = strings.get(str(help_id), "")

        # Parse bonuses from help text
        bonuses = _parse_civ_bonuses(help_text)

        civs.append({
            "name": civ_name,
            "key": civ_key,
            "bonuses": bonuses,
            "help_text": help_text[:500] if help_text else "",
        })

    # Extract technologies
    tech_data = data.get("data", {}).get("techs", {})
    for tech_id, tech in tech_data.items():
        name_id = tech.get("LanguageNameId", 0)
        help_id = tech.get("LanguageHelpId", 0)
        tech_name = strings.get(str(name_id), tech.get("internal_name", f"Tech {tech_id}"))
        tech_help = strings.get(str(help_id), "")

        cost = tech.get("Cost", {})

        techs.append({
            "id": int(tech_id),
            "name": tech_name,
            "internal_name": tech.get("internal_name", ""),
            "cost": {
                "food": cost.get("Food", 0),
                "wood": cost.get("Wood", 0),
                "gold": cost.get("Gold", 0),
                "stone": cost.get("Stone", 0),
            },
            "research_time": tech.get("ResearchTime", 0),
            "description": tech_help[:300] if tech_help else "",
        })

    return civs, techs


def _parse_civ_bonuses(help_text: str) -> list[str]:
    """Extract civilization bonuses from help text.

    The help_text format from aoe2techtree is HTML-like:
      "Infantry and Monk civilization<br>\\n<br>\\n• Bonus 1<br>\\n• Bonus 2<br>\\n..."
    """
    if not help_text:
        return []

    bonuses = []
    # Clean up HTML tags
    text = help_text.replace("<br>", "\n").replace("<b>", "").replace("</b>", "")
    lines = text.split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Bullet-point lines are bonuses (• or \u2022)
        if line.startswith("\u2022") or line.startswith("•"):
            bonus = line.lstrip("\u2022• ").strip()
            if bonus:
                bonuses.append(bonus)
        # "Team Bonus:" line
        elif "team bonus" in line.lower():
            continue  # The next bullet will be the team bonus
        # Stop at "Unique Unit:" or "Unique Techs:" sections
        elif ("unique unit" in line.lower() or "unique tech" in line.lower()) and ":" in line:
            break

    return bonuses[:10]


def generate_summary(units: list, buildings: list, civs: list) -> str:
    """Generate an LLM-readable summary markdown file."""
    lines = [
        "# AoE2 Strategic Reference",
        "",
        "Quick reference for the LLM agent's strategic decisions.",
        "",
        "## Unit Counter Matrix",
        "",
        "| Unit Type | Strong Against | Weak Against |",
        "|-----------|---------------|--------------|",
    ]

    for unit_type, info in COUNTER_MAP.items():
        strong = ", ".join(info["strong_vs"])
        weak = ", ".join(info["weak_vs"])
        lines.append(f"| {unit_type} | {strong} | {weak} |")

    lines.extend([
        "",
        "## Key Strategic Principles",
        "",
        "1. **Rock-Paper-Scissors**: Archers > Infantry > Cavalry > Archers",
        "2. **Economy wins games**: More villagers = more resources = bigger army",
        "3. **Never stop producing villagers** until 120+ (or you're all-in)",
        "4. **Scout early**: Know what your opponent is making, then counter it",
        "5. **Composition matters**: Mix unit types to cover weaknesses",
        "6. **Upgrades > Numbers**: Upgraded units beat larger un-upgraded armies",
        "",
        "## Common Build Orders",
        "",
        "### Fast Castle (Knights)",
        "- 6 sheep, 4 wood, 1 boar, 3 berries, 1 boar, 2 farms",
        "- Click Feudal at ~22 pop, build 2 stables, click Castle",
        "- Make Knights immediately",
        "",
        "### Scouts into Knights",
        "- 6 sheep, 3 wood, 1 boar, 1 berries, 1 boar, 3 farms",
        "- Click Feudal at ~21 pop, build stable + blacksmith",
        "- Make Scouts, transition to Knights in Castle Age",
        "",
        "### Archers (Feudal Rush)",
        "- 6 sheep, 4 wood, 1 boar, 3 gold, 1 boar, 2 farms",
        "- Click Feudal at ~22 pop, build 2 archery ranges",
        "- Mass archers, add skirmishers if they counter with archers",
        "",
        "## Age Advancement Costs",
        "",
        "| Age | Cost | Time |",
        "|-----|------|------|",
        "| Feudal | 500F | 130s |",
        "| Castle | 800F + 200G | 160s |",
        "| Imperial | 1000F + 800G | 190s |",
        "",
        f"## Statistics",
        "",
        f"- Total units in database: {len(units)}",
        f"- Total buildings in database: {len(buildings)}",
        f"- Total civilizations: {len(civs)}",
        "",
    ])

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("AoE2 Knowledge Base Builder")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Fetch data from sources
    print("\n1. Fetching data from halfon.aoe2.se...")
    try:
        halfon_data = fetch_json(HALFON_URL)
        units, buildings = process_halfon_data(halfon_data)
        print(f"   Got {len(units)} units, {len(buildings)} buildings")
    except Exception as e:
        print(f"   ERROR: {e}")
        units, buildings = [], []

    print("\n2. Fetching data from aoe2techtree...")
    try:
        techtree_data = fetch_json(TECHTREE_DATA_URL)
        techtree_strings = fetch_json(TECHTREE_STRINGS_URL)
        civs, techs = process_techtree_data(techtree_data, techtree_strings)
        print(f"   Got {len(civs)} civilizations, {len(techs)} technologies")
    except Exception as e:
        print(f"   ERROR: {e}")
        civs, techs = [], []

    # Write knowledge base files
    print("\n3. Writing knowledge base files...")

    # Units
    units_path = OUTPUT_DIR / "units.json"
    units_path.write_text(json.dumps(units, indent=2, ensure_ascii=False) + "\n")
    print(f"   {units_path}: {len(units)} units")

    # Buildings
    buildings_path = OUTPUT_DIR / "buildings.json"
    buildings_path.write_text(json.dumps(buildings, indent=2, ensure_ascii=False) + "\n")
    print(f"   {buildings_path}: {len(buildings)} buildings")

    # Civilizations
    civs_path = OUTPUT_DIR / "civs.json"
    civs_path.write_text(json.dumps(civs, indent=2, ensure_ascii=False) + "\n")
    print(f"   {civs_path}: {len(civs)} civilizations")

    # Technologies
    techs_path = OUTPUT_DIR / "techs.json"
    techs_path.write_text(json.dumps(techs, indent=2, ensure_ascii=False) + "\n")
    print(f"   {techs_path}: {len(techs)} technologies")

    # Counters
    counters_path = OUTPUT_DIR / "counters.json"
    counters_path.write_text(json.dumps(COUNTER_MAP, indent=2) + "\n")
    print(f"   {counters_path}: {len(COUNTER_MAP)} counter entries")

    # Summary
    summary = generate_summary(units, buildings, civs)
    summary_path = OUTPUT_DIR / "summary.md"
    summary_path.write_text(summary)
    print(f"   {summary_path}: LLM-readable reference")

    print("\n" + "=" * 60)
    print(f"Knowledge base written to: {OUTPUT_DIR}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
