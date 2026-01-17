#!/usr/bin/env python3
"""
One-time script to populate the AoE2 SQLite database from online sources.

Usage:
    python -m data.populate_db
    # or
    python data/populate_db.py
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.game_knowledge import GameKnowledge


def main():
    print("=" * 60)
    print("AoE2 Game Knowledge Database Population")
    print("=" * 60)

    db_path = Path(__file__).parent / "aoe2.db"
    print(f"\nDatabase location: {db_path}")

    # Remove existing database to start fresh
    if db_path.exists():
        print("Removing existing database...")
        db_path.unlink()

    # Create and populate database
    print("\nConnecting to database...")
    db = GameKnowledge(str(db_path))

    print("Fetching data from halfon.aoe2.se...")
    try:
        count = db.populate_from_halfon()
        print(f"Successfully imported {count} records!")
    except Exception as e:
        print(f"ERROR: Failed to populate database: {e}")
        db.close()
        return 1

    # Show sample data
    print("\n" + "-" * 40)
    print("Sample Units:")
    print("-" * 40)

    sample_resources = {"food": 100, "wood": 100, "gold": 100, "stone": 100}
    units = db.get_affordable_units(sample_resources, "dark", limit=5)
    for unit in units:
        name = unit.get("localized_name", "Unknown")
        food = unit.get("cost_food", 0)
        wood = unit.get("cost_wood", 0)
        gold = unit.get("cost_gold", 0)
        atk = unit.get("attack", 0)
        hp = unit.get("hit_points", 0)
        print(f"  {name}: F={food} W={wood} G={gold} | ATK={atk} HP={hp}")

    print("\n" + "-" * 40)
    print("Sample Buildings:")
    print("-" * 40)

    buildings = db.get_affordable_buildings(sample_resources, "dark", limit=5)
    for bldg in buildings:
        name = bldg.get("localized_name", "Unknown")
        wood = bldg.get("cost_wood", 0)
        stone = bldg.get("cost_stone", 0)
        hp = bldg.get("hit_points", 0)
        print(f"  {name}: W={wood} S={stone} | HP={hp}")

    print("\n" + "-" * 40)
    print("Sample Context Generation:")
    print("-" * 40)

    context = db.get_context_for_state("dark", sample_resources)
    print(context)

    print("\n" + "-" * 40)
    print("Early Game Priorities:")
    print("-" * 40)
    print(db.get_early_game_priorities())

    db.close()
    print("\n" + "=" * 60)
    print("Database population complete!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
