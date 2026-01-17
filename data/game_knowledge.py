"""
SQLite-based game knowledge database for AoE2.

Provides dynamic context injection to reduce prompt bloat while giving
the LLM access to relevant game data based on current game state.
"""

import sqlite3
import json
import urllib.request
from pathlib import Path
from typing import Optional


# Age progression mapping
AGE_ORDER = ["dark", "feudal", "castle", "imperial"]

# AoE2 Unit Class classifications (from game data mining)
# Class field determines the entity type
UNIT_CLASSES = {
    0: "archer",
    1: "artifact",
    4: "civilian",       # Villager
    6: "infantry",
    8: "cavalry",
    12: "cavalry",       # Mounted soldier
    13: "siege",
    14: "predator",      # Wolf, lion
    18: "monk",          # Priest
    19: "trade_unit",    # Trade cart
    22: "infantry",      # Phalanx
    23: "domestic",      # Sheep, turkey
    36: "ship",          # Fire ship
    52: "resource",      # Deep sea fish
    53: "resource",      # Gold mine
    54: "resource",      # Shore fish
    56: "resource",      # Forage/berries
    57: "boar",          # Wild boar
    58: "sheep",         # Sheep specifically
}

# Building classes
BUILDING_CLASSES = {
    3: "building",       # Generic building class
    11: "building",      # All buildings
    20: "trade_building", # Market
    21: "wall",          # Walls
    30: "tower",         # Towers
    51: "flag",          # Flags
}

# Known building IDs (hardcoded for reliability)
KNOWN_BUILDING_IDS = {
    12: ("Barracks", "military"),
    45: ("Dock", "economic"),
    49: ("Siege Workshop", "military"),
    68: ("Mill", "economic"),
    70: ("House", "economic"),
    79: ("Watch Tower", "defensive"),
    82: ("Castle", "defensive"),
    84: ("Market", "economic"),
    87: ("Archery Range", "military"),
    101: ("Stable", "military"),
    103: ("Blacksmith", "economic"),
    104: ("Monastery", "economic"),
    109: ("Town Center", "town_center"),
    117: ("Mining Camp", "economic"),
    155: ("Outpost", "defensive"),
    199: ("Fish Trap", "economic"),
    209: ("University", "economic"),
    234: ("Guard Tower", "defensive"),
    235: ("Keep", "defensive"),
    236: ("Bombard Tower", "defensive"),
    276: ("Wonder", "wonder"),
    487: ("Gate", "defensive"),
    562: ("Lumber Camp", "economic"),
    584: ("Stone Wall", "defensive"),
    598: ("Fortified Wall", "defensive"),
}

# Known unit IDs for key units
KNOWN_UNIT_IDS = {
    83: ("Villager", "civilian", "dark"),
    293: ("Villager", "civilian", "dark"),  # Female villager
    4: ("Archer", "archer", "feudal"),
    24: ("Crossbowman", "archer", "castle"),
    38: ("Knight", "cavalry", "castle"),
    74: ("Militia", "infantry", "dark"),
    75: ("Man-at-Arms", "infantry", "feudal"),
    77: ("Long Swordsman", "infantry", "castle"),
    93: ("Spearman", "infantry", "feudal"),
    358: ("Pikeman", "infantry", "castle"),
    440: ("Petard", "siege", "castle"),
    448: ("Scout Cavalry", "cavalry", "dark"),
    546: ("Hussar", "cavalry", "imperial"),
    751: ("Eagle Scout", "infantry", "feudal"),
    752: ("Eagle Warrior", "infantry", "castle"),
    1225: ("Sheep", "resource", "dark"),
}


class GameKnowledge:
    """SQLite database wrapper for AoE2 game data."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database. If None, uses default location.
        """
        if db_path is None:
            db_path = str(Path(__file__).parent / "aoe2.db")

        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
        self._init_tables()

    def _init_tables(self):
        """Create database tables if they don't exist."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS units (
                id INTEGER PRIMARY KEY,
                name TEXT,
                localized_name TEXT,
                hit_points INTEGER,
                attack INTEGER,
                melee_armor INTEGER DEFAULT 0,
                pierce_armor INTEGER DEFAULT 0,
                range INTEGER DEFAULT 0,
                cost_food INTEGER DEFAULT 0,
                cost_wood INTEGER DEFAULT 0,
                cost_gold INTEGER DEFAULT 0,
                cost_stone INTEGER DEFAULT 0,
                train_time INTEGER DEFAULT 0,
                age TEXT DEFAULT 'dark',
                type INTEGER,
                class INTEGER,
                line_of_sight INTEGER DEFAULT 4
            );

            CREATE TABLE IF NOT EXISTS buildings (
                id INTEGER PRIMARY KEY,
                name TEXT,
                localized_name TEXT,
                hit_points INTEGER,
                cost_wood INTEGER DEFAULT 0,
                cost_stone INTEGER DEFAULT 0,
                build_time INTEGER DEFAULT 0,
                age TEXT DEFAULT 'dark',
                type INTEGER,
                garrison_capacity INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS technologies (
                id INTEGER PRIMARY KEY,
                name TEXT,
                localized_name TEXT,
                cost_food INTEGER DEFAULT 0,
                cost_wood INTEGER DEFAULT 0,
                cost_gold INTEGER DEFAULT 0,
                cost_stone INTEGER DEFAULT 0,
                research_time INTEGER DEFAULT 0,
                age TEXT DEFAULT 'dark',
                building_id INTEGER,
                effect_description TEXT
            );

            CREATE TABLE IF NOT EXISTS counters (
                unit_id INTEGER,
                countered_by_id INTEGER,
                effectiveness TEXT,
                PRIMARY KEY (unit_id, countered_by_id),
                FOREIGN KEY (unit_id) REFERENCES units(id),
                FOREIGN KEY (countered_by_id) REFERENCES units(id)
            );

            CREATE INDEX IF NOT EXISTS idx_units_age ON units(age);
            CREATE INDEX IF NOT EXISTS idx_units_type ON units(type);
            CREATE INDEX IF NOT EXISTS idx_buildings_age ON buildings(age);
            CREATE INDEX IF NOT EXISTS idx_techs_age ON technologies(age);
        """)
        self.conn.commit()

    def _is_building(self, entity_id: int, entity: dict) -> bool:
        """Determine if an entity is a building based on multiple heuristics."""
        # Check known building IDs first
        if entity_id in KNOWN_BUILDING_IDS:
            return True

        entity_class = entity.get("class", -1)

        # Check building classes
        if entity_class in BUILDING_CLASSES:
            return True

        # Buildings typically have high HP, no attack, and garrison capacity
        hp = entity.get("hit_points", 0)
        attack = entity.get("attack", 0)
        garrison = entity.get("garrison_capacity", 0)

        # High HP with low/no attack is likely a building
        if hp > 500 and attack == 0:
            return True

        # Has garrison capacity
        if garrison > 0 and hp > 200:
            return True

        # Name-based heuristics
        name = str(entity.get("name", "")).lower()
        building_keywords = ["tower", "castle", "wall", "gate", "house", "mill",
                           "camp", "dock", "range", "stable", "barracks", "center",
                           "monastery", "market", "university", "blacksmith", "wonder",
                           "outpost", "workshop", "trap", "farm"]
        if any(kw in name for kw in building_keywords):
            return True

        return False

    def populate_from_halfon(self) -> int:
        """Download and populate database from halfon.aoe2.se.

        Returns:
            Number of records inserted.
        """
        url = "https://halfon.aoe2.se/data/units_buildings_techs.de.json"

        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                data = json.loads(response.read().decode('utf-8'))
        except Exception as e:
            raise RuntimeError(f"Failed to fetch data from {url}: {e}")

        count = 0
        unit_count = 0
        building_count = 0

        # Process units and buildings from the main data
        units_buildings = data.get("units_buildings", {})
        for id_str, entity in units_buildings.items():
            try:
                entity_id = int(id_str)
                cost = entity.get("cost", {})
                entity_class = entity.get("class", 0)

                # Determine if building or unit
                if self._is_building(entity_id, entity):
                    # It's a building
                    self.conn.execute("""
                        INSERT OR REPLACE INTO buildings
                        (id, name, localized_name, hit_points, cost_wood, cost_stone,
                         build_time, age, type, garrison_capacity)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        entity_id,
                        entity.get("name"),
                        entity.get("localised_name") or entity.get("localized_name"),
                        entity.get("hit_points", 0),
                        cost.get("wood", 0),
                        cost.get("stone", 0),
                        entity.get("build_time", 0),
                        self._infer_age(entity, entity_id),
                        entity.get("type", 0),
                        entity.get("garrison_capacity", 0)
                    ))
                    building_count += 1
                else:
                    # It's a unit
                    self.conn.execute("""
                        INSERT OR REPLACE INTO units
                        (id, name, localized_name, hit_points, attack, melee_armor,
                         pierce_armor, range, cost_food, cost_wood, cost_gold, cost_stone,
                         train_time, age, type, class, line_of_sight)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        entity_id,
                        entity.get("name"),
                        entity.get("localised_name") or entity.get("localized_name"),
                        entity.get("hit_points", 0),
                        entity.get("attack", 0),
                        entity.get("melee_armor", 0),
                        entity.get("pierce_armor", 0),
                        entity.get("range", 0),
                        cost.get("food", 0),
                        cost.get("wood", 0),
                        cost.get("gold", 0),
                        cost.get("stone", 0),
                        entity.get("train_time", 0),
                        self._infer_age(entity, entity_id),
                        entity.get("type", 0),
                        entity_class,
                        entity.get("line_of_sight", 4)
                    ))
                    unit_count += 1
                count += 1
            except (ValueError, TypeError) as e:
                continue  # Skip malformed entries

        # Also populate from hardcoded essential data for reliability
        self._populate_essential_units()
        self._populate_essential_buildings()

        self.conn.commit()
        print(f"Loaded {unit_count} units and {building_count} buildings")
        return count

    def _populate_essential_units(self):
        """Populate essential units that must always be present."""
        essential_units = [
            # id, name, localized_name, hp, attack, melee_armor, pierce_armor,
            # range, food, wood, gold, stone, train_time, age, type, class, los
            (83, "YOURNG", "Villager", 25, 3, 0, 0, 0, 50, 0, 0, 0, 25, "dark", 70, 4, 4),
            (448, "SCOUTG", "Scout Cavalry", 45, 3, 0, 2, 0, 0, 0, 0, 0, 30, "dark", 70, 12, 4),
            (1225, "SHEEP", "Sheep", 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, "dark", 70, 58, 2),
        ]

        for unit in essential_units:
            try:
                self.conn.execute("""
                    INSERT OR IGNORE INTO units
                    (id, name, localized_name, hit_points, attack, melee_armor,
                     pierce_armor, range, cost_food, cost_wood, cost_gold, cost_stone,
                     train_time, age, type, class, line_of_sight)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, unit)
            except Exception:
                pass

    def _populate_essential_buildings(self):
        """Populate essential buildings that must always be present."""
        essential_buildings = [
            # id, name, localized_name, hp, wood, stone, build_time, age, type, garrison
            (70, "HOUSEG", "House", 550, 25, 0, 25, "dark", 80, 0),
            (109, "TOWNCR", "Town Center", 2400, 275, 100, 150, "dark", 80, 15),
            (68, "MILLG", "Mill", 600, 100, 0, 35, "dark", 80, 0),
            (562, "LUMBRG", "Lumber Camp", 600, 100, 0, 35, "dark", 80, 0),
            (117, "MNGCMP", "Mining Camp", 600, 100, 0, 35, "dark", 80, 0),
            (12, "BARAKSG", "Barracks", 1200, 175, 0, 50, "dark", 80, 0),
        ]

        for building in essential_buildings:
            try:
                self.conn.execute("""
                    INSERT OR IGNORE INTO buildings
                    (id, name, localized_name, hit_points, cost_wood, cost_stone,
                     build_time, age, type, garrison_capacity)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, building)
            except Exception:
                pass

    def _infer_age(self, entity: dict, entity_id: int = 0) -> str:
        """Infer the age requirement for an entity.

        Args:
            entity: Entity dict from JSON data
            entity_id: Optional entity ID for known unit lookups
        """
        # Check known unit IDs first
        if entity_id in KNOWN_UNIT_IDS:
            return KNOWN_UNIT_IDS[entity_id][2]

        # Check if explicitly specified in data
        if "age" in entity:
            age_val = entity["age"]
            if isinstance(age_val, str):
                return age_val.lower()
            elif isinstance(age_val, int):
                return AGE_ORDER[min(age_val, 3)]

        # Infer from cost patterns or name
        name = str(entity.get("name", "")).lower()
        localized = str(entity.get("localised_name", "")).lower()
        cost = entity.get("cost", {})

        # Imperial age indicators
        imperial_keywords = ["elite", "heavy", "siege ram", "bombard", "champion",
                           "paladin", "arbalest", "hand cannon", "hussar"]
        if any(x in name or x in localized for x in imperial_keywords):
            return "imperial"

        # Castle age indicators
        castle_keywords = ["knight", "crossbow", "mangonel", "monk", "pikeman",
                         "long sword", "camel", "scorpion", "cavalier"]
        if any(x in name or x in localized for x in castle_keywords):
            return "castle"

        # Feudal age indicators
        feudal_keywords = ["archer", "scout", "spear", "skirmish", "man-at-arms",
                          "galley", "fire ship"]
        if any(x in name or x in localized for x in feudal_keywords):
            return "feudal"

        # High gold cost usually means later age
        if cost.get("gold", 0) > 50:
            return "castle"
        if cost.get("gold", 0) > 30:
            return "feudal"

        return "dark"

    def get_affordable_units(self, resources: dict, age: str = "dark", limit: int = 10) -> list[dict]:
        """Get units that can be afforded with current resources.

        Args:
            resources: Dict with food, wood, gold, stone values
            age: Current age (dark, feudal, castle, imperial)
            limit: Maximum number of results

        Returns:
            List of affordable unit dicts
        """
        age_index = AGE_ORDER.index(age.lower()) if age.lower() in AGE_ORDER else 0
        available_ages = AGE_ORDER[:age_index + 1]

        placeholders = ",".join("?" * len(available_ages))
        cursor = self.conn.execute(f"""
            SELECT localized_name, cost_food, cost_wood, cost_gold, attack, hit_points,
                   melee_armor, pierce_armor, range, train_time
            FROM units
            WHERE cost_food <= ? AND cost_wood <= ? AND cost_gold <= ? AND cost_stone <= ?
            AND age IN ({placeholders})
            AND localized_name IS NOT NULL
            ORDER BY attack DESC
            LIMIT ?
        """, (
            resources.get("food", 0),
            resources.get("wood", 0),
            resources.get("gold", 0),
            resources.get("stone", 0),
            *available_ages,
            limit
        ))

        return [dict(row) for row in cursor.fetchall()]

    def get_affordable_buildings(self, resources: dict, age: str = "dark", limit: int = 10) -> list[dict]:
        """Get buildings that can be afforded with current resources."""
        age_index = AGE_ORDER.index(age.lower()) if age.lower() in AGE_ORDER else 0
        available_ages = AGE_ORDER[:age_index + 1]

        placeholders = ",".join("?" * len(available_ages))
        cursor = self.conn.execute(f"""
            SELECT localized_name, cost_wood, cost_stone, hit_points, build_time
            FROM buildings
            WHERE cost_wood <= ? AND cost_stone <= ?
            AND age IN ({placeholders})
            AND localized_name IS NOT NULL
            ORDER BY hit_points DESC
            LIMIT ?
        """, (
            resources.get("wood", 0),
            resources.get("stone", 0),
            *available_ages,
            limit
        ))

        return [dict(row) for row in cursor.fetchall()]

    def get_unit_by_name(self, name: str) -> Optional[dict]:
        """Look up a unit by name (partial match)."""
        cursor = self.conn.execute("""
            SELECT * FROM units
            WHERE localized_name LIKE ? OR name LIKE ?
            LIMIT 1
        """, (f"%{name}%", f"%{name}%"))

        row = cursor.fetchone()
        return dict(row) if row else None

    def get_counter_info(self, unit_name: str) -> str:
        """Get counter information for a unit type."""
        # Hardcoded counter relationships (common knowledge)
        counters = {
            "archer": "Skirmisher, Mangonel, Knight",
            "knight": "Pikeman, Camel, Monk",
            "pikeman": "Archer, Mangonel, Knight (with micro)",
            "cavalry": "Pikeman, Camel, Monk",
            "infantry": "Archer, Hand Cannoneer",
            "siege": "Cavalry, Bombard Cannon",
            "monk": "Light Cavalry, Eagle Warrior",
        }

        unit_name_lower = unit_name.lower()
        for key, value in counters.items():
            if key in unit_name_lower:
                return value
        return "Unknown"

    def get_context_for_state(self, age: str, resources: dict,
                               detected_entities: Optional[list] = None) -> str:
        """Generate minimal context string for LLM injection.

        Args:
            age: Current age (dark, feudal, castle, imperial)
            resources: Dict with food, wood, gold, stone
            detected_entities: Optional list of detected entity dicts from YOLO

        Returns:
            Formatted context string (~200-500 tokens)
        """
        lines = []

        # Section 1: Detected entities (if provided)
        if detected_entities:
            lines.append("## Detected Entities")
            for entity in detected_entities[:15]:  # Limit to 15 entities
                eid = entity.get("id", "unknown")
                cls = entity.get("class", "unknown")
                center = entity.get("center", (0, 0))
                conf = entity.get("confidence", 0)
                lines.append(f"  {eid}: {cls} at ({int(center[0])},{int(center[1])}) [{conf:.0%}]")
            lines.append("")

        # Section 2: Affordable units
        affordable_units = self.get_affordable_units(resources, age, limit=5)
        if affordable_units:
            lines.append("## Trainable Units")
            for unit in affordable_units:
                name = unit.get("localized_name", "Unknown")
                food = unit.get("cost_food", 0)
                wood = unit.get("cost_wood", 0)
                gold = unit.get("cost_gold", 0)
                atk = unit.get("attack", 0)
                hp = unit.get("hit_points", 0)

                cost_parts = []
                if food: cost_parts.append(f"{food}F")
                if wood: cost_parts.append(f"{wood}W")
                if gold: cost_parts.append(f"{gold}G")

                lines.append(f"  {name}: {'/'.join(cost_parts)} (ATK:{atk}, HP:{hp})")
            lines.append("")

        # Section 3: Affordable buildings
        affordable_buildings = self.get_affordable_buildings(resources, age, limit=5)
        if affordable_buildings:
            lines.append("## Buildable Structures")
            for bldg in affordable_buildings:
                name = bldg.get("localized_name", "Unknown")
                wood = bldg.get("cost_wood", 0)
                stone = bldg.get("cost_stone", 0)
                hp = bldg.get("hit_points", 0)

                cost_parts = []
                if wood: cost_parts.append(f"{wood}W")
                if stone: cost_parts.append(f"{stone}S")

                lines.append(f"  {name}: {'/'.join(cost_parts)} (HP:{hp})")
            lines.append("")

        return "\n".join(lines)

    def get_early_game_priorities(self) -> str:
        """Get strategic priorities for early game (always inject this)."""
        return """## Early Game Priorities
1. FOOD FIRST: Send all villagers to sheep (50F each = constant villager production)
2. Keep TC producing: Queue villagers (H then Q)
3. Build house at 4/5 pop (before housed)
4. Villager cost: 50 food, 25 seconds train time
5. House cost: 25 wood, provides +5 population
"""

    def close(self):
        """Close database connection."""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Singleton instance for easy access
_instance: Optional[GameKnowledge] = None

def get_db() -> GameKnowledge:
    """Get or create the singleton database instance."""
    global _instance
    if _instance is None:
        _instance = GameKnowledge()
    return _instance
