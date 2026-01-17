#!/usr/bin/env python3
"""
Extract comprehensive sprite set for YOLO training.

This script extracts sprites grouped by gameplay-relevant categories,
not individual unit variants. For example, all archer-line units
(archer, crossbow, arbalest) are grouped as "archer_line".

This gives the agent enough information to make tactical decisions
while keeping the number of classes manageable for training.
"""

from pathlib import Path
from typing import Optional
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from detection.sld_extractor import extract_sprite


# =============================================================================
# SPRITE EXTRACTION CONFIGURATION
# =============================================================================
# Format: (class_name, [file_patterns], max_variants, description)
#
# Patterns can be:
#   - Exact filename: "u_vil_male_villager_idleA_x1.sld"
#   - Glob pattern: "*_house_age2_x1.sld"
#
# We extract multiple variants per class to improve model robustness
# =============================================================================

SPRITE_CONFIG = [
    # =========================================================================
    # ECONOMIC UNITS
    # =========================================================================
    ("villager", [
        "u_vil_male_villager_idle*_x1.sld",
        "u_vil_female_villager_idle*_x1.sld",
        "u_all_ant_male_villager_idle*_x1.sld",
        "u_all_ant_female_villager_idle*_x1.sld",
    ], 6, "Worker units"),

    ("trade_cart", [
        "u_trade_cart_idle*_x1.sld",
        "*trade*cart*idle*_x1.sld",
    ], 2, "Trade unit"),

    ("fishing_ship", [
        "u_shp_*fishing_ship*_x1.sld",
        "u_shp_fishing*_x1.sld",
    ], 2, "Water economy"),

    # =========================================================================
    # CAVALRY UNITS
    # =========================================================================
    ("scout_line", [
        "u_cav_scout_idle*_x1.sld",
        "u_cav_lightcavalry_idle*_x1.sld",
        "u_cav_hussar_idle*_x1.sld",
    ], 4, "Scout, Light Cav, Hussar"),

    ("knight_line", [
        "u_cav_knight_idle*_x1.sld",
        "u_cav_cavalier_idle*_x1.sld",
        "u_cav_paladin_idle*_x1.sld",
    ], 4, "Knight, Cavalier, Paladin"),

    ("camel_line", [
        "u_cam_camel_*idle*_x1.sld",
        "u_cam_camel_heavy_idle*_x1.sld",
    ], 3, "Camel units"),

    ("battle_elephant", [
        "u_cav_elephant*idle*_x1.sld",
        "*battle*elephant*idle*_x1.sld",
    ], 2, "Elephant units"),

    # =========================================================================
    # ARCHER UNITS
    # =========================================================================
    ("archer_line", [
        "u_arc_archer_idle*_x1.sld",
        "u_arc_crossbow*idle*_x1.sld",
        "u_arc_arbalest*idle*_x1.sld",
    ], 4, "Archer, Crossbow, Arbalest"),

    ("skirmisher_line", [
        "u_arc_skirmisher_idle*_x1.sld",
        "u_arc_eliteskirmisher*idle*_x1.sld",
    ], 3, "Skirmisher line"),

    ("cavalry_archer", [
        "u_cav_ant_archer_*idle*_x1.sld",
        "u_cav_ant_archer_heavy_idle*_x1.sld",
        "u_cav_cavalry_archer*idle*_x1.sld",
    ], 3, "Cavalry Archer line"),

    ("hand_cannoneer", [
        "u_arc_handcannoneer*idle*_x1.sld",
        "*hand*cannon*idle*_x1.sld",
    ], 2, "Gunpowder archer"),

    # =========================================================================
    # INFANTRY UNITS
    # =========================================================================
    ("militia_line", [
        "u_inf_militia_idle*_x1.sld",
        "u_inf_manatarms_idle*_x1.sld",
        "u_inf_longsword*idle*_x1.sld",
        "u_inf_twohanded*idle*_x1.sld",
        "u_inf_champion_idle*_x1.sld",
    ], 5, "Militia → Champion"),

    ("spearman_line", [
        "u_inf_spearman_idle*_x1.sld",
        "u_inf_pikeman_idle*_x1.sld",
        "u_inf_halberdier*idle*_x1.sld",
    ], 4, "Spearman, Pikeman, Halberdier"),

    ("eagle_line", [
        "u_inf_eagle*idle*_x1.sld",
        "*eagle*warrior*idle*_x1.sld",
    ], 3, "Meso eagle warriors"),

    # =========================================================================
    # SIEGE UNITS
    # =========================================================================
    ("ram", [
        "u_sie_batteringram*idle*_x1.sld",
        "u_sie_cappedram*idle*_x1.sld",
        "u_sie_siegeram*idle*_x1.sld",
        "*ram*idle*_x1.sld",
    ], 3, "Battering rams"),

    ("mangonel_line", [
        "u_sie_mangonel*idle*_x1.sld",
        "u_sie_onager*idle*_x1.sld",
        "*mangonel*idle*_x1.sld",
    ], 3, "Mangonel, Onager"),

    ("scorpion", [
        "u_sie_scorpion*idle*_x1.sld",
        "*scorpion*idle*_x1.sld",
    ], 2, "Scorpion"),

    ("trebuchet", [
        "u_sie_trebuchet*idle*_x1.sld",
        "*trebuchet*idle*_x1.sld",
    ], 2, "Trebuchet"),

    # =========================================================================
    # SPECIAL UNITS
    # =========================================================================
    ("monk", [
        "u_rel_monk_idle*_x1.sld",
        "*monk*idle*_x1.sld",
    ], 3, "Monks"),

    ("king", [
        "u_king*idle*_x1.sld",
        "*king*idle*_x1.sld",
    ], 1, "King (regicide)"),

    # =========================================================================
    # UNIQUE UNITS (selected important ones)
    # =========================================================================
    ("longbowman", [
        "*longbow*idle*_x1.sld",
    ], 2, "British UU"),

    ("mangudai", [
        "*mangudai*idle*_x1.sld",
    ], 2, "Mongol UU"),

    ("war_wagon", [
        "*war*wagon*idle*_x1.sld",
    ], 2, "Korean UU"),

    # =========================================================================
    # BUILDINGS - ECONOMY (Western European / Mediterranean style)
    # Using west/medi civs for generic European look (Franks, Britons, etc.)
    # =========================================================================
    ("town_center", [
        "b_west_town_center_age2_x1.sld",
        "b_west_town_center_age3_x1.sld",
        "b_medi_town_center_age2_x1.sld",
        "b_medi_town_center_age3_x1.sld",
    ], 4, "Town Center"),

    ("house", [
        "b_west_house_age2_x1.sld",
        "b_west_house_age3_x1.sld",
        "b_medi_house_age2_x1.sld",
        "b_medi_house_age3_x1.sld",
    ], 4, "Houses"),

    ("mill", [
        "b_west_mill_age2_x1.sld",
        "b_west_mill_age3_x1.sld",
        "b_medi_mill_age2_x1.sld",
    ], 3, "Mill"),

    ("lumber_camp", [
        "b_west_lumber_camp_age2_x1.sld",
        "b_west_lumber_camp_age3_x1.sld",
        "b_medi_lumber_camp_age2_x1.sld",
    ], 3, "Lumber Camp"),

    ("mining_camp", [
        "b_west_mining_camp_age2_x1.sld",
        "b_west_mining_camp_age3_x1.sld",
        "b_medi_mining_camp_age2_x1.sld",
    ], 3, "Mining Camp"),

    # Farm buildings are terrain textures in a different format, skipping

    ("market", [
        "b_west_market_age2_x1.sld",
        "b_west_market_age3_x1.sld",
        "b_medi_market_age2_x1.sld",
    ], 3, "Market"),

    ("blacksmith", [
        "b_west_blacksmith_age2_x1.sld",
        "b_west_blacksmith_age3_x1.sld",
        "b_medi_blacksmith_age2_x1.sld",
    ], 3, "Blacksmith"),

    # =========================================================================
    # BUILDINGS - MILITARY (Western European / Mediterranean style)
    # =========================================================================
    ("barracks", [
        "b_west_barracks_age2_x1.sld",
        "b_west_barracks_age3_x1.sld",
        "b_medi_barracks_age2_x1.sld",
    ], 3, "Barracks"),

    ("archery_range", [
        "b_west_archery_range_age2_x1.sld",
        "b_west_archery_range_age3_x1.sld",
        "b_medi_archery_range_age2_x1.sld",
    ], 3, "Archery Range"),

    ("stable", [
        "b_west_stable_age2_x1.sld",
        "b_west_stable_age3_x1.sld",
        "b_medi_stable_age2_x1.sld",
    ], 3, "Stable"),

    ("siege_workshop", [
        "b_west_siege_workshop_age3_x1.sld",
        "b_medi_siege_workshop_age3_x1.sld",
    ], 2, "Siege Workshop"),

    ("monastery", [
        "b_west_monastery_age3_x1.sld",
        "b_medi_monastery_age3_x1.sld",
    ], 2, "Monastery"),

    ("castle", [
        "b_west_castle_age3_x1.sld",
        "b_medi_castle_age3_x1.sld",
    ], 2, "Castle"),

    # Krepost is Bulgarian unique - skipping for generic training

    # =========================================================================
    # BUILDINGS - DEFENSE (Western European / Mediterranean style)
    # =========================================================================
    ("tower", [
        "b_west_watch_tower_age2_x1.sld",
        "b_west_guard_tower_age3_x1.sld",
        "b_west_keep_age4_x1.sld",
        "b_medi_watch_tower_age2_x1.sld",
    ], 4, "Towers"),

    ("wall", [
        "b_archaic_palisade_wall_x1.sld",
        "b_west_stone_wall_age3_x1.sld",
        "b_west_fortified_wall_age4_x1.sld",
    ], 3, "Walls"),

    ("gate", [
        "b_west_gate_palisade_e_closed_x1.sld",
        "b_west_gate_stone_e_closed_x1.sld",
    ], 2, "Gates"),

    # =========================================================================
    # BUILDINGS - OTHER (Western European / Mediterranean style)
    # =========================================================================
    ("dock", [
        "b_west_dock_age2_x1.sld",
        "b_west_dock_age3_x1.sld",
    ], 2, "Dock"),

    ("university", [
        "b_west_university_age3_x1.sld",
        "b_medi_university_age3_x1.sld",
    ], 2, "University"),

    ("wonder", [
        "b_west_wonder_britons_x1.sld",
        "b_west_wonder_franks_x1.sld",
    ], 2, "Wonder"),

    # =========================================================================
    # RESOURCES & NATURE
    # =========================================================================
    ("sheep", [
        "a_herd_sheep_idle*_x1.sld",
    ], 2, "Sheep"),

    ("deer", [
        "a_hunt_deer_idle*_x1.sld",
    ], 2, "Deer"),

    ("boar", [
        "a_hunt_boar_idle*_x1.sld",
        "a_hunt_javelina_idle*_x1.sld",
        "a_hunt_elephant_idle*_x1.sld",
        "a_hunt_rhino*idle*_x1.sld",
    ], 4, "Boar/Huntables"),

    ("wolf", [
        "a_pred_*wolf*idle*_x1.sld",
        "a_pred_arabian_wolf_idle*_x1.sld",
    ], 2, "Wolves (danger)"),

    ("gold_mine", [
        "n_*gold*_x1.sld",
    ], 2, "Gold"),

    ("stone_mine", [
        "n_*stone*_x1.sld",
    ], 2, "Stone"),

    ("berry_bush", [
        "n_*berry*_x1.sld",
        "n_*forage*_x1.sld",
    ], 2, "Berries"),

    ("tree", [
        "n_tree_oak*_x1.sld",
        "n_tree_pine*_x1.sld",
        "n_tree_palm*_x1.sld",
        "n_tree_jungle*_x1.sld",
    ], 4, "Trees (wood)"),

    ("relic", [
        "n_relic*_x1.sld",
        "*relic*_x1.sld",
    ], 2, "Relics"),
]


def find_matching_files(game_dir: Path, patterns: list[str], max_count: int) -> list[Path]:
    """Find SLD files matching patterns."""
    matches = []

    for pattern in patterns:
        if '*' in pattern:
            found = list(game_dir.glob(pattern))
        else:
            exact = game_dir / pattern
            found = [exact] if exact.exists() else []

        for f in sorted(found):  # Sort for consistency
            if f not in matches and f.suffix == '.sld':
                matches.append(f)
                if len(matches) >= max_count:
                    return matches

    return matches[:max_count]


def extract_sprites(
    game_graphics_dir: str,
    output_dir: str,
    verbose: bool = True
) -> dict:
    """Extract sprites for all configured classes."""
    game_dir = Path(game_graphics_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not game_dir.exists():
        print(f"Error: Directory not found: {game_dir}")
        return {"error": "Directory not found"}

    stats = {
        "total_extracted": 0,
        "total_failed": 0,
        "classes": 0,
        "by_class": {},
    }

    print(f"{'='*60}")
    print(f"Extracting AoE2 Sprites for YOLO Training")
    print(f"{'='*60}")
    print(f"Source: {game_dir}")
    print(f"Output: {out_dir}")
    print(f"Classes configured: {len(SPRITE_CONFIG)}")
    print(f"{'='*60}\n")

    for class_name, patterns, max_variants, description in SPRITE_CONFIG:
        matches = find_matching_files(game_dir, patterns, max_variants)

        if verbose:
            print(f"{class_name} ({description}):")

        if not matches:
            if verbose:
                print(f"  ⚠ No files found")
            stats["by_class"][class_name] = {"found": 0, "extracted": 0}
            continue

        extracted = 0
        for i, sld_file in enumerate(matches):
            out_file = out_dir / f"{class_name}_{i:02d}.png"

            try:
                success = extract_sprite(str(sld_file), str(out_file))
                if success:
                    extracted += 1
                    stats["total_extracted"] += 1
                    if verbose:
                        print(f"  ✓ {sld_file.name}")
                else:
                    stats["total_failed"] += 1
                    if verbose:
                        print(f"  ✗ {sld_file.name}")
            except Exception as e:
                stats["total_failed"] += 1
                if verbose:
                    print(f"  ✗ {sld_file.name}: {e}")

        if extracted > 0:
            stats["classes"] += 1

        stats["by_class"][class_name] = {
            "found": len(matches),
            "extracted": extracted,
            "description": description
        }

    # Summary
    print(f"\n{'='*60}")
    print("EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"Classes with sprites: {stats['classes']}/{len(SPRITE_CONFIG)}")
    print(f"Total sprites extracted: {stats['total_extracted']}")
    print(f"Total failed: {stats['total_failed']}")
    print(f"Output directory: {out_dir}")

    return stats


def print_config():
    """Print the extraction configuration."""
    print(f"\n{'='*60}")
    print("SPRITE EXTRACTION CONFIGURATION")
    print(f"{'='*60}")

    categories = {
        "Economic Units": ["villager", "trade_cart", "fishing_ship"],
        "Cavalry": ["scout_line", "knight_line", "camel_line", "battle_elephant"],
        "Archers": ["archer_line", "skirmisher_line", "cavalry_archer", "hand_cannoneer"],
        "Infantry": ["militia_line", "spearman_line", "eagle_line"],
        "Siege": ["ram", "mangonel_line", "scorpion", "trebuchet"],
        "Special": ["monk", "king", "longbowman", "mangudai", "war_wagon"],
        "Economy Buildings": ["town_center", "house", "mill", "lumber_camp", "mining_camp", "farm", "market", "blacksmith"],
        "Military Buildings": ["barracks", "archery_range", "stable", "siege_workshop", "monastery", "castle"],
        "Defense": ["tower", "wall", "gate"],
        "Resources": ["sheep", "deer", "boar", "wolf", "gold_mine", "stone_mine", "berry_bush", "tree", "relic"],
    }

    config_dict = {c[0]: c for c in SPRITE_CONFIG}

    for category, class_names in categories.items():
        print(f"\n{category}:")
        for name in class_names:
            if name in config_dict:
                _, _, max_var, desc = config_dict[name]
                print(f"  • {name} (max {max_var}): {desc}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract comprehensive sprite set for YOLO training"
    )
    parser.add_argument(
        "--game-dir", "-g",
        default="game_graphics",
        help="Path to game_graphics directory"
    )
    parser.add_argument(
        "--output", "-o",
        default="tmp/sprites",
        help="Output directory"
    )
    parser.add_argument(
        "--show-config", "-c",
        action="store_true",
        help="Show extraction configuration and exit"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce output"
    )

    args = parser.parse_args()

    if args.show_config:
        print_config()
        return 0

    agent_dir = Path(__file__).parent.parent
    game_dir = agent_dir / args.game_dir
    output_dir = agent_dir / args.output

    stats = extract_sprites(
        str(game_dir),
        str(output_dir),
        verbose=not args.quiet
    )

    return 0 if not stats.get("error") else 1


if __name__ == "__main__":
    exit(main())
