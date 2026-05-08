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

from .sld_extractor import extract_multiple_frames, extract_sprite

# Animation frames to extract for movement/action variation
# [0] = idle, [4,8,12] = walking/action frames (varies by unit)
DEFAULT_FRAMES_TO_EXTRACT = [0, 4, 8, 12]


# Substrings to exclude when matching building SLD files (destruction anims,
# rubble, shadows, foundation scaffolding, etc.)
# Note: TC multi-part layers (_back_, _center_, _front_, _main_) are already
# excluded by the glob patterns themselves, so we don't need to filter them.
# Excluding "_center_" would break town_center matching.
EXCLUDE_SUBSTRINGS = [
    "destruction",
    "rubble",
    "shadow",
    "constr",
    "foundation",
    "attackup",
    "defenseup",
    "bothup",
    "_snow_",
]


# =============================================================================
# SPRITE EXTRACTION CONFIGURATION
# =============================================================================
# Format: (class_name, [file_patterns], max_variants, description)
#
# Patterns can be:
#   - Exact filename: "u_vil_male_villager_idleA_x1.sld"
#   - Glob pattern: "*_house_age2_x1.sld"
#
# Building patterns use b_*_ wildcards to capture ALL architecture styles
# (afri, asia, ceas, east, greek, indi, medi, meso, orie, persian, puru,
# seas, slav, thracian, west, etc.)
#
# We extract multiple variants per class to improve model robustness
# =============================================================================

SPRITE_CONFIG = [
    # =========================================================================
    # ECONOMIC UNITS
    # =========================================================================
    (
        "villager",
        [
            "u_vil_male_villager_idle*_x1.sld",
            "u_vil_female_villager_idle*_x1.sld",
            "u_all_ant_male_villager_idle*_x1.sld",
            "u_all_ant_female_villager_idle*_x1.sld",
        ],
        6,
        "Worker units",
    ),
    (
        "trade_cart",
        [
            "u_trade_cart_idle*_x1.sld",
            "*trade*cart*idle*_x1.sld",
        ],
        2,
        "Trade unit",
    ),
    (
        "fishing_ship",
        [
            "u_shp_*fishing_ship*_x1.sld",
            "u_shp_fishing*_x1.sld",
        ],
        2,
        "Water economy",
    ),
    # =========================================================================
    # CAVALRY UNITS
    # =========================================================================
    (
        "scout_line",
        [
            "u_cav_scout_idle*_x1.sld",
            "u_cav_lightcavalry_idle*_x1.sld",
            "u_cav_hussar_idle*_x1.sld",
        ],
        4,
        "Scout, Light Cav, Hussar",
    ),
    (
        "knight_line",
        [
            "u_cav_knight_idle*_x1.sld",
            "u_cav_cavalier_idle*_x1.sld",
            "u_cav_paladin_idle*_x1.sld",
        ],
        4,
        "Knight, Cavalier, Paladin",
    ),
    (
        "camel_line",
        [
            "u_cam_camel_*idle*_x1.sld",
            "u_cam_camel_heavy_idle*_x1.sld",
        ],
        3,
        "Camel units",
    ),
    (
        "battle_elephant",
        [
            "u_cav_elephant*idle*_x1.sld",
            "*battle*elephant*idle*_x1.sld",
        ],
        2,
        "Elephant units",
    ),
    # =========================================================================
    # ARCHER UNITS
    # =========================================================================
    (
        "archer_line",
        [
            "u_arc_archer_idle*_x1.sld",
            "u_arc_crossbow*idle*_x1.sld",
            "u_arc_arbalest*idle*_x1.sld",
        ],
        4,
        "Archer, Crossbow, Arbalest",
    ),
    (
        "skirmisher_line",
        [
            "u_arc_skirmisher_idle*_x1.sld",
            "u_arc_eliteskirmisher*idle*_x1.sld",
        ],
        3,
        "Skirmisher line",
    ),
    (
        "cavalry_archer",
        [
            "u_cav_ant_archer_*idle*_x1.sld",
            "u_cav_ant_archer_heavy_idle*_x1.sld",
            "u_cav_cavalry_archer*idle*_x1.sld",
        ],
        3,
        "Cavalry Archer line",
    ),
    (
        "hand_cannoneer",
        [
            "u_arc_handcannoneer*idle*_x1.sld",
            "*hand*cannon*idle*_x1.sld",
        ],
        2,
        "Gunpowder archer",
    ),
    # =========================================================================
    # INFANTRY UNITS
    # =========================================================================
    (
        "militia_line",
        [
            "u_inf_militia_idle*_x1.sld",
            "u_inf_manatarms_idle*_x1.sld",
            "u_inf_longsword*idle*_x1.sld",
            "u_inf_twohanded*idle*_x1.sld",
            "u_inf_champion_idle*_x1.sld",
        ],
        5,
        "Militia → Champion",
    ),
    (
        "spearman_line",
        [
            "u_inf_spearman_idle*_x1.sld",
            "u_inf_pikeman_idle*_x1.sld",
            "u_inf_halberdier*idle*_x1.sld",
        ],
        4,
        "Spearman, Pikeman, Halberdier",
    ),
    (
        "eagle_line",
        [
            "u_inf_eagle*idle*_x1.sld",
            "*eagle*warrior*idle*_x1.sld",
        ],
        3,
        "Meso eagle warriors",
    ),
    # =========================================================================
    # SIEGE UNITS
    # =========================================================================
    (
        "ram",
        [
            "u_sie_batteringram*idle*_x1.sld",
            "u_sie_cappedram*idle*_x1.sld",
            "u_sie_siegeram*idle*_x1.sld",
            "*ram*idle*_x1.sld",
        ],
        3,
        "Battering rams",
    ),
    (
        "mangonel_line",
        [
            "u_sie_mangonel*idle*_x1.sld",
            "u_sie_onager*idle*_x1.sld",
            "*mangonel*idle*_x1.sld",
        ],
        3,
        "Mangonel, Onager",
    ),
    (
        "scorpion",
        [
            "u_sie_scorpion*idle*_x1.sld",
            "*scorpion*idle*_x1.sld",
        ],
        2,
        "Scorpion",
    ),
    (
        "trebuchet",
        [
            "u_sie_trebuchet*idle*_x1.sld",
            "*trebuchet*idle*_x1.sld",
        ],
        2,
        "Trebuchet",
    ),
    # =========================================================================
    # SPECIAL UNITS
    # =========================================================================
    (
        "monk",
        [
            "u_rel_monk_idle*_x1.sld",
            "*monk*idle*_x1.sld",
        ],
        3,
        "Monks",
    ),
    (
        "king",
        [
            "u_king*idle*_x1.sld",
            "*king*idle*_x1.sld",
        ],
        1,
        "King (regicide)",
    ),
    # =========================================================================
    # UNIQUE UNITS - Grouped by type (matching classes.yaml ids 50-54)
    # =========================================================================
    (
        "unique_archer",
        [
            "*longbow*idle*_x1.sld",
            "*plumedarcher*idle*_x1.sld",
            "*chukonu*idle*_x1.sld",
            "*mangudai*idle*_x1.sld",
            "*camel_archer*idle*_x1.sld",
            "*janissary*idle*_x1.sld",
            "*rattanarcher*idle*_x1.sld",
        ],
        8,
        "Unique archers (all civs)",
    ),
    (
        "unique_cavalry",
        [
            "u_cav_cataphract*idle*_x1.sld",
            "u_cav_boyar*idle*_x1.sld",
            "u_cav_leitis*idle*_x1.sld",
            "u_cav_keshik*idle*_x1.sld",
            "u_cav_konnik*idle*_x1.sld",
            "u_cav_tarkan*idle*_x1.sld",
            "u_cam_mameluke*idle*_x1.sld",
            "u_cav_magyar_huszar*idle*_x1.sld",
        ],
        8,
        "Unique cavalry (all civs)",
    ),
    (
        "unique_infantry",
        [
            "u_inf_berserk*idle*_x1.sld",
            "u_inf_samurai*idle*_x1.sld",
            "u_inf_jaguarwarrior*idle*_x1.sld",
            "u_inf_huskarl*idle*_x1.sld",
            "u_inf_throwingaxeman*idle*_x1.sld",
            "u_inf_woadraider*idle*_x1.sld",
            "u_inf_karambitwarrior*idle*_x1.sld",
            "u_inf_shotelwarrior*idle*_x1.sld",
            "u_inf_kamayuk*idle*_x1.sld",
            "u_inf_obuch*idle*_x1.sld",
            "u_inf_serjeant*idle*_x1.sld",
            "u_inf_gbeto*idle*_x1.sld",
            "u_inf_chakram_thrower*idle*_x1.sld",
            "u_inf_urumi_swordsman*idle*_x1.sld",
            "u_inf_ghulam*idle*_x1.sld",
        ],
        10,
        "Unique infantry (all civs)",
    ),
    (
        "unique_siege",
        [
            "u_cav_warwagon*idle*_x1.sld",
            "u_sie_organ*idle*_x1.sld",
            "u_sie_hussite_wagon*idle*_x1.sld",
            "u_sie_bombard_cannon*idle*_x1.sld",
        ],
        4,
        "Unique siege units",
    ),
    (
        "unique_ship",
        [
            "u_shp_longboat*_x1.sld",
            "u_shp_caravel*_x1.sld",
            "u_shp_turtle_ship*_x1.sld",
            "u_shp_thirisadai*_x1.sld",
        ],
        4,
        "Unique ships",
    ),
    # =========================================================================
    # BUILDINGS - ECONOMY (All ages, all architecture styles via wildcards)
    # Wildcards b_*_ capture all 17+ styles: afri, asia, ceas, east, greek,
    # indi, medi, meso, orie, persian, puru, seas, slav, thracian, west, etc.
    # =========================================================================
    (
        "town_center",
        [
            "b_*_town_center_age2_x1.sld",  # Feudal
            "b_*_town_center_age3_x1.sld",  # Castle
            "b_*_town_center_age4_x1.sld",  # Imperial
        ],
        15,
        "Town Center (all ages, all styles)",
    ),
    (
        "house",
        [
            "b_*_house_age1_x1.sld",  # Dark Age
            "b_*_house_age2_x1.sld",  # Feudal
            "b_*_house_age3_x1.sld",  # Castle
        ],
        15,
        "Houses (all ages, all styles)",
    ),
    (
        "mill",
        [
            "b_*_mill_age1_x1.sld",  # Dark Age
            "b_*_mill_age2_x1.sld",  # Feudal
            "b_*_mill_age3_x1.sld",  # Castle
        ],
        12,
        "Mill (all ages, all styles)",
    ),
    (
        "lumber_camp",
        [
            "b_*_lumber_camp_age1_x1.sld",  # Dark Age
            "b_*_lumber_camp_age2_x1.sld",  # Feudal
        ],
        10,
        "Lumber Camp (all ages, all styles)",
    ),
    (
        "mining_camp",
        [
            "b_*_mining_camp_age1_x1.sld",  # Dark Age
            "b_*_mining_camp_age2_x1.sld",  # Feudal
        ],
        10,
        "Mining Camp (all ages, all styles)",
    ),
    # Farm buildings are terrain textures in a different format, skipping
    (
        "market",
        [
            "b_*_market_age2_x1.sld",  # Feudal
            "b_*_market_age3_x1.sld",  # Castle
            "b_*_market_age4_x1.sld",  # Imperial
        ],
        12,
        "Market (all ages, all styles)",
    ),
    (
        "blacksmith",
        [
            "b_*_blacksmith_age2_x1.sld",  # Feudal
            "b_*_blacksmith_age3_x1.sld",  # Castle
        ],
        10,
        "Blacksmith (all ages, all styles)",
    ),
    # =========================================================================
    # BUILDINGS - MILITARY (All ages, all architecture styles)
    # =========================================================================
    (
        "barracks",
        [
            "b_*_barracks_age1_x1.sld",  # Dark Age
            "b_*_barracks_age2_x1.sld",  # Feudal
            "b_*_barracks_age3_x1.sld",  # Castle
        ],
        12,
        "Barracks (all ages, all styles)",
    ),
    (
        "archery_range",
        [
            "b_*_archery_range_age2_x1.sld",  # Feudal
            "b_*_archery_range_age3_x1.sld",  # Castle
        ],
        10,
        "Archery Range (all ages, all styles)",
    ),
    (
        "stable",
        [
            "b_*_stable_age2_x1.sld",  # Feudal
            "b_*_stable_age3_x1.sld",  # Castle
        ],
        10,
        "Stable (all ages, all styles)",
    ),
    (
        "siege_workshop",
        [
            "b_*_siege_workshop_age3_x1.sld",  # Castle
        ],
        8,
        "Siege Workshop (all styles)",
    ),
    (
        "monastery",
        [
            "b_*_monastery_age3_x1.sld",  # Castle
        ],
        10,
        "Monastery (all styles)",
    ),
    (
        "castle",
        [
            "b_*_castle_age3_x1.sld",  # Castle
        ],
        15,
        "Castle (all styles)",
    ),
    # =========================================================================
    # BUILDINGS - DEFENSE (All ages, all styles)
    # =========================================================================
    (
        "tower",
        [
            "b_*_outpost_age1_x1.sld",  # Dark Age outpost
            "b_*_tower_age2_x1.sld",  # Feudal (Watch Tower)
            "b_*_tower_age3_x1.sld",  # Castle (Guard Tower)
            "b_*_tower_age4_x1.sld",  # Imperial (Keep)
        ],
        12,
        "Towers (all ages, all styles)",
    ),
    (
        "wall",
        [
            "b_*_palisade_wall_x1.sld",  # Dark Age palisade
            "b_*_wall_stone_x1.sld",  # Castle stone wall
            "b_*_wall_fortified_x1.sld",  # Imperial fortified wall
        ],
        8,
        "Walls (all ages, all styles)",
    ),
    (
        "gate",
        [
            "b_*_gate_palisade_e_closed_x1.sld",  # Dark Age
            "b_*_gate_stone_e_closed_x1.sld",  # Castle
            "b_*_gate_fortified_e_closed_x1.sld",  # Imperial
        ],
        8,
        "Gates (all ages, all styles)",
    ),
    # =========================================================================
    # BUILDINGS - OTHER (All ages, all styles)
    # =========================================================================
    (
        "dock",
        [
            "b_*_dock_age1_x1.sld",  # Dark Age
            "b_*_dock_age2_x1.sld",  # Feudal
            "b_*_dock_age3_x1.sld",  # Castle
        ],
        10,
        "Dock (all ages, all styles)",
    ),
    (
        "university",
        [
            "b_*_university_age3_x1.sld",  # Castle
            "b_*_university_age4_x1.sld",  # Imperial
        ],
        10,
        "University (all ages, all styles)",
    ),
    (
        "wonder",
        [
            "b_*_wonder_*_x1.sld",  # All civ wonders
        ],
        12,
        "Wonder (all civs)",
    ),
    # =========================================================================
    # RESOURCES & NATURE
    # =========================================================================
    (
        "sheep",
        [
            "a_herd_sheep_idle*_x1.sld",
        ],
        2,
        "Sheep",
    ),
    (
        "deer",
        [
            "a_hunt_deer_idle*_x1.sld",
        ],
        2,
        "Deer",
    ),
    (
        "boar",
        [
            "a_hunt_boar_idle*_x1.sld",
            "a_hunt_javelina_idle*_x1.sld",
            "a_hunt_elephant_idle*_x1.sld",
            "a_hunt_rhino*idle*_x1.sld",
        ],
        4,
        "Boar/Huntables",
    ),
    (
        "wolf",
        [
            "a_pred_*wolf*idle*_x1.sld",
            "a_pred_arabian_wolf_idle*_x1.sld",
        ],
        2,
        "Wolves (danger)",
    ),
    (
        "gold_mine",
        [
            "n_*gold*_x1.sld",
        ],
        2,
        "Gold",
    ),
    (
        "stone_mine",
        [
            "n_*stone*_x1.sld",
        ],
        2,
        "Stone",
    ),
    (
        "berry_bush",
        [
            "n_*berry*_x1.sld",
            "n_*forage*_x1.sld",
        ],
        2,
        "Berries",
    ),
    (
        "tree",
        [
            "n_tree_oak*_x1.sld",
            "n_tree_pine*_x1.sld",
            "n_tree_palm*_x1.sld",
            "n_tree_jungle*_x1.sld",
        ],
        4,
        "Trees (wood)",
    ),
    (
        "relic",
        [
            "n_relic*_x1.sld",
            "*relic*_x1.sld",
        ],
        2,
        "Relics",
    ),
]


def find_matching_files(
    game_dir: Path,
    patterns: list[str],
    max_count: int,
    exclude_substrings: list[str] | None = None,
) -> list[Path]:
    """Find SLD files matching patterns.

    Args:
        game_dir: Directory to search in.
        patterns: Glob patterns or exact filenames.
        max_count: Maximum files to return.
        exclude_substrings: If provided, skip files whose name contains
            any of these substrings (e.g. "destruction", "rubble").
    """
    matches = []

    for pattern in patterns:
        if "*" in pattern:
            found = list(game_dir.glob(pattern))
        else:
            exact = game_dir / pattern
            found = [exact] if exact.exists() else []

        for f in sorted(found):  # Sort for consistency
            if f not in matches and f.suffix == ".sld":
                # Apply exclusion filter
                if exclude_substrings:
                    name_lower = f.name.lower()
                    if any(ex in name_lower for ex in exclude_substrings):
                        continue
                matches.append(f)

    # Shuffle to get style diversity instead of alphabetical bias
    if len(matches) > max_count:
        import random as _rng

        _rng.seed(42)  # Deterministic for reproducibility
        _rng.shuffle(matches)

    return matches[:max_count]


def extract_sprites(
    game_graphics_dir: str,
    output_dir: str,
    verbose: bool = True,
    extract_multiple_frames_flag: bool = False,
    apply_player_colors: bool = False,
    frame_indices: list[int] | None = None,
) -> dict:
    """Extract sprites for all configured classes.

    Args:
        game_graphics_dir: Path to game graphics directory containing .sld files
        output_dir: Output directory for extracted PNG sprites
        verbose: Whether to print detailed progress
        extract_multiple_frames_flag: If True, extract multiple animation frames per sprite
        apply_player_colors: If True, apply random player colors to unit sprites
        frame_indices: List of frame indices to extract (default: [0, 4, 8, 12])

    Returns:
        Statistics dictionary with extraction results
    """
    game_dir = Path(game_graphics_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if frame_indices is None:
        frame_indices = DEFAULT_FRAMES_TO_EXTRACT

    if not game_dir.exists():
        print(f"Error: Directory not found: {game_dir}")
        return {"error": "Directory not found"}

    # Classes that should have player colors applied (units, not buildings/resources)
    PLAYER_COLOR_CLASSES = {
        "villager",
        "trade_cart",
        "fishing_ship",
        "scout_line",
        "knight_line",
        "camel_line",
        "battle_elephant",
        "archer_line",
        "skirmisher_line",
        "cavalry_archer",
        "hand_cannoneer",
        "militia_line",
        "spearman_line",
        "eagle_line",
        "ram",
        "mangonel_line",
        "scorpion",
        "trebuchet",
        "monk",
        "king",
        "unique_archer",
        "unique_cavalry",
        "unique_infantry",
        "unique_siege",
        "unique_ship",
    }

    stats = {
        "total_extracted": 0,
        "total_failed": 0,
        "classes": 0,
        "by_class": {},
    }

    print(f"{'=' * 60}")
    print("Extracting AoE2 Sprites for YOLO Training (v2)")
    print(f"{'=' * 60}")
    print(f"Source: {game_dir}")
    print(f"Output: {out_dir}")
    print(f"Classes configured: {len(SPRITE_CONFIG)}")
    print(f"Multiple frames: {extract_multiple_frames_flag}")
    print(f"Player colors: {apply_player_colors}")
    if extract_multiple_frames_flag:
        print(f"Frame indices: {frame_indices}")
    print(f"{'=' * 60}\n")

    for class_name, patterns, max_variants, description in SPRITE_CONFIG:
        # Apply exclusion filter for building sprites to skip destruction/rubble/shadow files
        is_building = any(p.startswith("b_") for p in patterns)
        matches = find_matching_files(
            game_dir,
            patterns,
            max_variants,
            exclude_substrings=EXCLUDE_SUBSTRINGS if is_building else None,
        )

        if verbose:
            print(f"{class_name} ({description}):")

        if not matches:
            if verbose:
                print("  ⚠ No files found")
            stats["by_class"][class_name] = {"found": 0, "extracted": 0}
            continue

        # Determine if this class should have player colors
        should_apply_color = apply_player_colors and class_name in PLAYER_COLOR_CLASSES

        extracted = 0
        for i, sld_file in enumerate(matches):
            try:
                if extract_multiple_frames_flag:
                    # Extract multiple animation frames
                    count = extract_multiple_frames(
                        str(sld_file),
                        str(out_dir),
                        class_name,
                        variant_idx=i,
                        frame_indices=frame_indices,
                        apply_player_color=should_apply_color,
                    )
                    if count > 0:
                        extracted += count
                        stats["total_extracted"] += count
                        if verbose:
                            print(f"  ✓ {sld_file.name} ({count} frames)")
                    else:
                        stats["total_failed"] += 1
                        if verbose:
                            print(f"  ✗ {sld_file.name} (no frames)")
                else:
                    # Extract single frame (original behavior)
                    out_file = out_dir / f"{class_name}_{i:02d}.png"
                    success = extract_sprite(
                        str(sld_file),
                        str(out_file),
                        apply_player_color=should_apply_color,
                    )
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
            "description": description,
        }

    # Summary
    print(f"\n{'=' * 60}")
    print("EXTRACTION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Classes with sprites: {stats['classes']}/{len(SPRITE_CONFIG)}")
    print(f"Total sprites extracted: {stats['total_extracted']}")
    print(f"Total failed: {stats['total_failed']}")
    print(f"Output directory: {out_dir}")

    return stats


def print_config():
    """Print the extraction configuration."""
    print(f"\n{'=' * 60}")
    print("SPRITE EXTRACTION CONFIGURATION")
    print(f"{'=' * 60}")

    categories = {
        "Economic Units": ["villager", "trade_cart", "fishing_ship"],
        "Cavalry": ["scout_line", "knight_line", "camel_line", "battle_elephant"],
        "Archers": ["archer_line", "skirmisher_line", "cavalry_archer", "hand_cannoneer"],
        "Infantry": ["militia_line", "spearman_line", "eagle_line"],
        "Siege": ["ram", "mangonel_line", "scorpion", "trebuchet"],
        "Special": ["monk", "king"],
        "Unique Units": [
            "unique_archer",
            "unique_cavalry",
            "unique_infantry",
            "unique_siege",
            "unique_ship",
        ],
        "Economy Buildings": [
            "town_center",
            "house",
            "mill",
            "lumber_camp",
            "mining_camp",
            "farm",
            "market",
            "blacksmith",
        ],
        "Military Buildings": [
            "barracks",
            "archery_range",
            "stable",
            "siege_workshop",
            "monastery",
            "castle",
        ],
        "Defense": ["tower", "wall", "gate"],
        "Resources": [
            "sheep",
            "deer",
            "boar",
            "wolf",
            "gold_mine",
            "stone_mine",
            "berry_bush",
            "tree",
            "relic",
        ],
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
        description="Extract comprehensive sprite set for YOLO training (v2 with animation frames and player colors)"
    )
    parser.add_argument(
        "--game-dir", "-g", default="game_graphics", help="Path to game_graphics directory"
    )
    parser.add_argument("--output", "-o", default="tmp/sprites", help="Output directory")
    parser.add_argument(
        "--show-config", "-c", action="store_true", help="Show extraction configuration and exit"
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="Reduce output")
    parser.add_argument(
        "--multi-frame",
        "-m",
        action="store_true",
        help="Extract multiple animation frames per sprite (v2 feature)",
    )
    parser.add_argument(
        "--player-colors",
        "-p",
        action="store_true",
        help="Apply random player colors to unit sprites (v2 feature)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        nargs="+",
        default=[0, 4, 8, 12],
        help="Frame indices to extract when using --multi-frame (default: 0 4 8 12)",
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
        verbose=not args.quiet,
        extract_multiple_frames_flag=args.multi_frame,
        apply_player_colors=args.player_colors,
        frame_indices=args.frames,
    )

    return 0 if not stats.get("error") else 1


if __name__ == "__main__":
    exit(main())
