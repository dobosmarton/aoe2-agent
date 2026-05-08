#!/usr/bin/env python3
"""
Training Data Generator for AoE2 YOLO Model

Generates synthetic training data by compositing extracted game sprites
onto backgrounds with automatic bounding box labels.

Usage:
    python generate_training_data.py --num-images 1000 --output training_data

The generated dataset can be used directly with YOLOv8:
    yolo train model=yolov8n.pt data=training_data/dataset.yaml epochs=100
"""

import argparse
import io
import random
from dataclasses import dataclass
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
except ImportError:
    print("Error: Pillow required. Install with: pip install Pillow")
    exit(1)


@dataclass
class SpriteConfig:
    """Configuration for a sprite class."""

    class_id: int
    class_name: str
    sprite_patterns: list[str]  # Glob patterns to match sprite files
    scale_range: tuple[float, float] = (0.8, 1.2)
    count_range: tuple[int, int] = (1, 3)
    z_order: int = 0  # Higher = rendered on top
    # Placement constraints
    avoid_edges: bool = True
    min_spacing: int = 20  # Minimum pixels between same-class sprites


# Biome terrain color palettes for diverse synthetic backgrounds
BIOME_PALETTES = {
    "grass": {
        "colors": [(34, 89, 34), (46, 102, 46), (58, 115, 58), (72, 99, 52), (85, 107, 47)],
        "weight": 0.25,
    },
    "desert": {
        "colors": [
            (194, 178, 128),
            (210, 190, 140),
            (180, 160, 110),
            (220, 200, 150),
            (170, 150, 100),
        ],
        "weight": 0.15,
    },
    "snow": {
        "colors": [
            (220, 225, 230),
            (200, 210, 220),
            (235, 238, 240),
            (190, 200, 210),
            (210, 215, 225),
        ],
        "weight": 0.10,
    },
    "autumn": {
        "colors": [(120, 100, 40), (140, 110, 50), (100, 85, 35), (130, 95, 45), (110, 90, 38)],
        "weight": 0.10,
    },
    "jungle": {
        "colors": [(20, 70, 20), (30, 80, 25), (15, 60, 15), (25, 75, 22), (35, 85, 30)],
        "weight": 0.10,
    },
    "dirt": {
        "colors": [(139, 110, 72), (150, 120, 80), (125, 100, 65), (160, 130, 90), (145, 115, 75)],
        "weight": 0.10,
    },
    "water_shore": {
        "colors": [(60, 100, 120), (50, 90, 110), (70, 110, 130), (45, 85, 105), (55, 95, 115)],
        "weight": 0.05,
    },
    "dark_forest": {
        "colors": [(20, 50, 20), (30, 60, 30), (15, 45, 18), (25, 55, 25), (18, 48, 22)],
        "weight": 0.05,
    },
    "mixed": {
        "colors": None,  # Randomly merges 2-3 other biome palettes
        "weight": 0.10,
    },
}


# Sprite configurations using classes.yaml IDs directly (60 classes)
# Organized by category with appropriate z-order and spawn rates
SPRITE_CONFIGS = [
    # =========================================================================
    # RESOURCES & NATURE (rendered first, bottom layer)
    # =========================================================================
    SpriteConfig(
        class_id=0,
        class_name="tree",
        sprite_patterns=["tree_*.png"],
        scale_range=(0.8, 1.2),
        count_range=(4, 12),
        z_order=0,
    ),
    SpriteConfig(
        class_id=1,
        class_name="gold_mine",
        sprite_patterns=["gold_mine_*.png"],
        scale_range=(0.9, 1.1),
        count_range=(0, 2),
        z_order=0,
    ),
    SpriteConfig(
        class_id=2,
        class_name="stone_mine",
        sprite_patterns=["stone_mine_*.png"],
        scale_range=(0.9, 1.1),
        count_range=(0, 2),
        z_order=0,
    ),
    SpriteConfig(
        class_id=3,
        class_name="berry_bush",
        sprite_patterns=["berry_bush_*.png"],
        scale_range=(0.9, 1.1),
        count_range=(0, 4),
        z_order=0,
    ),
    SpriteConfig(
        class_id=4,
        class_name="relic",
        sprite_patterns=["relic_*.png"],
        scale_range=(0.9, 1.1),
        count_range=(0, 1),
        z_order=0,
    ),
    # =========================================================================
    # ANIMALS (z_order 0 - ground layer)
    # =========================================================================
    SpriteConfig(
        class_id=5,
        class_name="deer",
        sprite_patterns=["deer_*.png"],
        scale_range=(0.9, 1.1),
        count_range=(0, 3),
        z_order=0,
    ),
    SpriteConfig(
        class_id=6,
        class_name="boar",
        sprite_patterns=["boar_*.png"],
        scale_range=(0.9, 1.1),
        count_range=(0, 2),
        z_order=0,
    ),
    SpriteConfig(
        class_id=7,
        class_name="wolf",
        sprite_patterns=["wolf_*.png"],
        scale_range=(0.9, 1.1),
        count_range=(0, 2),
        z_order=0,
    ),
    SpriteConfig(
        class_id=8,
        class_name="sheep",
        sprite_patterns=["sheep_*.png"],
        scale_range=(0.9, 1.1),
        count_range=(2, 5),
        z_order=0,
    ),
    # =========================================================================
    # ECONOMY BUILDINGS (z_order 1)
    # =========================================================================
    SpriteConfig(
        class_id=9,
        class_name="town_center",
        sprite_patterns=["town_center_*.png"],
        scale_range=(0.9, 1.1),
        count_range=(1, 1),
        z_order=1,
    ),
    SpriteConfig(
        class_id=10,
        class_name="house",
        sprite_patterns=["house_*.png"],
        scale_range=(0.85, 1.15),
        count_range=(0, 4),
        z_order=1,
    ),
    SpriteConfig(
        class_id=11,
        class_name="lumber_camp",
        sprite_patterns=["lumber_camp_*.png"],
        scale_range=(0.9, 1.1),
        count_range=(0, 1),
        z_order=1,
    ),
    SpriteConfig(
        class_id=12,
        class_name="mining_camp",
        sprite_patterns=["mining_camp_*.png"],
        scale_range=(0.9, 1.1),
        count_range=(0, 1),
        z_order=1,
    ),
    SpriteConfig(
        class_id=13,
        class_name="mill",
        sprite_patterns=["mill_*.png"],
        scale_range=(0.9, 1.1),
        count_range=(0, 1),
        z_order=1,
    ),
    SpriteConfig(
        class_id=14,
        class_name="market",
        sprite_patterns=["market_*.png"],
        scale_range=(0.9, 1.1),
        count_range=(0, 1),
        z_order=1,
    ),
    SpriteConfig(
        class_id=15,
        class_name="dock",
        sprite_patterns=["dock_*.png"],
        scale_range=(0.9, 1.1),
        count_range=(0, 1),
        z_order=1,
    ),
    # =========================================================================
    # MILITARY BUILDINGS
    # =========================================================================
    SpriteConfig(
        class_id=17,
        class_name="barracks",
        sprite_patterns=["barracks_*.png"],
        scale_range=(0.9, 1.1),
        count_range=(0, 1),
        z_order=1,
    ),
    SpriteConfig(
        class_id=18,
        class_name="archery_range",
        sprite_patterns=["archery_range_*.png"],
        scale_range=(0.9, 1.1),
        count_range=(0, 1),
        z_order=1,
    ),
    SpriteConfig(
        class_id=19,
        class_name="stable",
        sprite_patterns=["stable_*.png"],
        scale_range=(0.9, 1.1),
        count_range=(0, 1),
        z_order=1,
    ),
    SpriteConfig(
        class_id=20,
        class_name="blacksmith",
        sprite_patterns=["blacksmith_*.png"],
        scale_range=(0.9, 1.1),
        count_range=(0, 1),
        z_order=1,
    ),
    SpriteConfig(
        class_id=21,
        class_name="siege_workshop",
        sprite_patterns=["siege_workshop_*.png"],
        scale_range=(0.9, 1.1),
        count_range=(0, 1),
        z_order=1,
    ),
    SpriteConfig(
        class_id=22,
        class_name="monastery",
        sprite_patterns=["monastery_*.png"],
        scale_range=(0.9, 1.1),
        count_range=(0, 1),
        z_order=1,
    ),
    SpriteConfig(
        class_id=23,
        class_name="castle",
        sprite_patterns=["castle_*.png"],
        scale_range=(0.9, 1.1),
        count_range=(0, 1),
        z_order=1,
    ),
    SpriteConfig(
        class_id=24,
        class_name="university",
        sprite_patterns=["university_*.png"],
        scale_range=(0.9, 1.1),
        count_range=(0, 1),
        z_order=1,
    ),
    # =========================================================================
    # DEFENSE STRUCTURES
    # =========================================================================
    SpriteConfig(
        class_id=25,
        class_name="gate",
        sprite_patterns=["gate_*.png"],
        scale_range=(0.9, 1.1),
        count_range=(0, 1),
        z_order=1,
    ),
    SpriteConfig(
        class_id=26,
        class_name="wall",
        sprite_patterns=["wall_*.png"],
        scale_range=(0.9, 1.1),
        count_range=(0, 2),
        z_order=1,
    ),
    SpriteConfig(
        class_id=27,
        class_name="tower",
        sprite_patterns=["tower_*.png"],
        scale_range=(0.9, 1.1),
        count_range=(0, 2),
        z_order=1,
    ),
    # =========================================================================
    # SPECIAL BUILDINGS
    # =========================================================================
    SpriteConfig(
        class_id=28,
        class_name="wonder",
        sprite_patterns=["wonder_*.png"],
        scale_range=(0.9, 1.1),
        count_range=(0, 1),
        z_order=1,
    ),
    # =========================================================================
    # CIVILIAN UNITS (z_order 3 - top layer)
    # =========================================================================
    SpriteConfig(
        class_id=30,
        class_name="villager",
        sprite_patterns=["villager_*.png"],
        scale_range=(0.9, 1.1),
        count_range=(3, 8),
        z_order=3,
    ),
    SpriteConfig(
        class_id=31,
        class_name="trade_cart",
        sprite_patterns=["trade_cart_*.png"],
        scale_range=(0.9, 1.1),
        count_range=(0, 2),
        z_order=3,
    ),
    SpriteConfig(
        class_id=32,
        class_name="fishing_ship",
        sprite_patterns=["fishing_ship_*.png"],
        scale_range=(0.9, 1.1),
        count_range=(0, 1),
        z_order=3,
    ),
    # =========================================================================
    # CAVALRY UNITS
    # =========================================================================
    SpriteConfig(
        class_id=33,
        class_name="scout_line",
        sprite_patterns=["scout_line_*.png"],
        scale_range=(0.9, 1.1),
        count_range=(0, 1),
        z_order=3,
    ),
    SpriteConfig(
        class_id=34,
        class_name="knight_line",
        sprite_patterns=["knight_line_*.png"],
        scale_range=(0.9, 1.1),
        count_range=(0, 3),
        z_order=3,
    ),
    SpriteConfig(
        class_id=35,
        class_name="camel_line",
        sprite_patterns=["camel_line_*.png"],
        scale_range=(0.9, 1.1),
        count_range=(0, 2),
        z_order=3,
    ),
    SpriteConfig(
        class_id=36,
        class_name="battle_elephant",
        sprite_patterns=["battle_elephant_*.png"],
        scale_range=(0.9, 1.1),
        count_range=(0, 1),
        z_order=3,
    ),
    # =========================================================================
    # ARCHER UNITS
    # =========================================================================
    SpriteConfig(
        class_id=37,
        class_name="archer_line",
        sprite_patterns=["archer_line_*.png"],
        scale_range=(0.9, 1.1),
        count_range=(0, 5),
        z_order=3,
    ),
    SpriteConfig(
        class_id=38,
        class_name="skirmisher_line",
        sprite_patterns=["skirmisher_line_*.png"],
        scale_range=(0.9, 1.1),
        count_range=(0, 3),
        z_order=3,
    ),
    SpriteConfig(
        class_id=39,
        class_name="cavalry_archer",
        sprite_patterns=["cavalry_archer_*.png"],
        scale_range=(0.9, 1.1),
        count_range=(0, 2),
        z_order=3,
    ),
    SpriteConfig(
        class_id=40,
        class_name="hand_cannoneer",
        sprite_patterns=["hand_cannoneer_*.png"],
        scale_range=(0.9, 1.1),
        count_range=(0, 2),
        z_order=3,
    ),
    # =========================================================================
    # INFANTRY UNITS
    # =========================================================================
    SpriteConfig(
        class_id=41,
        class_name="militia_line",
        sprite_patterns=["militia_line_*.png"],
        scale_range=(0.9, 1.1),
        count_range=(0, 5),
        z_order=3,
    ),
    SpriteConfig(
        class_id=42,
        class_name="spearman_line",
        sprite_patterns=["spearman_line_*.png"],
        scale_range=(0.9, 1.1),
        count_range=(0, 4),
        z_order=3,
    ),
    SpriteConfig(
        class_id=43,
        class_name="eagle_line",
        sprite_patterns=["eagle_line_*.png"],
        scale_range=(0.9, 1.1),
        count_range=(0, 3),
        z_order=3,
    ),
    # =========================================================================
    # SIEGE UNITS
    # =========================================================================
    SpriteConfig(
        class_id=44,
        class_name="ram",
        sprite_patterns=["ram_*.png"],
        scale_range=(0.9, 1.1),
        count_range=(0, 1),
        z_order=3,
    ),
    SpriteConfig(
        class_id=45,
        class_name="mangonel_line",
        sprite_patterns=["mangonel_line_*.png"],
        scale_range=(0.9, 1.1),
        count_range=(0, 1),
        z_order=3,
    ),
    SpriteConfig(
        class_id=46,
        class_name="scorpion",
        sprite_patterns=["scorpion_*.png"],
        scale_range=(0.9, 1.1),
        count_range=(0, 1),
        z_order=3,
    ),
    SpriteConfig(
        class_id=47,
        class_name="trebuchet",
        sprite_patterns=["trebuchet_*.png"],
        scale_range=(0.9, 1.1),
        count_range=(0, 1),
        z_order=3,
    ),
    # =========================================================================
    # SPECIAL UNITS
    # =========================================================================
    SpriteConfig(
        class_id=48,
        class_name="monk",
        sprite_patterns=["monk_*.png"],
        scale_range=(0.9, 1.1),
        count_range=(0, 2),
        z_order=3,
    ),
    SpriteConfig(
        class_id=49,
        class_name="king",
        sprite_patterns=["king_*.png"],
        scale_range=(0.9, 1.1),
        count_range=(0, 1),
        z_order=3,
    ),
    # =========================================================================
    # UNIQUE UNITS - Grouped by type (matching classes.yaml ids 50-54)
    # =========================================================================
    SpriteConfig(
        class_id=50,
        class_name="unique_archer",
        sprite_patterns=["unique_archer_*.png"],
        scale_range=(0.9, 1.1),
        count_range=(0, 3),
        z_order=3,
    ),
    SpriteConfig(
        class_id=51,
        class_name="unique_cavalry",
        sprite_patterns=["unique_cavalry_*.png"],
        scale_range=(0.9, 1.1),
        count_range=(0, 2),
        z_order=3,
    ),
    SpriteConfig(
        class_id=52,
        class_name="unique_infantry",
        sprite_patterns=["unique_infantry_*.png"],
        scale_range=(0.9, 1.1),
        count_range=(0, 3),
        z_order=3,
    ),
    SpriteConfig(
        class_id=53,
        class_name="unique_siege",
        sprite_patterns=["unique_siege_*.png"],
        scale_range=(0.9, 1.1),
        count_range=(0, 1),
        z_order=3,
    ),
    SpriteConfig(
        class_id=54,
        class_name="unique_ship",
        sprite_patterns=["unique_ship_*.png"],
        scale_range=(0.9, 1.1),
        count_range=(0, 1),
        z_order=3,
    ),
]


class TrainingDataGenerator:
    """Generates synthetic training images with YOLO labels."""

    def __init__(
        self,
        sprites_dir: Path,
        output_dir: Path,
        backgrounds_dir: Path | None = None,
        real_screenshots_dir: Path | None = None,
        image_size: tuple[int, int] = (1280, 720),
        configs: list[SpriteConfig] | None = None,
        real_background_ratio: float = 0.5,  # 50% real backgrounds
        enable_enhanced_augmentations: bool = True,
    ):
        self.sprites_dir = Path(sprites_dir)
        self.output_dir = Path(output_dir)
        self.backgrounds_dir = Path(backgrounds_dir) if backgrounds_dir else None
        self.real_screenshots_dir = Path(real_screenshots_dir) if real_screenshots_dir else None
        self.image_size = image_size
        self.configs = configs or SPRITE_CONFIGS
        self.real_background_ratio = real_background_ratio
        self.enable_enhanced_augmentations = enable_enhanced_augmentations

        # Create output structure
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load assets
        self.sprites: dict[str, list[Image.Image]] = {}
        self.backgrounds: list[Image.Image] = []
        self.real_backgrounds: list[Image.Image] = []
        self._load_sprites()
        self._load_backgrounds()
        self._load_real_backgrounds()

    def _load_sprites(self):
        """Load sprite images for each class."""
        for config in self.configs:
            self.sprites[config.class_name] = []
            for pattern in config.sprite_patterns:
                for path in self.sprites_dir.glob(pattern):
                    try:
                        img = Image.open(path).convert("RGBA")
                        self.sprites[config.class_name].append(img)
                    except Exception as e:
                        print(f"Warning: Failed to load {path}: {e}")

            count = len(self.sprites[config.class_name])
            if count == 0:
                print(f"Warning: No sprites found for {config.class_name}")
            else:
                print(f"Loaded {count} sprites for {config.class_name}")

    def _load_backgrounds(self):
        """Load background images."""
        if self.backgrounds_dir and self.backgrounds_dir.exists():
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                for path in self.backgrounds_dir.glob(ext):
                    try:
                        img = Image.open(path).convert("RGB")
                        self.backgrounds.append(img)
                    except Exception as e:
                        print(f"Warning: Failed to load background {path}: {e}")

        print(f"Loaded {len(self.backgrounds)} background images")

    def _load_real_backgrounds(self):
        """Load real game screenshots as backgrounds."""
        if self.real_screenshots_dir and self.real_screenshots_dir.exists():
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                for path in self.real_screenshots_dir.glob(ext):
                    try:
                        img = Image.open(path).convert("RGB")
                        # Optionally blur slightly to reduce overfitting to specific screenshots
                        img = img.filter(ImageFilter.GaussianBlur(radius=1))
                        self.real_backgrounds.append(img)
                    except Exception as e:
                        print(f"Warning: Failed to load real background {path}: {e}")

        print(f"Loaded {len(self.real_backgrounds)} real game screenshots as backgrounds")

    def _create_background(self) -> Image.Image:
        """Create or select a background image.

        Mixes real game screenshots with synthetic backgrounds based on
        real_background_ratio for improved model generalization.
        """
        # Decide whether to use real background
        use_real = self.real_backgrounds and random.random() < self.real_background_ratio

        if use_real:
            bg = random.choice(self.real_backgrounds).copy()
            bg = bg.resize(self.image_size, Image.Resampling.LANCZOS)
        elif self.backgrounds:
            bg = random.choice(self.backgrounds).copy()
            bg = bg.resize(self.image_size, Image.Resampling.LANCZOS)
        else:
            # Generate terrain-like background
            bg = self._generate_terrain_background()

        return bg

    def _select_biome(self) -> str:
        """Select a biome weighted by probability."""
        biomes = list(BIOME_PALETTES.keys())
        weights = [BIOME_PALETTES[b]["weight"] for b in biomes]
        return random.choices(biomes, weights=weights, k=1)[0]

    def _get_biome_colors(self, biome: str) -> list[tuple[int, int, int]]:
        """Get color palette for a biome, handling 'mixed' biome."""
        if biome == "mixed":
            other = [b for b in BIOME_PALETTES if b != "mixed" and BIOME_PALETTES[b]["colors"]]
            chosen = random.sample(other, k=random.randint(2, 3))
            colors = []
            for b in chosen:
                colors.extend(BIOME_PALETTES[b]["colors"])
            return colors
        return BIOME_PALETTES[biome]["colors"]

    def _generate_terrain_background(self) -> Image.Image:
        """Generate a biome-diverse terrain background."""
        bg = Image.new("RGB", self.image_size)

        biome = self._select_biome()
        colors = self._get_biome_colors(biome)

        # Store base color for padding in scale zoom
        self._current_biome_base_color = colors[0]

        draw = ImageDraw.Draw(bg)

        # Large patches
        for _ in range(20):
            color = random.choice(colors)
            x = random.randint(-100, self.image_size[0])
            y = random.randint(-100, self.image_size[1])
            w = random.randint(200, 500)
            h = random.randint(200, 500)
            draw.ellipse([x, y, x + w, y + h], fill=color)

        # Apply slight blur for smoother terrain
        return bg.filter(ImageFilter.GaussianBlur(radius=3))

    def _check_overlap(
        self,
        new_box: tuple[int, int, int, int],
        placed_boxes: list[tuple[int, int, int, int]],
        min_overlap: float = 0.3,
    ) -> bool:
        """Check if new box overlaps too much with existing boxes."""
        x1, y1, x2, y2 = new_box
        new_area = (x2 - x1) * (y2 - y1)

        for px1, py1, px2, py2 in placed_boxes:
            # Calculate intersection
            ix1 = max(x1, px1)
            iy1 = max(y1, py1)
            ix2 = min(x2, px2)
            iy2 = min(y2, py2)

            if ix1 < ix2 and iy1 < iy2:
                intersection = (ix2 - ix1) * (iy2 - iy1)
                overlap_ratio = intersection / new_area
                if overlap_ratio > min_overlap:
                    return True

        return False

    def generate_image(self, index: int) -> tuple[Image.Image, list[str]]:
        """Generate a single training image with labels.

        Returns:
            Tuple of (image, list of YOLO format labels)
        """
        bg = self._create_background()
        placed_boxes = []
        # Track (box, z_order, label) for post-rendering occlusion filtering
        placed_items: list[tuple[tuple[int, int, int, int], int, str]] = []

        # Sort configs by z_order
        sorted_configs = sorted(self.configs, key=lambda c: c.z_order)

        for config in sorted_configs:
            sprites = self.sprites.get(config.class_name, [])
            if not sprites:
                continue

            count = random.randint(*config.count_range)

            for _ in range(count):
                sprite = random.choice(sprites).copy()

                # Apply random scale
                scale = random.uniform(*config.scale_range)
                new_w = max(10, int(sprite.width * scale))
                new_h = max(10, int(sprite.height * scale))
                sprite = sprite.resize((new_w, new_h), Image.Resampling.LANCZOS)

                # Calculate valid placement area
                margin = 20 if config.avoid_edges else 0
                max_x = self.image_size[0] - new_w - margin
                max_y = self.image_size[1] - new_h - margin

                if max_x <= margin or max_y <= margin:
                    continue

                # Try to find non-overlapping position
                # Buildings never overlap in-game (grid); resources rarely; units can group
                overlap_thresholds = {0: 0.15, 1: 0.10, 3: 0.35}
                max_overlap = overlap_thresholds.get(config.z_order, 0.3)
                max_attempts = 50 if config.z_order <= 1 else 20
                for _ in range(max_attempts):
                    x = random.randint(margin, max_x)
                    y = random.randint(margin, max_y)

                    box = (x, y, x + new_w, y + new_h)
                    if not self._check_overlap(box, placed_boxes, min_overlap=max_overlap):
                        break
                else:
                    # Skip — better to have fewer entities than unrealistic overlap
                    continue

                # Paste sprite
                bg.paste(sprite, (x, y), sprite)
                placed_boxes.append(box)

                # Create YOLO label (normalized coordinates)
                x_center = (x + new_w / 2) / self.image_size[0]
                y_center = (y + new_h / 2) / self.image_size[1]
                norm_w = new_w / self.image_size[0]
                norm_h = new_h / self.image_size[1]

                label = f"{config.class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"
                placed_items.append((box, config.z_order, label))

        # Filter labels for sprites occluded by higher z_order sprites
        labels = []
        for i, (box_i, z_i, label_i) in enumerate(placed_items):
            bx1, by1, bx2, by2 = box_i
            box_area = (bx2 - bx1) * (by2 - by1)
            if box_area <= 0:
                continue

            total_occluded = 0.0
            for j, (box_j, z_j, _) in enumerate(placed_items):
                if j == i or z_j <= z_i:
                    continue
                ox1, oy1, ox2, oy2 = box_j
                ix1, iy1 = max(bx1, ox1), max(by1, oy1)
                ix2, iy2 = min(bx2, ox2), min(by2, oy2)
                if ix1 < ix2 and iy1 < iy2:
                    total_occluded += (ix2 - ix1) * (iy2 - iy1)

            if total_occluded / box_area < 0.5:
                labels.append(label_i)

        # Apply augmentations (non-spatial, but may occlude regions)
        bg, occluded = self._augment(bg)

        # Filter labels hidden by fog/UI overlays
        labels = self._filter_occluded_labels(labels, occluded)

        # Apply scale zoom (spatial — needs label update)
        if self.enable_enhanced_augmentations:
            bg, labels = self._apply_scale_zoom(bg, labels)

        return bg, labels

    def _augment(self, image: Image.Image) -> tuple[Image.Image, list[tuple[int, int, int, int]]]:
        """Apply random augmentations including enhanced game-realistic effects.

        Returns:
            Tuple of (augmented image, list of occluded pixel rectangles).
        """
        # Basic augmentations (always applied)
        # Brightness
        if random.random() < 0.5:
            factor = random.uniform(0.7, 1.3)
            image = ImageEnhance.Brightness(image).enhance(factor)

        # Contrast
        if random.random() < 0.5:
            factor = random.uniform(0.8, 1.2)
            image = ImageEnhance.Contrast(image).enhance(factor)

        # Saturation
        if random.random() < 0.3:
            factor = random.uniform(0.8, 1.2)
            image = ImageEnhance.Color(image).enhance(factor)

        # Slight blur (simulates motion/focus)
        if random.random() < 0.2:
            image = image.filter(ImageFilter.GaussianBlur(radius=0.5))

        # Enhanced augmentations (for v2 training)
        occluded_rects: list[tuple[int, int, int, int]] = []
        if self.enable_enhanced_augmentations:
            image, occluded_rects = self._apply_enhanced_augmentations(image)

        return image, occluded_rects

    def _apply_enhanced_augmentations(
        self, image: Image.Image
    ) -> tuple[Image.Image, list[tuple[int, int, int, int]]]:
        """Apply realistic game-like augmentations.

        Includes fog of war, UI elements, compression artifacts, and scale variation.

        Returns:
            Tuple of (augmented image, list of occluded pixel rects as (x1,y1,x2,y2)).
        """
        occluded_rects: list[tuple[int, int, int, int]] = []

        # Ensure we're working with RGB for augmentations
        if image.mode == "RGBA":
            # Create a composite with white background
            bg = Image.new("RGB", image.size, (255, 255, 255))
            bg.paste(image, mask=image.split()[3])
            image = bg
        elif image.mode != "RGB":
            image = image.convert("RGB")

        # 1. Fog of war overlay (simulates unexplored areas)
        if random.random() < 0.3:
            fog_overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
            fog_draw = ImageDraw.Draw(fog_overlay)

            # Random fog patches
            num_patches = random.randint(1, 4)
            for _ in range(num_patches):
                # Random corner or edge
                x = random.choice([0, image.width - 300])
                y = random.choice([0, image.height - 200])
                w = random.randint(200, 400)
                h = random.randint(150, 300)
                opacity = random.randint(80, 150)
                fog_draw.rectangle([x, y, x + w, y + h], fill=(0, 0, 0, opacity))

                # Track high-opacity patches that truly hide objects
                if opacity >= 120:
                    occluded_rects.append((x, y, x + w, y + h))

            image = Image.alpha_composite(image.convert("RGBA"), fog_overlay)
            image = image.convert("RGB")

        # 2. UI-like elements overlay
        if random.random() < 0.2:
            draw = ImageDraw.Draw(image)

            # Simulate minimap corner (bottom-right or top-right)
            minimap_size = random.randint(130, 180)
            if random.random() < 0.5:
                # Bottom-right
                mm_rect = (
                    image.width - minimap_size,
                    image.height - minimap_size,
                    image.width,
                    image.height,
                )
                draw.rectangle([mm_rect[0], mm_rect[1], mm_rect[2], mm_rect[3]], fill=(30, 30, 30))
            else:
                # Top-right (for some UI configs)
                mm_rect = (
                    image.width - minimap_size,
                    0,
                    image.width,
                    minimap_size,
                )
                draw.rectangle([mm_rect[0], mm_rect[1], mm_rect[2], mm_rect[3]], fill=(25, 25, 25))
            occluded_rects.append(mm_rect)

            # Simulate resource bar (top of screen)
            if random.random() < 0.5:
                bar_height = random.randint(25, 40)
                draw.rectangle([0, 0, image.width, bar_height], fill=(20, 20, 20))
                occluded_rects.append((0, 0, image.width, bar_height))

        # 3. Compression artifacts (simulate JPEG compression)
        if random.random() < 0.3:
            buffer = io.BytesIO()
            quality = random.randint(70, 90)
            image.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            image = Image.open(buffer).convert("RGB")

        # 4. Scale variation — moved to _apply_scale_zoom() for label-aware transform

        # 5. Color temperature shift (simulates different map types)
        if random.random() < 0.2:
            # Shift towards warm (desert) or cool (winter) tones
            r, g, b = image.split()
            if random.random() < 0.5:
                # Warm shift
                r = r.point(lambda x: min(255, int(x * 1.1)))
                b = b.point(lambda x: int(x * 0.9))
            else:
                # Cool shift
                r = r.point(lambda x: int(x * 0.9))
                b = b.point(lambda x: min(255, int(x * 1.1)))
            image = Image.merge("RGB", (r, g, b))

        # 6. Vignette effect (darker edges)
        if random.random() < 0.15:
            vignette = Image.new("L", image.size, 255)
            vignette_draw = ImageDraw.Draw(vignette)
            # Radial gradient approximation
            center_x, center_y = image.width // 2, image.height // 2
            max_dist = (center_x**2 + center_y**2) ** 0.5
            for i in range(0, 255, 5):
                radius = int(max_dist * (255 - i) / 255)
                vignette_draw.ellipse(
                    [center_x - radius, center_y - radius, center_x + radius, center_y + radius],
                    fill=255 - i // 4,
                )
            image = Image.composite(image, Image.new("RGB", image.size, (0, 0, 0)), vignette)

        return image, occluded_rects

    def _filter_occluded_labels(
        self,
        labels: list[str],
        occluded_rects: list[tuple[int, int, int, int]],
    ) -> list[str]:
        """Remove labels where >50% of bbox is covered by fog/UI overlays."""
        if not occluded_rects:
            return labels

        W, H = self.image_size
        filtered = []
        for label in labels:
            parts = label.split()
            xc, yc, w, h = (
                float(parts[1]),
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
            )

            # Convert YOLO normalized to pixel coords
            bx1 = (xc - w / 2) * W
            by1 = (yc - h / 2) * H
            bx2 = (xc + w / 2) * W
            by2 = (yc + h / 2) * H
            box_area = (bx2 - bx1) * (by2 - by1)

            # Sum intersection with all occluded rects
            total_occluded = 0.0
            for ox1, oy1, ox2, oy2 in occluded_rects:
                ix1 = max(bx1, ox1)
                iy1 = max(by1, oy1)
                ix2 = min(bx2, ox2)
                iy2 = min(by2, oy2)
                if ix1 < ix2 and iy1 < iy2:
                    total_occluded += (ix2 - ix1) * (iy2 - iy1)

            if box_area > 0 and total_occluded / box_area < 0.5:
                filtered.append(label)

        return filtered

    def _apply_scale_zoom(
        self, image: Image.Image, labels: list[str]
    ) -> tuple[Image.Image, list[str]]:
        """Apply scale zoom to both image and labels.

        Unlike other augmentations, scale zoom is spatial and must transform
        the YOLO labels to match the new object positions.
        """
        if random.random() >= 0.3:
            return image, labels

        scale = random.uniform(0.7, 1.3)
        new_w = int(image.width * scale)
        new_h = int(image.height * scale)
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        if scale > 1:
            # Crop center
            left = (new_w - self.image_size[0]) // 2
            top = (new_h - self.image_size[1]) // 2
            image = image.crop((left, top, left + self.image_size[0], top + self.image_size[1]))
        else:
            # Pad with current biome color
            pad_color = getattr(self, "_current_biome_base_color", (45, 80, 45))
            padded = Image.new("RGB", self.image_size, pad_color)
            left = (self.image_size[0] - new_w) // 2
            top = (self.image_size[1] - new_h) // 2
            padded.paste(image, (left, top))
            image = padded

        # Transform labels: new_coord = coord * scale + (1 - scale) / 2
        offset = (1 - scale) / 2
        new_labels = []
        for label in labels:
            parts = label.split()
            class_id = parts[0]
            xc = float(parts[1])
            yc = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

            xc_new = xc * scale + offset
            yc_new = yc * scale + offset
            w_new = w * scale
            h_new = h * scale

            # Clip box to [0, 1]
            x1 = max(0.0, xc_new - w_new / 2)
            y1 = max(0.0, yc_new - h_new / 2)
            x2 = min(1.0, xc_new + w_new / 2)
            y2 = min(1.0, yc_new + h_new / 2)

            visible_w = x2 - x1
            visible_h = y2 - y1

            # Skip if box is outside frame or less than 30% visible
            if visible_w <= 0 or visible_h <= 0:
                continue
            if (visible_w * visible_h) / (w_new * h_new) < 0.3:
                continue

            # Recompute center from clipped box
            xc_clipped = (x1 + x2) / 2
            yc_clipped = (y1 + y2) / 2
            new_labels.append(
                f"{class_id} {xc_clipped:.6f} {yc_clipped:.6f} {visible_w:.6f} {visible_h:.6f}"
            )

        return image, new_labels

    def generate_dataset(
        self, num_images: int = 1000, train_split: float = 0.8, seed: int = 42
    ) -> Path:
        """Generate complete YOLO training dataset.

        Args:
            num_images: Total images to generate
            train_split: Fraction for training (rest for validation)
            seed: Random seed

        Returns:
            Path to dataset.yaml
        """
        random.seed(seed)

        # Create directories
        train_img_dir = self.output_dir / "train" / "images"
        train_lbl_dir = self.output_dir / "train" / "labels"
        val_img_dir = self.output_dir / "val" / "images"
        val_lbl_dir = self.output_dir / "val" / "labels"

        for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Generate images
        num_train = int(num_images * train_split)
        num_val = num_images - num_train

        print(f"\nGenerating {num_train} training images...")
        for i in range(num_train):
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{num_train}")

            img, labels = self.generate_image(i)
            img.save(train_img_dir / f"img_{i:05d}.jpg", quality=90)
            with (train_lbl_dir / f"img_{i:05d}.txt").open("w") as f:
                f.write("\n".join(labels))

        print(f"\nGenerating {num_val} validation images...")
        for i in range(num_val):
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{num_val}")

            img, labels = self.generate_image(num_train + i)
            img.save(val_img_dir / f"img_{i:05d}.jpg", quality=90)
            with (val_lbl_dir / f"img_{i:05d}.txt").open("w") as f:
                f.write("\n".join(labels))

        # Create dataset.yaml
        yaml_path = self._create_yaml()

        print(f"\n{'=' * 50}")
        print("Dataset generated successfully!")
        print(f"  Training images: {num_train}")
        print(f"  Validation images: {num_val}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Dataset config: {yaml_path}")
        print("\nTo train YOLOv8:")
        print(f"  yolo train model=yolov8n.pt data={yaml_path} epochs=100 imgsz=640")

        return yaml_path

    def _create_yaml(self) -> Path:
        """Create YOLO dataset configuration file."""
        # Get classes that have sprites
        active_classes = []
        for config in sorted(self.configs, key=lambda c: c.class_id):
            if self.sprites.get(config.class_name):
                active_classes.append(config)

        yaml_content = f"""# AoE2 Object Detection Dataset
# Generated by generate_training_data.py

path: {self.output_dir.absolute()}
train: train/images
val: val/images

# Classes
names:
"""
        for config in active_classes:
            yaml_content += f"  {config.class_id}: {config.class_name}\n"

        yaml_path = self.output_dir / "dataset.yaml"
        yaml_path.write_text(yaml_content)

        return yaml_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data for AoE2 YOLO model (v2 with enhanced augmentations)"
    )
    parser.add_argument(
        "--sprites",
        "-s",
        default="tmp/sprites",
        help="Directory containing extracted sprites (default: tmp/sprites)",
    )
    parser.add_argument(
        "--backgrounds",
        "-b",
        default=None,
        help="Directory containing background images (optional)",
    )
    parser.add_argument(
        "--real-screenshots",
        "-r",
        default=None,
        help="Directory containing real game screenshots for backgrounds (v2 feature)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="detection/training_data",
        help="Output directory for dataset (default: detection/training_data)",
    )
    parser.add_argument(
        "--num-images",
        "-n",
        type=int,
        default=1000,
        help="Number of images to generate (default: 1000)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[1280, 720],
        help="Image size as width height (default: 1280 720)",
    )
    parser.add_argument(
        "--train-split", type=float, default=0.8, help="Fraction for training set (default: 0.8)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--real-bg-ratio",
        type=float,
        default=0.5,
        help="Ratio of real backgrounds to use (0.0-1.0, default: 0.5)",
    )
    parser.add_argument(
        "--no-enhanced-aug",
        action="store_true",
        help="Disable enhanced augmentations (fog of war, UI, compression, etc.)",
    )

    args = parser.parse_args()

    # Resolve paths relative to script location
    script_dir = Path(__file__).parent.parent  # agent/
    sprites_dir = script_dir / args.sprites
    output_dir = script_dir / args.output
    backgrounds_dir = Path(args.backgrounds) if args.backgrounds else None
    real_screenshots_dir = Path(args.real_screenshots) if args.real_screenshots else None

    print("AoE2 YOLO Training Data Generator v2")
    print("=" * 50)
    print(f"Sprites directory: {sprites_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Image size: {args.image_size[0]}x{args.image_size[1]}")
    print(f"Number of images: {args.num_images}")
    print(f"Real background ratio: {args.real_bg_ratio:.0%}")
    print(f"Enhanced augmentations: {'disabled' if args.no_enhanced_aug else 'enabled'}")

    if real_screenshots_dir:
        print(f"Real screenshots directory: {real_screenshots_dir}")

    if not sprites_dir.exists():
        print(f"\nError: Sprites directory not found: {sprites_dir}")
        print("Run the SLD extractor first to extract sprites.")
        return 1

    generator = TrainingDataGenerator(
        sprites_dir=sprites_dir,
        output_dir=output_dir,
        backgrounds_dir=backgrounds_dir,
        real_screenshots_dir=real_screenshots_dir,
        image_size=tuple(args.image_size),
        real_background_ratio=args.real_bg_ratio,
        enable_enhanced_augmentations=not args.no_enhanced_aug,
    )

    generator.generate_dataset(
        num_images=args.num_images,
        train_split=args.train_split,
        seed=args.seed,
    )

    return 0


if __name__ == "__main__":
    exit(main())
