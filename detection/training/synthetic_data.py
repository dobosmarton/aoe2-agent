"""
Synthetic training data generator for AoE2 YOLO model.

Generates training images by compositing extracted sprite images onto
screenshot backgrounds, with auto-generated bounding box labels.
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import shutil


@dataclass
class SpriteConfig:
    """Configuration for a sprite type."""
    class_id: int           # YOLO class ID
    class_name: str         # Class name
    sprite_files: list[str] # List of sprite file paths
    scale_range: tuple[float, float] = (0.8, 1.2)  # Scale variation
    count_range: tuple[int, int] = (1, 4)  # How many to place per image
    priority: int = 0       # Higher = placed later (on top)


# Default sprite configurations for AoE2
DEFAULT_SPRITE_CONFIGS = [
    SpriteConfig(
        class_id=0,
        class_name="sheep",
        sprite_files=["sheep_1.png", "sheep_2.png"],
        scale_range=(0.9, 1.1),
        count_range=(2, 4),
        priority=1
    ),
    SpriteConfig(
        class_id=1,
        class_name="villager",
        sprite_files=["villager_m.png", "villager_f.png"],
        scale_range=(0.9, 1.1),
        count_range=(3, 6),
        priority=2
    ),
    SpriteConfig(
        class_id=2,
        class_name="town_center",
        sprite_files=["tc.png"],
        scale_range=(0.95, 1.05),
        count_range=(1, 1),
        priority=0
    ),
    SpriteConfig(
        class_id=3,
        class_name="house",
        sprite_files=["house.png"],
        scale_range=(0.9, 1.1),
        count_range=(0, 3),
        priority=0
    ),
    SpriteConfig(
        class_id=8,
        class_name="scout",
        sprite_files=["scout.png"],
        scale_range=(0.9, 1.1),
        count_range=(1, 1),
        priority=2
    ),
]


class SyntheticDataGenerator:
    """Generates synthetic training data for YOLO object detection."""

    def __init__(
        self,
        sprites_dir: str,
        backgrounds_dir: str,
        output_dir: str,
        sprite_configs: Optional[list[SpriteConfig]] = None,
        image_size: tuple[int, int] = (1280, 800)
    ):
        """Initialize the generator.

        Args:
            sprites_dir: Directory containing extracted sprite PNGs
            backgrounds_dir: Directory containing screenshot backgrounds
            output_dir: Directory for generated training data
            sprite_configs: List of sprite configurations (uses defaults if None)
            image_size: Target image size (width, height)
        """
        self.sprites_dir = Path(sprites_dir)
        self.backgrounds_dir = Path(backgrounds_dir)
        self.output_dir = Path(output_dir)
        self.sprite_configs = sprite_configs or DEFAULT_SPRITE_CONFIGS
        self.image_size = image_size

        # Create output directories
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        # Load sprites and backgrounds
        self._sprites: dict[str, list] = {}
        self._backgrounds: list = []
        self._load_assets()

    def _load_assets(self):
        """Load sprite and background images."""
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("PIL/Pillow required. Install with: pip install Pillow")

        # Load sprites
        for config in self.sprite_configs:
            self._sprites[config.class_name] = []
            for sprite_file in config.sprite_files:
                sprite_path = self.sprites_dir / sprite_file
                if sprite_path.exists():
                    img = Image.open(sprite_path).convert("RGBA")
                    self._sprites[config.class_name].append(img)
                else:
                    print(f"Warning: Sprite not found: {sprite_path}")

        # Load backgrounds
        if self.backgrounds_dir.exists():
            for bg_path in self.backgrounds_dir.glob("*.jpg"):
                img = Image.open(bg_path).convert("RGB")
                self._backgrounds.append(img)
            for bg_path in self.backgrounds_dir.glob("*.png"):
                img = Image.open(bg_path).convert("RGB")
                self._backgrounds.append(img)

        print(f"Loaded sprites: {sum(len(v) for v in self._sprites.values())} images")
        print(f"Loaded backgrounds: {len(self._backgrounds)} images")

    def generate_image(self, index: int, seed: Optional[int] = None) -> tuple[Path, Path]:
        """Generate a single training image with labels.

        Args:
            index: Image index (used in filename)
            seed: Random seed for reproducibility

        Returns:
            Tuple of (image_path, label_path)
        """
        from PIL import Image

        if seed is not None:
            random.seed(seed)

        # Select or create background
        if self._backgrounds:
            bg = random.choice(self._backgrounds).copy()
            bg = bg.resize(self.image_size, Image.Resampling.LANCZOS)
        else:
            # Create solid color background if no backgrounds available
            bg = Image.new("RGB", self.image_size, color=(34, 89, 34))  # Dark green

        labels = []

        # Sort configs by priority (lower = placed first/bottom)
        sorted_configs = sorted(self.sprite_configs, key=lambda c: c.priority)

        # Place sprites
        for config in sorted_configs:
            sprites = self._sprites.get(config.class_name, [])
            if not sprites:
                continue

            # Determine how many to place
            count = random.randint(*config.count_range)

            for _ in range(count):
                sprite = random.choice(sprites).copy()

                # Apply random scale
                scale = random.uniform(*config.scale_range)
                new_width = int(sprite.width * scale)
                new_height = int(sprite.height * scale)
                sprite = sprite.resize((new_width, new_height), Image.Resampling.LANCZOS)

                # Random position (avoid edges)
                margin = 50
                max_x = self.image_size[0] - sprite.width - margin
                max_y = self.image_size[1] - sprite.height - margin

                if max_x <= margin or max_y <= margin:
                    continue  # Skip if sprite too large

                x = random.randint(margin, max_x)
                y = random.randint(margin, max_y)

                # Paste sprite onto background (using alpha channel)
                bg.paste(sprite, (x, y), sprite)

                # Calculate YOLO format label
                # YOLO format: class_id x_center y_center width height (all normalized)
                x_center = (x + sprite.width / 2) / self.image_size[0]
                y_center = (y + sprite.height / 2) / self.image_size[1]
                norm_width = sprite.width / self.image_size[0]
                norm_height = sprite.height / self.image_size[1]

                labels.append(
                    f"{config.class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"
                )

        # Apply random augmentations
        bg = self._augment(bg)

        # Save image
        image_path = self.images_dir / f"image_{index:05d}.jpg"
        bg.save(image_path, "JPEG", quality=90)

        # Save labels
        label_path = self.labels_dir / f"image_{index:05d}.txt"
        with open(label_path, "w") as f:
            f.write("\n".join(labels))

        return image_path, label_path

    def _augment(self, image: "Image.Image") -> "Image.Image":
        """Apply random augmentations to image."""
        from PIL import ImageEnhance

        # Random brightness (0.8 to 1.2)
        if random.random() < 0.5:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))

        # Random contrast (0.9 to 1.1)
        if random.random() < 0.5:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(random.uniform(0.9, 1.1))

        # Random saturation (0.8 to 1.2)
        if random.random() < 0.3:
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))

        return image

    def generate_dataset(
        self,
        num_images: int = 1000,
        train_split: float = 0.8,
        seed: int = 42
    ) -> dict:
        """Generate a complete training dataset.

        Args:
            num_images: Total number of images to generate
            train_split: Fraction of images for training (rest for validation)
            seed: Random seed for reproducibility

        Returns:
            Dictionary with dataset statistics
        """
        random.seed(seed)

        # Generate all images
        all_images = []
        for i in range(num_images):
            if (i + 1) % 100 == 0:
                print(f"Generating image {i + 1}/{num_images}...")
            img_path, label_path = self.generate_image(i, seed=seed + i)
            all_images.append((img_path, label_path))

        # Split into train/val
        random.shuffle(all_images)
        split_idx = int(len(all_images) * train_split)
        train_images = all_images[:split_idx]
        val_images = all_images[split_idx:]

        # Create train/val directories
        train_images_dir = self.output_dir / "train" / "images"
        train_labels_dir = self.output_dir / "train" / "labels"
        val_images_dir = self.output_dir / "val" / "images"
        val_labels_dir = self.output_dir / "val" / "labels"

        for d in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Move files to appropriate directories
        for img_path, label_path in train_images:
            shutil.move(str(img_path), str(train_images_dir / img_path.name))
            shutil.move(str(label_path), str(train_labels_dir / label_path.name))

        for img_path, label_path in val_images:
            shutil.move(str(img_path), str(val_images_dir / img_path.name))
            shutil.move(str(label_path), str(val_labels_dir / label_path.name))

        # Remove temporary directories
        shutil.rmtree(self.images_dir)
        shutil.rmtree(self.labels_dir)

        # Create dataset YAML for YOLO training
        yaml_content = self._create_dataset_yaml()
        yaml_path = self.output_dir / "dataset.yaml"
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        stats = {
            "total_images": num_images,
            "train_images": len(train_images),
            "val_images": len(val_images),
            "classes": [c.class_name for c in self.sprite_configs],
            "yaml_path": str(yaml_path)
        }

        print(f"\nDataset generated:")
        print(f"  Train: {stats['train_images']} images")
        print(f"  Val: {stats['val_images']} images")
        print(f"  YAML: {yaml_path}")

        return stats

    def _create_dataset_yaml(self) -> str:
        """Create YOLO dataset configuration YAML."""
        class_names = {c.class_id: c.class_name for c in self.sprite_configs}

        yaml_lines = [
            f"path: {self.output_dir.absolute()}",
            "train: train/images",
            "val: val/images",
            "",
            "names:"
        ]

        for class_id in sorted(class_names.keys()):
            yaml_lines.append(f"  {class_id}: {class_names[class_id]}")

        return "\n".join(yaml_lines)


def extract_sprites_from_game(
    game_graphics_dir: str,
    output_dir: str,
    sprite_mapping: Optional[dict] = None
) -> int:
    """Extract sprite images from AoE2 game files.

    NOTE: This requires SLX Studio or similar tool to be installed.
    This function provides guidance on the extraction process.

    Args:
        game_graphics_dir: Path to game graphics directory
            Typically: Steam/steamapps/common/AoE2DE/resources/_common/drs/graphics
        output_dir: Output directory for extracted sprites
        sprite_mapping: Optional dict mapping unit names to sprite file IDs

    Returns:
        Number of sprites extracted (or 0 if extraction tool not available)
    """
    print("""
Sprite Extraction Guide
=======================

To extract sprites from AoE2 game files:

1. Install SLX Studio (search "SLX Studio AoE2")

2. Locate game graphics:
   - Steam: Steam/steamapps/common/AoE2DE/resources/_common/drs/graphics
   - Files are in .smx format

3. Key sprite files to extract:
   - u_vill_*.smx: Villagers (male/female, various actions)
   - u_sheep_*.smx: Sheep
   - u_scout_*.smx: Scout cavalry
   - b_tc_*.smx: Town Center
   - b_house_*.smx: Houses

4. Use SLX Studio to convert .smx → .png
   - Export with transparency (alpha channel)
   - Use "standing" or "idle" animation frame

5. Use AdvancedGenieEditor3 for unit ID → sprite file mapping:
   https://github.com/Tapsa/AGE

For quick testing without extraction, you can:
- Use placeholder sprites (colored rectangles)
- Download sprite sheets from Spriters Resource:
  https://www.spriters-resource.com/pc_computer/ageofempiresii/
""")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create placeholder sprites if no real sprites available
    try:
        from PIL import Image, ImageDraw

        placeholders = [
            ("sheep_1.png", (30, 20), (200, 180, 140)),  # Tan sheep
            ("sheep_2.png", (30, 20), (220, 200, 160)),  # Lighter sheep
            ("villager_m.png", (24, 30), (100, 80, 60)),  # Brown villager
            ("villager_f.png", (24, 30), (120, 90, 70)),  # Lighter villager
            ("tc.png", (120, 100), (139, 90, 43)),  # Brown TC
            ("house.png", (50, 45), (160, 120, 80)),  # Brown house
            ("scout.png", (35, 40), (80, 60, 40)),  # Dark horse
        ]

        count = 0
        for filename, size, color in placeholders:
            img = Image.new("RGBA", size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            # Draw filled shape with slight transparency
            draw.ellipse([2, 2, size[0]-2, size[1]-2], fill=(*color, 230))
            img.save(output_path / filename)
            count += 1

        print(f"\nCreated {count} placeholder sprites in {output_dir}")
        print("Replace these with real game sprites for better training results.")

        return count

    except ImportError:
        print("PIL not available. Cannot create placeholder sprites.")
        return 0


if __name__ == "__main__":
    import sys

    print("Synthetic Data Generator for AoE2 YOLO Training")
    print("=" * 50)

    # Default paths
    base_dir = Path(__file__).parent
    sprites_dir = base_dir / "sprites"
    backgrounds_dir = base_dir / "backgrounds"
    output_dir = base_dir / "training_data"

    # Create placeholder sprites if needed
    if not sprites_dir.exists():
        print("\nNo sprites directory found. Creating placeholders...")
        extract_sprites_from_game("", str(sprites_dir))

    # Create placeholder background if needed
    if not backgrounds_dir.exists():
        backgrounds_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nCreated backgrounds directory: {backgrounds_dir}")
        print("Add screenshot JPGs here for better training data.")

    # Check if we have enough assets
    if not list(sprites_dir.glob("*.png")):
        print("\nNo sprite files found. Run sprite extraction first.")
        sys.exit(1)

    print(f"\nSprites: {sprites_dir}")
    print(f"Backgrounds: {backgrounds_dir}")
    print(f"Output: {output_dir}")

    generator = SyntheticDataGenerator(
        sprites_dir=str(sprites_dir),
        backgrounds_dir=str(backgrounds_dir),
        output_dir=str(output_dir)
    )

    # Generate small test dataset
    print("\nGenerating test dataset (100 images)...")
    stats = generator.generate_dataset(num_images=100, train_split=0.8)

    print("\n" + "=" * 50)
    print("Dataset generation complete!")
    print(f"Train with: yolo train model=yolov8n.pt data={stats['yaml_path']} epochs=50")
