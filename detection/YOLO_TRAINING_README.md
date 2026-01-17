# YOLO Object Detection for Age of Empires II

This document describes the complete pipeline for training a YOLO object detection model on AoE2:DE game sprites, from sprite extraction to synthetic training data generation.

## Overview

The detection system enables an AI agent to identify game entities (units, buildings, resources) from screenshots with precise bounding box coordinates. This replaces unreliable LLM coordinate guessing with accurate computer vision.

```
┌─────────────────────────────────────────────────────────────────┐
│                    DETECTION PIPELINE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Game Graphics (.sld)                                           │
│        │                                                        │
│        ▼                                                        │
│  ┌──────────────┐                                               │
│  │ SLD Extractor│  extract_training_sprites.py                  │
│  │ (DXT1 decode)│  sld_extractor.py                             │
│  └──────────────┘                                               │
│        │                                                        │
│        ▼                                                        │
│  Sprite PNGs (tmp/sprites/)                                     │
│        │                                                        │
│        ▼                                                        │
│  ┌──────────────┐                                               │
│  │ Synthetic    │  generate_training_data.py                    │
│  │ Data Gen     │                                               │
│  └──────────────┘                                               │
│        │                                                        │
│        ▼                                                        │
│  Training Dataset (training_data/)                              │
│  ├── train/images/*.jpg                                         │
│  ├── train/labels/*.txt                                         │
│  ├── val/images/*.jpg                                           │
│  ├── val/labels/*.txt                                           │
│  └── dataset.yaml                                               │
│        │                                                        │
│        ▼                                                        │
│  ┌──────────────┐                                               │
│  │ YOLOv8 Train │  ultralytics                                  │
│  └──────────────┘                                               │
│        │                                                        │
│        ▼                                                        │
│  Trained Model (models/aoe2_yolo.pt)                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component 1: SLD Sprite Extractor

### Purpose
Extracts sprite graphics from AoE2:DE's proprietary SLD (SLDX) file format into PNG images with transparency.

### Files
- `sld_extractor.py` - Core SLD parser and DXT1 decoder
- `SLD_EXTRACTOR_README.md` - Detailed format specification

### Technical Details

#### SLD File Format
The SLD format uses DXT1 (BC1) block compression with run-length encoding for sparse sprites:

```
SLD File Structure:
├── File Header (16 bytes)
│   ├── Signature: "SLDX"
│   ├── Version: uint16
│   ├── Frame count: uint16
│   └── Reserved fields
│
└── Frames (variable)
    ├── Frame Header (12 bytes)
    │   ├── Canvas dimensions
    │   ├── Hotspot coordinates
    │   └── Layer flags (bitmask)
    │
    └── Layers (based on flags)
        ├── Main graphics (DXT1)
        ├── Shadow
        ├── Damage mask
        └── Player color mask
```

#### DXT1 Decompression
Each 4x4 pixel block is encoded as 8 bytes:
- 2 bytes: Color 0 (RGB565)
- 2 bytes: Color 1 (RGB565)
- 4 bytes: 16 2-bit indices

The decoder interpolates a 4-color palette:
```python
if color0 > color1:  # Opaque mode
    palette = [c0, c1, (2*c0+c1)/3, (c0+2*c1)/3]
else:  # Transparent mode
    palette = [c0, c1, (c0+c1)/2, transparent]
```

#### Run-Length Encoding
Command arrays specify skip/draw pairs to handle sparse sprites efficiently:
```python
for skip_count, draw_count in commands:
    block_index += skip_count  # Skip transparent blocks
    for _ in range(draw_count):
        decode_dxt1_block(block_index)
        block_index += 1
```

### Usage
```bash
# Extract single sprite
python sld_extractor.py input.sld output.png

# Extract specific frame
python sld_extractor.py input.sld output.png --frame 5
```

### Python API
```python
from detection.sld_extractor import SLDExtractor, extract_sprite

# Simple extraction
extract_sprite("sheep.sld", "sheep.png")

# Advanced usage
extractor = SLDExtractor("villager.sld")
frames = extractor.extract_all()
for frame in frames:
    extractor.save_as_png(frame, f"frame_{frame.index}.png")
```

---

## Component 2: Training Sprite Extraction

### Purpose
Extracts a curated set of game sprites organized by gameplay-relevant categories for YOLO training.

### File
- `extract_training_sprites.py` - Batch extraction with class configuration

### Design Philosophy

**Grouped Classes**: Similar units are grouped into single detection classes:
- `archer_line` = Archer + Crossbowman + Arbalester
- `knight_line` = Knight + Cavalier + Paladin
- `militia_line` = Militia + Man-at-Arms + Long Swordsman + Two-Handed + Champion

This approach:
1. Reduces class count (46 vs 200+)
2. Provides sufficient tactical information
3. Improves model accuracy with more training examples per class

**Civilization Filtering**: Uses Western European (`b_west_*`) and Mediterranean (`b_medi_*`) building styles for generic appearance compatible with common civilizations (Franks, Britons, etc.).

**Exclusions**:
- Destruction animations (`*_destruction_*`)
- Rubble sprites (`*_rubble_*`)
- Civilization-specific unique buildings (Krepost, etc.)

### Class Configuration

```python
SPRITE_CONFIG = [
    # (class_name, [file_patterns], max_variants, description)
    ("villager", [
        "u_vil_male_villager_idle*_x1.sld",
        "u_vil_female_villager_idle*_x1.sld",
    ], 6, "Worker units"),

    ("knight_line", [
        "u_cav_knight_idle*_x1.sld",
        "u_cav_cavalier_idle*_x1.sld",
        "u_cav_paladin_idle*_x1.sld",
    ], 4, "Knight, Cavalier, Paladin"),

    ("town_center", [
        "b_west_town_center_age2_x1.sld",
        "b_medi_town_center_age2_x1.sld",
    ], 4, "Town Center"),
    # ... 46 classes total
]
```

### Current Classes (46)

| Category | Classes |
|----------|---------|
| Resources | tree, gold_mine, stone_mine, berry_bush, relic |
| Economy Buildings | town_center, house, lumber_camp, mining_camp, blacksmith, dock, university |
| Military Buildings | barracks, archery_range, stable, monastery, castle, wonder, gate |
| Animals | sheep, deer, boar, wolf |
| Economic Units | villager, trade_cart, fishing_ship |
| Cavalry | scout_line, knight_line, camel_line, battle_elephant |
| Archers | archer_line, skirmisher_line, cavalry_archer, hand_cannoneer |
| Infantry | militia_line, spearman_line, eagle_line |
| Siege | ram, mangonel_line, scorpion, trebuchet |
| Special | monk, king, longbowman, mangudai, war_wagon |

### Usage
```bash
# Show configuration
python extract_training_sprites.py --show-config

# Extract sprites
python extract_training_sprites.py --output tmp/sprites

# Custom game graphics location
python extract_training_sprites.py --game-dir /path/to/game_graphics
```

---

## Component 3: Synthetic Training Data Generator

### Purpose
Creates labeled training images by compositing extracted sprites onto generated backgrounds with automatic YOLO-format bounding box annotations.

### File
- `generate_training_data.py` - Synthetic data generator

### Why Synthetic Data?

| Approach | Pros | Cons |
|----------|------|------|
| Manual labeling | Realistic | Time-consuming, expensive |
| Screenshot mining | Authentic game images | Requires manual labeling |
| **Synthetic generation** | **Automatic labels, unlimited data** | **Less realistic** |

Synthetic data works well for AoE2 because:
1. Sprites are consistent (no perspective variation)
2. Isometric view has fixed camera angle
3. Ground truth coordinates are known at generation time

### Technical Implementation

#### Sprite Configuration
```python
@dataclass
class SpriteConfig:
    class_id: int
    class_name: str
    sprite_patterns: list[str]  # Glob patterns
    scale_range: tuple[float, float] = (0.8, 1.2)
    count_range: tuple[int, int] = (1, 3)
    z_order: int = 0  # Rendering layer
    avoid_edges: bool = True
    min_spacing: int = 20
```

#### Z-Order Layering
Sprites are rendered in z-order to simulate depth:
```
z_order 0: Resources (trees, mines, bushes)
z_order 1: Buildings (rendered behind units)
z_order 2: Animals (sheep, deer, boar)
z_order 3: Units (villagers, military)
```

#### Overlap Management
The generator prevents excessive overlap:
```python
def _check_overlap(new_box, placed_boxes, min_overlap=0.3):
    """Reject placement if >30% overlap with existing boxes."""
    for placed in placed_boxes:
        intersection = calculate_intersection(new_box, placed)
        if intersection / new_box.area > min_overlap:
            return True  # Too much overlap
    return False
```

#### Background Generation
When no real backgrounds are provided, the generator creates terrain-like backgrounds:
```python
def _generate_terrain_background(self):
    grass_colors = [
        (34, 89, 34),   # Dark green
        (46, 102, 46),  # Medium green
        (58, 115, 58),  # Light green
    ]
    # Draw overlapping ellipses for natural appearance
    for _ in range(20):
        color = random.choice(grass_colors)
        draw.ellipse([x, y, x+w, y+h], fill=color)
    # Blur for smooth transitions
    return bg.filter(GaussianBlur(radius=3))
```

#### Data Augmentation
Applied to increase variety:
- **Brightness**: 0.7x - 1.3x
- **Contrast**: 0.8x - 1.2x
- **Saturation**: 0.8x - 1.2x
- **Blur**: Occasional Gaussian blur (radius 0.5)

### YOLO Label Format
Labels are normalized coordinates:
```
<class_id> <x_center> <y_center> <width> <height>
```

Example (`train/labels/img_00000.txt`):
```
5 0.689453 0.543750 0.319531 0.381944   # town_center
6 0.509766 0.739583 0.147656 0.206944   # house
23 0.212891 0.821528 0.028906 0.062500  # villager
19 0.522266 0.386806 0.030469 0.037500  # sheep
```

### Usage
```bash
# Generate 1000 images (default)
python generate_training_data.py

# Custom options
python generate_training_data.py \
    --num-images 2000 \
    --output detection/training_data \
    --image-size 1280 720 \
    --train-split 0.8 \
    --seed 42

# With real backgrounds
python generate_training_data.py --backgrounds path/to/screenshots/
```

---

## Dataset Structure

```
detection/training_data/
├── dataset.yaml          # YOLO configuration
├── train/
│   ├── images/
│   │   ├── img_00000.jpg
│   │   ├── img_00001.jpg
│   │   └── ...
│   └── labels/
│       ├── img_00000.txt
│       ├── img_00001.txt
│       └── ...
└── val/
    ├── images/
    └── labels/
```

### dataset.yaml
```yaml
path: /path/to/training_data
train: train/images
val: val/images

names:
  0: tree
  1: gold_mine
  2: stone_mine
  # ... 46 classes
  45: war_wagon
```

---

## Training YOLOv8

### Prerequisites
```bash
pip install ultralytics
```

### Training Command
```bash
# Train YOLOv8-nano (fastest, smallest)
yolo train model=yolov8n.pt \
    data=detection/training_data/dataset.yaml \
    epochs=100 \
    imgsz=640

# Train YOLOv8-small (better accuracy)
yolo train model=yolov8s.pt \
    data=detection/training_data/dataset.yaml \
    epochs=100 \
    imgsz=640 \
    batch=16
```

### Recommended Hyperparameters
```yaml
epochs: 100-200
imgsz: 640
batch: 16 (adjust based on GPU memory)
patience: 50 (early stopping)
lr0: 0.01
lrf: 0.01
```

### Training Output
```
runs/detect/train/
├── weights/
│   ├── best.pt    # Best validation mAP
│   └── last.pt    # Final epoch
├── results.png    # Training curves
└── confusion_matrix.png
```

---

## Inference Integration

### Python API
```python
from ultralytics import YOLO

class EntityDetector:
    def __init__(self, model_path="models/aoe2_yolo.pt"):
        self.model = YOLO(model_path)
        self.class_names = [...]  # 46 classes

    def detect(self, screenshot) -> list[dict]:
        results = self.model(screenshot, conf=0.5)
        entities = []

        for box, cls, conf in zip(
            results[0].boxes.xyxy,
            results[0].boxes.cls,
            results[0].boxes.conf
        ):
            x1, y1, x2, y2 = box.tolist()
            entities.append({
                "id": f"{self.class_names[int(cls)]}_{len(entities)}",
                "class": self.class_names[int(cls)],
                "bbox": [x1, y1, x2, y2],
                "center": ((x1 + x2) / 2, (y1 + y2) / 2),
                "confidence": float(conf)
            })

        return entities
```

### Action Resolution
```python
# LLM outputs target IDs instead of coordinates
action = {"type": "right_click", "target_id": "sheep_0"}

# Executor resolves to pixel coordinates
entity = find_entity(action["target_id"], detected_entities)
x, y = entity["center"]
pyautogui.click(x, y)
```

---

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Detection latency | <50ms | YOLOv8-nano on GPU |
| mAP@0.5 | >0.85 | On synthetic validation set |
| Click accuracy | >90% | Measured by successful actions |
| Classes | 46 | Covers core gameplay entities |

---

## File Locations

### Source Files
```
detection/
├── sld_extractor.py           # SLD format parser
├── extract_training_sprites.py # Batch sprite extraction
├── generate_training_data.py   # Synthetic data generator
├── SLD_EXTRACTOR_README.md     # Format specification
└── YOLO_TRAINING_README.md     # This file
```

### Generated Files
```
tmp/sprites/                    # Extracted PNG sprites
detection/training_data/        # YOLO training dataset
models/aoe2_yolo.pt            # Trained model (after training)
```

### Game Resources
```
Steam/steamapps/common/AoE2DE/resources/_common/drs/graphics/
```

---

## Troubleshooting

### SLD Extraction Fails
Some SLD files have format variations:
- Try different frames: `--frame 1`, `--frame 2`
- Some civilization buildings fail consistently
- Frame 0 (idle pose) is most reliable

### Poor Detection Accuracy
1. Increase training epochs
2. Add real screenshot backgrounds
3. Increase synthetic data count
4. Fine-tune on real game screenshots

### GPU Memory Issues
- Reduce batch size: `batch=8` or `batch=4`
- Use smaller model: `yolov8n.pt`
- Reduce image size: `imgsz=416`

---

## Future Improvements

1. **Real Screenshot Integration**: Mix synthetic and real screenshots for better generalization
2. **Player Color Masks**: Apply player color to unit sprites for color-based team detection
3. **Animation Frames**: Include non-idle poses for better detection during combat
4. **Fog of War Handling**: Train on partially visible sprites
5. **Resolution Scaling**: Support multiple game resolutions

---

## References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [openage SLD Format](https://github.com/SFTtech/openage/blob/master/doc/media/sld-files.md)
- [DXT1/BC1 Compression](https://docs.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-block-compression)
