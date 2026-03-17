# AoE2 Entity Detection System

YOLO-based object detection for identifying game entities (units, buildings, resources) in Age of Empires II: Definitive Edition screenshots.

## Overview

The detection system enables the AI agent to understand game state by detecting and localizing entities with bounding boxes. It uses a custom-trained YOLO model on synthetic training data generated from extracted game sprites.

```
Screenshot → YOLO Model → Detected Entities (class, bbox, confidence)
```

**Key Features:**
- 60 entity classes (units, buildings, resources, animals)
- Real-time inference at 1280px resolution (~234ms/image, negligible vs LLM latency)
- 92.2% mAP50 accuracy (v5 model, 18,520-image hybrid dataset)
- Persistent entity tracking across frames via IoU matching
- NMS applied to all backends (PyTorch, ONNX, Mock)
- SAHI sliced inference by default for large screenshots (640x640 tiles, 20% overlap)

## Quick Start

### Using the Detector

```python
from detection import EntityDetector, get_detector

# Get singleton detector instance
detector = get_detector()

# Detect entities in a screenshot
entities = detector.detect("screenshot.png")

for entity in entities:
    print(f"{entity.class_name}: {entity.center} (conf: {entity.confidence:.2f})")
```

### From the Agent

The agent automatically uses detection when available:

```python
# In game_loop.py, detection is used like this:
from detection.inference.detector import EntityDetector, get_detector

detector = get_detector(imgsz=1280)  # Configurable inference resolution
entities = detector.detect(screenshot_bytes)
# Entity IDs persist across frames via IoU matching (e.g., sheep_0 stays sheep_0)
```

## Directory Structure

```
detection/
├── __init__.py                  # Package exports (EntityDetector, get_detector)
│
├── inference/                   # Runtime detection
│   ├── detector.py              # EntityDetector class (NMS, persistent tracking)
│   ├── ownership.py             # Blue-dominance ownership classifier (own vs enemy)
│   └── models/
│       ├── aoe2_yolo_v5.pt      # PyTorch model weights (latest, preferred)
│       ├── aoe2_yolo_v2.onnx    # ONNX model (fallback)
│       └── aoe2_yolo26.pt       # v1 model (last resort)
│
├── training/                    # Training pipeline
│   ├── train_yolo.py            # YOLO training script
│   ├── generate_training_data.py # Synthetic image generator
│   ├── synthetic_data.py        # Data generation utilities
│   └── config/
│       └── classes.yaml         # Class definitions (single source of truth)
│
├── extraction/                  # Sprite & screenshot extraction
│   ├── sld_extractor.py         # AoE2 SLD sprite format parser
│   ├── extract_sprites.py       # Batch sprite extraction
│   └── capture_replay.py        # Screenshot capture from game replays
│
├── testing/                     # Validation & testing
│   └── test_real_detection.py   # Test model on real screenshots
│
├── docs/                        # Documentation
│   ├── SLD_FORMAT.md            # SLD file format specification
│   └── TRAINING_GUIDE.md        # Complete training pipeline guide
│
├── training_data/               # Generated training dataset (gitignored)
└── real_screenshots/            # Captured game screenshots (gitignored)
```

## Documentation

| Document | Purpose |
|----------|---------|
| [TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) | Complete guide for the detection pipeline: sprite extraction, synthetic data generation, cloud training on Lambda Labs, and inference integration. Includes training results, cost comparisons, and troubleshooting. |
| [SLD_FORMAT.md](docs/SLD_FORMAT.md) | Technical specification for AoE2:DE's SLD (SLDX) sprite file format. Documents the binary structure, DXT1 compression, and layer parsing. Essential for understanding or modifying the sprite extractor. |
| [classes.yaml](training/config/classes.yaml) | Single source of truth for all 60 detection classes. Organized by category with IDs and examples for unique unit groups. |

## Entity Classes (60 total)

Classes are organized by gameplay category. See `training/config/classes.yaml` for the complete list.

| Category | Count | Examples |
|----------|-------|----------|
| Resources & Nature | 9 | tree, gold_mine, stone_mine, berry_bush, sheep, boar, wolf |
| Economy Buildings | 8 | town_center, house, lumber_camp, mill, market, dock, farm |
| Military Buildings | 8 | barracks, archery_range, stable, blacksmith, siege_workshop, castle |
| Defensive | 3 | gate, wall, tower |
| Special Buildings | 2 | wonder, krepost |
| Civilian Units | 3 | villager, trade_cart, fishing_ship |
| Cavalry | 4 | scout_line, knight_line, camel_line, battle_elephant |
| Archers | 4 | archer_line, skirmisher_line, cavalry_archer, hand_cannoneer |
| Infantry | 3 | militia_line, spearman_line, eagle_line |
| Siege | 5 | ram, mangonel_line, scorpion, trebuchet, siege_tower |
| Monks & Special | 2 | monk, king |
| Unique Units | 5 | unique_archer, unique_cavalry, unique_infantry, unique_siege, unique_ship |
| Naval | 3 | fish, galley, fire_galley |
| Animals | 1 | goose |

**Note:** Unit upgrade lines are grouped (e.g., `militia_line` includes militia through champion). Civilization-specific unique units are categorized by type rather than individually.

## Training Pipeline

The full training pipeline is documented in [TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md).

### Summary

1. **Extract Sprites** - Parse SLD files from game directory
   ```bash
   python -m detection.extraction.extract_sprites --output tmp/sprites
   ```

2. **Generate Synthetic Data** - Composite sprites onto backgrounds
   ```bash
   python -m detection.training.generate_training_data --num-images 3000
   ```

3. **Train Model** - Use Lambda Labs A100 for fast training (~1 hour, ~$1.30)
   ```bash
   python -m detection.training.train_yolo
   ```

4. **Validate** - Test on real screenshots
   ```bash
   python -m detection.testing.test_real_detection --images real_screenshots/raw
   ```

## Capturing Real Screenshots

Use the replay capture script to collect training/validation data:

```bash
# Capture 200 screenshots at 5-second intervals from game replay
python -m detection.extraction.capture_replay --count 200 --interval 5
```

**Workflow:**
1. Load a replay in AoE2 DE (Single Player → Replays)
2. Set replay speed to 8x
3. Run the capture script
4. Move camera around during capture for diverse angles

## Model Performance (v5 — latest)

| Metric | v5 | v4 (previous) |
|--------|-----|---------------|
| mAP50 | **92.2%** | 86.8% |
| mAP50-95 | **85.4%** | 72.3% |
| Precision | **94.8%** | 87.1% |
| Recall | **89.2%** | 78.5% |
| Inference (imgsz=1280) | ~234ms | ~7ms (at 640) |

**v5 dataset:** 18,520 images (8,000 synthetic train + 7,120 real train + 2,000 synthetic val + 1,400 real val)

**Inference resolution:** 1280px (up from 640). Benchmarked on 1920x1080 gameplay: imgsz=640 found 12 entities, imgsz=1280 found 55 entities (+4.6x detections, only +160ms). The LLM API call dominates at 1-3s, so detection latency is negligible.

## Dependencies

```
ultralytics>=8.0.0    # YOLO implementation
Pillow>=9.0.0         # Image processing
numpy>=1.21.0         # Array operations
```

Install with:
```bash
pip install ultralytics Pillow numpy
```

## Ownership Classification

`detection/inference/ownership.py` classifies detected military units as own (Player 1, blue) or enemy using pixel color analysis.

In AoE2:DE, Player 1 is always blue. The classifier samples two regions per entity:
1. **Health bar zone** — narrow band above the bounding box (health bars are tinted in player color)
2. **Unit body zone** — top 30% of the bounding box (livery/clothing color)

A pixel is "blue" if `B > 120 AND B > R * 1.5 AND B > G * 1.5`. If blue pixel ratio exceeds 4%, the unit is classified as own.

```python
from detection.inference.ownership import classify_entities, Owner

results = classify_entities(screenshot_bytes, entities, threat_classes)
for entity_id, (owner, blue_ratio) in results.items():
    print(f"{entity_id}: {owner.value} (blue_ratio={blue_ratio:.3f})")
    # e.g., "scout_line_0: own (blue_ratio=0.065)"
    # e.g., "archer_line_0: enemy (blue_ratio=0.000)"
```

Used by the alarm system (`src/goals.py`) to avoid false alarms from own military units.

## Integration with Agent

The detection module integrates with the main agent in `src/game_loop.py`:

```python
# Detection is optional - agent falls back gracefully if unavailable
try:
    from detection.inference.detector import EntityDetector, get_detector
    DETECTION_AVAILABLE = True
except ImportError:
    DETECTION_AVAILABLE = False

# During game loop, entities are detected and passed to LLM
if DETECTION_AVAILABLE:
    detector = get_detector(imgsz=config.detection_imgsz)  # default 1280
    entities = detector.detect(screenshot_bytes)
    # Entity IDs persist across frames: sheep_0 stays sheep_0 between turns
    # LLM can reference entities by ID: "right_click on sheep_0"
```

The executor resolves entity IDs to screen coordinates automatically. Window offset is re-fetched before each action to handle window movement.

## License

Part of the AoE2 LLM Arena project. For research and educational purposes.
