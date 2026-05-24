# YOLO Object Detection for Age of Empires II

This document describes the complete pipeline for training a YOLO object detection model on AoE2:DE game sprites, from sprite extraction to cloud-based training.

## Overview

The detection system enables an AI agent to identify game entities (units, buildings, resources) from screenshots with precise bounding box coordinates.

```
┌─────────────────────────────────────────────────────────────────┐
│                    DETECTION PIPELINE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Game Graphics (.sld)                                           │
│        │                                                        │
│        ▼                                                        │
│  ┌──────────────┐                                               │
│  │ SLD Extractor│  extraction/extract_sprites.py                │
│  │ (DXT1 decode)│  extraction/sld_extractor.py                  │
│  └──────────────┘                                               │
│        │                                                        │
│        ▼                                                        │
│  Sprite PNGs (tmp/sprites/)                                     │
│        │                                                        │
│        ▼                                                        │
│  ┌──────────────┐                                               │
│  │ Synthetic    │  training/generate_training_data.py           │
│  │ Data Gen     │                                               │
│  └──────────────┘                                               │
│        │                                                        │
│        ▼                                                        │
│  Training Dataset (training_data/)                              │
│        │                                                        │
│        ▼                                                        │
│  ┌──────────────┐                                               │
│  │ YOLO Train   │  training/train_yolo.py                       │
│  └──────────────┘                                               │
│        │                                                        │
│        ▼                                                        │
│  Trained Model (inference/models/aoe2_yolo26.pt)                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Training Results

### v5 (Latest)

**Model:** YOLO11n (nano)

**Training Configuration:**
- Dataset: 18,520 images (8,000 synthetic train + 7,120 real train + 2,000 synthetic val + 1,400 real val)
- Epochs: 150 on Lambda Labs A100
- Image size: 640x640
- Batch size: 32
- Classes: 60 (see `training/config/classes.yaml`)

**Final Metrics (v5):**

| Metric | v5 | v4 (previous) |
|--------|-----|---------------|
| **mAP50** | **92.2%** | 86.8% |
| **mAP50-95** | **85.4%** | 72.3% |
| Precision | **94.8%** | 87.1% |
| Recall | **89.2%** | 78.5% |

**Key improvements over v4:**
- 2x larger dataset (18,520 vs 8,520 images)
- Z-order-aware overlap thresholds in synthetic data (buildings 10%, resources 15%, units 35%)
- Skip-instead-of-force-place when overlap limit exceeded
- Better balance of synthetic and real data

**Training Time:** ~1 hour on Lambda Labs A100

### v4 (Previous)

| Metric | Value |
|--------|-------|
| mAP50 | 86.8% |
| mAP50-95 | 72.3% |
| Precision | 87.1% |
| Recall | 78.5% |

Dataset: 8,520 images (5,520 real tiles + 3,000 synthetic)

---

## Quick Start

### 1. Extract Sprites
```bash
# From agent/ directory
python -m detection.extraction.extract_sprites --output tmp/sprites
```

### 2. Generate Training Data
```bash
python -m detection.training.generate_training_data --num-images 3000 --output detection/training_data
```

### 3. Train Model (Cloud)
See [Cloud Training](#cloud-training-lambda-labs) section below.

### 4. Use Model
```python
from ultralytics import YOLO

model = YOLO("detection/inference/models/aoe2_yolo26.pt")
results = model("screenshot.png", conf=0.5)
```

---

## Component 1: SLD Sprite Extractor

Extracts sprite graphics from AoE2:DE's proprietary SLD (SLDX) file format into PNG images.

### Files
- `extraction/sld_extractor.py` - Core SLD parser and DXT1 decoder
- `docs/SLD_FORMAT.md` - Detailed format specification

### Usage
```bash
python -m detection.extraction.sld_extractor input.sld output.png --frame 0
```

---

## Component 2: Training Sprite Extraction

Extracts curated game sprites organized by gameplay-relevant categories.

### Design Philosophy

**Grouped Classes**: Similar units grouped into single detection classes:
- `archer_line` = Archer + Crossbowman + Arbalester
- `knight_line` = Knight + Cavalier + Paladin
- `militia_line` = Militia + Man-at-Arms + Long Swordsman + Two-Handed + Champion

**Architecture Diversity**: Uses wildcard patterns (`b_*_{building}_age{N}_x1.sld`) to extract all 17+ architecture styles (African, Asian, Eastern European, Mediterranean, Western European, etc.) with multi-age variants.

**Exclusions**: Destruction animations, rubble sprites, shadow sprites, construction states, and multi-part building components are filtered via `EXCLUDE_SUBSTRINGS`.

### Classes (60 total)

> **Note:** Class definitions are maintained in `training/config/classes.yaml` as the single source of truth. All synthetic data, prelabels, and CVAT annotations use these IDs directly.

| Category | Classes |
|----------|---------|
| Resources & Nature | tree, gold_mine, stone_mine, berry_bush, relic, deer, boar, wolf, sheep |
| Economy Buildings | town_center, house, lumber_camp, mining_camp, mill, market, dock, farm |
| Military Buildings | barracks, archery_range, stable, blacksmith, siege_workshop, monastery, castle, university |
| Defensive | gate, wall, tower |
| Special Buildings | wonder, krepost |
| Civilian Units | villager, trade_cart, fishing_ship |
| Cavalry | scout_line, knight_line, camel_line, battle_elephant |
| Archers | archer_line, skirmisher_line, cavalry_archer, hand_cannoneer |
| Infantry | militia_line, spearman_line, eagle_line |
| Siege | ram, mangonel_line, scorpion, trebuchet, siege_tower |
| Monks & Special | monk, king |
| Unique Units | unique_archer, unique_cavalry, unique_infantry, unique_siege, unique_ship |
| Naval | fish, galley, fire_galley |
| Animals | goose |

### Usage
```bash
python -m detection.extraction.extract_sprites --show-config
python -m detection.extraction.extract_sprites --output tmp/sprites
```

---

## Component 3: Synthetic Training Data Generator

Creates labeled training images by compositing sprites onto generated backgrounds.

### Why Synthetic Data?

Synthetic data works well for AoE2 because:
1. Sprites are consistent (no perspective variation)
2. Isometric view has fixed camera angle
3. Ground truth coordinates are known at generation time

### Features
- Z-order layering (resources → buildings → animals → units)
- Z-order-aware overlap thresholds: buildings 10%, resources 15%, units 35% (skip placement when exceeded)
- Data augmentation (brightness, contrast, saturation, blur)
- Automatic YOLO-format label generation

### Usage
```bash
python -m detection.training.generate_training_data \
    --num-images 3000 \
    --output detection/training_data \
    --image-size 1280 720 \
    --train-split 0.8
```

---

## Dataset Structure

```
detection/training_data/
├── dataset.yaml
├── train/
│   ├── images/  (2400 images)
│   └── labels/  (2400 label files)
└── val/
    ├── images/  (600 images)
    └── labels/  (600 label files)
```

---

## Cloud Training (Lambda Labs)

Local training on Apple M2 Pro takes ~6.5 days. Cloud training on Lambda Labs A100 takes ~1 hour for ~$1.30.

### Lambda Labs Setup

**Instance Configuration:**
| Setting | Value |
|---------|-------|
| Instance Type | 1x A100 (40 GB SXM4) |
| Price | $1.29/hour |
| Region | us-east-1 (Virginia) |
| Base Image | Lambda Stack 22.04 |
| Filesystem | None (use local SSD) |

### Step 1: Prepare Training Data

```bash
# Create tarball of training data
tar -czvf training_data.tar.gz detection/training_data

# Create dataset.yaml for Lambda (update paths)
cat > lambda_dataset.yaml << 'EOF'
path: /home/ubuntu/training_data
train: train/images
val: val/images

names:
  0: tree
  1: gold_mine
  2: stone_mine
  3: berry_bush
  4: relic
  5: deer
  6: boar
  7: wolf
  8: sheep
  9: town_center
  10: house
  11: lumber_camp
  12: mining_camp
  13: mill
  14: market
  15: dock
  16: farm
  17: barracks
  18: archery_range
  19: stable
  20: blacksmith
  21: siege_workshop
  22: monastery
  23: castle
  24: university
  25: gate
  26: wall
  27: tower
  28: wonder
  29: krepost
  30: villager
  31: trade_cart
  32: fishing_ship
  33: scout_line
  34: knight_line
  35: camel_line
  36: battle_elephant
  37: archer_line
  38: skirmisher_line
  39: cavalry_archer
  40: hand_cannoneer
  41: militia_line
  42: spearman_line
  43: eagle_line
  44: ram
  45: mangonel_line
  46: scorpion
  47: trebuchet
  48: monk
  49: king
  50: unique_archer
  51: unique_cavalry
  52: unique_infantry
  53: unique_siege
  54: unique_ship
  55: fish
  56: galley
  57: fire_galley
  58: siege_tower
  59: goose
EOF
```

### Step 2: Launch Instance & Upload

```bash
# Upload files to Lambda instance (replace <IP> with your instance IP)
scp -i ~/.ssh/your-key.pem training_data.tar.gz ubuntu@<IP>:/home/ubuntu/
scp -i ~/.ssh/your-key.pem lambda_dataset.yaml ubuntu@<IP>:/home/ubuntu/

# SSH into instance
ssh -i ~/.ssh/your-key.pem ubuntu@<IP>
```

### Step 3: Run Training on Lambda

```bash
# On the Lambda instance:

# Create virtual environment (avoids package conflicts)
python3 -m venv ~/yolo_env
source ~/yolo_env/bin/activate

# Install ultralytics
pip install --upgrade pip
pip install numpy ultralytics

# Extract training data
cd /home/ubuntu
tar -xzf training_data.tar.gz
mv detection/training_data /home/ubuntu/training_data
cp lambda_dataset.yaml /home/ubuntu/training_data/dataset.yaml

# Start training
python -c "
from ultralytics import YOLO
model = YOLO('yolo26n.pt')
model.train(
    data='/home/ubuntu/training_data/dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=32,
    device=0,
    workers=8,
    project='runs',
    name='aoe2_yolo26',
    exist_ok=True
)
"
```

### Step 4: Download Model & Terminate

```bash
# From your local machine:
scp -i ~/.ssh/your-key.pem ubuntu@<IP>:/home/ubuntu/runs/aoe2_yolo26/weights/best.pt ./detection/inference/models/aoe2_yolo26.pt

# IMPORTANT: Terminate the instance in Lambda dashboard to stop billing!
```

### Cost Comparison

| Platform | Time | Cost |
|----------|------|------|
| Apple M2 Pro (local) | ~6.5 days | $0 (electricity) |
| Lambda Labs A100 | ~1 hour | ~$1.30 |
| Lambda Labs H100 | ~30 min | ~$1.65 |

---

## Inference Integration

### Python API
```python
from ultralytics import YOLO

class EntityDetector:
    def __init__(self, model_path="detection/inference/models/aoe2_yolo26.pt"):
        self.model = YOLO(model_path)
        # Class names loaded from classes.yaml (60 classes)
        # See detection/training/config/classes.yaml for the full list
        self.class_names = DEFAULT_CLASSES  # from detector.py

    def detect(self, screenshot, conf=0.5) -> list[dict]:
        results = self.model(screenshot, conf=conf)
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

### SAHI for High-Resolution Prelabeling

For prelabeling high-resolution screenshots (e.g., 3024x1964 retina), direct inference even at imgsz=1280 misses many objects because the image is still downscaled ~2.4x. SAHI (Slicing Aided Hyper Inference) solves this:

```bash
python -m detection.labeling.prelabel --model aoe2_yolo_v5.pt --sahi --conf 0.15
```

SAHI cuts the image into overlapping 640x640 tiles, runs inference on each, and merges results with NMS. Benchmarked on 3024x1964 retina screenshots:
- Without SAHI: 475 total detections (0 town centers)
- With SAHI: 8,108 total detections across 104 images

SAHI is used for **offline prelabeling only** — it takes ~978ms/image, too slow for real-time gameplay. Real-time detection uses `imgsz=1280` directly (~234ms, negligible vs 1-3s LLM call).

---

## File Locations

```
detection/
├── __init__.py                  # Package exports
├── inference/                   # Runtime detection
│   ├── detector.py              # EntityDetector class
│   └── models/
│       └── aoe2_yolo26.pt       # Trained model
├── training/                    # Training pipeline
│   ├── train_yolo.py            # YOLO training script
│   ├── generate_training_data.py # Synthetic data generator
│   ├── synthetic_data.py        # Data generation utilities
│   └── config/
│       └── classes.yaml         # Class definitions (60 classes)
├── extraction/                  # Sprite extraction
│   ├── sld_extractor.py         # SLD format parser
│   ├── extract_sprites.py       # Batch sprite extraction
│   └── capture_replay.py        # Screenshot capture from replays
├── testing/                     # Test scripts
│   └── test_real_detection.py   # Real screenshot validation
├── docs/                        # Documentation
│   ├── SLD_FORMAT.md            # SLD format specification
│   └── TRAINING_GUIDE.md        # This file
├── training_data/               # Generated dataset (gitignored)
└── real_screenshots/            # Captured screenshots (gitignored)
```

---

## Troubleshooting

### SLD Extraction Fails
- Try different frames: `--frame 1`, `--frame 2`
- Frame 0 (idle pose) is most reliable

### Poor Detection Accuracy
1. Increase training data (more synthetic images)
2. Add real screenshot backgrounds
3. Fine-tune on real game screenshots

### Lambda Training Issues
- Use virtual environment to avoid package conflicts
- Check GPU usage: `nvidia-smi`
- Monitor training: `tail -f training.log`

---

## References

- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [Lambda Labs Cloud](https://lambdalabs.com/)
- [openage SLD Format](https://github.com/SFTtech/openage/blob/master/doc/media/sld-files.md)
