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

**Model:** YOLO26n (2.5M parameters, 5.9 GFLOPs)

**Training Configuration:**
- Dataset: 3000 synthetic images (2400 train / 600 val)
- Epochs: 100
- Image size: 640x640
- Batch size: 32
- Classes: 55 (see `training/config/classes.yaml`)

**Final Metrics:**

| Metric | Value |
|--------|-------|
| **mAP50** | **86.0%** |
| **mAP50-95** | **71.9%** |
| Precision | 86.7% |
| Recall | 77.8% |
| Inference speed | 7.1ms/image |

**Per-Class Performance (Top 10):**

| Class | mAP50 | Class | mAP50 |
|-------|-------|-------|-------|
| trade_cart | 97.4% | castle | 96.9% |
| war_wagon | 97.1% | wonder | 96.8% |
| knight_line | 95.8% | fishing_ship | 94.4% |
| mangudai | 94.6% | camel_line | 95.1% |
| town_center | 92.9% | battle_elephant | 95.3% |

**Training Time:** ~1 hour on Lambda Labs A100

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

**Civilization Filtering**: Uses Western European (`b_west_*`) and Mediterranean (`b_medi_*`) building styles with multi-age variants (Dark, Feudal, Castle, Imperial).

**Exclusions**: Destruction animations, rubble sprites, civilization-specific unique buildings.

### Classes (55 total)

> **Note:** Class definitions are now maintained in `training/config/classes.yaml` as the single source of truth.

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
- Overlap management (max 30% overlap)
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
  5: town_center
  6: house
  7: lumber_camp
  8: mining_camp
  9: blacksmith
  10: dock
  11: university
  12: barracks
  13: archery_range
  14: stable
  15: monastery
  16: castle
  17: wonder
  18: gate
  19: sheep
  20: deer
  21: boar
  22: wolf
  23: villager
  24: trade_cart
  25: fishing_ship
  26: scout_line
  27: knight_line
  28: camel_line
  29: battle_elephant
  30: archer_line
  31: skirmisher_line
  32: cavalry_archer
  33: hand_cannoneer
  34: militia_line
  35: spearman_line
  36: eagle_line
  37: ram
  38: mangonel_line
  39: scorpion
  40: trebuchet
  41: monk
  42: king
  43: longbowman
  44: mangudai
  45: war_wagon
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
        self.class_names = [
            "tree", "gold_mine", "stone_mine", "berry_bush", "relic",
            "town_center", "house", "lumber_camp", "mining_camp", "blacksmith",
            "dock", "university", "barracks", "archery_range", "stable",
            "monastery", "castle", "wonder", "gate", "sheep", "deer", "boar",
            "wolf", "villager", "trade_cart", "fishing_ship", "scout_line",
            "knight_line", "camel_line", "battle_elephant", "archer_line",
            "skirmisher_line", "cavalry_archer", "hand_cannoneer", "militia_line",
            "spearman_line", "eagle_line", "ram", "mangonel_line", "scorpion",
            "trebuchet", "monk", "king", "longbowman", "mangudai", "war_wagon"
        ]

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
│       └── classes.yaml         # Class definitions (55 classes)
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
