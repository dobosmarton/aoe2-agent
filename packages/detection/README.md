# AoE2 Entity Detection System

YOLO-based object detection for identifying game entities (units, buildings, resources) in Age of Empires II: Definitive Edition screenshots.

## Overview

The detection system enables the AI agent to understand game state by detecting and localizing entities with bounding boxes. It targets a custom-trained YOLO26n model (Ultralytics, NMS-free) on synthetic training data generated from extracted game sprites.

```
Screenshot → YOLO Model → Detected Entities (class, bbox, confidence)
```

**Key Features:**
- 60 entity classes (units, buildings, resources, animals)
- Real-time inference: **single-pass at 1280px** (the deployed mode)
- Current model **v9** (YOLO26n, NMS-free): cleaned synthetic + real CVAT labels @1280; real-frame F1 ≈ 0.67 (see Model Performance). The served version is configured in `apps/agent/src/config.py` (`detection_model` / `AOE2_DETECTION_MODEL`) — that file is the single source of truth.
- **Adaptive SAHI** — smart tiling that only runs SAHI on regions with entities (~3-8 tiles vs ~18); *available in code but currently disabled* — single-pass at the training resolution wins because SAHI's tile scale doesn't match the training scale
- **Kalman filter object tracking** — 6D state vector with Hungarian algorithm assignment for persistent entity IDs
- **Tracker prediction mode** — extrapolate entity positions without inference (~0ms) when confidence is high
- **Frame differencing** — skip redundant rescans when the screen hasn't changed
- **ONNX batched SAHI** — all tiles in a single batched inference call (~3-5x faster than sequential)
- Dedup NMS across detections/SAHI tiles (the YOLO26 model head is itself NMS-free, but the detector still dedupes overlapping detections across detections and tiles on all backends — PyTorch, ONNX, Mock)
- SAHI sliced inference for large screenshots (640x640 tiles, 64px overlap)

## Quick Start

### Using the Detector

```python
from detection import EntityDetector, get_detector

# Get singleton detector instance
detector = get_detector()

# Detect entities — the deployed default is fast single-pass @1280
entities = detector.detect_fast(screenshot_bytes)

for entity in entities:
    print(f"{entity.entity_id}: {entity.class_name} at {entity.center} (conf: {entity.confidence:.2f})")
```

### From the Agent

The agent automatically uses detection when available:

```python
from detection.inference.detector import EntityDetector, get_detector

detector = get_detector(imgsz=1280)

# Fast single-pass @1280 — the DEPLOYED default (config.adaptive_sahi=False)
entities = detector.detect_fast(screenshot_bytes)

# Adaptive SAHI — fast scan + targeted SAHI on entity regions (only when adaptive_sahi=True)
entities = detector.detect_adaptive(screenshot_bytes, force_full=False)

# Full SAHI — tiles entire image (used on first turn and periodically when SAHI is on)
entities = detector.detect(screenshot_bytes)

# Tracker prediction — extrapolate positions without inference
predicted = detector.tracker.predict()
```

## Detection Modes

| Mode | Method | Tiles | Latency | When Used |
|------|--------|-------|---------|-----------|
| Fast single-pass | `detect_fast()` | 1 | ~50ms | **Deployed default** (`config.adaptive_sahi=False`) — every turn @1280 |
| Full SAHI | `detect()` | ~18 | ~234ms | Available; used when SAHI is enabled (first turn, every N turns, alarm) |
| Adaptive SAHI | `detect_adaptive()` | ~3-8 | ~100-200ms | Available when `adaptive_sahi=True` |
| Prediction | `tracker.predict()` | 0 | ~0ms | Rescan skip (confidence > 80%) |

**The deployed agent runs fast single-pass at `imgsz=1280`** — `config.adaptive_sahi` defaults to `False` because SAHI hurt v9's accuracy at retina resolution (scale mismatch; single-pass @1280 wins on real micro-F1). Adaptive and full SAHI remain in the library for when a model benefits from tiling: adaptive runs a fast single-pass scan, clusters detected entities into ROI regions, then runs SAHI only on those regions, falling back to full SAHI on the first turn, every `full_sahi_interval` turns (default 5), and on alarm. The latency figures above predate v9 and are indicative only.

## Directory Structure

```
detection/
├── __init__.py                  # Package exports (EntityDetector, get_detector)
│
├── inference/                   # Runtime detection
│   ├── detector.py              # EntityDetector (SAHI, adaptive SAHI, dedup NMS, tracking)
│   ├── onnx_layout.py           # Shared YOLO26 (N,6) ONNX decoder (decode_example, DetectionRow)
│   ├── tracker.py               # Kalman filter multi-object tracker
│   ├── frame_diff.py            # Frame differencing (skip unchanged frames)
│   ├── ownership.py             # Blue-dominance ownership classifier (own vs enemy)
│   └── models/
│       ├── aoe2_yolo_v9.onnx    # ONNX model (v9, served — single-pass @1280)
│       └── aoe2_yolo_v9.pt      # PyTorch model weights (v9)
│
├── training/                    # Training pipeline
│   ├── train_yolo.py            # YOLO training script (defaults to YOLO26n; --cls-gain/--box-gain/--dfl-gain knobs)
│   ├── export_onnx.py           # Export trained model to ONNX format
│   ├── generate_training_data.py # Synthetic image generator (rare-unit/cavalry rebalancing, distant-unit rendering)
│   ├── spike_yolo26_onnx.py     # YOLO26 ONNX output-shape + ARM64 latency spike (v6 go/no-go gate)
│   └── config/
│       └── classes.yaml         # Class definitions (single source of truth)
│
├── labeling/                    # Auto-labeling & re-labeling
│   ├── prelabel.py              # Auto-labeler (--open-vocab {yoloe,dinox})
│   ├── open_vocab.py            # Open-vocab backends (YOLOE local default / DINO-X hosted)
│   ├── open_vocab_mapping.py    # Open-vocab prompt → class mapping
│   └── hard_negatives.py        # Surface low-confidence cavalry confusions for re-labeling
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

## Model Performance (current: v9, YOLO26n @1280)

The served model is **`aoe2_yolo_v9`** (YOLO26n, NMS-free), trained on cleaned synthetic + real CVAT labels at imgsz=1280 and run **single-pass at 1280** (`adaptive_sahi=False`). SAHI tiling is disabled: it presents objects at a different scale than training, which *lowers* real-frame accuracy (the scale-match rule). The served version is set in `apps/agent/src/config.py` (`detection_model`); the detection server takes its model via `--model` at launch — keep the two in sync.

The metric of record is **real-frame** detection (via `evaluate_real.py`, single-pass at the training resolution) — not synthetic-validation mAP, which is optimistic because the validation set is synthetic-heavy:

| Metric (real frames) | v9 (@1280) | v7 (@640) | v6 (@640) |
|---|-----|-----|-----|
| F1 | **~0.67** | ~0.54 | ~0.41 |
| Recall | ~0.665 | ~0.45 | ~0.30 |
| Precision | ~0.676 | ~0.69 | ~0.67 |

v6 was 100% synthetic (≈0 real recall on animals/berries); v7 added real CVAT labels, which lifted real recall; v9 retrained on a cleaned synthetic set at imgsz=1280, enlarging small objects (berries/sheep) for another recall step. Known blind spot: military-unit recall on real frames is still near zero (knight/cavalry-archer/militia lines) — see `IMPROVEMENT-PLAN.md` P1.

> **Measure through the deployment path.** `evaluate_real.py` loaded via ultralytics *mismeasures* dynamic-axes ONNX exports (v9 read 0.21 that way); the raw-onnxruntime path — what the detection server runs — gives the true ~0.67.

## Object Tracking

The detection module includes a Kalman filter-based multi-object tracker (`detection/inference/tracker.py`) that provides persistent entity IDs across frames. This replaces the previous greedy IoU-only matching.

**How it works:**
1. Each tracked entity has a 6D state vector: `[x_center, y_center, vx, vy, width, height]`
2. On each cycle, the tracker **predicts** where entities moved using constant-velocity kinematics over the wall-clock seconds actually elapsed (velocities are px/second; extrapolation saturates at 1s)
3. New detections are **matched** to predicted tracks via the Hungarian algorithm (cost = `1 - IoU`, same-class constraint)
4. Matched tracks are **updated** with Kalman gain correction; unmatched detections create new tracks
5. Tracks are pruned after more than 3 consecutive misses

**Camera motion is not modelled.** A pan or zoom displaces every box at once, which no per-entity velocity can express, and a pan of roughly one box width can match tracks to their *neighbours* — a silent ID swap. Callers must call `tracker.reset()` when they move the camera; the agent wires this through `executor.set_tracks_invalidator()`.

**Prediction mode:** When the tracker has high confidence (`prediction_confidence() > 0.8`), the game loop can call `tracker.predict()` to extrapolate entity positions without running YOLO inference at all (~0ms). This is used in mid-turn rescans to save ~50-234ms per rescan.

**Fallback:** If scipy is unavailable for the Hungarian algorithm, a greedy IoU matcher is used. If the tracker module fails to import entirely, the detector falls back to the legacy `_assign_persistent_ids()` method.

For detailed Kalman filter math and tuning parameters, see [Chapter 7: Detector Architecture](../../docs/part3-entity-detection/07-detector-architecture.md).

## Dependencies

```
ultralytics>=8.0.0    # YOLO implementation
onnxruntime>=1.17.0   # ONNX batched inference (faster SAHI)
scipy>=1.11.0         # Hungarian algorithm for tracker (optional, has greedy fallback)
Pillow>=9.0.0         # Image processing
numpy>=1.21.0         # Array operations
pyyaml>=6.0           # Class config parsing
```

Install with:
```bash
pip install ultralytics onnxruntime scipy Pillow numpy pyyaml
```

Open-vocabulary auto-labeling (D3, `prelabel.py --open-vocab`) needs the optional `autolabel` extra and runs offline only:
```bash
pip install -e '.[autolabel]'
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

The detection module integrates with the main agent in `apps/agent/src/detection_phase.py`:

```python
# Detection is optional - agent falls back gracefully if unavailable
try:
    from detection.inference.detector import EntityDetector, get_detector
    DETECTION_AVAILABLE = True
except ImportError:
    DETECTION_AVAILABLE = False

# During game loop
if DETECTION_AVAILABLE:
    detector = get_detector(imgsz=config.detection_imgsz)  # default 1280

    # Main detection — adaptive SAHI by default
    if config.adaptive_sahi:
        force_full = (iteration == 1 or iteration % config.full_sahi_interval == 0 or alarm)
        entities = detector.detect_adaptive(screenshot_bytes, force_full=force_full)
    else:
        entities = detector.detect(screenshot_bytes)

    # Entity IDs persist across frames via Kalman tracker
    # LLM can reference entities by ID: "right_click on sheep_0"

    # Mid-turn rescan callback (inside action execution)
    if detector.tracker and detector.tracker.prediction_confidence() > 0.8:
        predicted = detector.tracker.predict()  # ~0ms, no YOLO inference
    else:
        rescan_entities = detector.detect_fast(new_screenshot)  # ~50ms single-pass
```

The executor resolves entity IDs to screen coordinates automatically. Window offset is re-fetched before each action to handle window movement.

## License

Part of the AoE2 LLM Arena project. For research and educational purposes.
