# Chapter 8: Training Pipeline

The YOLO model is trained on a hybrid dataset: synthetic images generated from extracted game sprites, plus real screenshots labeled in CVAT. This chapter covers synthetic data generation, augmentation, and the YOLO training process.

## 8.1 Pipeline Overview

```mermaid
flowchart LR
    SLD["SLD Game Files<br/>(game assets)"] -->|sld_extractor.py| PNG["Sprite PNGs<br/>(RGBA, per-class)"]
    BG["Real Screenshots<br/>(blurred backgrounds)"] --> GEN
    PNG -->|generate_training_data.py| GEN["Synthetic Dataset<br/>(2400 train / 600 val)"]
    CVAT["CVAT Labeled<br/>Real Screenshots"] -->|prepare_training.py| MERGE["Hybrid Dataset<br/>(classes.yaml IDs)"]
    GEN -->|"copy (same IDs)"| MERGE
    MERGE -->|train_yolo.py| MODEL["YOLO Model<br/>(60 classes)"]
    MODEL -->|export| ONNX["aoe2_yolo.onnx"]
    MODEL -->|copy| PT["aoe2_yolo.pt"]
```

## 8.2 Synthetic Data Generation

`packages/detection/src/training/generate_training_data.py` generates labeled training images by compositing sprites onto backgrounds.

### Sprite Configurations

53 sprite configurations define how each entity type appears in generated images. Each config specifies:

| Field | Example | Purpose |
|-------|---------|---------|
| `class_id` | `8` | YOLO class ID (matches classes.yaml directly) |
| `class_name` | `"sheep"` | Human-readable name |
| `sprite_patterns` | `["u_sheep_idle*_x1.sld"]` | Glob patterns for sprite files |
| `scale_range` | `(0.8, 1.2)` | Random size variation |
| `count_range` | `(2, 6)` | Min/max instances per image |
| `z_order` | `2` | Rendering layer (0=back, 3=front) |
| `avoid_edges` | `True` | Keep sprites away from image borders |
| `min_spacing` | `30` | Minimum distance between same-class instances |

### Z-Order Layering

Sprites are rendered in z-order to simulate realistic occlusion:

| z_order | Category | Examples |
|---------|----------|----------|
| 0 | Resources | trees, gold mines, stone mines |
| 1 | Buildings | town center, barracks, houses |
| 2 | Animals | sheep, deer, boar, wolf |
| 3 | Units | villagers, scouts, military units |

### Placement Algorithm

For each image generation:

1. Sort sprite configs by z_order
2. For each config, pick random count from `count_range`
3. For each sprite instance:
   - Apply random scale from `scale_range`
   - Try up to 20 random positions
   - Check overlap with z-order-aware thresholds: buildings 10%, resources 15%, units 35%
   - **Skip placement** if overlap limit exceeded (no force-place)
4. Paste sprite with alpha transparency
5. Generate YOLO-format label: `class_id x_center y_center width height` (all normalized 0-1)

> **v5 improvement**: Z-order-aware overlap thresholds replaced the flat 40% IoU threshold from earlier versions. Buildings overlap less (10%) since they're large and static, while units tolerate more overlap (35%) since they cluster in groups. Sprites that can't find a valid position are skipped entirely rather than force-placed, reducing label noise.

### Background Sources

Three background types, selected randomly per image:

1. **Real screenshots** (50% probability via `real_background_ratio=0.5`) -- actual game screenshots from `packages/detection/src/real_screenshots/raw/`, Gaussian-blurred with radius=1 to reduce overfitting on specific game states while preserving terrain colors and textures.

2. **Synthetic backgrounds** -- pre-generated terrain images.

3. **Procedural terrain** -- generated at runtime with biome-aware color palettes. A biome is selected randomly (weighted) from 9 types: grass (25%), desert (15%), snow (10%), autumn (10%), jungle (10%), dirt (10%), mixed (10%), water_shore (5%), dark_forest (5%). Each biome defines 5 terrain colors used for 20 elliptical patches (200-500px) with Gaussian blur (radius=3). The "mixed" biome merges colors from 2-3 random biomes.

## 8.3 Augmentation Pipeline

### Basic Augmentations

Applied with independent probabilities per image:

| Augmentation | Probability | Parameters | Purpose |
|--------------|-------------|------------|---------|
| Brightness | 50% | 0.7-1.3x | Day/night, shadows |
| Contrast | 50% | 0.8-1.2x | Monitor variation |
| Saturation | 30% | 0.8-1.2x | Color variation |
| Gaussian blur | 20% | radius=0.5 | Slight defocus |

### Enhanced Augmentations (v2)

Game-realistic effects that simulate actual screenshot conditions:

**Fog of War** (30% chance) -- 1-4 dark patches at image edges with opacity 80-150, simulating unexplored areas at map borders.

**UI Element Simulation** (20% chance) -- dark rectangles mimicking the minimap (130-180px at corner) and resource bar (25-40px at top). Teaches the model to ignore UI overlays.

**JPEG Compression** (30% chance) -- re-encodes at quality 70-90, simulating screenshot compression artifacts.

**Scale Variation** (30% chance) -- 0.7-1.3x zoom with center crop or padding. Simulates different camera zoom levels and screen resolutions.

**Color Temperature** (20% chance) -- warm shift (boost R, reduce B) for desert maps or cool shift (reduce R, boost B) for winter maps.

**Vignette** (15% chance) -- radial gradient darkening at edges, simulating viewport effects.

> **Key Insight**: The `flipud=0.0` setting in YOLO training is deliberate. AoE2 uses an isometric camera at a fixed angle -- units never appear upside-down. Vertical flipping would create unrealistic training samples with upside-down buildings and units, confusing the model. Horizontal flip (`fliplr=0.5`) is fine because units face both left and right.

## 8.4 YOLO Training

`packages/detection/src/training/train_yolo.py` trains a YOLO11 nano model:

### Model

Base model: `yolo11n.pt` (YOLO11 nano) -- ~6MB, optimized for real-time inference on consumer hardware. The nano variant was chosen for speed; each detection call needs to complete within the 2-second loop cycle.

### Hyperparameters

Tuned for isometric game graphics:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `epochs` | 150 | Sufficient for convergence with early stopping |
| `batch` | 16 | Fits in GPU memory (A100 40GB) |
| `imgsz` | 640 | Standard YOLO input size |
| `patience` | 20 | Early stopping patience |
| `hsv_h` | 0.015 | Slight hue variation |
| `hsv_s` | 0.7 | Saturation augmentation |
| `hsv_v` | 0.4 | Brightness augmentation |
| `degrees` | 10 | Small rotation (units face different directions) |
| `translate` | 0.1 | Position shift |
| `scale` | 0.5 | Zoom variation |
| `flipud` | 0.0 | **No vertical flip** (isometric constraint) |
| `fliplr` | 0.5 | Horizontal flip OK |
| `mosaic` | 1.0 | Full mosaic augmentation |
| `mixup` | 0.1 | Light MixUp for regularization |

### Dataset Structure

```
training_data/
├── train/
│   ├── images/   # hybrid: synthetic + real tiles
│   └── labels/   # YOLO .txt files (class_id cx cy w h)
├── val/
│   ├── images/
│   └── labels/
└── dataset.yaml  # Paths + 60 class names (classes.yaml IDs)
```

### Output

Training produces:
- `runs/aoe2_yolo_v2/weights/best.pt` -- best validation mAP checkpoint
- Optionally exported to ONNX with `--export-onnx` flag
- Copied to `packages/detection/src/inference/models/aoe2_yolo_v2.pt` and `.onnx`

### Results

**v5 model (latest):** **92.2% mAP50**, **85.4% mAP50-95** on validation data, 60 classes.

| Metric | v5 | v4 (previous) |
|--------|-----|---------------|
| mAP50 | **92.2%** | 86.8% |
| mAP50-95 | **85.4%** | 72.3% |
| Precision | **94.8%** | 87.1% |
| Recall | **89.2%** | 78.5% |

**v5 dataset:** 18,520 images total (15,120 train + 3,400 val):
- 8,000 synthetic train + 2,000 synthetic val
- 7,120 real train + 1,400 real val

See [Chapter 12](../part5-operations/12-cloud-training.md) for cloud training details.

---

## Summary

- Synthetic data: sprite compositing with z-order, z-order-aware overlap thresholds (buildings 10%, resources 15%, units 35%)
- 53 sprite configs using classes.yaml IDs directly (no remapping needed)
- 17+ architecture styles per building via wildcard patterns
- 6 enhanced augmentations simulating real game conditions (fog, UI, compression, zoom, temperature, vignette)
- YOLO11 nano model: 150 epochs, isometric-tuned hyperparameters
- v5 model: 92.2% mAP50 on 18,520-image hybrid dataset (8k synthetic + 7.1k real train, 2k synthetic + 1.4k real val)

## Related Topics

- [Chapter 7: Detector Architecture](./07-detector-architecture.md) -- how the trained model is used at runtime
- [Chapter 9: Labeling & Active Learning](./09-labeling-and-active-learning.md) -- how real data is labeled and merged
- [Chapter 11: Sprite Extraction](../part4-game-knowledge/11-sprite-extraction.md) -- how sprites are extracted from game files
- [Chapter 12: Cloud Training](../part5-operations/12-cloud-training.md) -- Lambda Labs training workflow
