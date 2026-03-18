# Chapter 7: Detector Architecture

The entity detection system runs YOLO inference on game screenshots, producing labeled bounding boxes with semantic IDs like `sheep_0` or `town_center_0`. It supports three backends (PyTorch, ONNX, Mock), a 60-class taxonomy, Kalman filter-based object tracking, and adaptive SAHI for efficient high-resolution detection.

## 7.1 DetectedEntity

The core output type (`detection/inference/detector.py`):

```python
@dataclass
class DetectedEntity:
    id: str                          # "sheep_0", "villager_1"
    class_name: str                  # "sheep", "villager"
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixels
    center: tuple[float, float]      # (cx, cy) center point
    confidence: float                # 0-1
    area: float = 0                  # bbox area in pixels
```

`to_dict()` converts to a flat dict for the LLM context and executor cache. The `id` field follows the pattern `{class_name}_{counter}`, where counters are globally unique and persist across frames via Kalman filter tracking (see §7.9).

## 7.2 The 60-Class Taxonomy

Defined in `detection/training/config/classes.yaml` (source of truth). The detector loads classes dynamically: the PyTorch backend reads `model.names` at load time; ONNX and mock backends use `_load_default_classes()` which parses `classes.yaml` at import time (with a hardcoded fallback if YAML loading fails).

| Range | Category | Classes |
|-------|----------|---------|
| 0-8 | Resources & Nature | tree, gold_mine, stone_mine, berry_bush, relic, deer, boar, wolf, sheep |
| 9-16 | Economy Buildings | town_center, house, lumber_camp, mining_camp, mill, market, dock, farm |
| 17-24 | Military Buildings | barracks, archery_range, stable, blacksmith, siege_workshop, monastery, castle, university |
| 25-27 | Defensive | gate, wall, tower |
| 28-29 | Special Buildings | wonder, krepost |
| 30-32 | Civilian Units | villager, trade_cart, fishing_ship |
| 33-36 | Cavalry | scout_line, knight_line, camel_line, battle_elephant |
| 37-40 | Archers | archer_line, skirmisher_line, cavalry_archer, hand_cannoneer |
| 41-43 | Infantry | militia_line, spearman_line, eagle_line |
| 44-47 | Siege | ram, mangonel_line, scorpion, trebuchet |
| 48-49 | Monks & Special | monk, king |
| 50-54 | Unique Units | unique_archer, unique_cavalry, unique_infantry, unique_siege, unique_ship |
| 55-57 | Naval | fish, galley, fire_galley |
| 58 | Additional Siege | siege_tower |
| 59 | Animals | goose |

The `_line` suffix denotes unit upgrade paths (e.g., `militia_line` covers Militia through Champion). The `unique_` prefix groups civilization-specific units by combat type rather than by civilization -- there are too many unique units to have a class per civ.

## 7.3 EntityDetector Class

Defined at `detection/inference/detector.py`. Key initialization parameters:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `model_path` | auto-detect | Path to .pt or .onnx model file |
| `confidence_threshold` | `0.35` | Minimum confidence for detections |
| `class_names` | loaded from `classes.yaml` | 60-class name list (PyTorch overrides with `model.names`) |
| `use_mock` | `False` | Use mock detections for testing |
| `imgsz` | `1280` | Inference resolution (used for fast scan and standard inference) |
| `use_sahi` | `True` | Use SAHI sliced inference for large images |
| `tracker` | auto-init | Kalman filter tracker for persistent IDs (see §7.9) |

### Model Loading

The detector supports two model formats:

- **PyTorch (.pt)** -- loaded via ultralytics YOLO library. Requires `torch` and `ultralytics` packages. At load time, `_load_pytorch()` reads `model.names` from the model file to set `self.class_names` — this is authoritative and overrides the YAML-loaded defaults.
- **ONNX (.onnx)** -- loaded via `onnxruntime`. Cross-platform, works on ARM64 Windows where PyTorch may not be available. Uses `DEFAULT_CLASSES` loaded from `classes.yaml`. Session options: `ORT_ENABLE_ALL` graph optimization, 4 intra-op threads, auto-detected execution provider (DML or CPU).

If `model_path` is not specified, `get_detector()` auto-detects using a priority chain:

1. `models/aoe2_yolo_v6.onnx` -- v6 ONNX (YOLO26, NMS-free, fastest)
2. `models/aoe2_yolo_v6.pt` -- v6 PyTorch
3. `models/aoe2_yolo_v5.onnx` -- v5 ONNX (batched SAHI support)
4. `models/aoe2_yolo_v5.pt` -- v5 PyTorch
5. `models/aoe2_yolo_v2.onnx` -- v2 ONNX (hybrid trained, fallback)
6. `models/aoe2_yolo_v2.pt` -- v2 PyTorch
7. `models/aoe2_yolo26.onnx` -- v1 ONNX (last resort)
8. `models/aoe2_yolo26.pt` -- v1 PyTorch
9. Mock mode -- if no model file found

### Detection Modes

The detector provides three detection methods:

| Method | Tiles | Speed | When Used |
|--------|-------|-------|-----------|
| `detect()` | ~18 (full SAHI) | ~300ms | Forced full scans |
| `detect_adaptive()` | ~3-8 (ROI only) | ~100-200ms | Default per-turn detection |
| `detect_fast()` | 1 (no SAHI) | ~50-100ms | Mid-turn rescans |

All three methods apply NMS and persistent ID assignment (Kalman tracker or greedy IoU fallback) before returning results.

## 7.4 Backend: PyTorch

### Full SAHI Sliced Inference

When `use_sahi=True` (default), images wider than 640px use SAHI (Slicing Aided Hyper Inference). On Retina displays (3024x1672), standard inference resizes to `imgsz=1280`, shrinking small entities like sheep to ~21px — below the model's reliable detection threshold. SAHI solves this:

1. Tiles the screenshot into overlapping **640x640** chunks (model's training resolution)
2. Overlap: **64px** (10%) — prevents missing entities at tile boundaries
3. Runs YOLO on each tile at native resolution (~40ms/tile)
4. Offsets each tile's detections by `(x_start, y_start)` to get original-image coordinates
5. Returns all entities; the unified NMS in `detect()` deduplicates overlapping detections

For a 3024x1672 screenshot: ~18 tiles × ~40ms ≈ 720ms total. This is used for full scans; most turns use adaptive SAHI instead (see §7.11).

> **Note**: Full SAHI is now the fallback. The default detection path uses adaptive SAHI (§7.11), which runs SAHI only on ROI regions around detected entities, reducing tile count to ~3-8.

### Standard Inference (fallback)

For images ≤640px wide, or when `use_sahi=False`, the detector falls back to standard ultralytics inference:

```python
results = self.model(image, conf=self.confidence_threshold, imgsz=self.input_size, verbose=False)
```

The `imgsz` parameter (default 1280, configurable via `config.detection_imgsz`) controls inference resolution.

## 7.5 Backend: ONNX

The ONNX backend handles two different output formats because ultralytics exports vary by version:

**Format 1 (post-NMS)**: Output shape `(batch, N, 6)` where each detection is `[x1, y1, x2, y2, confidence, class_id]`. These are ready to use after coordinate scaling.

**Format 2 (raw predictions)**: Output shape `(batch, 4+num_classes, N)`. Requires:
1. Transpose to `(N, 4+num_classes)`
2. Extract bbox `[x_center, y_center, width, height]` from first 4 columns
3. Class scores from remaining columns -- `argmax` for class ID, `max` for confidence
4. Filter by confidence threshold
5. Convert from center format to corner format `(x1, y1, x2, y2)`
6. Apply per-class NMS (IoU-based suppression)

Both formats require scaling coordinates from the model input resolution back to the original screenshot dimensions.

> **Key Insight**: The ONNX backend auto-detects which output format it receives by checking `output.shape[2]`. If shape[2] == 6, it's post-NMS. If shape[1] == 4+num_classes, it's raw predictions. This makes the detector resilient to ultralytics version changes without requiring model re-export.

### ONNX Batched SAHI

When using the ONNX backend with SAHI, all tiles are batched into a single inference call via `_onnx_sahi_detect()`:

1. Tiles the image into 640×640 chunks (same overlap=64 as PyTorch SAHI)
2. Pads edge tiles to 640×640 with black pixels
3. Stacks all tiles into a single `(N, 3, 640, 640)` batch tensor
4. Runs **one** `session.run()` call for all tiles
5. Parses results per tile via `_parse_onnx_tile()` and offsets coordinates

This provides ~3-5x speedup over sequential PyTorch SAHI. The ONNX model must be exported with `dynamic=True` to support variable batch sizes (see `detection/training/export_onnx.py`).

## 7.6 Backend: Mock

For testing without a trained model. Generates plausible Dark Age detections:

- 1 town_center at center-ish position
- 2-4 sheep scattered nearby
- 3 villagers near the TC
- 1 scout offset from the TC

Uses deterministic positions (not random) so test results are reproducible.

## 7.7 Spatial Queries

Utility methods for finding specific entities:

**`find_entity_by_id(entity_id)`** -- linear search by ID string. Returns `DetectedEntity` or `None`.

**`find_entities_by_class(class_name)`** -- filter all detections by class. Returns list.

**`find_nearest_entity(x, y, class_name=None)`** -- Euclidean distance search. Optionally filtered by class. Returns the closest entity.

These are available for any code that needs to query detection results beyond the basic cache used by the executor.

## 7.8 NMS for All Backends

Non-maximum suppression is applied in all detection methods (`detect()`, `detect_fast()`, `detect_adaptive()`) after backend-specific inference, ensuring consistent duplicate removal regardless of whether PyTorch, ONNX, or Mock is used:

```python
def detect(self, screenshot):
    # ... backend dispatch ...
    entities = self._nms(entities, iou_threshold=0.5)
    if self.tracker:
        entities = self.tracker.update(entities)
    else:
        entities = self._assign_persistent_ids(entities)
    return entities
```

The `_nms()` method sorts entities by confidence (highest first) and removes lower-confidence boxes that overlap >50% IoU with a higher-confidence box of the same class.

## 7.9 Entity Tracking & ID Persistence

Entity IDs persist across detection frames so the LLM can consistently reference the same entity across turns (e.g., `sheep_0` remains `sheep_0`). The detector uses a two-tier tracking system: a **Kalman filter tracker** (primary) with a **greedy IoU matcher** (fallback).

### Kalman Filter Tracker (Primary)

Implemented in `detection/inference/tracker.py`. The `EntityTracker` maintains a list of `TrackedEntity` objects, each with a Kalman filter state that estimates position and velocity.

**State Vector** (6D):

```
state = [x_center, y_center, vx, vy, width, height]
```

- Position `(x, y)`: bounding box center in pixel coordinates
- Velocity `(vx, vy)`: estimated motion per frame (used for prediction)
- Size `(w, h)`: bounding box dimensions

**Kalman Filter Matrices**:

| Matrix | Dimensions | Purpose |
|--------|-----------|---------|
| `F` (transition) | 6×6 | Constant velocity model: `x += vx`, `y += vy` |
| `H` (measurement) | 4×6 | Observes `[x, y, w, h]` from YOLO detections |
| `Q` (process noise) | 6×6 | `diag([10, 10, 5, 5, 2, 2])²` — tuned for AoE2 unit speeds (~5-20 px/frame) |
| `R` (measurement noise) | 4×4 | `diag([5, 5, 3, 3])²` — tuned for YOLO bbox jitter (~3-5 px) |
| `P₀` (initial covariance) | 6×6 | `diag([10, 10, 100, 100, 10, 10])²` — high velocity uncertainty initially |

**Per-frame cycle:**

1. **Predict**: For each existing track, advance state using the constant velocity model: `state = F @ state`, `P = F @ P @ Fᵀ + Q`. This projects where each entity should be *before* seeing new detections.

2. **Match**: Build a cost matrix `(num_tracks × num_detections)` using `1 - IoU` between predicted track bounding boxes and new YOLO detections. Same-class constraint: cost is set to 1.0 (maximum) for class mismatches, ensuring a sheep track never matches a villager detection.

3. **Assign**: Solve the assignment problem using the **Hungarian algorithm** (`scipy.optimize.linear_sum_assignment`) for globally optimal matching. Falls back to greedy matching if scipy is not installed. A match is accepted only if the cost is below `1 - iou_threshold` (default: IoU > 0.3).

4. **Update**: For matched tracks, apply the Kalman update step: compute innovation `y = z - H @ state`, Kalman gain `K`, and correct the state estimate. Reset `misses = 0`, increment `hits`.

5. **Handle unmatched**: Unmatched tracks get `misses += 1`. Unmatched detections spawn new tracks with zero initial velocity and high covariance. Tracks with `misses > max_misses` (default 3) are pruned.

6. **Output**: Return `DetectedEntity` list from all active tracks (misses = 0), with stable IDs that persist across frames.

**Track Lifecycle**:

```
Detection → New Track (id="sheep_0", velocity=0)
     ↓ (matched next frame)
Active Track (Kalman update, velocity estimated)
     ↓ (matched next frame)
Active Track (velocity refined, position predicted)
     ↓ (unmatched — entity temporarily occluded)
Missing Track (misses=1, still predicting position)
     ↓ (unmatched again)
Missing Track (misses=2)
     ↓ (unmatched again)
Dead Track (misses=3, pruned from tracker)
```

### Prediction Mode

`tracker.predict()` extrapolates entity positions using the Kalman predict step *without* new detections. This is used in the game loop's rescan callback: when tracker confidence exceeds 80%, the rescan skips screenshot capture and YOLO inference entirely, using predicted positions instead (~0ms vs ~100ms).

```python
# In game_loop.py rescan callback:
if detector.tracker and detector.tracker.get_confidence() > 0.8:
    predicted = detector.tracker.predict()  # Instant — no inference
    set_detected_entities(predicted)
    return
```

Confidence is computed as `active_tracks / total_tracks`. If many tracks are lost (misses > 0), confidence drops and actual detection is triggered.

### Greedy IoU Fallback

If the Kalman tracker is unavailable (e.g., scipy not installed), the detector falls back to `_assign_persistent_ids()`, a simpler greedy IoU matcher:

- For each new detection, find the best IoU match among previous same-class entities
- If IoU > 0.4, reuse the old entity's ID
- Otherwise, assign a new globally unique ID (counter never resets)

This provides basic ID persistence but lacks velocity estimation and optimal assignment.

## 7.10 Singleton Access

`get_detector()` at the bottom of `detector.py` provides a singleton with auto-detection:

```python
_instance: Optional[EntityDetector] = None

def get_detector(model_path=None, use_mock=False, imgsz=1280) -> EntityDetector:
    global _instance
    if _instance is None:
        # Auto-detect model file in priority order (v6 > v5 > v2 > v1)...
        _instance = EntityDetector(model_path=path, use_mock=use_mock, imgsz=imgsz)
    return _instance
```

The game loop calls `get_detector(imgsz=config.detection_imgsz)` once during initialization. The same instance is reused for all subsequent detection calls, preserving the Kalman tracker state across frames.

## 7.11 Adaptive SAHI (Smart Tiling)

Full SAHI tiles the entire screenshot (~18 tiles for 3024×1672). Most tiles cover static terrain with no entities. Adaptive SAHI reduces this to ~3-8 tiles by running SAHI only on regions of interest around detected entities.

### How It Works

`detect_adaptive(screenshot, force_full=False)` implements a two-phase detection:

```
Phase 1: Fast Scan        Phase 2: Targeted SAHI
┌──────────────────┐      ┌──────────────────┐
│ Full screenshot   │      │                  │
│ at imgsz=1280    │      │  ┌───┐    ┌────┐ │
│                  │  →   │  │ROI│    │ROI │ │
│ Single-pass YOLO │      │  │ 1 │    │ 2  │ │
│ (~50-100ms)      │      │  └───┘    └────┘ │
│                  │      │  SAHI on ROIs     │
│                  │      │  (~3-8 tiles)     │
└──────────────────┘      └──────────────────┘
```

1. **Fast scan**: Run single-pass YOLO at `imgsz=1280` on the full screenshot. This detects most entities but may miss very small objects (~20px sheep at 3024px width).

2. **ROI computation** (`_compute_sahi_rois()`): Cluster detected entities into groups and compute padded bounding regions:
   - **Union-Find clustering**: Entities within 200px of each other are grouped into the same cluster
   - **Disappeared entities**: Previous-frame entities not found in the fast scan are included (they may have moved just beyond fast-pass detection range)
   - **Padding**: 128px added around each cluster's bounding box
   - **ROI merging**: Overlapping ROI regions are merged to avoid redundant tiles

3. **Targeted SAHI** (`_sahi_detect_rois()`): Tile only the ROI regions into 640×640 chunks with 64px overlap. If using ONNX, all ROI tiles are batched into a single inference call. If using PyTorch, tiles are processed sequentially.

4. **Merge** (`_merge_detections()`): Combine results from both phases:
   - Fast entities whose center falls **outside** all ROIs → kept (reliable at full resolution)
   - SAHI entities from **inside** ROI regions → kept (more accurate at native 640)
   - NMS deduplicates at ROI boundaries

5. **Post-processing**: Apply NMS and Kalman tracker (or greedy IoU fallback) to assign persistent IDs.

### Force-Full Triggers

Adaptive SAHI reverts to full SAHI via `detect()` when:

- **First iteration**: No previous entities to guide ROI placement
- **Periodic interval**: Every `full_sahi_interval` turns (default 5) to catch entities the fast scan may consistently miss
- **Alarm**: When enemy threats were detected on the previous turn (need maximum detection coverage)

```python
# In game_loop.py:
force_full = (
    iteration == 1
    or iteration % config.full_sahi_interval == 0
    or alarm
)
detected_entities = detector.detect_adaptive(screenshot, force_full=force_full)
```

### Configuration

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `config.adaptive_sahi` | `True` | Enable adaptive SAHI (falls back to full SAHI if `False`) |
| `config.full_sahi_interval` | `5` | Force full SAHI scan every N turns |

### Performance

| Mode | Tiles | Approx. Time |
|------|-------|-------------|
| Full SAHI | ~18 | ~300ms (ONNX batch) |
| Adaptive SAHI | ~3-8 | ~100-200ms |
| Fast scan only (no ROIs) | 1 | ~50-100ms |

## 7.12 Frame Differencing

`detection/inference/frame_diff.py` provides `FrameDiffer`, which compares consecutive screenshots to skip redundant mid-turn rescans.

### How It Works

1. **Downscale**: Convert screenshot to 320×180 grayscale
2. **Crop**: Remove top 4% (resource bar — changes frequently with resource counts but doesn't indicate game state changes)
3. **Compare**: Compute Mean Absolute Difference (MAD) against the previous frame
4. **Threshold**: If MAD < 0.03 (3% average pixel change), the frame is considered unchanged

```python
differ = FrameDiffer(threshold=0.03)

# In the rescan callback:
if not differ.has_changed(screenshot):
    return  # Skip detection, reuse previous entities
entities = detector.detect_fast(screenshot)
differ.update(screenshot)
```

Used in the game loop's rescan callback, after the tracker prediction check and before `detect_fast()`. Saves ~50-100ms per skipped rescan.

---

## Summary

- 60-class taxonomy organized by category (resources, buildings, units, siege, naval, animals)
- Three backends: PyTorch (ultralytics), ONNX Runtime, Mock
- Auto-detects model with fallback chain: v6 (YOLO26) > v5 (YOLO11) > v2 > v1
- **Adaptive SAHI** (default): fast scan + targeted SAHI on entity clusters (~3-8 tiles vs ~18)
- **ONNX batched SAHI**: all tiles in one inference call (~3-5x faster than sequential)
- **Kalman filter tracking**: 6D state with Hungarian algorithm matching for stable entity IDs
- **Tracker prediction**: skip rescans entirely when confidence > 80% (~0ms)
- **Frame differencing**: skip rescans when screenshot hasn't changed (MAD threshold 3%)
- NMS applied to all backends and detection modes
- Greedy IoU ID assignment as tracker fallback

## Related Topics

- [Chapter 2: Game Loop Pipeline](../part1-architecture/02-game-loop-pipeline.md) -- detection integration, rescan flow
- [Chapter 3: Action Model & Execution](../part1-architecture/03-action-model-and-execution.md) -- how detections become click targets
- [Chapter 8: Training Pipeline](./08-training-pipeline.md) -- how the model is created
- [Chapter 13: Class Schema Evolution](../part5-operations/13-class-schema-evolution.md) -- the class taxonomy history and unified schema
