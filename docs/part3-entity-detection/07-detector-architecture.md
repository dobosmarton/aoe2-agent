# Chapter 7: Detector Architecture

The entity detection system runs YOLO inference on game screenshots, producing labeled bounding boxes with semantic IDs like `sheep_0` or `town_center_0`. It supports three backends (PyTorch, ONNX, Mock), a 60-class taxonomy, Kalman filter-based object tracking, and (optional, currently disabled) SAHI tiling.

> **What the agent actually runs (v6).** A **single forward pass at `imgsz=640`** — the model's training resolution. `config.detection_imgsz=640` and `config.adaptive_sahi=False`; the per-turn detection in `apps/agent/src/detection_phase.py` builds the detector with `use_sahi=config.adaptive_sahi` and calls `detect_fast`/`detect_fast_multi`. SAHI tiling (§7.4, §7.11) is still implemented but **off by default**, because on real screenshots it *lowers* accuracy (§7.4 — Why single-pass @640). The SAHI sections below document a path the agent keeps for a future model retrained at SAHI-native scale, not the deployed one.

<aside class="prereqs">

**New to YOLO?** Read [Appendix A — YOLO and object detection](../appendix/01-yolo-and-object-detection.md) first — it covers anchor boxes, NMS, IoU, mAP, and how the network actually turns pixels into bounding boxes. Also useful: [Chapter 2 — Game Loop Pipeline](../part1-architecture/02-game-loop-pipeline.md) for how this detector fits into the agent's per-turn loop.

</aside>

## 7.1 DetectedEntity

The core output type (`packages/detection/src/inference/detector.py`):

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

Defined in `packages/detection/src/training/config/classes.yaml` (source of truth). The detector loads classes dynamically: the PyTorch backend reads `model.names` at load time; ONNX and mock backends use `_load_default_classes()` which parses `classes.yaml` at import time (with a hardcoded fallback if YAML loading fails).

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

Defined at `packages/detection/src/inference/detector.py`. Key initialization parameters:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `model_path` | auto-detect | Path to .pt or .onnx model file |
| `confidence_threshold` | `0.35` | Minimum confidence for detections |
| `class_names` | loaded from `classes.yaml` | 60-class name list (PyTorch overrides with `model.names`) |
| `use_mock` | `False` | Use mock detections for testing |
| `imgsz` | `1280` | Constructor default. The agent overrides it to **`640`** via `config.detection_imgsz` (match training resolution) |
| `use_sahi` | `True` | Constructor default. The agent passes `use_sahi=config.adaptive_sahi`, i.e. **`False`** — single-pass, no tiling |
| `tracker` | auto-init | Kalman filter tracker for persistent IDs (see §7.9) |

### Model Loading

The trained artifact is **YOLO26n** (NMS-free), shipped as `aoe2_yolo_v6.onnx` / `aoe2_yolo_v6.pt`. The detector supports two model formats:

- **PyTorch (.pt)** -- loaded via ultralytics YOLO library. Requires `torch` and `ultralytics` packages. At load time, `_load_pytorch()` reads `model.names` from the model file to set `self.class_names` — this is authoritative and overrides the YAML-loaded defaults.
- **ONNX (.onnx)** -- loaded via `onnxruntime`. Cross-platform, works on ARM64 Windows where PyTorch may not be available. Uses `DEFAULT_CLASSES` loaded from `classes.yaml`. Session options: `ORT_ENABLE_ALL` graph optimization, 4 intra-op threads, auto-detected execution provider (DML or CPU).

If `model_path` is not specified, `get_detector()` resolves the model in this order (there is no multi-version fallback ladder — old model files were dropped, so there's no backward compatibility to maintain):

1. `models/aoe2_yolo_v6.onnx` -- preferred; this is the ARM64 deploy path (YOLO26, NMS-free)
2. `models/aoe2_yolo_v6.pt` -- used only if the ONNX export is absent
3. Mock mode -- if neither model file is found

### Detection Modes

The detector provides three detection methods:

| Method | Tiles | Relative cost | When Used |
|--------|-------|---------------|-----------|
| `detect_fast()` / `detect_fast_multi()` | 1 (no SAHI) | cheapest — one forward pass | **The deployed per-turn path** (`adaptive_sahi=False`) |
| `detect_adaptive()` | ~3-8 (ROI only) | moderate | Only when `config.adaptive_sahi=True` (off by default) |
| `detect()` | ~18 (full SAHI) | most expensive (an order of magnitude slower) | Only when `adaptive_sahi=True`, on forced full scans |

All three methods apply NMS and persistent ID assignment (Kalman tracker or greedy IoU fallback) before returning results. Exact latency is hardware- and backend-dependent (ONNX on the deploy VM vs. PyTorch on a dev Mac differ by an order of magnitude), so the table ranks the modes rather than pinning millisecond figures. `detect_fast_multi()` adds a center-crop second pass at 640 to recover small objects without paying for full tiling.

## 7.4 Why single-pass @640 (and why SAHI is off)

This is the design decision that drives the deployed detection path, so it comes first.

The intuition that bigger inputs detect more is wrong for *this* model. v6 (YOLO26n) was **trained at `imgsz=640`**, and a YOLO model detects best when inference objects appear at the scale it trained on. The agent therefore resizes the whole 3024×1672 screenshot down to 640 and runs **one** forward pass.

We validated this on held-out **real** screenshots with `evaluate_real.py` (per-class precision/recall/F1 by IoU matching — see §7.13). On the real split, the three candidate inference modes ranked:

| Inference mode | Real micro-F1 |
|----------------|---------------|
| **Single-pass @640** (training resolution) | **≈ 0.42** (P 0.65 / R 0.31) |
| Single-pass @1280 | ≈ 0.21 |
| Full SAHI (3024px → 640 tiles) | ≈ 0.04 |

SAHI is *worse*, not better, and the reason is scale. Tiling a 3024px screenshot into 640 crops shows the model objects at roughly **2.4× the size** they had in training (a 50px sheep becomes ~120px). The model never saw entities that large, so it misses or hallucinates them and real F1 collapses. This is why the agent pins `config.detection_imgsz=640` and `config.adaptive_sahi=False`.

> **The SAHI code stays — disabled.** The tiling machinery below (§7.4.1, §7.11) is fully implemented and tested. It is the right tool only once the model is retrained at a resolution whose SAHI tiles match the training scale; until then it is off. Treat the SAHI sections as "built, measured, parked," not as the current path.

### 7.4.1 Full SAHI Sliced Inference (disabled by default)

When `use_sahi=True`, images wider than 640px use SAHI (Slicing Aided Hyper Inference):

1. Tiles the screenshot into overlapping **640x640** chunks (model's training resolution)
2. Overlap: **64px** (10%) — prevents missing entities at tile boundaries
3. Runs YOLO on each tile at native resolution
4. Offsets each tile's detections by `(x_start, y_start)` to get original-image coordinates
5. Returns all entities; the unified NMS in `detect()` deduplicates overlapping detections

For a 3024x1672 screenshot that's ~18 tiles — an order of magnitude slower than a single pass, on top of the accuracy regression above. So even setting accuracy aside, it's the most expensive mode.

### Standard (single-pass) Inference — the deployed path

The agent runs this path. For images ≤640px wide, or whenever `use_sahi=False` (the agent's setting), the detector runs one standard ultralytics pass:

```python
results = self.model(image, conf=self.confidence_threshold, imgsz=self.input_size, verbose=False)
```

The `imgsz` parameter (constructor default 1280, set to **640** by the agent via `config.detection_imgsz`) controls inference resolution.

## 7.5 Backend: ONNX

YOLO26 is an end-to-end, **NMS-free** model: the ONNX graph already decodes and filters its own predictions, so it emits a single, fully-decoded layout. There is no longer a raw, pre-NMS output to post-process.

**Output layout**: shape `(num_boxes, 6)`, where each row is `[x1, y1, x2, y2, conf, class]` — corner (xyxy) coordinates in model-input pixels. The only work the backend does is scale those coordinates from the model input resolution back to the original screenshot dimensions.

Parsing lives in one shared module, **`packages/detection/src/inference/onnx_layout.py`**. Its `decode_example()` reads the raw ONNX output and returns a list of typed `DetectionRow`s. Both detection paths call into it:

- the single-image path, `detector._onnx_detect`, and
- the batched-SAHI path, `sahi.parse_onnx_tile`,

so the two can't drift apart in how they interpret model output.

`decode_example()` accepts **only** the `(num_boxes, 6)` layout. Any other shape raises **`UnknownOnnxLayoutError`** — it does not silently guess or auto-detect alternative formats. (The old dual-format handling, which sniffed `output.shape` to tell a post-NMS `(N, 6)` tensor apart from a raw `(4+num_classes, N)` tensor and ran `argmax` + thresholding + NMS to decode the latter, has been removed entirely along with the model's NMS head.)

> **Key Insight**: This is a deliberate trade of flexibility for correctness — a *single source of truth* for ONNX parsing plus *fail-loud* behavior on unexpected exports. If a future export changes shape, the detector throws immediately instead of silently mis-decoding boxes.

### ONNX Batched SAHI

When using the ONNX backend with SAHI, all tiles are batched into a single inference call via `_onnx_sahi_detect()`:

1. Tiles the image into 640×640 chunks (same overlap=64 as PyTorch SAHI)
2. Pads edge tiles to 640×640 with black pixels
3. Stacks all tiles into a single `(N, 3, 640, 640)` batch tensor
4. Runs **one** `session.run()` call for all tiles
5. Parses results per tile via `sahi.parse_onnx_tile` (which delegates to the shared `onnx_layout.decode_example()`) and offsets coordinates

This provides ~3-5x speedup over sequential PyTorch SAHI. The ONNX model must be exported with `dynamic=True` to support variable batch sizes (see `packages/detection/src/training/export_onnx.py`).

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

Implemented in `packages/detection/src/inference/tracker.py`. The `EntityTracker` maintains a list of `TrackedEntity` objects, each with a Kalman filter state that estimates position and velocity.

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
        # Resolve model file: aoe2_yolo_v6.onnx, else aoe2_yolo_v6.pt, else mock...
        _instance = EntityDetector(model_path=path, use_mock=use_mock, imgsz=imgsz)
    return _instance
```

The per-turn detection phase (`apps/agent/src/detection_phase.py`) calls `get_detector(use_mock=False, imgsz=config.detection_imgsz, use_sahi=config.adaptive_sahi)` once — i.e. `imgsz=640`, `use_sahi=False`. The same instance is reused for all subsequent detection calls, preserving the Kalman tracker state across frames.

## 7.11 Adaptive SAHI (Smart Tiling) — disabled by default

> **Off in v6.** Everything in this section is gated behind `config.adaptive_sahi`, which defaults to `False` (see §7.4 for why). The agent does **not** run adaptive SAHI; this documents the parked path for a future model retrained at SAHI-native scale.

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
| `config.adaptive_sahi` | **`False`** | Master switch for all SAHI. `False` ⇒ single-pass @640 (the deployed path); `True` ⇒ adaptive SAHI with full-SAHI fallback |
| `config.detection_imgsz` | `640` | Inference resolution — pinned to the training resolution |
| `config.full_sahi_interval` | `5` | Force full SAHI scan every N turns (only consulted when `adaptive_sahi=True`) |

### Relative cost (when SAHI is enabled)

| Mode | Tiles | Relative cost |
|------|-------|---------------|
| Full SAHI | ~18 | most expensive |
| Adaptive SAHI | ~3-8 | moderate |
| Single pass (deployed) | 1 | cheapest |

## 7.12 Frame Differencing

`packages/detection/src/inference/frame_diff.py` provides `FrameDiffer`, which compares consecutive screenshots to skip redundant mid-turn rescans.

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

Used in the game loop's rescan callback, after the tracker prediction check and before `detect_fast()`. Skips a redundant inference pass when nothing moved.

## 7.13 Measuring real performance & per-class thresholds

The metric of record is **real** F1, not synthetic mAP. A blended mAP over a ~95%-synthetic validation set hides real-world behaviour, so `packages/detection/src/testing/evaluate_real.py` scores **per-class precision/recall/F1 by IoU-matching** against the ground-truth labels in a `training_data_vN/val/` split, and reports real images **separately** from synthetic. It runs at the model's training resolution single-pass (`--mode detect_fast --imgsz 640`) — the realistic number. The latest real micro-F1 is **≈ 0.42** (precision 0.65, recall 0.31); the per-class breakdown (in `eval_real_summary.json`) is what exposes the rare-class tail — several cavalry and unique-unit lines still sit near zero recall on real frames.

`--conf-sweep` finds the best-F1 confidence per class and writes it to `recommended_thresholds`. Those promote into **`packages/detection/src/inference/thresholds.py`** — the single source both the detection server and the local detector read — via `detection.inference.sync_thresholds`, so each change lands as a reviewable git diff:

```python
DEFAULT_CONFIDENCE = 0.35
CLASS_THRESHOLDS = {           # lower thresholds for small / hard-to-detect classes
    "berry_bush": 0.25, "deer": 0.20, "relic": 0.20, "sheep": 0.20, "villager": 0.25,
}
```

See [Chapter 8 — Training Pipeline](./08-training-pipeline.md) for the sim-to-real levers (water-scene mode, real-data oversampling) that move these numbers, and [the v6 retrain runbook](../runbooks/retrain-detection-v6.md) for the end-to-end loop.

---

## Summary

- 60-class taxonomy organized by category (resources, buildings, units, siege, naval, animals)
- Three backends: PyTorch (ultralytics), ONNX Runtime, Mock
- Resolves the model as `aoe2_yolo_v6.onnx` (preferred, ARM64 deploy path), else `aoe2_yolo_v6.pt`, else mock — YOLO26n, NMS-free, no legacy version fallback
- **Deployed path: single forward pass at `imgsz=640`** (`detect_fast`/`detect_fast_multi`, `adaptive_sahi=False`) — matches training resolution; beats @1280 and SAHI on real F1
- **SAHI (full + adaptive)**: implemented and tested but **disabled** — scale mismatch lowers real accuracy; parked for a future SAHI-native retrain
- **ONNX batched SAHI**: all tiles in one inference call (~3-5x faster than sequential) — only relevant when SAHI is re-enabled
- **Kalman filter tracking**: 6D state with Hungarian algorithm matching for stable entity IDs
- **Tracker prediction**: skip rescans entirely when confidence > 80% (~0ms)
- **Frame differencing**: skip rescans when screenshot hasn't changed (MAD threshold 3%)
- **Real-eval harness + per-class thresholds**: `evaluate_real.py` (real F1 is the metric of record), thresholds synced into `thresholds.py`
- NMS applied to all backends and detection modes
- Greedy IoU ID assignment as tracker fallback

## Related Topics

- [Chapter 2: Game Loop Pipeline](../part1-architecture/02-game-loop-pipeline.md) -- detection integration, rescan flow
- [Chapter 3: Action Model & Execution](../part1-architecture/03-action-model-and-execution.md) -- how detections become click targets
- [Chapter 8: Training Pipeline](./08-training-pipeline.md) -- how the model is created
- [Chapter 13: Class Schema Evolution](../part5-operations/13-class-schema-evolution.md) -- the class taxonomy history and unified schema
