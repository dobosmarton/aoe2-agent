# Chapter 7: Detector Architecture

The entity detection system runs YOLO inference on game screenshots, producing labeled bounding boxes with semantic IDs like `sheep_0` or `town_center_0`. It supports three backends (PyTorch, ONNX, Mock) and a 60-class taxonomy covering resources, buildings, and military units.

## 7.1 DetectedEntity

The core output type (`detection/inference/detector.py:23-41`):

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

`to_dict()` converts to a flat dict for the LLM context and executor cache. The `id` field follows the pattern `{class_name}_{counter}`, where counters are globally unique and persist across frames via IoU-based tracking (see §7.9).

## 7.2 The 60-Class Taxonomy

Defined in `detection/training/config/classes.yaml` (source of truth) and mirrored in `detection/inference/detector.py` as `DEFAULT_CLASSES`:

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

Defined at `detection/inference/detector.py:123+`. Key initialization parameters:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `model_path` | auto-detect | Path to .pt or .onnx model file |
| `confidence_threshold` | `0.35` | Minimum confidence for detections |
| `class_names` | `DEFAULT_CLASSES` | 60-class name list |
| `use_mock` | `False` | Use mock detections for testing |
| `imgsz` | `1280` | Inference resolution (higher = more detections on large screenshots) |

### Model Loading

The detector supports two model formats:

- **PyTorch (.pt)** -- loaded via ultralytics YOLO library. Requires `torch` and `ultralytics` packages.
- **ONNX (.onnx)** -- loaded via `onnxruntime`. Cross-platform, works on ARM64 Windows where PyTorch may not be available.

If `model_path` is not specified, `get_detector()` auto-detects using a priority chain:

1. `models/aoe2_yolo_v5.pt` -- v5 PyTorch (preferred, latest)
2. `models/aoe2_yolo_v2.onnx` -- v2 ONNX
3. `models/aoe2_yolo_v2.pt` -- v2 PyTorch
4. `models/aoe2_yolo26.onnx` -- v1 ONNX (fallback)
5. `models/aoe2_yolo26.pt` -- v1 PyTorch (fallback)
6. Mock mode -- if no model file found

### Detection Pipeline

`detect(screenshot)` accepts JPEG bytes or a PIL Image:

1. Resets per-class entity counters
2. Routes to the appropriate backend based on loaded model type
3. Applies NMS (IoU threshold 0.5) to remove duplicate overlapping detections -- applied to **all** backends (PyTorch, ONNX, Mock)
4. Assigns persistent entity IDs via IoU-based tracking across frames (see §7.9)
5. Returns `list[DetectedEntity]` sorted by class name, then confidence descending

## 7.4 Backend: PyTorch

Uses ultralytics YOLO for inference:

```python
results = self.model(image, conf=self.confidence_threshold, imgsz=self.input_size, verbose=False)
for box in results[0].boxes:
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    confidence = float(box.conf[0])
    class_id = int(box.cls[0])
```

The `imgsz` parameter (default 1280, configurable via `config.detection_imgsz`) controls inference resolution. Ultralytics handles image preprocessing (resize to imgsz) and post-processing internally. An additional NMS pass is applied in the unified `detect()` method across all backends.

## 7.5 Backend: ONNX

The ONNX backend handles two different output formats because ultralytics exports vary by version:

**Format 1 (post-NMS)**: Output shape `(1, N, 6)` where each detection is `[x1, y1, x2, y2, confidence, class_id]`. These are ready to use after coordinate scaling.

**Format 2 (raw predictions)**: Output shape `(1, 4+num_classes, N)`. Requires:
1. Transpose to `(N, 4+num_classes)`
2. Extract bbox `[x_center, y_center, width, height]` from first 4 columns
3. Class scores from remaining columns -- `argmax` for class ID, `max` for confidence
4. Filter by confidence threshold
5. Convert from center format to corner format `(x1, y1, x2, y2)`
6. Apply per-class NMS (IoU-based suppression)

Both formats require scaling coordinates from the 640x640 model input back to the original screenshot dimensions.

> **Key Insight**: The ONNX backend auto-detects which output format it receives by checking `output.shape[2]`. If shape[2] == 6, it's post-NMS. If shape[1] == 4+num_classes, it's raw predictions. This makes the detector resilient to ultralytics version changes without requiring model re-export.

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

Non-maximum suppression is applied in the unified `detect()` method after backend-specific inference, ensuring consistent duplicate removal regardless of whether PyTorch, ONNX, or Mock is used:

```python
def detect(self, screenshot):
    # ... backend dispatch ...
    entities = self._nms(entities, iou_threshold=0.5)
    entities = self._assign_persistent_ids(entities)
    return entities
```

The `_nms()` method sorts entities by confidence (highest first) and removes lower-confidence boxes that overlap >50% IoU with a higher-confidence box of the same class.

## 7.9 Persistent Entity Tracking

Entity IDs persist across detection frames via IoU-based matching. This allows the LLM to consistently reference the same entity across turns (e.g., `sheep_0` remains `sheep_0` as long as the sheep is visible and hasn't moved dramatically).

```python
def _assign_persistent_ids(self, new_entities):
    for entity in new_entities:
        # Find best IoU match among previous frame's same-class entities
        best_iou, best_match = find_best_match(entity, self._previous_entities)
        if best_iou > 0.4:
            entity.id = best_match.id  # Reuse old ID
        else:
            entity.id = f"{entity.class_name}_{self._global_id_counter}"
            self._global_id_counter += 1  # Never resets
    self._previous_entities = new_entities
```

Key design decisions:
- **IoU threshold 0.4**: Matches entities that overlap 40%+ with a same-class entity from the previous frame
- **Global counter**: IDs like `sheep_5` are globally unique and never repeat, even if `sheep_0` through `sheep_4` have disappeared
- **Same-class constraint**: Only matches entities of the same class (a villager can't "become" a sheep)

## 7.10 Singleton Access

`get_detector()` at the bottom of `detector.py` provides a singleton with auto-detection:

```python
_instance: Optional[EntityDetector] = None

def get_detector(model_path=None, use_mock=False, imgsz=1280) -> EntityDetector:
    global _instance
    if _instance is None:
        # Auto-detect model file in priority order (v5 > v2 > v1)...
        _instance = EntityDetector(model_path=path, use_mock=use_mock, imgsz=imgsz)
    return _instance
```

The game loop calls `get_detector(imgsz=config.detection_imgsz)` once during initialization. The same instance is reused for all subsequent detection calls, preserving the persistent entity tracking state.

---

## Summary

- 60-class taxonomy organized by category (resources, buildings, units, siege, naval, animals)
- Three backends: PyTorch (ultralytics), ONNX Runtime, Mock
- v5 model preferred, auto-detects with fallback chain (v5 > v2 > v1)
- Configurable inference resolution (`imgsz`, default 1280)
- NMS applied to all backends in unified `detect()` method
- Persistent entity IDs across frames via IoU-based tracking (`_assign_persistent_ids`)
- ONNX handles two output formats transparently

## Related Topics

- [Chapter 3: Action Model & Execution](../part1-architecture/03-action-model-and-execution.md) -- how detections become click targets
- [Chapter 8: Training Pipeline](./08-training-pipeline.md) -- how the model is created
- [Chapter 13: Class Schema Evolution](../part5-operations/13-class-schema-evolution.md) -- the class taxonomy history and unified schema
