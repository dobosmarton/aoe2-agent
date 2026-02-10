# Chapter 9: Labeling and Active Learning

Real screenshots are labeled in CVAT, exported in COCO format, converted to YOLO labels, and merged with synthetic data. An active learning pipeline prioritizes the most informative images for labeling.

## 9.1 The Labeling Workflow

```
Raw Screenshots              Pre-label with YOLO         CVAT
(220 images in               (generate initial           (manual correction
 real_screenshots/raw/)       bounding boxes)             + new annotations)
        ↓                           ↓                          ↓
   prelabel.py              Import to CVAT              Export as COCO 1.0
                            (with classes.txt)                  ↓
                                                     prepare_training.py
                                                        (COCO → YOLO +
                                                         remap v1→v2 +
                                                         merge with synthetic)
                                                              ↓
                                                      training_data_v2/
```

## 9.2 Pre-Labeling

`detection/labeling/prelabel.py` bootstraps annotation by running the existing model on unlabeled screenshots:

1. Loads the current YOLO model (v1 or v2)
2. Runs inference on each image in `real_screenshots/raw/`
3. Exports predictions as YOLO `.txt` label files
4. Generates `classes.txt` for CVAT project import
5. Optionally saves preview images with drawn bounding boxes

Pre-labels are not training-quality -- they provide a starting point for human annotators to correct rather than drawing everything from scratch. Confidence thresholds are set low (0.15) to catch more potential objects.

## 9.3 CVAT Integration

### Export Format: COCO 1.0 (Not YOLO)

> **Key Insight**: CVAT's YOLO 1.1 export format silently drops polygon annotations -- only rectangles survive the export. Since some entities are labeled with polygon shapes in CVAT (for precise outlines), the project uses COCO 1.0 export format instead. `prepare_training.py` handles the COCO-to-YOLO conversion, computing bounding boxes from polygon vertices.

### Export Format Detection

`prepare_training.py` auto-detects the export format at `detection/labeling/prepare_training.py:43-66`:

```python
def detect_export_format(cvat_dir) -> "coco" | "yolo":
    # Check for COCO JSON
    if (cvat_dir / "annotations" / "instances_default.json").exists():
        return "coco"
    # Fall back to YOLO text files
    return "yolo"
```

### COCO to YOLO Conversion

For COCO exports (`prepare_training.py:69-200`):

1. Reads `annotations/instances_default.json`
2. Maps COCO category IDs to v2 class IDs **by name** (not by numeric ID)
3. For each annotation:
   - If it has a direct `bbox`: uses `[x, y, width, height]` directly
   - If it has `segmentation` (polygon): computes bounding box from min/max of polygon vertices
4. Converts to YOLO normalized format: `class_id x_center y_center w_norm h_norm`

COCO categories are **1-indexed** while YOLO classes are 0-indexed. The name-matching approach avoids this pitfall entirely -- both sides are looked up by name, not ID.

### CVAT Directory Structure

The code handles multiple CVAT export directory layouts:
- `obj_train_data/` -- standard YOLO export
- `obj_Train_data/` -- case variant (observed in some CVAT versions)
- `labels/` -- alternative layout
- `.txt` files at root level

## 9.4 Class Schema Remapping

When merging synthetic (v1 schema) and real (v2 schema) data, class IDs must be remapped. See [Chapter 13](../part5-operations/13-class-schema-evolution.md) for the full schema story.

`detection/labeling/class_mapping.py` provides the mapping utilities:

**`build_v1_to_v2_mapping()`** -- builds a dict mapping v1 class IDs to v2 IDs by matching class names. Special handling for unique units:

| v1 Name | v1 ID | v2 Name | v2 ID |
|---------|-------|---------|-------|
| longbowman | 43 | unique_archer | 50 |
| mangudai | 44 | unique_archer | 50 |
| war_wagon | 45 | unique_siege | 53 |

**`convert_label_file()`** -- rewrites a YOLO `.txt` file replacing each class ID with its mapped counterpart. Lines with unmappable classes are dropped with a warning.

## 9.5 Hybrid Dataset Merge

`prepare_training()` at `detection/labeling/prepare_training.py:248-445` orchestrates the full merge:

1. **Scan local images** -- builds an index of all raw screenshots by filename
2. **Detect export format** -- auto-detects COCO or YOLO from the CVAT export directory
3. **Convert COCO to YOLO** -- if COCO format, generates temp YOLO label files
4. **Match labels to images** -- pairs each label file with its corresponding image
5. **Split real data** -- 85/15 train/val split with `seed=42` for reproducibility
6. **Copy synthetic data** -- copies synthetic images and **remaps v1 labels to v2 IDs** during copy
7. **Copy real data** -- copies real images with `real_` prefix to avoid filename collisions
8. **Generate dataset.yaml** -- writes the YOLO training config with all 59 classes

Output structure:
```
training_data_v2/
├── train/
│   ├── images/
│   │   ├── img_00000.jpg          # synthetic
│   │   ├── real_screenshot_001.jpg # real
│   └── labels/
│       ├── img_00000.txt          # remapped v1→v2
│       ├── real_screenshot_001.txt # already v2
├── val/
│   └── ...
├── dataset.yaml
└── merge_summary.json             # statistics
```

## 9.6 Active Learning Pipeline

`detection/labeling/active_learning.py` optimizes which images to label next.

### Triage: Scoring Images by Informativeness

Runs the current model on all unlabeled images and scores each by how "interesting" it is to the model:

| Condition | Score | Rationale |
|-----------|-------|-----------|
| Detection with confidence < 0.15 | +3 | Model is confused |
| Detection with 0.15 <= confidence < 0.7 | +2 | Model is uncertain |
| No detections at all | +15 | Completely novel content |
| Fewer than expected detections | +5 | Missing entities |

High-scoring images are the most informative for training -- they represent cases where the model struggles.

### Prepare Batch

Selects the top-N highest-scoring images and creates a CVAT-ready batch:
- Copies images to an output directory
- Generates pre-labels for CVAT import
- Writes `classes.txt` with v2 class names

### Integrate

After manual correction in CVAT, `integrate()` copies the corrected labels into the training dataset.

## 9.7 Current Dataset Scale

| Source | Train | Val | Total |
|--------|-------|-----|-------|
| Synthetic | 2,400 | 600 | 3,000 |
| Real (labeled) | 50 | 8 | 58 |
| **Total** | **2,450** | **608** | **3,058** |

220 raw screenshots exist in `detection/real_screenshots/raw/`. 58 have been labeled in CVAT so far.

---

## Summary

- CVAT labels exported as COCO 1.0 (not YOLO, which drops polygons)
- Automatic COCO-to-YOLO conversion with name-based class mapping
- v1-to-v2 class ID remapping during synthetic data merge
- Active learning scores unlabeled images by model uncertainty
- Hybrid dataset: 2450 train (2400 synthetic + 50 real), 608 val

## Related Topics

- [Chapter 8: Training Pipeline](./08-training-pipeline.md) -- synthetic data generation and YOLO training
- [Chapter 13: Class Schema Evolution](../part5-operations/13-class-schema-evolution.md) -- the v1/v2 mapping in depth
- [Chapter 7: Detector Architecture](./07-detector-architecture.md) -- how the trained model runs at inference time
