# Chapter 9: Labeling and Active Learning

Real screenshots are labeled in CVAT, exported in COCO format, converted to YOLO labels, and merged with synthetic data. An active learning pipeline prioritizes the most informative images for labeling.

<aside class="prereqs">

[Chapter 8 — Training Pipeline](./08-training-pipeline.md) for where labels feed into training. If [COCO vs YOLO label formats](../glossary.md#c) are new terms, the [Glossary](../glossary.md) defines them.

</aside>

> **Also see [Chapter 24 — Detection Training Tracker](./24-training-tracker.md).** The CVAT loop described here is still how bulk annotation gets done, but the in-repo tracker now owns *dataset visibility* (per-class coverage) and *incremental review* (approve/correct model predictions in the browser). Reach for CVAT when labeling a fresh batch from scratch; reach for the tracker to see what's missing, and to review model-proposed boxes one at a time.

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
                                                         merge with synthetic)
                                                              ↓
                                                      training_data/
```

## 9.2 Pre-Labeling

`packages/detection/src/labeling/prelabel.py` bootstraps annotation by running the existing model on unlabeled screenshots:

1. Loads the current YOLO model (v1 or v2)
2. Runs inference on each image in `real_screenshots/raw/`
3. Exports predictions as YOLO `.txt` label files
4. Generates `classes.txt` for CVAT project import
5. Optionally saves preview images with drawn bounding boxes

Pre-labels are not training-quality -- they provide a starting point for human annotators to correct rather than drawing everything from scratch. Confidence thresholds are set low (0.15) to catch more potential objects.

## 9.3 CVAT Integration

### Export Format: COCO 1.0 (Not YOLO)

> **Key Insight**: CVAT's YOLO 1.1 export format silently drops polygon annotations -- only rectangles survive the export. Since some entities are labeled with polygon shapes in CVAT (for precise outlines), the project uses COCO 1.0 export format instead. `prepare_training.py` handles the COCO-to-YOLO conversion, computing bounding boxes from polygon vertices.

<aside class="concept" data-title="COCO vs YOLO label formats">

**COCO** is JSON-based: one file per dataset, with `images`, `categories`, and `annotations` arrays. Each annotation references an image by ID, carries a `bbox` `[x, y, width, height]` *in absolute pixels*, optionally a `segmentation` (polygon vertices), and a `category_id` that points into the global `categories` table. **1-indexed.**

**YOLO** is plain-text: one `.txt` per image, one annotation per line, `class_id x_center y_center width height` *normalized to [0, 1]*. No category names in the file — they live separately in `dataset.yaml`. **0-indexed.**

Three traps that bite people doing CVAT → YOLO conversions:

1. **Polygons silently dropped.** CVAT's YOLO export only writes rectangles. If you labeled with polygons for precision, you lose them. Export as COCO, convert to YOLO yourself, and compute bboxes from the polygon vertices' min/max.
2. **Off-by-one class IDs.** COCO is 1-indexed; YOLO is 0-indexed. The safe move is *never* convert by numeric ID — convert by class *name* through your canonical `classes.yaml` taxonomy. That's why this chapter's converter looks up COCO categories by name and writes YOLO class IDs from the taxonomy.
3. **Coordinate-format confusion.** COCO is top-left + width/height. YOLO is center + width/height. The numbers look interchangeable until they're not.

</aside>

### Export Format Detection

`prepare_training.py` auto-detects the export format at `packages/detection/src/labeling/prepare_training.py:43-66`:

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

## 9.4 Class Schema

All data sources now use `classes.yaml` IDs directly (60 classes). See [Chapter 13](../part5-operations/13-class-schema-evolution.md) for schema history.

`packages/detection/src/labeling/class_mapping.py` provides the loader (`load_classes_yaml()`) and the CVAT helpers (`get_classes_for_cvat()`, `write_classes_txt()`). The model (YOLO26/v6) emits classes.yaml IDs natively, so there is no class remapping — `prelabel.py` and `prepare_training.py` write classes.yaml IDs directly. (The legacy v1→v2 mapping helpers were removed.)

## 9.5 Hybrid Dataset Merge

`prepare_training()` at `packages/detection/src/labeling/prepare_training.py:248-445` orchestrates the full merge:

1. **Scan local images** -- builds an index of all raw screenshots by filename
2. **Detect export format** -- auto-detects COCO or YOLO from the CVAT export directory
3. **Convert COCO to YOLO** -- if COCO format, generates temp YOLO label files
4. **Match labels to images** -- pairs each label file with its corresponding image
5. **Split real data** -- 85/15 train/val split with `seed=42` for reproducibility
6. **Copy synthetic data** -- copies synthetic images and labels directly (both use classes.yaml IDs)
7. **Copy real data** -- copies real images with `real_` prefix to avoid filename collisions
8. **Generate dataset.yaml** -- writes the YOLO training config with all 60 classes

Output structure:
```
training_data/
├── train/
│   ├── images/
│   │   ├── img_00000.jpg          # synthetic
│   │   ├── real_screenshot_001.jpg # real
│   └── labels/
│       ├── img_00000.txt          # classes.yaml IDs
│       ├── real_screenshot_001.txt # classes.yaml IDs
├── val/
│   └── ...
├── dataset.yaml
└── merge_summary.json             # statistics
```

## 9.6 Active Learning Pipeline

`packages/detection/src/labeling/active_learning.py` optimizes which images to label next.

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

As of the `v9` build (`packages/detection/src/training_data_v9_slim/`):

| Source | Train | Val | Total |
|--------|-------|-----|-------|
| Synthetic (`img_*`) | 2,400 | 600 | 3,000 |
| Real (`real_*`, incl. duplicates) | 1,870 | 32 | 1,902 |
| **Total** | **4,270** | **632** | **4,902** |

The real-image row needs a caveat: those 1,902 files come from **219 distinct screenshots**. Train-side real images are oversampled 10× (187 distinct stems × 10, carrying a `__dupN` filename suffix); val is not oversampled (32 distinct, 32 files). Labeling has progressed from 58 screenshots at the v5/v6 era to 219 of the 220 raw captures in `packages/detection/src/real_screenshots/raw/`.

<aside class="concept" data-title="Why duplicate real images instead of reweighting the loss?">

Real images are outnumbered ~13:1 by synthetic ones here. Left alone, the gradient is dominated by synthetic data whose appearance the model already fits easily — and the real screenshots, which are the *only* samples drawn from the deployment distribution, contribute a rounding error to each epoch.

Two standard fixes:

1. **Weight the loss** per sample by source. Principled, but Ultralytics' training loop has no per-sample weight hook — you'd be patching the loss function and maintaining that patch across upgrades.
2. **Duplicate the files** so the sampler draws them more often. Crude, costs disk, and shows up as inflated image counts that mislead anyone reading the dataset stats (exactly the confusion this table's caveat exists to prevent).

We take (2) because it's zero-code and framework-agnostic. The important detail is that duplication is applied to **train only**. Duplicating validation images would be actively harmful: it doesn't change which distinct images are measured, but it re-weights the val metric toward the duplicated ones, so your reported mAP drifts from the mAP on real, distinct data. Any oversampling scheme has the same rule — **resample the training set, never the evaluation set.**

</aside>

Live per-class coverage — including which classes have *zero* real examples — is served by the training tracker's `/stats` endpoint and its `/training/coverage` page ([Chapter 24](./24-training-tracker.md)), which is more current than any table checked into a document.

---

## Summary

- CVAT labels exported as COCO 1.0 (not YOLO, which drops polygons)
- Automatic COCO-to-YOLO conversion with name-based class mapping
- All data sources use classes.yaml IDs directly (no remapping needed for v5+ synthetic data)
- Active learning scores unlabeled images by model uncertainty
- Coverage and incremental review now live in the [training tracker](./24-training-tracker.md); CVAT remains the bulk-annotation tool

## Related Topics

- [Chapter 8: Training Pipeline](./08-training-pipeline.md) -- synthetic data generation and YOLO training
- [Chapter 13: Class Schema Evolution](../part5-operations/13-class-schema-evolution.md) -- schema history and unified class IDs
- [Chapter 7: Detector Architecture](./07-detector-architecture.md) -- how the trained model runs at inference time
- [Chapter 24: Detection Training Tracker](./24-training-tracker.md) -- coverage stats and in-browser prelabel review
