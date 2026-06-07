# Appendix A — YOLO and object detection

This appendix is the reference behind every chapter that talks about the detector — [Chapter 7 (Detector Architecture)](../part3-entity-detection/07-detector-architecture.md), [Chapter 8 (Training Pipeline)](../part3-entity-detection/08-training-pipeline.md), [Chapter 12 (Cloud Training)](../part5-operations/12-cloud-training.md). If you understand how a YOLO-family detector actually works end-to-end, those chapters become operational walkthroughs of a system you already understand. If you don't, this appendix builds the mental model from the ground up: what the network outputs, how raw predictions become a final list of boxes, how training pushes the network toward better predictions, and how we know when it's improving.

We use **YOLO11n** (Ultralytics) — `n` for nano, the smallest variant in the family. This appendix is written for the YOLOv5/8/11 generation; the broad strokes apply to the whole lineage.

---

## A.1 The shape of the problem

You have an image. You want a list: *what objects are in it, and where?* "Where" means a bounding box: `(x_center, y_center, width, height)`, all in pixels. "What" means a class label drawn from a fixed set (our 60-class taxonomy: villager, militia, sheep, town_center, …).

The naive approach is to slide a classifier across the image at many positions and scales (R-CNN, ~2014). Slow — thousands of forward passes per image. The YOLO insight (*You Only Look Once*, Redmon 2016) is: **predict everything in one forward pass**. The network looks at the entire image once and emits a structured grid where every cell is responsible for predicting any objects centered in it.

Modern YOLO is just that idea, refined for a decade: a better backbone, multi-scale prediction, anchor-free heads, and a much better loss function. The cost: one forward pass through a ~3 MB network. That's why we can run it ~10 times per second on a laptop.

---

## A.2 What the network outputs

A single forward pass through YOLO11n on a 640×640 image produces a tensor of shape roughly `(num_predictions, 4 + num_classes)`. Each row is one candidate prediction:

- **4 box parameters** — `(x_center, y_center, width, height)` in pixels.
- **`num_classes` class scores** — one logit per class. After a sigmoid (or in some YOLOs, a softmax), each becomes a probability.

There are *thousands* of candidates per image — modern YOLOs predict at multiple scales (e.g. a 80×80 grid for small objects, 40×40 for medium, 20×20 for large), giving `80² + 40² + 20² = 8400` candidate predictions per image at default resolution. **Most of them are garbage** — predictions for empty patches of grass. The class scores tell you which to keep.

### Anchor boxes (older YOLOs) vs anchor-free (modern)

Older YOLOs (v3–v5) used **anchor boxes**: each grid cell predicted offsets relative to a small set of pre-defined box shapes (tall, square, wide). The anchors were chosen by k-means clustering of the training set's ground-truth boxes. This worked but required you to pick good anchors for each dataset.

YOLO11 is **anchor-free**: each cell directly predicts box parameters relative to its own location. Simpler, no per-dataset tuning, and competitive accuracy. You'll still see anchor-box language in older tutorials and the YOLOv5 codebase.

---

## A.3 From raw predictions to a clean list of boxes — NMS

Even after thresholding away the low-confidence garbage, you'll have multiple overlapping boxes for the same object — three cells near a sheep all "see" the sheep and emit boxes. **Non-Maximum Suppression (NMS)** is the post-processing that collapses these:

```python
def nms(boxes, scores, iou_threshold=0.5):
    keep = []
    order = scores.argsort(descending=True)
    while len(order) > 0:
        best = order[0]
        keep.append(best)
        ious = iou(boxes[best], boxes[order[1:]])
        order = order[1:][ious < iou_threshold]
    return keep
```

Sort by confidence, greedily keep the top box, drop anything overlapping it too much, repeat. The `iou_threshold` (typically 0.5) controls how aggressive the suppression is — too high and you keep duplicates; too low and you'll erase legitimate adjacent objects (two villagers standing next to each other).

### IoU: the universal box-similarity score

`IoU = area(A ∩ B) / area(A ∪ B)`. Always between 0 (no overlap) and 1 (identical). It's everywhere in detection:

- NMS uses IoU to decide whether to suppress.
- mAP uses IoU thresholds to decide what counts as a correct detection.
- Trackers use IoU as the cost in the Hungarian matching step.

Two non-overlapping boxes have IoU=0. Two identical boxes have IoU=1. IoU=0.5 means the boxes overlap by about half — a reasonable "this is probably the same object" threshold for most domains.

---

## A.4 How training pushes predictions toward truth

Training a YOLO is supervised: you have images with ground-truth boxes and class labels, and you compute a loss that punishes wrong predictions.

The loss is a sum of three components:

1. **Box regression loss** — typically *CIoU loss* (Complete IoU): penalizes both how badly the predicted box overlaps the ground truth and how far apart their centers are. Smoother gradient than raw MSE on box coordinates.
2. **Classification loss** — binary cross-entropy on each class's predicted probability. Sigmoid'd, not softmax'd, so the network can predict multiple labels per box (though we don't use that).
3. **Objectness / distribution-focal loss** — penalizes wrong confidence in whether a cell contains an object at all.

Modern YOLOs assign each ground-truth box to the *k* best-matching predictions (Task-Aligned Assignment, TAL) rather than to a single cell — the loss is computed against those, and unmatched predictions are pushed toward zero confidence.

Backprop, repeat for thousands of mini-batches over hundreds of epochs.

### Augmentations: data multipliers

Real training data is finite. **Augmentations** synthesize variations: random crops, scaling, horizontal flip, color jitter, mosaic (four images stitched together in a 2×2), MixUp (two images blended). They make the network robust to natural variation it didn't see literally during training.

Our domain has a constraint most YOLO recipes don't: **isometric perspective**. The game camera is fixed; villagers never appear upside-down. So we set `flipud=0.0` (vertical flip would create unnatural training images), `degrees=10` (gentle rotation only), and rely heavily on mosaic + color jitter. Synthetic data generation does most of the augmentation work for us — see [Chapter 8](../part3-entity-detection/08-training-pipeline.md).

---

## A.5 mAP: the standard detection metric

Once trained, you measure quality on a held-out test set. The metric is **mean Average Precision (mAP)**.

For each class, on each test image, you sort the model's predictions by confidence and walk down the list. At each rank:

- A prediction is **correct** if it matches a ground-truth box at the chosen IoU threshold and the class label is right (and that ground-truth box hasn't already been claimed by a higher-ranked prediction).
- Compute precision (`tp / (tp + fp)`) and recall (`tp / (tp + fn)`) at this rank.

You now have a precision-recall curve per class. Its area is the *Average Precision (AP)* for that class. Average across classes → *mAP*.

### mAP50 vs mAP50-95 — what the numbers mean

- **mAP50** — uses a single IoU threshold of 0.5. "The predicted box has to overlap the ground truth by ≥50% to count as correct." Looser, easier to improve, the headline number in older papers.
- **mAP50-95** — averages mAP across 10 IoU thresholds from 0.5 to 0.95 in 0.05 steps. Much harder: at IoU 0.95 you need an almost-pixel-perfect box. The COCO benchmark default.

For our domain, mAP50 is more representative — we don't need pixel-perfect boxes (the LLM only needs to know roughly where to click), but we need recall on small entities like sheep that are ~20 pixels across at our typical zoom level. Small objects naturally drive mAP50-95 down because a few pixels of box error pushes IoU below 0.95 quickly.

### A useful rule of thumb

| mAP50 | What it means in practice |
|---|---|
| < 0.5 | Detector is unreliable — too many misses, too many wrong-class predictions. |
| 0.5–0.7 | Workable for low-stakes recommendation systems; not workable as a sole input to a planning agent. |
| 0.7–0.85 | Good for most production CV systems. Most rare-class errors. |
| > 0.85 | Excellent. At this point you're chasing edge cases. |

Our 60-class model sits around 0.78 mAP50 with most error concentrated on rare or visually similar classes (light cavalry vs heavy cavalry from a distance).

---

## A.6 Why "single-shot" beats "two-stage" for our use case

The older Faster R-CNN family is *two-stage*: first a Region Proposal Network suggests where things might be, then a classifier scores each proposal. Higher accuracy historically, but ~5× slower.

YOLO is *single-shot*: one network produces everything in one pass. Faster, slightly less accurate at the high end, but modern YOLOs have closed most of the gap. For an interactive agent that needs to act every second, single-shot is the only viable choice.

---

## A.7 What we glossed over

- **The backbone** — the CNN that extracts features before the detection head. YOLO11n uses a custom C3k2 backbone, but you can swap in MobileNet, EfficientNet, etc. The detection head is largely backbone-agnostic.
- **Multi-scale prediction (FPN)** — the detection head outputs at three resolutions so the network can handle objects of very different sizes. The feature-pyramid architecture is why YOLO works on both villagers (small) and town centers (huge) in the same model.
- **Quantization and ONNX export** — for deployment we sometimes export to ONNX and run on CPU via onnxruntime or CoreML. See [Chapter 7](../part3-entity-detection/07-detector-architecture.md) for our backend abstraction.

---

## Further reading

- Redmon et al., *You Only Look Once: Unified, Real-Time Object Detection* (2016) — the original paper. Worth reading for the framing even though the architecture is dated.
- Ultralytics, [*YOLO11 documentation*](https://docs.ultralytics.com/) — current reference for the model and training CLI we use.
- Glenn Jocher et al., *YOLOv5 docs* — still the most accessible writeup of the modern recipe (mosaic, anchor-free head, NMS variants).

---

**Cross-references:**

- [Chapter 7 — Detector Architecture](../part3-entity-detection/07-detector-architecture.md) — how we wrap YOLO behind our `EntityDetector` abstraction.
- [Chapter 8 — Training Pipeline](../part3-entity-detection/08-training-pipeline.md) — how we generate synthetic training data and tune the augmentation recipe.
- [Chapter 12 — Cloud Training](../part5-operations/12-cloud-training.md) — the Lambda Labs workflow for full retraining.
- [Glossary](../glossary.md) — one-liners for anchor box, IoU, mAP, NMS, and friends.
