# Real-frame validation protocol

This report describes a **fixed operating point of the application detector**,
not a general-purpose object-detection benchmark or a game-play success rate.
The result is appropriate to publish with this private model as a transparent
validation result, but not to advertise as an independent test score.

## Data and provenance

- 32 hand-labeled real screenshots from January 18–19, 2026; 1,212 labeled
  entity boxes across the 60-class schema.
- Images: the `real_*` files in
  `packages/detection/src/training_data_v9_slim/val/images` (1280 × 831 JPEG).
  Labels: matching YOLO text files in `val/labels`.
- The ordered image-and-label manifest SHA-256 is
  `3a21b93f150f155d0538a45c0fa6da8fed0ec4466703cf21560badfb38d2bfd0`.
  File names and file bytes both contribute to that digest. The model SHA-256
  is `515a018bc2190fdf5427a01ff21e294331324929c8603d870c18255626cee8fd`.
- The split was created by shuffling and dividing images, not by holding out a
  capture session. Its training side contains 187 unique real images, each
  oversampled 10 times. Training and validation therefore share capture
  sessions and nearby game states. This can overstate generalization.
- Screenshots and labels are not distributed in this model repository. The
  hashes make the result auditable by someone with the private source data;
  they do not make the evaluation publicly reproducible.

## Inference and scoring

1. Load the exact `model.onnx` bytes with ONNX Runtime's CPU provider.
2. Run the detection server's single-pass `letterbox` and ONNX decoder at
   1280 × 1280 input. Apply its committed confidence floors: 0.35 default;
   0.25 for berry bush and villager; 0.20 for deer, relic, and sheep; 0.55
   for mill. Apply the remote client's same-class NMS at IoU 0.50.
3. Match predictions to ground truth **within each image and class**, in
   descending confidence order, using one-to-one greedy IoU ≥ 0.50 matching.
   Unmatched predictions are false positives; unmatched labels are false
   negatives. Report micro precision = TP/(TP+FP), recall = TP/(TP+FN), and
   F1 = 2PR/(P+R). All classes, including classes with no labels, contribute
   their false positives to the micro count.
4. Thresholds were fixed for this evaluation run, but the project's existing
   per-class thresholds were selected using its validation workflow. They may
   therefore reflect tuning on these images. No temporal tracker, cached
   boxes, OCR, owner classifier, or action logic is evaluated. This is not
   mAP and should not be compared with mAP scores.

The exact thresholds, per-class TP/FP/FN, source hashes, and full-precision
metrics are in [`evaluation.json`](evaluation.json).

## Results

| Images | Labels | TP | FP | FN | Precision | Recall | F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 1,212 | 767 | 251 | 445 | 0.753 | 0.633 | 0.688 |

### Selected per-class results

The support column counts labeled boxes in the **real validation screenshots
only**, not examples in the synthetic or real training data.

| Class | Real validation boxes | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Villager | 218 | 0.721 | 0.757 | 0.738 |
| Town center | 18 | 0.875 | 0.778 | 0.824 |
| Sheep | 6 | 0.750 | 0.500 | 0.600 |
| Berry bush | 6 | 0.667 | 0.667 | 0.667 |
| Farm | 80 | 0.786 | 0.413 | 0.541 |
| Knight line | 29 | 0.333 | 0.379 | 0.355 |

The full 60-class breakdown is in [`evaluation.json`](evaluation.json).
Results for classes with only a few labeled boxes are unstable.

The model misses 47 of 80 labeled farms and 3 of 6 labeled sheep. It also
finds only 11 of 29 knight-line units, with 22 false knight-line detections.
The 7 labeled mills are all detected with no false positives **in this split**,
but that small count is not evidence of reliable mill recognition in new
runs. Per-class results with zero labeled examples are not recall estimates.

## Reproduction by a data holder

From the agent repository root, with the private validation files present:

```bash
uv run --no-sync python -m detection_server.evaluate \
  --model packages/detection/src/inference/models/aoe2_yolo_v9.onnx \
  --data packages/detection/src/training_data_v9_slim \
  --output packages/detection/registry/aoe2-entity-detector/evaluation.json
```

The command rejects missing or malformed labels and checks that the server and
training class names agree. Verify the two input hashes before comparing
numbers. ONNX Runtime CPU was chosen for repeatability; the Mac deployment
may use the CoreML execution provider. Provider parity on these frames remains
untested.

## What a release-quality generalization test still needs

Freeze the model and thresholds, then capture and independently annotate new
game sessions (including Dark Age openings and both Arabia and Highland) at
the agent's actual capture resolution. Hold out entire sessions and include
negative/empty scenes, small food targets, crowded armies, and UI variations.
Report the session count, image count, per-class support, false positives,
latency, and error examples. Do not choose thresholds or retrain on that test
set; if it is used for iteration, create another untouched holdout.
