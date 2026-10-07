# Detector quality baseline — 2026-10-05

This is the frozen **development reference** for the detector-retraining plan. It
describes the currently served ONNX path, not just the weights. The score below
was reproduced locally from the available model and validation files and
matched the published `evaluation.json` field for field. The model and labeled
screenshots are not tracked in this Git checkout; their hashes identify the
required bytes. No new model was trained or released in this step.

## Frozen inputs and inference contract

| Component | Repository path | SHA-256 |
| --- | --- | --- |
| ONNX weights | `packages/detection/src/inference/models/aoe2_yolo_v9.onnx` | `515a018bc2190fdf5427a01ff21e294331324929c8603d870c18255626cee8fd` |
| 60-class schema | `packages/detection/src/training/config/classes.yaml` | `5365dcf538d16f9b237070a5e9c7609028314794dd8e544233eecb76a09de717` |
| Confidence thresholds | `packages/detection/src/inference/thresholds.py` | `2257a09e901f14364d3e7156e90bf41f0a6e780e51d6993a7a91d72c3933929e` |
| Letterbox preprocessing | `packages/detection/src/inference/preprocess.py` | `8df0e13f6656f5a155f460a5e05b7b9488a6dea9e724712a0c790751d9cb4c4b` |
| Classwise NMS | `packages/detection/src/inference/postprocess.py` | `dade6f3f9a220ff8f938544bbca812bf86258939b1577a0bc80d0527d48d87a2` |
| Server decoder | `apps/detection-server/src/app.py` | `e2b13a712cd0964591f1c599c4ab0980d7f67405ae2c92a70490af852e1efa01` |
| Real-frame evaluator | `apps/detection-server/src/evaluate.py` | `2fb48f40e750d4e5f0c74cd37eb07a3eccc2d19dcedca62addad2936f618d80e` |
| Box matching/scoring | `packages/detection/src/testing/evaluate_real.py` | `eb55a81ab1b29d25f8b52134f5f0270dd6b72a8360455d09a304e8c9e4302b82` |

The model input is a static 1280 × 1280 tensor. The evaluator sends each full
image through the server's letterbox/ONNX decoder, applies the served confidence
thresholds, then the client's same-class NMS at IoU 0.50. Confidence floors are
0.35 by default, 0.25 for berry bush and villager, 0.20 for deer, relic, and
sheep, and 0.55 for mill. There is no tracker, camera cache, crop pass, or agent
action in this score. The full thresholds and all 60 class results are in
[`evaluation.json`](../../packages/detection/registry/aoe2-entity-detector/evaluation.json).

## Reproduced development result

The evaluation set contains 32 `real_*` screenshots at 1280 × 831 pixels and
1,212 labeled boxes. Its ordered image-and-label manifest SHA-256 is
`3a21b93f150f155d0538a45c0fa6da8fed0ec4466703cf21560badfb38d2bfd0`.
These January 2026 frames share capture sessions with training and therefore
are **not** an independent or release-quality test. Exact zoom settings were
not recorded for this set, so no zoom-robustness claim follows from it.
The reproduced registry report itself has SHA-256
`72e9502b1136566dfc8c24291551c2231b2d3ceda784794e99551ec33380cc7d`.

| Images | Labels | TP | FP | FN | Precision | Recall | F1 | FP/frame |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 1,212 | 767 | 251 | 445 | 0.753 | 0.633 | 0.688 | 7.84 |

| Action-critical class | Labeled boxes | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Sheep | 6 | 0.750 | 0.500 | 0.600 |
| Berry bush | 6 | 0.667 | 0.667 | 0.667 |
| Villager | 218 | 0.721 | 0.757 | 0.738 |
| Town center | 18 | 0.875 | 0.778 | 0.824 |
| House | 110 | 0.895 | 0.618 | 0.731 |
| Mill | 7 | 1.000 | 1.000 | 1.000 |
| Farm | 80 | 0.786 | 0.413 | 0.541 |
| Knight line | 29 | 0.333 | 0.379 | 0.355 |

For each image and class, detections are sorted by confidence and greedily
matched one-to-one to labels at box IoU ≥ 0.50. Unmatched detections are false
positives; unmatched labels are false negatives. Micro precision is
`TP / (TP + FP)`, recall is `TP / (TP + FN)`, and F1 is their harmonic mean.
False positives from classes with no labeled examples still count in the
overall score. FP/frame is total false positives divided by evaluated frames.
This is a single operating point, **not mAP**. A class with no ground-truth
boxes has no interpretable recall; support of six or seven boxes is too small
to claim reliable behavior.

## Latency diagnostic

The new `detection_server.benchmark_latency` command measured the same
full-image path on this Mac: 32 decoded images, three warmup calls, one timed
call per image, ONNX Runtime CPU provider. It measured **109.5 ms median** and
**147.2 ms p95** from decoded image to postprocessed boxes, on
macOS 26.6.2 arm64 with Python 3.11.9, ONNX Runtime 1.26.0, and Pillow 12.2.0.
These are a single-pass, machine-specific diagnostic, not a performance claim
for the Windows VM. JPEG decoding, screen capture, network transfer, tracking,
and agent scheduling are excluded. Requested target-refresh p50/p95 cannot be
measured from this Mac baseline; record them on the Windows runtime in step 2.

## Reproduction

Run from the repository root with the local model and validation data present:

```bash
uv run --no-sync python -m detection_server.evaluate \
  --model packages/detection/src/inference/models/aoe2_yolo_v9.onnx \
  --data packages/detection/src/training_data_v9_slim \
  --output tmp/evaluations/current-development.json

uv run --no-sync python -m detection_server.benchmark_latency \
  --model packages/detection/src/inference/models/aoe2_yolo_v9.onnx \
  --data packages/detection/src/training_data_v9_slim \
  --output tmp/evaluations/current-latency.json
```

Compare the evaluator's parsed JSON to the registry's
[`evaluation.json`](../../packages/detection/registry/aoe2-entity-detector/evaluation.json)
and verify the model/schema/data hashes before interpreting a different
result. CPU latency will vary by host and load; compare it only with a candidate
run on the same host, provider, input set, and measurement scope.

## Predeclared comparison for later candidates

1. Keep this reference untouched. On a future session-held-out real-image test,
   run both reference and candidate over the **same entire frames** with the
   same class schema, label convention, and one-to-one IoU 0.50 scorer. Never
   tune thresholds on the final-test sessions.
2. For a *weights/data* experiment, hold preprocessing, postprocessing,
   confidence thresholds, and full-frame inference fixed so changes can be
   attributed to the checkpoint. If the *inference package* changes (for
   example, adds crops), report that as a separate package comparison with its
   extra detections, duplicate boxes, coordinate errors, and latency included.
3. The primary quality target is higher held-out **micro precision, recall,
   and F1** than this reference measured on those same new sessions. Also
   report all 60 per-class results and support, especially the action-critical
   classes above; FP/frame; metrics by actual zoom and model-input object-size
   band; and missed/wrong action targets. Do not mask a food-target regression
   behind a gain from common tree boxes. If support is too small, call that
   class unqualified, not improved.
4. Measure decoded-image CPU latency as above and whole requested-refresh
   p50/p95 on Windows, including capture, transport, detection, and target
   handoff. A candidate must meet the existing ≤2-second p95 requested-refresh
   target and show no increase in wrong-click targets before deployment.

The independent test, zoom coverage, and Windows requested-refresh baseline
are deliberately **pending**; they require the captures and instrumentation of
later steps. An apparent gain on the 32 development images alone does not meet
the release gate.
