---
license: agpl-3.0
pipeline_tag: object-detection
library_name: ultralytics
tags:
  - age-of-empires-ii
  - yolo26
  - onnx
---

# AoE2 entity detector

This model detects 60 entity classes in Age of Empires II: Definitive Edition
screenshots. `model.onnx` contains the weights used by this project's Mac-hosted
detector server. The model repository is an artifact registry, not a hosted
inference API. Future versions replace `model.onnx` in a new commit;
consumers must pin a commit or release tag.

## Intended use

The supported use is screen-based perception for the project's game agent. The
model was trained on synthetic scenes generated from game sprites and annotated
real-game screenshots. Neither the training images nor extracted sprites are
included in this artifact release.

The model is fine-tuned from the Ultralytics YOLO26n base model and exported to
ONNX. Its metadata identifies the Ultralytics AGPL-3.0 license.

## License and terms

The model weights are available under the [GNU Affero General Public License
v3.0 (AGPL-3.0)](LICENSE). You may use, modify, and redistribute them,
including commercially, subject to that license. This project imposes no
additional noncommercial-use restriction or fee.

Users integrating the model into an application or service must meet applicable
AGPL-3.0 obligations, including corresponding-source requirements. For
different terms covering Ultralytics technology, contact Ultralytics about its
Enterprise license. This is an independent fine-tune, not an official
Ultralytics release. The model is provided as-is, without warranty.

## Inference contract

See `inference-contract.json` and `classes.yaml`. Input is a 1280 × 1280 RGB
letterboxed image, normalized to float32 values in `[0, 1]` and transposed to
NCHW. The model returns up to 300 rows of
`[x1, y1, x2, y2, confidence, class_id]` in model-input pixel coordinates.
The detector server applies class-specific confidence thresholds and maps boxes
back to the original screenshot; its client handles tracking and deduplication.
Loading the ONNX file alone does not reproduce those application-level results.

## Evaluation and limitations

On the project's **32-image real-frame validation split** (1,212 labeled
objects), the served ONNX checkpoint reaches micro **precision 0.753, recall
0.633, and F1 0.688** at IoU ≥ 0.50. This uses the detector server's
single-pass 1280-pixel preprocessing, its deployed per-class confidence
thresholds, and the client's classwise NMS. The evaluation used ONNX Runtime's
CPU provider and excludes temporal tracking, cached detections, and gameplay.

**This is validation, not an independent test benchmark.** Training and
validation images came from the same January 2026 capture sessions and can
contain nearby frames; the split was by image rather than game/session. The
existing confidence thresholds were also developed using the project's
validation workflow. The labels and screenshots are not released with the
model. This overall score is dominated by common classes and does not establish
reliable detection of rare objects. The full-screen 1280 × 831 validation
images also do not establish performance on other game
captures, UI scales, maps, or versions. The model's ability to support
reliable gathering or combat has **not** been validated by this score.

See [the complete evaluation protocol and per-class results](EVALUATION.md) and
[machine-readable counts and hashes](evaluation.json). A future independent,
session-held-out test should be reported separately rather than replacing this
validation result.

## Provenance

- Served registry artifact: `model.onnx`
- Model SHA-256: `515a018bc2190fdf5427a01ff21e294331324929c8603d870c18255626cee8fd`
- Class schema SHA-256: `5365dcf538d16f9b237070a5e9c7609028314794dd8e544233eecb76a09de717`
- Detector source revision: `ca4b06310a49d46381c043bec3a35618c0e5dee2`

For reproducible game runs, download a full-commit-pinned revision and verify
both checksums before starting the detector server. Do not download weights in
the agent's frame-processing loop.
