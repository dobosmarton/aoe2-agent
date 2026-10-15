# AoE2 entity detection

This package contains the AoE2:DE entity detector, model-training tools, labeling utilities, and sprite extraction. The deployed model is a YOLO26n checkpoint served as ONNX by `apps/detection-server`; its class schema lives in `src/training/config/classes.yaml`.

The detector adapters return **stateless, single-frame** classes, boxes, and scores. The gameplay agent owns short-term association and camera alignment in `apps/agent/src/temporal_entities.py`. Every published spatial observation uses a fresh inference result; historical predictions are never click targets. See [streaming perception](../../docs/streaming-perception.md) for the live observation contract.

## Main components

- `src/inference/detector.py` — local single-frame PyTorch, ONNX, and mock inference.
- `src/inference/remote_detector.py` — HTTP client for the macOS detection server.
- `src/inference/thresholds.py` — deployed per-class confidence thresholds.
- `src/training/` — synthetic scene generation, model training, and ONNX export.
- `src/labeling/` — prelabeling, class mapping, and CVAT dataset preparation.
- `src/extraction/` — game-sprite extraction.

The shipped v9 release evaluation measures the exact ONNX deployment path on 32 labeled real validation screenshots (1,212 boxes): micro precision **0.753**, recall **0.633**, F1 **0.688** at IoU 0.50. This is validation, not an independent test set; rare classes have very small samples. See the [evaluation report](registry/aoe2-entity-detector/EVALUATION.md) and [machine-readable results](registry/aoe2-entity-detector/evaluation.json) before comparing a retrained model.

## Development

The local detector API can inspect one screenshot:

```python
from detection.inference.detector import get_detector

detector = get_detector(imgsz=1280, use_sahi=False)
entities = detector.detect_fast(screenshot_bytes)
```

The agent's production path captures game pixels on Windows and sends selected JPEG frames to the remote detector. The server returns single-frame results; the agent then publishes camera-aware spatial observations independently of HUD readings. Interactive runs can use the local detector fallback, while recorded experiments fail when their required remote detector is unavailable.

The current model is trained for single-pass inference at `imgsz=1280`. SAHI inference remains available in the library for model experiments, but it is not the deployed mode because tiling changes object scale relative to this checkpoint's training distribution.

For dataset preparation and retraining, start with [the training pipeline](../../docs/part3-entity-detection/08-training-pipeline.md), [labeling workflow](../../docs/part3-entity-detection/09-labeling-and-active-learning.md), and [detector quality runbook](../../docs/runbooks/detector-quality-retraining.md).
