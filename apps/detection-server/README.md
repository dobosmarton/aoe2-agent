# `detection-server/` — Macos-Hosted YOLO Inference Endpoint

A tiny FastAPI server that hosts the YOLO detector and answers POST requests
with bounding-box detections. Lives in its own package so the slim agent
install (Windows VM) doesn't pull `fastapi`/`uvicorn`, and so the macOS host
can run the heavy inference with CoreML/ANE while the VM does only screen
capture + click execution.

## What's here

```
packages/detection-server/src/
├── app.py            # FastAPI app: POST /detect, POST /detect/sahi, GET /health
└── classes.yaml      # Class mapping for the deployed checkpoint
```

## Common entry points

```bash
just server --model packages/detection/src/inference/models/aoe2_yolo_v9.onnx
just server --model packages/detection/src/inference/models/aoe2_yolo_v5.mlpackage  # CoreML build (only v5 is exported to .mlpackage; v9 serves via ONNX)
```

Or directly:

```bash
uv run --package detection-server aoe2-server --host 0.0.0.0 --port 8420 \
    --model packages/detection/src/inference/models/aoe2_yolo_v9.onnx
```

Health check from a peer machine (replace with your Mac's IP):

```bash
curl http://192.168.64.1:8420/health
# {"backend": "onnx_cpu", "classes": 60, "model": "aoe2_yolo_v9.onnx"}
```

## Conventions

- `classes.yaml` is bundled inside the wheel via `package-data`. It must match
  the checkpoint being served. The training schema now has 65 classes, while
  the deployed checkpoint still has 60; append the five new classes here when
  deploying a model trained with them.
- CoreML inference is opt-in via the `coreml` extra on `detection`
  (`coremltools` is macOS-only, so it's not in the default deps).

## Where to read more

- [Deployment Guide](../../docs/deployment-guide.md) — Mac + Windows VM setup, IP discovery, firewall.
- [Chapter 7 — Detector Architecture](../../docs/part3-entity-detection/07-detector-architecture.md) — the inference backends this server exposes.
