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
└── classes.yaml      # Bundled class list (synced from packages/detection/src/training/config/classes.yaml)
```

## Common entry points

```bash
just server --model packages/detection/src/inference/models/aoe2_yolo_v5.onnx
just server --model packages/detection/src/inference/models/aoe2_yolo_v5.mlpackage
```

Or directly:

```bash
uv run --package detection-server aoe2-server --host 0.0.0.0 --port 8420 \
    --model packages/detection/src/inference/models/aoe2_yolo_v5.onnx
```

Health check from a peer machine (replace with your Mac's IP):

```bash
curl http://192.168.64.1:8420/health
# {"backend": "onnx_cpu", "classes": 60, "model": "aoe2_yolo_v5.onnx"}
```

## Conventions

- `classes.yaml` is bundled inside the wheel via `package-data`. Keep it
  synced with `packages/detection/src/training/config/classes.yaml` via
  `just sync-classes` after any class-list change in training.
- CoreML inference is opt-in via the `coreml` extra on `detection`
  (`coremltools` is macOS-only, so it's not in the default deps).

## Where to read more

- [Deployment Guide](../../docs/deployment-guide.md) — Mac + Windows VM setup, IP discovery, firewall.
- [Chapter 7 — Detector Architecture](../../docs/part3-entity-detection/07-detector-architecture.md) — the inference backends this server exposes.
