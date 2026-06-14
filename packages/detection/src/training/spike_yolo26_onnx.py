"""D1 go/no-go spike: inspect YOLO26's ONNX output layout and benchmark latency.

YOLO26 is NMS-free, so its exported ONNX tensor differs from YOLO11's. Before
trusting the shared decoder (`inference/onnx_layout.py`) or deleting the v5
model, run this on the GPU box (to export) and on the **ARM64 VM** (to time):

    # Export yolo26n and describe + benchmark its ONNX output
    python -m detection.training.spike_yolo26_onnx --model yolo26n.pt --imgsz 1280 \
        --sample-image detection/real_screenshots/raw/<some>.jpg \
        --baseline detection/inference/models/aoe2_yolo_v5.onnx

This script does not modify anything. It answers two questions:

  1. **Layout** — is the output ``(num_boxes, 6)`` end-to-end (and are the boxes
     xyxy or xywh, padded top-k?) or the raw ``(4 + num_classes, num_boxes)``?
     The first rows are printed so a human can confirm before the decoder ships.
  2. **Latency** — single 1280 image and an N-tile SAHI batch, versus the v5
     baseline. CPU-only on ARM64 must meet budget (the ~43% CPU-ONNX claim).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import onnxruntime as ort

SAHI_TILE_SIZE = 640
NUM_CLASSES = 60


@dataclass(frozen=True, slots=True)
class LatencyResult:
    """Mean wall-clock latency in milliseconds for the two inference shapes."""

    single_ms: float
    sahi_batch_ms: float


def resolve_onnx(model: str, imgsz: int) -> str:
    """Return an ONNX path, exporting a .pt model first if needed."""
    if model.endswith(".onnx"):
        return model
    from detection.training.export_onnx import export_to_onnx

    return export_to_onnx(model, imgsz)


def make_session(onnx_path: str) -> ort.InferenceSession:
    """Create an inference session and report the active execution providers."""
    session = ort.InferenceSession(onnx_path)
    print(f"Providers: {session.get_providers()}")
    return session


def load_input(sample_image: str | None, imgsz: int) -> np.ndarray:
    """Build a (1, 3, imgsz, imgsz) float32 input from a screenshot or noise.

    A real screenshot yields meaningful detection rows for layout inspection;
    noise still reveals the tensor shape but usually decodes to empty boxes.
    """
    if sample_image is None:
        rng = np.random.default_rng(0)
        return cast("np.ndarray", rng.random((1, 3, imgsz, imgsz), dtype=np.float32))

    from PIL import Image

    image = Image.open(sample_image).convert("RGB").resize((imgsz, imgsz))
    array = cast("np.ndarray", np.asarray(image).astype(np.float32) / 255.0)
    chw = cast("np.ndarray", np.transpose(array, (2, 0, 1)))
    return cast("np.ndarray", np.expand_dims(chw, axis=0))


def _run(session: ort.InferenceSession, batch: np.ndarray) -> list[np.ndarray]:
    input_name = cast("str", session.get_inputs()[0].name)
    return cast("list[np.ndarray]", session.run(None, {input_name: batch}))


def describe_output(session: ort.InferenceSession, model_input: np.ndarray) -> None:
    """Print the output tensor shape and first rows, with a layout reading."""
    outputs = _run(session, model_input)
    primary = outputs[0]
    print(f"\nOutputs: {len(outputs)}; primary shape: {primary.shape}")

    if primary.ndim != 3:
        print("  Unexpected: primary output is not 3D — inspect manually.")
        return

    example = cast("np.ndarray", primary[0])
    preview = cast("list[list[float]]", cast("np.ndarray", example[:5]).tolist())
    if example.shape[-1] == 6:
        print("  Layout: end-to-end (num_boxes, 6) — NMS-free, as YOLO26 should be.")
        print("  First rows [c0 c1 c2 c3 conf class]; confirm xyxy (c2>c0, c3>c1) vs xywh:")
        for row in preview:
            print(f"    {[round(v, 2) for v in row]}")
    elif example.shape[0] == 4 + NUM_CLASSES:
        print(f"  Layout: raw (4 + {NUM_CLASSES}, num_boxes) — needs transpose + argmax.")
    else:
        print(f"  Layout: UNKNOWN example shape {example.shape} — decoder must be updated.")


def benchmark(session: ort.InferenceSession, imgsz: int, tiles: int, iters: int) -> LatencyResult:
    """Time a single full-image pass and an N-tile SAHI batch pass."""
    single = cast("np.ndarray", np.zeros((1, 3, imgsz, imgsz), dtype=np.float32))
    sahi_batch = cast(
        "np.ndarray", np.zeros((tiles, 3, SAHI_TILE_SIZE, SAHI_TILE_SIZE), dtype=np.float32)
    )
    return LatencyResult(
        single_ms=_mean_ms(session, single, iters),
        sahi_batch_ms=_mean_ms(session, sahi_batch, iters),
    )


def _mean_ms(session: ort.InferenceSession, batch: np.ndarray, iters: int) -> float:
    _run(session, batch)  # warm up
    start = time.perf_counter()
    for _ in range(iters):
        _run(session, batch)
    return (time.perf_counter() - start) / iters * 1000.0


def _report(label: str, result: LatencyResult, tiles: int) -> None:
    print(f"\n{label}:")
    print(f"  single full image: {result.single_ms:.1f} ms")
    print(f"  SAHI batch ({tiles} tiles): {result.sahi_batch_ms:.1f} ms")


class _SpikeArgs(argparse.Namespace):
    model: str
    imgsz: int
    sample_image: str | None
    baseline: str | None
    tiles: int
    iters: int


def main() -> int:
    parser = argparse.ArgumentParser(description="YOLO26 ONNX layout + latency spike")
    parser.add_argument("--model", default="yolo26n.pt", help="Path to .pt (exported) or .onnx")
    parser.add_argument("--imgsz", type=int, default=1280, help="Full-image input size")
    parser.add_argument("--sample-image", default=None, help="Screenshot for layout inspection")
    parser.add_argument("--baseline", default=None, help="v5 .onnx to compare latency against")
    parser.add_argument("--tiles", type=int, default=18, help="SAHI batch tile count")
    parser.add_argument("--iters", type=int, default=20, help="Timed iterations per shape")
    args = parser.parse_args(namespace=_SpikeArgs())

    onnx_path = resolve_onnx(args.model, args.imgsz)
    session = make_session(onnx_path)
    describe_output(session, load_input(args.sample_image, args.imgsz))
    _report(
        f"YOLO26 ({Path(onnx_path).name})",
        benchmark(session, args.imgsz, args.tiles, args.iters),
        args.tiles,
    )

    if args.baseline is not None:
        baseline_session = make_session(args.baseline)
        _report(
            f"v5 baseline ({Path(args.baseline).name})",
            benchmark(baseline_session, args.imgsz, args.tiles, args.iters),
            args.tiles,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
