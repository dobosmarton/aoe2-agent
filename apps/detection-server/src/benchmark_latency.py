"""Measure the served detector's single-image CPU path on labeled real frames.

The images are decoded before timing. Each timed call includes letterboxing,
ONNX inference, output decoding, confidence filtering, and classwise NMS.
Network transfer, screenshot capture, tracking, and agent scheduling are outside
this measurement.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

from detection_server.evaluate import (
    discover_cases,
    load_cpu_model,
    manifest_sha256,
    predict,
    sha256_file,
)
from PIL import Image


@dataclass(frozen=True, slots=True)
class LatencyReport:
    measurement: str
    image_count: int
    warmup_calls: int
    median_ms: float
    p95_ms: float
    model_sha256: str
    dataset_manifest_sha256: str
    machine: str
    python: str
    onnxruntime: str
    pillow: str


def percentile(values: list[float], fraction: float) -> float:
    """Linearly interpolate the requested percentile of measured times."""
    if not values or not 0 <= fraction <= 1:
        raise ValueError("Expected nonempty measurements and a fraction in [0, 1]")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * fraction
    lower = math.floor(rank)
    upper = math.ceil(rank)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def benchmark(model: Path, data: Path, *, warmup_calls: int = 3) -> LatencyReport:
    """Time one inference per real validation image after model warmup."""
    if warmup_calls < 0:
        raise ValueError("warmup_calls must be nonnegative")
    cases = discover_cases(data)
    state = load_cpu_model(model)
    images: list[Image.Image] = []
    for case in cases:
        with Image.open(case.image) as source:
            images.append(source.convert("RGB"))

    for _ in range(warmup_calls):
        predict(state, images[0])

    elapsed_ms: list[float] = []
    for image in images:
        start = time.perf_counter_ns()
        predict(state, image)
        elapsed_ms.append((time.perf_counter_ns() - start) / 1_000_000)

    return LatencyReport(
        measurement="decoded_image_to_postprocessed_boxes_cpu",
        image_count=len(images),
        warmup_calls=warmup_calls,
        median_ms=percentile(elapsed_ms, 0.5),
        p95_ms=percentile(elapsed_ms, 0.95),
        model_sha256=sha256_file(model),
        dataset_manifest_sha256=manifest_sha256(cases),
        machine=platform.platform(),
        python=platform.python_version(),
        onnxruntime=version("onnxruntime"),
        pillow=version("Pillow"),
    )


class _Args(argparse.Namespace):
    model: Path
    data: Path
    output: Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(namespace=_Args())
    report = benchmark(args.model, args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(asdict(report), indent=2, sort_keys=True) + "\n")
    print(
        f"Wrote {args.output}: {report.image_count} images, "
        f"median={report.median_ms:.1f} ms, p95={report.p95_ms:.1f} ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
