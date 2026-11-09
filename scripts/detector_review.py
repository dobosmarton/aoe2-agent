"""Side-by-side review of the detector on Mac frames and on the VM test frames.

Plan rule R3: at matched object size, are Mac frames clearly worse for the detector
than VM frames? There are no Mac labels, so the frames are reviewed by eye: the
same model's boxes are drawn on both sets in the same layout, and the wrong boxes
and obvious misses are counted per frame. Two numbers come without labels: boxes
per frame, and how much the served model and the previous one agree (a box from one with a same-class box of
IoU >= 0.5 from the other).

    # Lossless frames from the running game (Terminal.app, game in front):
    uv run python scripts/detector_review.py capture --count 20 --every 6 --out logs/detector-review/mac
    # Review sheets and the label-free numbers for a set of frames:
    uv run python scripts/detector_review.py render --frames logs/detector-review/mac \\
        --out logs/detector-review/mac-sheets --limit 20

Frames and sheets are game screenshots: keep them under logs/, which git ignores.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image, ImageDraw

if TYPE_CHECKING:
    from collections.abc import Sequence

    from core import DetectedEntity

MODELS = ("aoe2_yolo_v12_nms736", "aoe2_yolo_v11")  # The served model, and the previous one.
SHEET_WIDTH = 1512  # Points on the Mac display: one sheet fills it.
# The VM frames carry no band; Mac frames are the game area already. Both are
# 3024 wide, so one sheet scale fits both.
AGREEMENT_IOU = 0.5


@dataclass(frozen=True, slots=True)
class FrameRow:
    frame: str
    boxes: dict[str, int]
    agreement: float


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def agreement(first: Sequence[DetectedEntity], second: Sequence[DetectedEntity]) -> float:
    """Share of boxes, from either model, that the other model also found."""
    if not first and not second:
        return 1.0

    def matched(source: Sequence[DetectedEntity], other: Sequence[DetectedEntity]) -> int:
        return sum(
            any(
                o.class_name == s.class_name and _iou(s.bbox, o.bbox) >= AGREEMENT_IOU
                for o in other
            )
            for s in source
        )

    return (matched(first, second) + matched(second, first)) / (len(first) + len(second))


def _sheet(image: Image.Image, entities: Sequence[DetectedEntity]) -> Image.Image:
    scale = SHEET_WIDTH / image.width
    sheet = image.resize((SHEET_WIDTH, round(image.height * scale)))
    draw = ImageDraw.Draw(sheet)
    for entity in entities:
        x0, y0, x1, y1 = (value * scale for value in entity.bbox)
        draw.rectangle((x0, y0, x1, y1), outline=(255, 0, 255), width=1)
        draw.text((x0 + 1, max(0.0, y0 - 10)), entity.class_name, fill=(255, 255, 0))
    return sheet


def capture(count: int, every: float, out: Path) -> int:
    from gameplay_agent.screen import capture_native, close_capture
    from gameplay_agent.window_macos import MacBackend

    out.mkdir(parents=True, exist_ok=True)
    if not MacBackend().focus(2):
        print("the game couldn't be brought to the front")
        return 1
    try:
        for index in range(count):
            capture_native().image().save(out / f"mac_{index:02d}.png")
            print(f"captured {index + 1}/{count}", flush=True)
            time.sleep(every)
    finally:
        close_capture()
    return 0


def render(frames: Path, out: Path, limit: int) -> int:
    from detection.inference.detector import EntityDetector, resolve_model_path

    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in frames.iterdir() if p.suffix.lower() in {".png", ".jpg"})[:limit]
    detectors = {
        name: EntityDetector(model_path=resolve_model_path(name), use_sahi=False, imgsz=1280)
        for name in MODELS
    }
    rows: list[FrameRow] = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        found = {name: detector.detect_fast(image) for name, detector in detectors.items()}
        for name, entities in found.items():
            _sheet(image, entities).save(out / f"{path.stem}.{name}.jpg", quality=90)
        rows.append(
            FrameRow(
                path.name,
                {name: len(entities) for name, entities in found.items()},
                round(agreement(*found.values()), 3),
            )
        )
    count = max(len(rows), 1)
    summary: dict[str, float | int] = {
        "frames": len(rows),
        **{
            f"mean_boxes_{name}": round(sum(row.boxes[name] for row in rows) / count, 1)
            for name in MODELS
        },
        "mean_agreement": round(sum(row.agreement for row in rows) / count, 3),
    }
    frames_out = [
        {"frame": row.frame, "boxes": row.boxes, "agreement": row.agreement} for row in rows
    ]
    (out / "summary.json").write_text(
        json.dumps({"summary": summary, "frames": frames_out}, indent=2)
    )
    print(json.dumps(summary))
    return 0


class _Args(argparse.Namespace):
    command: str
    count: int
    every: float
    out: Path
    frames: Path
    limit: int


def main() -> None:
    parser = argparse.ArgumentParser(description="Detector review on Mac and VM frames")
    commands = parser.add_subparsers(dest="command", required=True)
    take = commands.add_parser("capture", help="save lossless frames from the running game")
    take.add_argument("--count", type=int, default=20)
    take.add_argument("--every", type=float, default=6.0)
    take.add_argument("--out", type=Path, required=True)
    draw = commands.add_parser("render", help="draw both models' boxes and count agreement")
    draw.add_argument("--frames", type=Path, required=True)
    draw.add_argument("--out", type=Path, required=True)
    draw.add_argument("--limit", type=int, default=20)
    args = parser.parse_args(namespace=_Args())
    if args.command == "capture":
        raise SystemExit(capture(args.count, args.every, args.out))
    raise SystemExit(render(args.frames, args.out, args.limit))


if __name__ == "__main__":
    main()
