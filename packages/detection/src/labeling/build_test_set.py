"""Build a scored test split from a reviewed X-AnyLabeling batch.

The batch manifest marks every frame ``test``, so ``prepare_training`` refuses
it. This module is the only path from those frames to YOLO labels, and it
refuses a session that any training dataset manifest already used. Every image
must carry a sidecar marked checked: an unreviewed prelabel is not ground truth.
Fine reviewer labels (``ibex``) become their class (``deer``); the original
label of each box is kept in ``test/fine_labels`` for per-label recall.

Run from the repository root with::

    uv run python -m detection.labeling.build_test_set \
        --batch packages/detection/src/real_screenshots/labeling_batch_20261029_independent_test \
        --training-manifest packages/detection/src/training_data_reviewed_20261018/dataset_manifest.json \
        --output tmp/test_data_independent_20261029

Score it with ``python -m detection_server.evaluate --split test``.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, cast

from PIL import Image

from .class_mapping import load_label_aliases
from .prepare_training import _classes, _sha256, _source_name, _yolo_line

if TYPE_CHECKING:
    from collections.abc import Sequence


def training_sessions(manifests: Sequence[Path]) -> set[str]:
    """Sessions that trained or selected any candidate model."""
    sessions: set[str] = set()
    for path in manifests:
        data = cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
        for key in ("train_sessions", "validation_sessions"):
            sessions.update(cast("list[str]", data[key]))
    return sessions


def _box(points: list[list[float]], width: int, height: int) -> tuple[float, float, float, float]:
    """Return a COCO box clipped to the image; drawing tools allow edge overshoot."""
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x1, x2 = max(0.0, min(xs)), min(float(width), max(xs))
    y1, y2 = max(0.0, min(ys)), min(float(height), max(ys))
    return x1, y1, x2 - x1, y2 - y1


def _labels(
    sidecar: Path, classes: tuple[str, ...], aliases: dict[str, str], width: int, height: int
) -> list[tuple[str, str]]:
    """YOLO lines paired with the reviewer's own label for each box."""
    if not sidecar.is_file():
        raise ValueError(f"Missing review sidecar: {sidecar}")
    data = cast("dict[str, object]", json.loads(sidecar.read_text(encoding="utf-8")))
    if data.get("checked") is not True:
        raise ValueError(f"Image has not been marked reviewed: {sidecar}")
    if (data.get("imageWidth"), data.get("imageHeight")) != (width, height):
        raise ValueError(f"Sidecar dimensions differ from the image: {sidecar}")
    lines: list[tuple[str, str]] = []
    for shape in cast("list[dict[str, object]]", data["shapes"]):
        label = cast("str", shape["label"])
        if shape["shape_type"] != "rectangle":
            raise ValueError(f"Only rectangles are scored, got {shape['shape_type']} in {sidecar}")
        class_name = aliases.get(label, label)
        if class_name not in classes:
            raise ValueError(f"Unknown class {label!r} in {sidecar}")
        points = cast("list[list[float]]", shape["points"])
        box = _box(points, width, height)
        lines.append((_yolo_line(classes.index(class_name), box, width, height), label))
    return lines


def build_test_set(
    batch: Path, output: Path, training_manifests: Sequence[Path]
) -> dict[str, object]:
    """Validate the reviewed batch and write ``output/test`` in YOLO format."""
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    if not training_manifests:
        raise ValueError("At least one training dataset manifest is required")
    classes = _classes()
    aliases = load_label_aliases()
    used = training_sessions(training_manifests)
    with (batch / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Batch manifest is empty: {batch}")

    staged: list[tuple[Path, str, list[tuple[str, str]], dict[str, object]]] = []
    for row in rows:
        name = _source_name(Path(row["image"]).name)
        session = row["source_session"].strip()
        if row["split"] != "test":
            raise ValueError(f"Only test frames may be built: {name}")
        if session in used:
            raise ValueError(f"Session {session} already trained or selected a model: {name}")
        image = batch / "images" / name
        if not image.is_file() or _sha256(image) != row["sha256"]:
            raise ValueError(f"Missing or changed image: {image}")
        with Image.open(image) as opened:
            width, height = opened.size
        lines = _labels(image.with_suffix(".json"), classes, aliases, width, height)
        entry = {
            "image": f"test/images/real_{name}",
            "source_session": session,
            "image_sha256": row["sha256"],
            "boxes": len(lines),
        }
        staged.append((image, name, lines, entry))

    images_dir = output / "test" / "images"
    labels_dir = output / "test" / "labels"
    fine_dir = output / "test" / "fine_labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir()
    fine_dir.mkdir()
    for image, name, lines, _ in staged:
        stem = f"real_{Path(name).stem}"
        shutil.copy2(image, images_dir / f"real_{name}")
        (labels_dir / f"{stem}.txt").write_text("".join(f"{line}\n" for line, _ in lines))
        (fine_dir / f"{stem}.txt").write_text("".join(f"{label}\n" for _, label in lines))
    summary: dict[str, object] = {
        "batch_manifest_sha256": _sha256(batch / "manifest.csv"),
        "excluded_training_manifests": {str(path): _sha256(path) for path in training_manifests},
        "test_sessions": sorted({cast("str", entry["source_session"]) for *_, entry in staged}),
        "images": [entry for *_, entry in staged],
    }
    (output / "dataset_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


class _Args(argparse.Namespace):
    batch: Path
    output: Path
    training_manifest: list[Path]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New directory; must not exist")
    parser.add_argument(
        "--training-manifest",
        action="append",
        required=True,
        type=Path,
        help="dataset_manifest.json of a training set; repeat for each candidate model",
    )
    args = parser.parse_args(namespace=_Args())
    summary = build_test_set(args.batch, args.output, args.training_manifest)
    images = cast("list[dict[str, object]]", summary["images"])
    boxes = sum(cast("int", entry["boxes"]) for entry in images)
    print(f"Wrote {args.output}: {len(images)} images, {boxes} boxes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
