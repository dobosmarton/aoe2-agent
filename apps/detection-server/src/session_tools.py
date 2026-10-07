"""Prepare and audit session-scoped detector labels without touching the test split.

Development frames can receive ONNX prelabels for correction in CVAT. Final-test
frames must be annotated blind. Corrected COCO boxes are imported as YOLO labels;
only a later human-review seal makes them eligible for evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import cast

from detection.labeling.class_mapping import load_classes_yaml
from detection.labeling.session_dataset import (
    REVIEWED_MANIFEST,
    CaptureSession,
    label_path,
    list_sessions,
    load_reviewed_labels,
    seal_reviewed_labels,
    sha256_file,
    validate_yolo_label,
)
from PIL import Image, ImageDraw

from .evaluate import load_cpu_model, predict

_CRITICAL_CLASSES = (
    "sheep",
    "berry_bush",
    "villager",
    "town_center",
    "house",
    "mill",
    "farm",
    "knight_line",
)


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError("Expected a JSON object")
    return cast("dict[str, object]", value)


def _array(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError("Expected a JSON array")
    return cast("list[object]", value)


def _integer(value: object, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Expected integer {description}")
    return value


def _text(value: object, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Expected text {description}")
    return value


def _session(root: Path, session_id: str) -> tuple[Path, CaptureSession]:
    for directory, session in list_sessions(root):
        if session.session_id == session_id:
            return directory, session
    raise ValueError(f"Unknown capture session: {session_id}")


def prelabel_session(root: Path, session_id: str, model: Path) -> Path:
    """Emit COCO suggestions for one development task, never the final test."""
    directory, session = _session(root, session_id)
    if session.split != "development":
        raise ValueError("Final-test images must be annotated blind, without model prelabels")
    output = directory / "prelabels.coco.json"
    if output.exists():
        raise FileExistsError(output)
    state = load_cpu_model(model)
    classes = load_classes_yaml()
    if tuple(classes[index] for index in sorted(classes)) != state.class_names:
        raise ValueError("Model and annotation class schemas differ")

    images: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    for image_id, frame in enumerate(session.frames, start=1):
        path = directory / frame.image
        with Image.open(path) as source:
            image = source.convert("RGB")
        images.append(
            {
                "id": image_id,
                "file_name": path.name,
                "width": image.width,
                "height": image.height,
            }
        )
        for prediction in predict(state, image):
            left, top, right, bottom = prediction.box
            if not all(math.isfinite(value) for value in (left, top, right, bottom)):
                continue
            left = max(0.0, min(left, image.width))
            top = max(0.0, min(top, image.height))
            right = max(0.0, min(right, image.width))
            bottom = max(0.0, min(bottom, image.height))
            if right <= left or bottom <= top:
                continue
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": prediction.class_id + 1,
                    "bbox": [left, top, right - left, bottom - top],
                    "area": (right - left) * (bottom - top),
                    "segmentation": [],
                    "iscrowd": 0,
                    "score": prediction.confidence,
                }
            )
    coco = {
        "info": {
            "description": "Model suggestions only; every visible object needs human review",
            "session_id": session_id,
            "model_sha256": sha256_file(model),
        },
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": class_id + 1, "name": name, "supercategory": ""}
            for class_id, name in sorted(classes.items())
        ],
    }
    output.write_text(json.dumps(coco, indent=2) + "\n", encoding="utf-8")
    return output


def import_corrected_coco(
    root: Path, session_id: str, coco_path: Path, *, replace_unsealed: bool = False
) -> int:
    """Import a complete CVAT COCO export; empty scenes receive empty labels."""
    directory, session = _session(root, session_id)
    if (directory / REVIEWED_MANIFEST).exists():
        raise ValueError("This session is already sealed; do not overwrite reviewed labels")
    data: object = json.loads(coco_path.read_text(encoding="utf-8"))
    coco = _object(data)
    classes = load_classes_yaml()
    name_to_id = {name: class_id for class_id, name in classes.items()}
    categories: dict[int, int] = {}
    for raw in _array(coco.get("categories")):
        category = _object(raw)
        identifier = _integer(category.get("id"), "category ID")
        name = _text(category.get("name"), "category name")
        if identifier in categories or name not in name_to_id:
            raise ValueError(f"Duplicate or unknown COCO category: {name}")
        categories[identifier] = name_to_id[name]

    frames_by_name = {Path(frame.image).name: frame for frame in session.frames}
    images_by_id: dict[int, str] = {}
    for raw in _array(coco.get("images")):
        image = _object(raw)
        identifier = _integer(image.get("id"), "image ID")
        raw_name = _text(image.get("file_name"), "image filename")
        name = PurePosixPath(raw_name.replace("\\", "/")).name
        if (
            identifier in images_by_id
            or name not in frames_by_name
            or name in images_by_id.values()
        ):
            raise ValueError(f"Unknown or duplicate COCO image: {name}")
        if (
            _integer(image.get("width"), "image width") != session.width
            or _integer(image.get("height"), "image height") != session.height
        ):
            raise ValueError(f"COCO image dimensions disagree with capture: {name}")
        images_by_id[identifier] = name
    if set(images_by_id.values()) != set(frames_by_name):
        raise ValueError("COCO export must contain every captured image, including empty scenes")

    lines: dict[str, list[str]] = {name: [] for name in frames_by_name}
    for raw in _array(coco.get("annotations")):
        annotation = _object(raw)
        image_id = _integer(annotation.get("image_id"), "annotation image ID")
        category_id = _integer(annotation.get("category_id"), "annotation category ID")
        if image_id not in images_by_id or category_id not in categories:
            raise ValueError("Annotation references an unknown image or category")
        box = _array(annotation.get("bbox"))
        if len(box) != 4 or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool) for value in box
        ):
            raise ValueError("COCO annotation requires a four-number bbox")
        x, y, width, height = (float(value) for value in box)
        if not all(math.isfinite(value) for value in (x, y, width, height)) or not (
            0 <= x < x + width <= session.width and 0 <= y < y + height <= session.height
        ):
            raise ValueError("COCO annotation bbox leaves the captured image")
        lines[images_by_id[image_id]].append(
            f"{categories[category_id]} "
            f"{(x + width / 2) / session.width:.9f} "
            f"{(y + height / 2) / session.height:.9f} "
            f"{width / session.width:.9f} {height / session.height:.9f}"
        )

    labels_dir = directory / "labels"
    if labels_dir.exists() and not replace_unsealed:
        raise FileExistsError("Unsealed labels exist; use --replace-unsealed to archive them")
    if labels_dir.exists() and not labels_dir.is_dir():
        raise ValueError(f"Labels path is not a directory: {labels_dir}")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    incoming = directory / f".labels_incoming_{stamp}"
    incoming.mkdir()
    for name, frame in frames_by_name.items():
        path = incoming / f"{Path(frame.image).stem}.txt"
        path.write_text("\n".join(lines[name]) + ("\n" if lines[name] else ""), encoding="utf-8")
        validate_yolo_label(path, len(classes))
    if labels_dir.exists():
        history = directory / "annotation_history"
        history.mkdir(exist_ok=True)
        labels_dir.rename(history / f"labels_{stamp}")
    incoming.rename(labels_dir)
    return sum(len(items) for items in lines.values())


def _difference_hash(path: Path) -> int:
    """Coarse visual similarity check; flags pairs for review, never deletes."""
    with Image.open(path) as source:
        image = source.convert("L").resize((9, 8))
    values = image.tobytes()
    bits = 0
    for row in range(8):
        for column in range(8):
            bits = (bits << 1) | (values[row * 9 + column] > values[row * 9 + column + 1])
    return bits


def audit_dataset(root: Path) -> dict[str, object]:
    """Count coverage, missing review, and cross-split near-duplicates."""
    sessions = list_sessions(root)
    classes = load_classes_yaml()
    split_frames: Counter[str] = Counter()
    by_class: dict[str, Counter[str]] = defaultdict(Counter)
    sources: dict[str, Counter[str]] = defaultdict(Counter)
    maps: dict[str, Counter[str]] = defaultdict(Counter)
    civilizations: dict[str, Counter[str]] = defaultdict(Counter)
    game_versions: dict[str, Counter[str]] = defaultdict(Counter)
    graphics_presets: dict[str, Counter[str]] = defaultdict(Counter)
    ui_scales: dict[str, Counter[str]] = defaultdict(Counter)
    zooms: dict[str, Counter[str]] = defaultdict(Counter)
    stages: dict[str, Counter[str]] = defaultdict(Counter)
    resolutions: dict[str, Counter[str]] = defaultdict(Counter)
    pending: list[str] = []
    review_samples: list[str] = []
    hashes: dict[str, list[str]] = defaultdict(list)
    perceptual: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for directory, session in sessions:
        split = session.split
        split_frames[split] += len(session.frames)
        sources[split][session.source] += 1
        maps[split][session.map_name] += 1
        civilizations[split][session.civilization] += 1
        game_versions[split][session.game_version] += 1
        graphics_presets[split][session.graphics_preset] += 1
        ui_scales[split][session.ui_scale] += 1
        for index, frame in enumerate(session.frames):
            image = directory / frame.image
            zooms[split][frame.zoom] += 1
            stages[split][frame.game_stage] += 1
            resolutions[split][f"{session.width}x{session.height}"] += 1
            hashes[frame.image_sha256].append(f"{session.session_id}/{frame.image}")
            perceptual[split].append(
                (f"{session.session_id}/{frame.image}", _difference_hash(image))
            )
            if index in {0, len(session.frames) // 2, len(session.frames) - 1}:
                review_samples.append(f"{session.session_id}/{frame.image}")
        if not (directory / REVIEWED_MANIFEST).exists():
            pending.append(session.session_id)
            continue
        reviewed = load_reviewed_labels(directory, session)
        for item in reviewed.labels:
            for class_id in validate_yolo_label(directory / item.label, len(classes)):
                by_class[split][classes[class_id]] += 1

    near_duplicates: list[dict[str, object]] = []
    for dev_path, dev_hash in perceptual["development"]:
        for test_path, test_hash in perceptual["final_test"]:
            distance = (dev_hash ^ test_hash).bit_count()
            if distance <= 4:
                near_duplicates.append(
                    {"development": dev_path, "final_test": test_path, "hash_distance": distance}
                )
    return {
        "schema_version": 1,
        "sessions": {
            split: [s.session_id for _, s in sessions if s.split == split] for split in split_frames
        },
        "frames": dict(split_frames),
        "labeled_boxes": {split: dict(counts) for split, counts in by_class.items()},
        "critical_class_support": {
            split: {name: by_class[split][name] for name in _CRITICAL_CLASSES}
            for split in split_frames
        },
        "sources": {split: dict(counts) for split, counts in sources.items()},
        "maps": {split: dict(counts) for split, counts in maps.items()},
        "civilizations": {split: dict(counts) for split, counts in civilizations.items()},
        "game_versions": {split: dict(counts) for split, counts in game_versions.items()},
        "graphics_presets": {split: dict(counts) for split, counts in graphics_presets.items()},
        "ui_scales": {split: dict(counts) for split, counts in ui_scales.items()},
        "zooms": {split: dict(counts) for split, counts in zooms.items()},
        "game_stages": {split: dict(counts) for split, counts in stages.items()},
        "resolutions": {split: dict(counts) for split, counts in resolutions.items()},
        "unreviewed_sessions": pending,
        "identical_capture_groups": [paths for paths in hashes.values() if len(paths) > 1],
        "near_duplicate_cross_split_pairs": near_duplicates[:50],
        "near_duplicate_cross_split_count": len(near_duplicates),
        "human_qa_sample": review_samples,
    }


def render_qa_gallery(root: Path, output: Path) -> list[Path]:
    """Render imported labels before sealing, or sealed labels for reinspection."""
    classes = load_classes_yaml()
    rendered: list[Path] = []
    for directory, session in list_sessions(root):
        if not all(label_path(directory, frame).is_file() for frame in session.frames):
            continue
        if (directory / REVIEWED_MANIFEST).exists():
            load_reviewed_labels(directory, session)
        indices = {0, len(session.frames) // 2, len(session.frames) - 1}
        for index in sorted(indices):
            frame = session.frames[index]
            label = label_path(directory, frame)
            validate_yolo_label(label, len(classes))
            destination = output / f"{session.session_id}_{Path(frame.image).stem}.png"
            if destination.exists():
                raise FileExistsError(destination)
            with Image.open(directory / frame.image) as source:
                image = source.convert("RGB")
            drawing = ImageDraw.Draw(image)
            for line in label.read_text(encoding="utf-8").splitlines():
                class_id_text, cx_text, cy_text, width_text, height_text = line.split()
                cx, cy, width, height = (
                    float(value) for value in (cx_text, cy_text, width_text, height_text)
                )
                box = (
                    (cx - width / 2) * image.width,
                    (cy - height / 2) * image.height,
                    (cx + width / 2) * image.width,
                    (cy + height / 2) * image.height,
                )
                drawing.rectangle(box, outline="red", width=2)
                drawing.text((box[0], box[1]), classes[int(class_id_text)], fill="yellow")
            output.mkdir(parents=True, exist_ok=True)
            image.save(destination)
            rendered.append(destination)
    return rendered


class _Args(argparse.Namespace):
    command: str
    root: Path
    session_id: str
    model: Path
    coco: Path
    reviewer: str
    output: Path
    output_dir: Path
    replace_unsealed: bool


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    prelabel = commands.add_parser("prelabel")
    prelabel.add_argument("--session-id", required=True)
    prelabel.add_argument("--model", type=Path, required=True)
    imported = commands.add_parser("import-coco")
    imported.add_argument("--session-id", required=True)
    imported.add_argument("--coco", type=Path, required=True)
    imported.add_argument("--replace-unsealed", action="store_true")
    sealed = commands.add_parser("seal")
    sealed.add_argument("--session-id", required=True)
    sealed.add_argument("--reviewer", required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--output", type=Path, required=True)
    gallery = commands.add_parser("qa-gallery")
    gallery.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(namespace=_Args())

    if args.command == "prelabel":
        print(prelabel_session(args.root, args.session_id, args.model))
    elif args.command == "import-coco":
        count = import_corrected_coco(
            args.root, args.session_id, args.coco, replace_unsealed=args.replace_unsealed
        )
        print(f"Imported {count} corrected boxes; inspect QA gallery before sealing")
    elif args.command == "seal":
        reviewed = seal_reviewed_labels(args.root / args.session_id, args.reviewer)
        print(f"Sealed {len(reviewed.labels)} reviewed frames")
    elif args.command == "qa-gallery":
        images = render_qa_gallery(args.root, args.output_dir)
        print(f"Rendered {len(images)} images for manual QA at {args.output_dir}")
    else:
        report = audit_dataset(args.root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
