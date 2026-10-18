"""Build a reviewed, session-separated YOLO dataset from COCO or YOLO exports.

The input manifests record the source session and image hash for every frame.
Only development frames are accepted; an independent final-test session never
enters a training dataset. An existing output is never modified or replaced.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Literal, cast

import yaml
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from .class_mapping import load_classes_yaml

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp"})
_SCHEMA_PATH = Path(__file__).resolve().parents[1] / "training/config/classes.yaml"
Split = Literal["train", "val", "synthetic_val"]


class _CocoCategory(BaseModel):
    model_config = ConfigDict(strict=True)

    id: int
    name: str


class _CocoImage(BaseModel):
    model_config = ConfigDict(strict=True)

    id: int
    file_name: str = Field(min_length=1)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class _CocoAnnotation(BaseModel):
    model_config = ConfigDict(strict=True)

    id: int
    image_id: int
    category_id: int
    bbox: tuple[float, float, float, float] | None = None
    segmentation: list[list[float]] | None = None
    iscrowd: int = 0


class _CocoExport(BaseModel):
    model_config = ConfigDict(strict=True)

    images: list[_CocoImage]
    annotations: list[_CocoAnnotation]
    categories: list[_CocoCategory]


@dataclass(frozen=True, slots=True)
class SourceFrame:
    name: str
    session: str
    image_sha256: str
    width: int
    height: int
    manifest: Path


@dataclass(frozen=True, slots=True)
class LabeledFrame:
    image: Path
    image_sha256: str
    label_lines: tuple[str, ...]
    session: str
    source: str


@dataclass(frozen=True, slots=True)
class DatasetPlan:
    real_train: tuple[LabeledFrame, ...]
    real_val: tuple[LabeledFrame, ...]
    historical_train: tuple[LabeledFrame, ...]
    synthetic_train: tuple[LabeledFrame, ...]
    synthetic_val: tuple[LabeledFrame, ...]
    classes: tuple[str, ...]
    oversample_real: int
    source_hashes: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class DatasetSummary:
    reviewed_train_unique: int
    historical_train_unique: int
    real_train_unique: int
    real_train_on_disk: int
    real_val: int
    synthetic_train: int
    synthetic_val_diagnostic: int
    total_train_on_disk: int
    validation_images: int
    oversample_real: int
    per_class: dict[str, dict[str, int]]
    zero_real_classes: tuple[str, ...]


class _PrepareTrainingArgs(argparse.Namespace):
    cvat_export: list[str]
    images_dir: str
    manifest: list[str]
    output: str
    synthetic: str | None
    historical_real: str | None
    val_split: float
    seed: int
    oversample_real: int
    reviewed_export: bool
    dry_run: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_name(raw: str) -> str:
    path = Path(raw)
    if not raw or path.is_absolute() or ".." in path.parts or "\\" in raw:
        raise ValueError(f"Unsafe image name: {raw!r}")
    if path.suffix.lower() not in _IMAGE_SUFFIXES:
        raise ValueError(f"Unsupported image name: {raw!r}")
    return path.name


def _classes() -> tuple[str, ...]:
    schema = load_classes_yaml()
    if set(schema) != set(range(len(schema))) or len(set(schema.values())) != len(schema):
        raise ValueError("Training class IDs must be unique and contiguous from zero")
    return tuple(schema[index] for index in range(len(schema)))


def _read_manifests(paths: Sequence[Path]) -> dict[str, SourceFrame]:
    if not paths:
        raise ValueError("At least one source-session manifest is required")
    frames: dict[str, SourceFrame] = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                name = _source_name(row["image"])
                if row["split"] != "development":
                    raise ValueError(f"Only development frames may be prepared: {name}")
                session = row["source_session"].strip()
                digest = row["sha256"].lower()
                if (
                    not session
                    or len(digest) != 64
                    or any(c not in "0123456789abcdef" for c in digest)
                ):
                    raise ValueError(f"Invalid session or SHA-256 for {name} in {path}")
                frame = SourceFrame(
                    name=name,
                    session=session,
                    image_sha256=digest,
                    width=int(row["width"]),
                    height=int(row["height"]),
                    manifest=path,
                )
                if frame.width <= 0 or frame.height <= 0 or name in frames:
                    raise ValueError(f"Invalid or duplicate manifest image: {name}")
                frames[name] = frame
    if not frames:
        raise ValueError("Source manifests contain no frames")
    return frames


def _validate_image(path: Path, frame: SourceFrame) -> None:
    if not path.is_file() or _sha256(path) != frame.image_sha256:
        raise ValueError(f"Missing or changed image: {path}")
    with Image.open(path) as image:
        if image.size != (frame.width, frame.height):
            raise ValueError(f"Image dimensions differ from manifest: {path}")
        image.verify()


def _reviewed(path: Path, *, trust_export: bool) -> None:
    sidecar = path.with_suffix(".json")
    if sidecar.exists():
        data = cast("object", json.loads(sidecar.read_text(encoding="utf-8")))
        if not isinstance(data, dict) or data.get("checked") is not True:
            raise ValueError(f"Image has not been marked reviewed: {path}")
    elif not trust_export:
        raise ValueError(
            f"No review evidence for {path}; use --reviewed-export only after manual review"
        )


def _validate_box(x: float, y: float, width: float, height: float, iw: int, ih: int) -> None:
    if (
        not all(math.isfinite(value) for value in (x, y, width, height))
        or width <= 0
        or height <= 0
        or x < 0
        or y < 0
        or x + width > iw
        or y + height > ih
    ):
        raise ValueError(f"Invalid box {(x, y, width, height)} in {iw}x{ih} image")


def _yolo_line(class_id: int, bbox: tuple[float, float, float, float], iw: int, ih: int) -> str:
    x, y, width, height = bbox
    _validate_box(x, y, width, height, iw, ih)
    return (
        f"{class_id} {(x + width / 2) / iw:.9f} {(y + height / 2) / ih:.9f} "
        f"{width / iw:.9f} {height / ih:.9f}"
    )


def _coco_bbox(annotation: _CocoAnnotation) -> tuple[float, float, float, float]:
    if annotation.bbox is not None:
        return annotation.bbox
    polygons = annotation.segmentation
    if not polygons or any(len(polygon) < 6 or len(polygon) % 2 for polygon in polygons):
        raise ValueError(f"Annotation {annotation.id} has no valid box or polygon")
    coordinates = [value for polygon in polygons for value in polygon]
    xs, ys = coordinates[::2], coordinates[1::2]
    return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)


def _coco_path(directory: Path) -> Path | None:
    candidates = sorted((directory / "annotations").glob("*.json")) + sorted(
        directory.glob("*.json")
    )
    coco_files: list[Path] = []
    for path in candidates:
        data = cast("object", json.loads(path.read_text(encoding="utf-8")))
        if isinstance(data, dict) and {"images", "annotations", "categories"} <= data.keys():
            coco_files.append(path)
    if len(coco_files) > 1:
        raise ValueError(f"Multiple COCO exports in {directory}; provide one export per directory")
    return coco_files[0] if coco_files else None


def _read_coco(
    path: Path, classes: tuple[str, ...], sources: dict[str, SourceFrame]
) -> dict[str, tuple[str, ...]]:
    export = _CocoExport.model_validate_json(path.read_bytes())
    if not export.images:
        raise ValueError(f"COCO export has no images: {path}")
    category_ids = [item.id for item in export.categories]
    image_ids = [item.id for item in export.images]
    annotation_ids = [item.id for item in export.annotations]
    if any(len(items) != len(set(items)) for items in (category_ids, image_ids, annotation_ids)):
        raise ValueError(f"Duplicate COCO IDs in {path}")
    name_to_id = {name: index for index, name in enumerate(classes)}
    categories: dict[int, int] = {}
    for category in export.categories:
        if category.name not in name_to_id:
            raise ValueError(f"Unknown COCO class {category.name!r} in {path}")
        categories[category.id] = name_to_id[category.name]
    if len(set(categories.values())) != len(categories):
        raise ValueError(f"Duplicate COCO class names in {path}")
    images = {item.id: item for item in export.images}
    lines: dict[str, list[str]] = {}
    for item in export.images:
        name = _source_name(item.file_name)
        if name in lines:
            raise ValueError(f"Duplicate image name in {path}: {name}")
        source = sources.get(name)
        if source is None or (item.width, item.height) != (source.width, source.height):
            raise ValueError(f"COCO image missing from manifest or dimensions changed: {name}")
        lines[name] = []
    for item in export.annotations:
        if item.image_id not in images or item.category_id not in categories:
            raise ValueError(f"Annotation {item.id} references an unknown image or class")
        if item.iscrowd:
            raise ValueError(f"Crowd annotation {item.id} cannot become a single detection box")
        image = images[item.image_id]
        line = _yolo_line(categories[item.category_id], _coco_bbox(item), image.width, image.height)
        lines[_source_name(image.file_name)].append(line)
    return {name: tuple(items) for name, items in lines.items()}


def _parse_yolo(text: str, classes: tuple[str, ...], source: Path) -> tuple[str, ...]:
    lines: list[str] = []
    for number, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Malformed YOLO label at {source}:{number}")
        try:
            class_id = int(parts[0])
            x, y, width, height = (float(part) for part in parts[1:])
        except ValueError as exc:
            raise ValueError(f"Malformed YOLO label at {source}:{number}") from exc
        if not 0 <= class_id < len(classes):
            raise ValueError(f"Invalid class ID at {source}:{number}")
        # Source YOLO files use six decimals; a clipped edge can round 5e-7 outside.
        tolerance = 1e-6
        if (
            not all(math.isfinite(value) for value in (x, y, width, height))
            or width <= 0
            or height <= 0
            or x - width / 2 < -tolerance
            or y - height / 2 < -tolerance
            or x + width / 2 > 1 + tolerance
            or y + height / 2 > 1 + tolerance
        ):
            raise ValueError(f"Invalid normalized box at {source}:{number}")
        lines.append(" ".join(parts))
    return tuple(lines)


def _read_yolo(
    directory: Path, names: Iterable[str], classes: tuple[str, ...]
) -> dict[str, tuple[str, ...]]:
    labels_dir = next(
        (
            path
            for path in (
                directory / "obj_train_data",
                directory / "labels",
                directory / "train/labels",
            )
            if path.is_dir()
        ),
        None,
    )
    if labels_dir is None:
        raise ValueError(f"No COCO file or YOLO labels in {directory}")
    class_file = next(
        (path for path in (directory / "classes.txt", directory / "obj.names") if path.is_file()),
        None,
    )
    if class_file is None or tuple(class_file.read_text(encoding="utf-8").splitlines()) != classes:
        raise ValueError(f"YOLO export class mapping does not match training schema: {directory}")
    result: dict[str, tuple[str, ...]] = {}
    expected_stems = {Path(name).stem for name in names}
    for name in names:
        label = labels_dir / f"{Path(name).stem}.txt"
        result[name] = (
            _parse_yolo(label.read_text(encoding="utf-8"), classes, label) if label.exists() else ()
        )
    extra = {path.stem for path in labels_dir.glob("*.txt")} - expected_stems
    if extra:
        raise ValueError(f"YOLO labels have no matching image: {sorted(extra)[:5]}")
    return result


def _real_frames(
    export_dirs: Sequence[Path],
    images_dir: Path,
    sources: dict[str, SourceFrame],
    classes: tuple[str, ...],
    *,
    trust_export: bool,
) -> tuple[tuple[LabeledFrame, ...], tuple[tuple[str, str], ...]]:
    if not export_dirs:
        raise ValueError("At least one annotation export is required")
    all_labels: dict[str, tuple[str, ...]] = {}
    hashes: list[tuple[str, str]] = []
    for directory in export_dirs:
        coco_path = _coco_path(directory)
        if coco_path is not None:
            labels = _read_coco(coco_path, classes, sources)
            hashes.append((str(coco_path), _sha256(coco_path)))
        else:
            if len(export_dirs) != 1:
                raise ValueError("A YOLO export must cover the full source manifest by itself")
            labels = _read_yolo(directory, sources, classes)
        if set(all_labels) & set(labels):
            raise ValueError("The same image appears in multiple annotation exports")
        all_labels.update(labels)
    if set(all_labels) != set(sources):
        raise ValueError(
            f"Export/manifest image mismatch: missing {sorted(set(sources) - set(all_labels))[:5]}, "
            f"extra {sorted(set(all_labels) - set(sources))[:5]}"
        )
    if len({Path(name).stem for name in all_labels}) != len(all_labels):
        raise ValueError("Different source images share a label filename stem")
    frames: list[LabeledFrame] = []
    for name, source in sorted(sources.items()):
        image = images_dir / name
        _validate_image(image, source)
        _reviewed(image, trust_export=trust_export)
        frames.append(
            LabeledFrame(image, source.image_sha256, all_labels[name], source.session, "real")
        )
    hashes.extend(
        (str(path), _sha256(path)) for path in sorted({s.manifest for s in sources.values()})
    )
    return tuple(frames), tuple(hashes)


def _split_sessions(
    frames: tuple[LabeledFrame, ...], val_fraction: float, seed: int
) -> tuple[tuple[LabeledFrame, ...], tuple[LabeledFrame, ...]]:
    if not 0 < val_fraction < 1:
        raise ValueError("Validation fraction must be between zero and one")
    by_session: dict[str, list[LabeledFrame]] = defaultdict(list)
    for frame in frames:
        by_session[frame.session].append(frame)
    if len(by_session) < 2:
        raise ValueError("At least two independent sessions are required for train/val")
    sessions = sorted(by_session)
    random.Random(seed).shuffle(sessions)
    choices: dict[int, tuple[str, ...]] = {0: ()}
    for session in sessions:
        count = len(by_session[session])
        for total, selected in list(choices.items()):
            candidate = (*selected, session)
            existing = choices.get(total + count)
            if existing is None or len(candidate) > len(existing):
                choices[total + count] = candidate
    target = len(frames) * val_fraction
    valid = [(count, selection) for count, selection in choices.items() if 0 < count < len(frames)]
    _, chosen = min(valid, key=lambda item: (abs(item[0] - target), -len(item[1]), item[0]))
    val_sessions = set(chosen)
    train = tuple(frame for frame in frames if frame.session not in val_sessions)
    val = tuple(frame for frame in frames if frame.session in val_sessions)
    return train, val


def _dataset_names(directory: Path) -> tuple[str, ...]:
    raw = cast("object", yaml.safe_load((directory / "dataset.yaml").read_text(encoding="utf-8")))
    if not isinstance(raw, dict) or "names" not in raw:
        raise ValueError(f"Dataset class schema is missing: {directory}")
    names: object = raw["names"]
    if isinstance(names, list):
        result = tuple(names)
    elif isinstance(names, dict) and set(names) == set(range(len(names))):
        result = tuple(names[index] for index in range(len(names)))
    else:
        raise ValueError(f"Invalid dataset class IDs: {directory}")
    if not all(isinstance(name, str) for name in result) or len(set(result)) != len(result):
        raise ValueError(f"Invalid dataset class names: {directory}")
    return cast("tuple[str, ...]", result)


def _historical_frames(directory: Path, classes: tuple[str, ...]) -> tuple[LabeledFrame, ...]:
    """Import each distinct historical real image as training-only evidence.

    The historical split was not session-disjoint. Its validation images are
    therefore usable for training, but never for a new validation score.
    """
    names = _dataset_names(directory)
    unknown = set(names) - set(classes)
    if unknown:
        raise ValueError(f"Historical classes absent from training schema: {sorted(unknown)}")
    class_ids = {index: classes.index(name) for index, name in enumerate(names)}
    frames: list[LabeledFrame] = []
    seen_stems: set[str] = set()
    seen_hashes: set[str] = set()
    for split in ("train", "val"):
        images_dir = directory / split / "images"
        labels_dir = directory / split / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            raise ValueError(f"Historical {split} images and labels are required: {directory}")
        images = sorted(
            path for path in images_dir.glob("real_*") if path.suffix.lower() in _IMAGE_SUFFIXES
        )
        image_stems = {image.stem for image in images}
        label_stems = {label.stem for label in labels_dir.glob("real_*.txt")}
        if image_stems != label_stems:
            raise ValueError(f"Historical {split} images and labels differ: {directory}")
        for image in images:
            label = labels_dir / f"{image.stem}.txt"
            if not label.is_file():
                raise ValueError(f"Historical image has no label: {image}")
            duplicate = re.search(r"__dup\d+$", image.stem)
            if duplicate is not None:
                original = images_dir / f"{image.stem[: duplicate.start()]}{image.suffix}"
                original_label = labels_dir / f"{original.stem}.txt"
                if (
                    not original.is_file()
                    or not original_label.is_file()
                    or _sha256(image) != _sha256(original)
                    or label.read_bytes() != original_label.read_bytes()
                ):
                    raise ValueError(f"Historical oversample differs from its source: {image}")
                continue
            if image.stem in seen_stems:
                raise ValueError(f"Historical real image stem occurs twice: {image.stem}")
            seen_stems.add(image.stem)
            with Image.open(image) as source:
                source.verify()
            digest = _sha256(image)
            if digest in seen_hashes:
                raise ValueError(f"Identical historical image appears more than once: {image}")
            seen_hashes.add(digest)
            source_lines = _parse_yolo(label.read_text(encoding="utf-8"), names, label)
            mapped = tuple(
                f"{class_ids[int(line.split()[0])]} {line.partition(' ')[2]}"
                for line in source_lines
            )
            frames.append(
                LabeledFrame(image, digest, mapped, "historical_unverified", "historical_real")
            )
    if not frames:
        raise ValueError(f"No historical real images found: {directory}")
    return tuple(frames)


def _synthetic_frames(
    directory: Path, split: Literal["train", "val"], classes: tuple[str, ...]
) -> tuple[LabeledFrame, ...]:
    images_dir = directory / split / "images"
    labels_dir = directory / split / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise ValueError(f"Synthetic {split} images and labels are required: {directory}")
    names = _dataset_names(directory)
    unknown = set(names) - set(classes)
    if unknown:
        raise ValueError(f"Synthetic classes absent from training schema: {sorted(unknown)}")
    ids = {index: classes.index(name) for index, name in enumerate(names)}
    images = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in _IMAGE_SUFFIXES)
    stems = [path.stem for path in images]
    if len(stems) != len(set(stems)) or not images:
        raise ValueError(f"Missing or duplicate synthetic image stems: {images_dir}")
    if {path.stem for path in labels_dir.glob("*.txt")} != set(stems):
        raise ValueError(f"Synthetic images and labels differ in {directory / split}")
    result: list[LabeledFrame] = []
    for image in images:
        label = labels_dir / f"{image.stem}.txt"
        source_lines = _parse_yolo(label.read_text(encoding="utf-8"), names, label)
        mapped = tuple(
            f"{ids[int(line.split()[0])]} {line.partition(' ')[2]}" for line in source_lines
        )
        with Image.open(image) as source:
            source.verify()
        result.append(LabeledFrame(image, _sha256(image), mapped, image.stem, "synthetic"))
    return tuple(result)


def _class_counts(frames: Iterable[LabeledFrame], classes: tuple[str, ...]) -> Counter[str]:
    return Counter(classes[int(line.split()[0])] for frame in frames for line in frame.label_lines)


def _summary(plan: DatasetPlan) -> DatasetSummary:
    class_counts = {
        "real_train": _class_counts((*plan.real_train, *plan.historical_train), plan.classes),
        "historical_train": _class_counts(plan.historical_train, plan.classes),
        "real_val": _class_counts(plan.real_val, plan.classes),
        "synthetic_train": _class_counts(plan.synthetic_train, plan.classes),
        "synthetic_val": _class_counts(plan.synthetic_val, plan.classes),
    }
    per_class = {
        name: {split: counts[name] for split, counts in class_counts.items()}
        for name in plan.classes
    }
    return DatasetSummary(
        reviewed_train_unique=len(plan.real_train),
        historical_train_unique=len(plan.historical_train),
        real_train_unique=len(plan.real_train) + len(plan.historical_train),
        real_train_on_disk=(len(plan.real_train) + len(plan.historical_train))
        * plan.oversample_real,
        real_val=len(plan.real_val),
        synthetic_train=len(plan.synthetic_train),
        synthetic_val_diagnostic=len(plan.synthetic_val),
        total_train_on_disk=(len(plan.real_train) + len(plan.historical_train))
        * plan.oversample_real
        + len(plan.synthetic_train),
        validation_images=len(plan.real_val),
        oversample_real=plan.oversample_real,
        per_class=per_class,
        zero_real_classes=tuple(
            name
            for name in plan.classes
            if not per_class[name]["real_train"] and not per_class[name]["real_val"]
        ),
    )


def _write_frame(root: Path, frame: LabeledFrame, split: Split, name: str) -> dict[str, str]:
    images = root / split / "images"
    labels = root / split / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    destination = images / f"{name}{frame.image.suffix.lower()}"
    label_path = labels / f"{name}.txt"
    shutil.copy2(frame.image, destination)
    label_text = "\n".join(frame.label_lines) + ("\n" if frame.label_lines else "")
    label_path.write_text(label_text, encoding="utf-8")
    return {
        "split": split,
        "image": destination.relative_to(root).as_posix(),
        "label": label_path.relative_to(root).as_posix(),
        "source_image": str(frame.image),
        "source_session": frame.session,
        "source": frame.source,
        "image_sha256": frame.image_sha256,
        "label_sha256": hashlib.sha256(label_text.encode()).hexdigest(),
    }


def _write_dataset(root: Path, plan: DatasetPlan, summary: DatasetSummary) -> None:
    records: list[dict[str, str]] = []
    for frame in (*plan.real_train, *plan.historical_train):
        for copy_index in range(plan.oversample_real):
            suffix = "" if copy_index == 0 else f"__dup{copy_index}"
            prefix = "historical" if frame.source == "historical_real" else "real"
            records.append(
                _write_frame(root, frame, "train", f"{prefix}_{frame.image.stem}{suffix}")
            )
    for frame in plan.real_val:
        records.append(_write_frame(root, frame, "val", f"real_{frame.image.stem}"))
    for frame in plan.synthetic_train:
        records.append(_write_frame(root, frame, "train", f"synth_{frame.image.stem}"))
    for frame in plan.synthetic_val:
        records.append(_write_frame(root, frame, "synthetic_val", f"synth_{frame.image.stem}"))
    yaml_data = {
        "train": "train/images",
        "val": "val/images",
        "names": dict(enumerate(plan.classes)),
    }
    (root / "dataset.yaml").write_text(yaml.safe_dump(yaml_data, sort_keys=False), encoding="utf-8")
    (root / "merge_summary.json").write_text(
        json.dumps(asdict(summary), indent=2) + "\n", encoding="utf-8"
    )
    (root / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "schema_sha256": _sha256(_SCHEMA_PATH),
                "input_files": dict(plan.source_hashes),
                "validation_sessions": sorted({frame.session for frame in plan.real_val}),
                "train_sessions": sorted({frame.session for frame in plan.real_train}),
                "historical_real_train_only": bool(plan.historical_train),
                "images": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def prepare_training(
    export_dirs: Sequence[str | Path],
    *,
    images_dir: str | Path,
    manifest_paths: Sequence[str | Path],
    output_dir: str | Path,
    synthetic_dir: str | Path | None = None,
    historical_real_dir: str | Path | None = None,
    val_split: float = 0.15,
    seed: int = 42,
    oversample_real: int = 1,
    trust_export: bool = False,
    dry_run: bool = False,
) -> DatasetSummary:
    """Validate inputs, plan a session split, and publish a fresh dataset.

    A checked X-AnyLabeling sidecar is review evidence. For other reviewed
    exports, callers explicitly assert review with ``trust_export=True``.
    """
    if oversample_real < 1:
        raise ValueError("Real training oversampling must be at least one")
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Dataset output already exists: {output}")
    classes = _classes()
    sources = _read_manifests([Path(path) for path in manifest_paths])
    real, source_hashes = _real_frames(
        [Path(path) for path in export_dirs],
        Path(images_dir),
        sources,
        classes,
        trust_export=trust_export,
    )
    train, val = _split_sessions(real, val_split, seed)
    historical_train: tuple[LabeledFrame, ...] = ()
    if historical_real_dir is not None:
        historical = Path(historical_real_dir)
        historical_train = _historical_frames(historical, classes)
        source_hashes = (
            *source_hashes,
            (str(historical / "dataset.yaml"), _sha256(historical / "dataset.yaml")),
        )
    synthetic_train: tuple[LabeledFrame, ...] = ()
    synthetic_val: tuple[LabeledFrame, ...] = ()
    if synthetic_dir is not None:
        synthetic = Path(synthetic_dir)
        synthetic_train = _synthetic_frames(synthetic, "train", classes)
        synthetic_val = _synthetic_frames(synthetic, "val", classes)
        source_hashes = (
            *source_hashes,
            (str(synthetic / "dataset.yaml"), _sha256(synthetic / "dataset.yaml")),
        )
    training_hashes = {
        frame.image_sha256 for frame in (*train, *historical_train, *synthetic_train)
    }
    if training_hashes & {frame.image_sha256 for frame in val}:
        raise ValueError("Identical image bytes occur in training and validation")
    plan = DatasetPlan(
        train,
        val,
        historical_train,
        synthetic_train,
        synthetic_val,
        classes,
        oversample_real,
        source_hashes,
    )
    summary = _summary(plan)
    if dry_run:
        return summary
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staged = Path(temporary) / "dataset"
        staged.mkdir()
        _write_dataset(staged, plan, summary)
        staged.rename(output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cvat-export",
        action="append",
        required=True,
        help="COCO or YOLO export directory; repeat for multiple exports",
    )
    parser.add_argument(
        "--images-dir", required=True, help="Directory containing the original reviewed images"
    )
    parser.add_argument(
        "--manifest",
        action="append",
        required=True,
        help="Source-session CSV manifest; repeat for multiple batches",
    )
    parser.add_argument(
        "--output", required=True, help="New dataset directory; must not already exist"
    )
    parser.add_argument(
        "--synthetic", help="Explicit synthetic source with dataset.yaml, train, and val splits"
    )
    parser.add_argument(
        "--historical-real",
        help="Prior YOLO dataset; import distinct real_* images from both old splits as training-only",
    )
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--oversample-real", type=int, default=1)
    parser.add_argument(
        "--reviewed-export",
        action="store_true",
        help="Assert manual review when source images have no checked sidecars",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and report without writing files"
    )
    args = parser.parse_args(namespace=_PrepareTrainingArgs())
    summary = prepare_training(
        args.cvat_export,
        images_dir=args.images_dir,
        manifest_paths=args.manifest,
        output_dir=args.output,
        synthetic_dir=args.synthetic,
        historical_real_dir=args.historical_real,
        val_split=args.val_split,
        seed=args.seed,
        oversample_real=args.oversample_real,
        trust_export=args.reviewed_export,
        dry_run=args.dry_run,
    )
    print(
        f"Real train: {summary.real_train_unique} unique "
        f"({summary.reviewed_train_unique} reviewed, {summary.historical_train_unique} historical), "
        f"{summary.real_train_on_disk} on disk; "
        f"real validation: {summary.real_val}; synthetic train: {summary.synthetic_train}; "
        f"synthetic diagnostic: {summary.synthetic_val_diagnostic}"
    )
    print(f"Classes without real boxes: {len(summary.zero_real_classes)}")
    if args.dry_run:
        print("Dry run: no files written")
    else:
        print(f"Dataset written to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
