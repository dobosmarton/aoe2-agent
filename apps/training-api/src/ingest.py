"""Seed the tracker DB from on-disk data (idempotent).

Pure parsers turn the raw-screenshot dir and a YOLO dataset into domain records;
a single writer persists them. Re-running upserts by path and rebuilds the dataset
version, so it never duplicates rows.

Run directly to seed the default dataset version:
    uv run --package training-api python -m training_api.ingest
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

from .classes import ClassCatalog
from .config import TrackerConfig, load_config
from .db import connect, init_schema
from .domain import Annotation, ClassId, ImageRecord, Split
from .geometry import bbox_from_yolo
from .repository import SqliteTrackerRepository

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})
_DUP_SUFFIX = re.compile(r"__dup\d+$")
_DEFAULT_DATASET_DIRNAME = "training_data_v9_slim"
_DEFAULT_VERSION_NAME = "v9"


@dataclass(frozen=True, slots=True)
class YoloBox:
    class_id: ClassId
    cx: float
    cy: float
    w: float
    h: float


@dataclass(slots=True)
class DatasetLabels:
    """Mutable accumulator: real labels keyed by canonical raw stem, plus
    synthetic aggregates. Built up during a single dataset walk."""

    real: dict[str, tuple[Split, tuple[YoloBox, ...]]] = field(default_factory=dict)
    synth_class_counts: dict[ClassId, int] = field(default_factory=dict)
    synth_image_count: int = 0


@dataclass(frozen=True, slots=True)
class IngestReport:
    raw_images: int
    labeled_images: int
    unmatched_dataset_stems: int
    synth_images: int
    version_name: str


# ---------------------------------------------------------------------------
# Pure parsers
# ---------------------------------------------------------------------------


def canonical_real_stem(dataset_image_stem: str) -> str | None:
    """Map a dataset image stem (`real_<stem>[__dupN]`) back to its raw stem."""
    if not dataset_image_stem.startswith("real_"):
        return None
    core = dataset_image_stem[len("real_") :]
    return _DUP_SUFFIX.sub("", core)


def parse_yolo_line(line: str) -> YoloBox | None:
    parts = line.split()
    if len(parts) != 5:
        return None
    class_id = int(parts[0])
    cx, cy, w, h = (float(p) for p in parts[1:])
    return YoloBox(class_id=class_id, cx=cx, cy=cy, w=w, h=h)


def parse_label_file(path: Path) -> tuple[YoloBox, ...]:
    boxes = (parse_yolo_line(line) for line in path.read_text().splitlines() if line.strip())
    return tuple(box for box in boxes if box is not None)


def collect_dataset_labels(dataset_dir: Path) -> DatasetLabels:
    """Walk a YOLO dataset's train/val splits, separating real vs synthetic."""
    labels = DatasetLabels()
    for split in ("train", "val"):
        labels_dir = dataset_dir / split / "labels"
        if not labels_dir.is_dir():
            continue
        for label_path in sorted(labels_dir.glob("*.txt")):
            _accumulate_label(labels, label_path, split)
    return labels


def _accumulate_label(labels: DatasetLabels, label_path: Path, split: Split) -> None:
    stem = label_path.stem
    if stem.startswith("img_"):
        boxes = parse_label_file(label_path)
        for box in boxes:
            labels.synth_class_counts[box.class_id] = (
                labels.synth_class_counts.get(box.class_id, 0) + 1
            )
        labels.synth_image_count += 1
        return

    canonical = canonical_real_stem(stem)
    if canonical is None or canonical in labels.real:
        return  # first occurrence of a screenshot wins; dup copies are identical
    labels.real[canonical] = (split, parse_label_file(label_path))


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_raw_images(raw_dir: Path) -> Iterator[Path]:
    for path in sorted(raw_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
            yield path


def scan_raw_images(raw_dir: Path) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for path in iter_raw_images(raw_dir):
        with Image.open(path) as image:
            width, height = image.size
        records.append(
            ImageRecord(
                id=None,
                path=str(path),
                source="real",
                sha256=sha256_of(path),
                width=width,
                height=height,
                capture_meta=None,
            )
        )
    return records


def boxes_to_annotations(
    image_id: int, boxes: tuple[YoloBox, ...], width: int, height: int
) -> list[Annotation]:
    """Real dataset labels are CVAT-corrected, so mark them human/approved."""
    return [
        Annotation(
            id=None,
            image_id=image_id,
            class_id=box.class_id,
            geometry=bbox_from_yolo(box.cx, box.cy, box.w, box.h, width, height),
            source="human",
            status="approved",
        )
        for box in boxes
    ]


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def seed(config: TrackerConfig, *, dataset_dirname: str, version_name: str) -> IngestReport:
    catalog = ClassCatalog(config.classes_yaml)
    conn = connect(config.db_path)
    init_schema(conn)
    repo = SqliteTrackerRepository(conn, catalog)

    raw_records = scan_raw_images(config.raw_images_dir)
    stem_to_id: dict[str, int] = {}
    for record in raw_records:
        image_id = repo.upsert_image(record)
        stem_to_id[Path(record.path).stem] = image_id
    width_by_id = {stem_to_id[Path(r.path).stem]: (r.width, r.height) for r in raw_records}

    labels = collect_dataset_labels(config.dataset_root / dataset_dirname)
    version_id = repo.reset_dataset_version(
        version_name,
        notes=f"Seeded from {dataset_dirname}",
        val_split=None,
        synth_image_count=labels.synth_image_count,
        synth_class_counts=labels.synth_class_counts,
    )

    labeled = 0
    unmatched = 0
    for canonical, (split, boxes) in labels.real.items():
        image_id = stem_to_id.get(canonical)
        if image_id is None:
            unmatched += 1
            continue
        width, height = width_by_id[image_id]
        repo.replace_annotations(image_id, boxes_to_annotations(image_id, boxes, width, height))
        repo.add_dataset_image(version_id, image_id, split)
        labeled += 1

    report = IngestReport(
        raw_images=len(raw_records),
        labeled_images=labeled,
        unmatched_dataset_stems=unmatched,
        synth_images=labels.synth_image_count,
        version_name=version_name,
    )
    logger.info("ingest complete: %s", report)
    conn.close()
    return report


def main() -> None:
    import os

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(os.environ)
    report = seed(
        config,
        dataset_dirname=os.environ.get("TRAINING_API_SEED_DATASET", _DEFAULT_DATASET_DIRNAME),
        version_name=os.environ.get("TRAINING_API_SEED_VERSION", _DEFAULT_VERSION_NAME),
    )
    print(
        f"Seeded {report.raw_images} raw images, {report.labeled_images} labeled, "
        f"{report.synth_images} synthetic in '{report.version_name}' "
        f"({report.unmatched_dataset_stems} dataset stems unmatched)."
    )


if __name__ == "__main__":
    main()
