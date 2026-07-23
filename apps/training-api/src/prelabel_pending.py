"""Model-assisted prelabeling: seed `model`/`pending` boxes for human review.

Runs the current detector over every real screenshot that has no approved box
yet and writes its predictions as *pending* annotations. A reviewer then approves,
corrects, or rejects them through the PATCH/DELETE routes; approving turns a box
into training data. Re-running is safe — `set_model_prelabels` only ever replaces
an image's unreviewed model boxes, so a reviewer's decisions are never clobbered.

The detector is injected behind the `Detector` protocol so the conversion logic is
tested against a fake and the heavy ONNX runtime is imported only in `main`.

Run directly to prelabel the default work queue:
    uv run --package training-api python -m training_api.prelabel_pending --conf 0.25
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from PIL import Image

from .classes import ClassCatalog
from .config import TrackerConfig, load_config
from .db import connect, init_schema
from .domain import Annotation
from .geometry import BBox
from .repository import SqliteTrackerRepository

if TYPE_CHECKING:
    from collections.abc import Sequence

    from core import DetectedEntity

logger = logging.getLogger(__name__)

_DEFAULT_CONF = 0.25
# Single full-image pass, not SAHI tiling: the served weights are full-frame
# trained, and SAHI's tile scale mismatches that (see project detection eval
# findings). Override per-run if a SAHI-aware model is ever served.
_DEFAULT_IMGSZ = 1280
_DEFAULT_USE_SAHI = False


class Detector(Protocol):
    """The one detector capability the prelabeler needs (satisfied structurally
    by `detection.get_detector(...)`; a fake supplies it in tests)."""

    def detect(self, screenshot: Image.Image) -> list[DetectedEntity]: ...


@dataclass(frozen=True, slots=True)
class Converted:
    annotations: tuple[Annotation, ...]
    skipped_low_conf: int
    skipped_unknown_class: int


@dataclass(frozen=True, slots=True)
class PrelabelReport:
    images_processed: int
    boxes_written: int
    skipped_low_conf: int
    skipped_unknown_class: int


def detections_to_annotations(
    image_id: int,
    detections: Sequence[DetectedEntity],
    catalog: ClassCatalog,
    min_conf: float,
) -> Converted:
    """Map detector output to pending model annotations (pure).

    Two boxes are dropped rather than written: those below `min_conf`, and those
    whose class name isn't in the schema (an open-vocab detector can emit labels
    the 60-class training set doesn't cover). Each is counted so the caller can
    surface it — a silent drop reads as "nothing detected".
    """
    annotations: list[Annotation] = []
    skipped_low_conf = 0
    skipped_unknown_class = 0
    for det in detections:
        if det.confidence < min_conf:
            skipped_low_conf += 1
            continue
        class_id = catalog.id_of(det.class_name)
        if class_id is None:
            skipped_unknown_class += 1
            continue
        x1, y1, x2, y2 = det.bbox
        annotations.append(
            Annotation(
                id=None,
                image_id=image_id,
                class_id=class_id,
                geometry=BBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1),
                source="model",
                status="pending",
            )
        )
    return Converted(tuple(annotations), skipped_low_conf, skipped_unknown_class)


def run(
    config: TrackerConfig,
    detector: Detector,
    *,
    min_conf: float = _DEFAULT_CONF,
    limit: int | None = None,
) -> PrelabelReport:
    """Prelabel the work queue and return a summary."""
    catalog = ClassCatalog(config.classes_yaml)
    conn = connect(config.db_path)
    try:
        init_schema(conn)
        repo = SqliteTrackerRepository(conn, catalog)
        image_ids = repo.real_image_ids_needing_prelabel(limit)
        processed = written = low = unknown = 0
        for image_id in image_ids:
            detail = repo.get_image(image_id)
            if detail is None:
                continue
            image = _load_rgb(detail.record.path)
            if image is None:
                continue
            converted = detections_to_annotations(
                image_id, detector.detect(image), catalog, min_conf
            )
            repo.set_model_prelabels(image_id, converted.annotations)
            processed += 1
            written += len(converted.annotations)
            low += converted.skipped_low_conf
            unknown += converted.skipped_unknown_class
    finally:
        conn.close()
    report = PrelabelReport(processed, written, low, unknown)
    logger.info("prelabel complete: %s", report)
    return report


def _load_rgb(path: str) -> Image.Image | None:
    try:
        with Image.open(path) as handle:
            return handle.convert("RGB")
    except (FileNotFoundError, OSError) as exc:
        logger.warning("skipping unreadable image %s: %s", path, exc)
        return None


def _build_detector(model_name: str | None, imgsz: int, use_sahi: bool) -> Detector:
    # Imported here so the module (and its tests) never pull in onnxruntime.
    from detection import get_detector

    return get_detector(model_name=model_name, imgsz=imgsz, use_sahi=use_sahi)


class _PrelabelArgs(argparse.Namespace):
    conf: float
    limit: int | None
    model: str | None
    imgsz: int
    sahi: bool


def _parse_args(argv: Sequence[str] | None) -> _PrelabelArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conf", type=float, default=_DEFAULT_CONF, help="min confidence to keep")
    parser.add_argument("--limit", type=int, default=None, help="cap images this run")
    parser.add_argument("--model", default=None, help="bundled model name (default: newest)")
    parser.add_argument("--imgsz", type=int, default=_DEFAULT_IMGSZ, help="inference resolution")
    parser.add_argument(
        "--sahi",
        action="store_true",
        default=_DEFAULT_USE_SAHI,
        help="enable SAHI tiling (only for SAHI-aware weights)",
    )
    return parser.parse_args(argv, namespace=_PrelabelArgs())


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)
    config = load_config(os.environ)
    detector = _build_detector(args.model, args.imgsz, args.sahi)
    report = run(config, detector, min_conf=args.conf, limit=args.limit)
    print(
        f"Prelabeled {report.images_processed} images, wrote {report.boxes_written} pending "
        f"boxes ({report.skipped_low_conf} below conf, "
        f"{report.skipped_unknown_class} unknown-class dropped)."
    )


if __name__ == "__main__":
    main()
