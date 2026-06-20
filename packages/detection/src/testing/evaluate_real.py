#!/usr/bin/env python3
"""Score a detection model on REAL held-out screenshots against ground truth.

Unlike ``test_real_detection.py`` (which only counts detections), this computes
per-class precision / recall / F1 by greedy IoU matching against the YOLO label
files in a ``training_data_vN/val/`` split — and reports REAL images separately
from SYNTHETIC ones, because a blended mAP over a ~95%-synthetic val set hides
real-world performance.

The real metric of record. Run it after every retrain.

Usage:
    # real-only, single-pass at the model's training resolution (the realistic number)
    uv run --project packages/detection python -m detection.testing.evaluate_real \
        --model detection/inference/models/aoe2_yolo_v6.onnx \
        --data  detection/training_data_v7 --mode detect_fast --imgsz 640

    # also score synthetic, and sweep confidence to recommend per-class thresholds
    ... --split both --conf-sweep

Notes:
    * Real images are identified by the ``real_`` filename prefix that
      ``prepare_training.py`` applies.
    * Per-class thresholds from ``thresholds.py`` are intentionally disabled here
      so a single uniform ``--conf`` makes classes comparable.
    * ``--conf-sweep`` runs inference ONCE at a low floor, then re-scores at many
      thresholds by filtering — so a sweep costs one detection pass, not N.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple, NotRequired, TypedDict, get_args

from detection.inference.detector import EntityDetector
from detection.inference.postprocess import iou
from detection.labeling.class_mapping import load_classes_yaml
from PIL import Image

if TYPE_CHECKING:
    from collections.abc import Callable

    from core import DetectedEntity

# --- Types ------------------------------------------------------------------
BBox = tuple[float, float, float, float]
DetectMode = Literal["detect", "detect_fast", "detect_fast_multi"]
Split = Literal["real", "synth", "both"]


@dataclass(frozen=True, slots=True)
class Prediction:
    """A model detection mapped into class-id space."""

    class_id: int
    box: BBox
    confidence: float


@dataclass(frozen=True, slots=True)
class GroundTruth:
    """A ground-truth box parsed from a YOLO label file."""

    class_id: int
    box: BBox


Sample = tuple[list[Prediction], list[GroundTruth]]


@dataclass(slots=True)
class Tally:
    """Mutable per-class match accumulator (true/false positives, misses, GT)."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    gt: int = 0


class PRF(NamedTuple):
    """Precision / recall / F1 triple."""

    precision: float
    recall: float
    f1: float


class MicroSummary(TypedDict):
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


class ClassCounts(TypedDict):
    gt: int
    tp: int
    fp: int
    fn: int


class ThresholdRec(TypedDict):
    threshold: float
    f1: float


class GroupSummary(TypedDict):
    n_images: int
    micro: MicroSummary
    per_class: dict[str, ClassCounts]
    recommended_thresholds: NotRequired[dict[str, ThresholdRec]]


class EvalSummary(TypedDict):
    model: str
    data: str
    mode: str
    imgsz: int
    conf: float
    iou: float
    real: NotRequired[GroupSummary]
    synth: NotRequired[GroupSummary]


_SWEEP_FLOOR = 0.05
_SWEEP_GRID = [round(0.05 + 0.05 * i, 2) for i in range(12)]  # 0.05 .. 0.60
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


# --- Scoring ----------------------------------------------------------------
def _load_gt(label_path: Path, width: int, height: int) -> list[GroundTruth]:
    """Parse a YOLO label file into ground-truth pixel boxes."""
    if not label_path.exists():
        return []
    out: list[GroundTruth] = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        box: BBox = (
            (cx - bw / 2) * width,
            (cy - bh / 2) * height,
            (cx + bw / 2) * width,
            (cy + bh / 2) * height,
        )
        out.append(GroundTruth(class_id=class_id, box=box))
    return out


def _score(samples: list[Sample], conf: float, iou_thr: float) -> dict[int, Tally]:
    """Greedy IoU match across all samples at a confidence threshold."""
    per_class: defaultdict[int, Tally] = defaultdict(Tally)
    for preds, gts in samples:
        for gt in gts:
            per_class[gt.class_id].gt += 1
        used = [False] * len(gts)
        for pred in sorted((p for p in preds if p.confidence >= conf), key=lambda p: -p.confidence):
            best_idx = -1
            best_iou = iou_thr
            for gi, gt in enumerate(gts):
                if used[gi] or gt.class_id != pred.class_id:
                    continue
                overlap = iou(pred.box, gt.box)
                if overlap >= best_iou:
                    best_idx, best_iou = gi, overlap
            if best_idx >= 0:
                used[best_idx] = True
                per_class[pred.class_id].tp += 1
            else:
                per_class[pred.class_id].fp += 1
        for gi, gt in enumerate(gts):
            if not used[gi]:
                per_class[gt.class_id].fn += 1
    return dict(per_class)


def _micro(per_class: dict[int, Tally]) -> Tally:
    """Sum per-class tallies into a single micro-average tally."""
    total = Tally()
    for t in per_class.values():
        total.tp += t.tp
        total.fp += t.fp
        total.fn += t.fn
        total.gt += t.gt
    return total


def _prf(t: Tally) -> PRF:
    """Precision / recall / F1 from a tally (0.0 where undefined)."""
    precision = t.tp / (t.tp + t.fp) if (t.tp + t.fp) else 0.0
    recall = t.tp / (t.tp + t.fn) if (t.tp + t.fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return PRF(precision, recall, f1)


def _class_ids(samples: list[Sample]) -> set[int]:
    """Every class id appearing in predictions or ground truth."""
    ids: set[int] = set()
    for preds, gts in samples:
        ids.update(p.class_id for p in preds)
        ids.update(g.class_id for g in gts)
    return ids


def _sweep_thresholds(
    samples: list[Sample], iou_thr: float, id2name: dict[int, str]
) -> dict[str, ThresholdRec]:
    """Best-F1 confidence threshold per class (one re-score per grid point)."""
    scored = {threshold: _score(samples, threshold, iou_thr) for threshold in _SWEEP_GRID}
    best: dict[str, ThresholdRec] = {}
    for class_id in _class_ids(samples):
        best_threshold = _SWEEP_GRID[0]
        best_f1 = -1.0
        for threshold in _SWEEP_GRID:
            tally = scored[threshold].get(class_id)
            if tally is None:
                continue
            f1 = _prf(tally).f1
            if f1 > best_f1:
                best_f1, best_threshold = f1, threshold
        if best_f1 > 0:
            name = id2name.get(class_id, str(class_id))
            best[name] = ThresholdRec(threshold=best_threshold, f1=round(best_f1, 3))
    return best


# --- Inference --------------------------------------------------------------
def _build_detector(model_path: str, floor: float, imgsz: int) -> EntityDetector:
    """Detector with per-class thresholds disabled so a uniform conf applies."""
    det = EntityDetector(model_path=model_path, confidence_threshold=floor, imgsz=imgsz)
    det.class_thresholds = {}  # fall back to the uniform confidence_threshold
    if det.use_mock:
        raise SystemExit(f"Model failed to load (mock mode): {model_path}")
    return det


def _collect(
    det: EntityDetector,
    mode: DetectMode,
    images: list[Path],
    labels_dir: Path,
    name2id: dict[str, int],
) -> list[Sample]:
    """Run inference once per image; pair predictions with ground truth."""
    dispatch: dict[DetectMode, Callable[[bytes | Image.Image], list[DetectedEntity]]] = {
        "detect": det.detect,
        "detect_fast": det.detect_fast,
        "detect_fast_multi": det.detect_fast_multi,
    }
    run = dispatch[mode]
    samples: list[Sample] = []
    for index, img_path in enumerate(images):
        image = Image.open(img_path).convert("RGB")
        width, height = image.size
        preds: list[Prediction] = []
        for entity in run(image):
            class_id = name2id.get(entity.class_name)
            if class_id is not None:
                preds.append(
                    Prediction(class_id=class_id, box=entity.bbox, confidence=entity.confidence)
                )
        gts = _load_gt(labels_dir / f"{img_path.stem}.txt", width, height)
        samples.append((preds, gts))
        print(f"  [{index + 1}/{len(images)}] {img_path.name}: {len(preds)} preds / {len(gts)} gt")
    return samples


# --- Reporting --------------------------------------------------------------
def _print_table(title: str, rows: list[tuple[int, Tally]], id2name: dict[int, str]) -> None:
    """Print a per-class precision/recall/F1 table."""
    header = (
        f"\n{'id':>3} {'class':<16}{'gt':>5}{'tp':>5}{'fp':>5}{'fn':>5}"
        f"{'rec':>6}{'prec':>6}{'F1':>6}  [{title}]"
    )
    print(header)
    print("-" * 72)
    for class_id, tally in rows:
        m = _prf(tally)
        print(
            f"{class_id:>3} {id2name.get(class_id, '?'):<16}{tally.gt:>5}{tally.tp:>5}"
            f"{tally.fp:>5}{tally.fn:>5}{m.recall:>6.2f}{m.precision:>6.2f}{m.f1:>6.2f}"
        )


def _evaluate_group(
    det: EntityDetector,
    mode: DetectMode,
    label: Split,
    images: list[Path],
    labels_dir: Path,
    name2id: dict[str, int],
    id2name: dict[int, str],
    conf: float,
    iou_thr: float,
    sweep: bool,
) -> GroupSummary:
    """Score one image group (real or synthetic) and emit its summary."""
    print(f"\n=== Collecting predictions ({label}, {len(images)} images) ===")
    samples = _collect(det, mode, images, labels_dir, name2id)
    per_class = _score(samples, conf, iou_thr)
    micro = _micro(per_class)
    metrics = _prf(micro)
    rows = sorted(per_class.items(), key=lambda item: -item[1].gt)
    _print_table(f"{label} @conf{conf}", rows, id2name)
    print(
        f"\n[{label}] MICRO @conf{conf} iou{iou_thr}: TP={micro.tp} FP={micro.fp} FN={micro.fn}  "
        f"precision={metrics.precision:.3f} recall={metrics.recall:.3f} F1={metrics.f1:.3f}"
    )
    group: GroupSummary = {
        "n_images": len(images),
        "micro": {
            "tp": micro.tp,
            "fp": micro.fp,
            "fn": micro.fn,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
        },
        "per_class": {
            id2name.get(cid, str(cid)): {"gt": t.gt, "tp": t.tp, "fp": t.fp, "fn": t.fn}
            for cid, t in rows
        },
    }
    if sweep:
        recommended = _sweep_thresholds(samples, iou_thr, id2name)
        group["recommended_thresholds"] = recommended
        print(f"\n[{label}] recommended per-class thresholds (best F1):")
        for name, rec in sorted(recommended.items()):
            print(f"  {name:<16} conf={rec['threshold']:.2f}  F1={rec['f1']:.3f}")
    return group


class _EvaluateRealArgs(argparse.Namespace):
    model: str
    data: str
    mode: DetectMode
    imgsz: int
    conf: float
    iou: float
    split: Split
    conf_sweep: bool
    output: str | None


def _resolve_data_dir(repo: Path, data_arg: str) -> Path:
    """Resolve a training_data_vN dir from a repo-relative or src-relative arg."""
    data_dir = Path(data_arg) if Path(data_arg).exists() else repo / data_arg
    if not (data_dir / "val" / "images").exists():
        data_dir = repo / "packages" / "detection" / "src" / data_arg
    return data_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Per-class real-vs-synthetic detection eval")
    parser.add_argument(
        "--model", "-m", required=True, help="Path to .onnx/.pt model (repo-relative ok)"
    )
    parser.add_argument(
        "--data", "-d", required=True, help="training_data_vN dir (uses its val/ split)"
    )
    parser.add_argument(
        "--mode", choices=get_args(DetectMode), default="detect", help="Detector method"
    )
    parser.add_argument(
        "--imgsz", type=int, default=1280, help="Single-pass resolution (match training res!)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25, help="Confidence threshold for the report"
    )
    parser.add_argument("--iou", type=float, default=0.5, help="IoU match threshold")
    parser.add_argument(
        "--split", choices=get_args(Split), default="real", help="Which images to score"
    )
    parser.add_argument("--conf-sweep", action="store_true", help="Recommend per-class thresholds")
    parser.add_argument(
        "--output", "-o", default=None, help="Summary JSON (default: <data>/eval_real_summary.json)"
    )
    args = parser.parse_args(namespace=_EvaluateRealArgs())

    repo = Path(__file__).resolve().parents[3]  # agent/
    model_path = args.model if Path(args.model).exists() else str(repo / args.model)
    data_dir = _resolve_data_dir(repo, args.data)
    val_images = data_dir / "val" / "images"
    val_labels = data_dir / "val" / "labels"
    if not val_images.exists():
        raise SystemExit(f"val/images not found under {data_dir}")

    id2name = load_classes_yaml()
    name2id = {name: cid for cid, name in id2name.items()}

    all_imgs = sorted(p for p in val_images.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES)
    real_imgs = [p for p in all_imgs if p.name.startswith("real_")]
    synth_imgs = [p for p in all_imgs if not p.name.startswith("real_")]
    print(f"val images: {len(all_imgs)} (real={len(real_imgs)} synth={len(synth_imgs)})")

    floor = _SWEEP_FLOOR if args.conf_sweep else args.conf
    det = _build_detector(model_path, floor, args.imgsz)
    print(
        f"backend={det.backend} mode={args.mode} static_hw={det.onnx_input_hw} "
        f"batch_dynamic={det.onnx_batch_dynamic} floor={floor} iou={args.iou}"
    )

    summary: EvalSummary = {
        "model": model_path,
        "data": str(data_dir),
        "mode": args.mode,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
    }
    groups: list[tuple[Split, list[Path]]] = []
    if args.split in ("real", "both"):
        groups.append(("real", real_imgs))
    if args.split in ("synth", "both"):
        groups.append(("synth", synth_imgs))

    for label, imgs in groups:
        if not imgs:
            print(f"\n[{label}] no images — skipping")
            continue
        group = _evaluate_group(
            det,
            args.mode,
            label,
            imgs,
            val_labels,
            name2id,
            id2name,
            args.conf,
            args.iou,
            args.conf_sweep,
        )
        if label == "real":
            summary["real"] = group
        else:
            summary["synth"] = group

    out = Path(args.output) if args.output else data_dir / "eval_real_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
