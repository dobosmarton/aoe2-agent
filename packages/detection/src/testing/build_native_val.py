"""Rebuild a real-image val split at NATIVE capture resolution.

`training_data_vN/val/images` stores real screenshots already downscaled to
1280px wide, but the agent feeds the detector whatever `capture_screenshot()`
grabbed — 3024x1672 on the dev Mac, per the run logs. Any mode whose behaviour
depends on object *scale* (`detect_fast_multi`, SAHI) therefore measures a
situation the agent never encounters: objects appear ~2.4x larger in the val
set than in production, which flatters the result.

That gap is not hypothetical. Scored against the downscaled split, two-pass
detection loses to single-pass by 0.02 F1; scored at native resolution the same
comparison loses by 0.07, because the centre-crop pass overshoots the training
scale. The mismatch was invisible until the split matched the deployment.

This script pairs each `real_*` val image back to its full-resolution original
in `real_screenshots/raw/` and copies the label file across unchanged — YOLO
labels are normalised, so they transfer between resolutions with no rescaling.

Usage:
    uv run python packages/detection/src/testing/build_native_val.py \\
        --data packages/detection/src/training_data_v9_slim \\
        --raw packages/detection/src/real_screenshots/raw \\
        --out /tmp/native_val

    uv run python packages/detection/src/testing/evaluate_real.py \\
        -m packages/detection/src/inference/models/aoe2_yolo_v9.onnx \\
        -d /tmp/native_val --mode detect_fast --imgsz 1280 --split real
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

# evaluate_real.py selects the real split by filename prefix; keep it identical.
_REAL_PREFIX = "real_"
# generate_training_data.py appends this when a real image is oversampled.
_DUP_MARKER = "__dup"
_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def _raw_index(raw_dir: Path) -> dict[str, Path]:
    """Map every raw screenshot's stem to its path."""
    return {path.stem: path for suffix in _IMAGE_SUFFIXES for path in raw_dir.rglob(f"*{suffix}")}


def _original_stem(val_image: Path) -> str:
    """Recover the raw screenshot stem from a prefixed, possibly duplicated name."""
    name = val_image.name.removeprefix(_REAL_PREFIX)
    return Path(name.split(_DUP_MARKER)[0]).stem


def build(data_dir: Path, raw_dir: Path, out_dir: Path) -> int:
    """Write a val/ split of native-resolution images plus their labels."""
    out_images = out_dir / "val" / "images"
    out_labels = out_dir / "val" / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    raws = _raw_index(raw_dir)
    written = 0
    missing: list[str] = []

    for val_image in sorted((data_dir / "val" / "images").glob(f"{_REAL_PREFIX}*")):
        stem = _original_stem(val_image)
        raw_path = raws.get(stem)
        label = data_dir / "val" / "labels" / f"{val_image.stem}.txt"
        if raw_path is None or not label.exists():
            missing.append(val_image.name)
            continue
        target = f"{_REAL_PREFIX}{stem}"
        shutil.copy(raw_path, out_images / f"{target}{raw_path.suffix}")
        shutil.copy(label, out_labels / f"{target}.txt")
        written += 1

    if missing:
        print(f"Skipped {len(missing)} val images with no raw original or label:")
        for name in missing[:5]:
            print(f"  {name}")
    return written


class _BuildNativeValArgs(argparse.Namespace):
    data: str
    raw: str
    out: str


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", "-d", required=True, help="training_data_vN dir")
    parser.add_argument("--raw", "-r", required=True, help="real_screenshots/raw dir")
    parser.add_argument("--out", "-o", required=True, help="Destination dataset dir")
    args = parser.parse_args(namespace=_BuildNativeValArgs())

    written = build(Path(args.data), Path(args.raw), Path(args.out))
    if not written:
        raise SystemExit("No images written — check --data and --raw paths")
    print(f"Wrote {written} native-resolution val images to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
