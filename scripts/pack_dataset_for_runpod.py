"""Pack a YOLO dataset into a slim, upload-ready copy for cloud GPU training.

Images are downscaled to a max long-side (default 1280px) and re-encoded as
JPEG; label `.txt` files and `dataset.yaml` are copied verbatim. This is lossless
for training at `imgsz<=640`: ultralytics resizes every image to `imgsz` on load,
and YOLO labels are normalized (0..1), so shrinking the pixels never invalidates a
box. Typical effect: ~19 GB -> ~1 GB, turning an hours-long upload into minutes.

    uv run python scripts/pack_dataset_for_runpod.py \
        --src packages/detection/src/training_data_v8 \
        --out packages/detection/src/training_data_v8_slim
"""

from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

_SPLITS = ("train", "val")
_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".webp"})


class _Args(argparse.Namespace):
    src: Path
    out: Path
    max_side: int
    quality: int


def _resize_one(source: Path, target: Path, max_side: int, quality: int) -> int:
    """Resize one image to fit `max_side` and save as JPEG. Returns output bytes."""
    with Image.open(source) as raw:
        image = raw.convert("RGB")
        longest = max(image.size)
        if longest > max_side:
            scale = max_side / longest
            new_size = (round(image.width * scale), round(image.height * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        image.save(target, format="JPEG", quality=quality)
    return target.stat().st_size


def _pack_split(src_split: Path, out_split: Path, max_side: int, quality: int) -> tuple[int, int]:
    """Pack one split's images (resized) + labels (copied). Returns (count, bytes)."""
    out_images = out_split / "images"
    out_images.mkdir(parents=True, exist_ok=True)
    jobs = [
        (path, out_images / f"{path.stem}.jpg")
        for path in sorted((src_split / "images").iterdir())
        if path.suffix.lower() in _IMAGE_SUFFIXES
    ]
    with ThreadPoolExecutor() as pool:
        sizes = list(pool.map(lambda job: _resize_one(job[0], job[1], max_side, quality), jobs))

    src_labels = src_split / "labels"
    if src_labels.is_dir():
        shutil.copytree(src_labels, out_split / "labels", dirs_exist_ok=True)
    return len(jobs), sum(sizes)


def _parse_args() -> _Args:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="Source dataset root")
    parser.add_argument("--out", type=Path, required=True, help="Destination (slim) dataset root")
    parser.add_argument("--max-side", type=int, default=1280, help="Max image long-side (px)")
    parser.add_argument("--quality", type=int, default=90, help="JPEG quality (1-100)")
    return parser.parse_args(namespace=_Args())


def main() -> None:
    args = _parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.src / "dataset.yaml", args.out / "dataset.yaml")

    total_count = 0
    total_bytes = 0
    for split in _SPLITS:
        count, size = _pack_split(args.src / split, args.out / split, args.max_side, args.quality)
        total_count += count
        total_bytes += size
        print(f"  {split}: {count} images -> {size / 1e6:.0f} MB")

    print(f"Done: {total_count} images, {total_bytes / 1e9:.2f} GB total at {args.out}")


if __name__ == "__main__":
    main()
