#!/usr/bin/env python
"""Semi-automatic resource-bar calibration for a new capture resolution.

Given a few game screenshots at the target resolution, this:
  1. runs RapidOCR over the top band to detect the resource-bar text,
  2. classifies detections (4 resource numbers, population, age),
  3. aggregates boxes across frames and writes
     `resource_ocr_assets/calibration.<W>x<H>.yaml`,
  4. saves an overlay preview (`calibration_preview_<W>x<H>.png`) to verify by eye,
  5. (optional) extracts digit templates from one frame with known values.

SEMI-automatic on purpose: always eyeball the overlay and tweak the YAML boxes if
a field is off. RapidOCR is the production backend, so its detection boxes map
directly to what `read_resource_bar` will segment.

Usage:
  .venv/bin/python scripts/calibrate_resource_bar.py --screenshots 'logs/screenshots/*.jpg'

  # optional digit templates (lone-digit fallback) from one labeled frame,
  # values in on-screen order: wood food gold stone population
  .venv/bin/python scripts/calibrate_resource_bar.py --screenshots '...' \
      --template-frame shot.jpg --template-values "6133 7725 2590 2314 180/200"
"""

from __future__ import annotations

import argparse
import glob
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw

try:
    # The bar-localization logic now lives in the OCR module so the live agent
    # (autodetect_calibration) and this offline CLI share one implementation.
    from gameplay_agent.resource_ocr import (
        Box,
        _assign,
        _binarize_digits,
        _column_centers,
        _detect,
        _extract,
        _rapidocr_engine,
    )
except ImportError as exc:  # pragma: no cover
    sys.stderr.write(f"run with the project venv (need gameplay_agent + cv2): {exc}\n")
    raise SystemExit(2) from exc


def _aggregate(per_frame: list[dict[str, Box]], pad: int) -> dict[str, list[int]]:
    """Per field: median x0/y, max x1 (widest number seen), + pad.

    Boxes are already correctly assigned by x-column, so max x1 safely fits the
    widest value across frames without merge/shift artifacts. Fields seen in
    < half the frames are dropped as unstable. (Calibrate from frames spanning your
    game's value range; the self-check readout flags any box that clips.)
    """
    keys = ["wood", "food", "gold", "stone", "population", "age"]
    fields: dict[str, list[int]] = {}
    for key in keys:
        boxes = [f[key] for f in per_frame if key in f]
        if len(boxes) < max(1, len(per_frame) // 2):
            continue
        x0 = int(statistics.median(b[0] for b in boxes))
        y0 = int(statistics.median(b[1] for b in boxes)) - pad
        x1 = max(b[2] for b in boxes) + pad
        y1 = int(statistics.median(b[3] for b in boxes)) + pad
        fields[key] = [max(0, x0), max(0, y0), x1, y1]
    return fields


def _save_overlay(image_path: Path, fields: dict[str, list[int]], out: Path) -> None:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for name, (x0, y0, x1, y1) in fields.items():
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=3)
        draw.text((x0, max(0, y0 - 16)), name, fill=(255, 255, 0))
    # crop to the bar neighbourhood so the preview is legible
    bottom = max(y1 for *_, y1 in fields.values()) + 40 if fields else img.height
    img.crop((0, 0, img.width, min(img.height, bottom))).save(out)


def _extract_templates(
    frame: Path, values: list[str], fields: dict[str, list[int]], tdir: Path
) -> None:
    """Segment a labeled frame's fields into per-digit templates (lone-digit fallback)."""
    order = ["wood", "food", "gold", "stone", "population"]
    gray = np.asarray(Image.open(frame).convert("L"))
    tdir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, bool] = {}
    for name, value in zip(order, values, strict=False):
        if name not in fields:
            print(f"  template: skip {name} (no box)")
            continue
        x0, y0, x1, y1 = fields[name]
        binary = _binarize_digits(gray[y0:y1, x0:x1])
        n, _l, stats, _c = cv2.connectedComponentsWithStats(binary)
        fh = binary.shape[0]
        glyphs = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if h < 0.3 * fh or area < 6:
                continue
            glyphs.append((int(x), binary[y : y + h, x : x + w]))
        glyphs.sort(key=lambda g: g[0])
        if len(glyphs) != len(value):
            print(
                f"  template: {name} segmented {len(glyphs)} ≠ {len(value)} chars — skip, fix box"
            )
            continue
        for ch, (_x, g) in zip(value, glyphs, strict=False):
            key = "slash" if ch == "/" else ch
            if key not in saved:
                Image.fromarray(g).save(tdir / f"{key}.png")
                saved[key] = True
    print(f"  templates saved: {sorted(saved)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--screenshots", required=True, help="glob of game screenshots at the target resolution"
    )
    ap.add_argument(
        "--out-dir", type=Path, default=None, help="assets dir (default: resource_ocr_assets)"
    )
    ap.add_argument(
        "--top-frac", type=float, default=0.15, help="fraction of height to search for the bar"
    )
    ap.add_argument("--pad", type=int, default=4, help="pixels to pad each detected box")
    ap.add_argument("--template-frame", type=Path, default=None)
    ap.add_argument(
        "--template-values", default=None, help='"wood food gold stone population", on-screen order'
    )
    args = ap.parse_args()

    paths = [Path(p) for p in sorted(glob.glob(args.screenshots))]  # noqa: PTH207 (user glob pattern)
    if not paths:
        sys.stderr.write(f"no screenshots matched {args.screenshots!r}\n")
        return 1
    sizes = {Image.open(p).size for p in paths}
    if len(sizes) != 1:
        sys.stderr.write(
            f"screenshots are mixed resolutions {sizes}; calibrate one resolution at a time\n"
        )
        return 1
    (w, h) = next(iter(sizes))
    assets = args.out_dir or (
        Path(__file__).resolve().parent.parent / "apps/agent/src/resource_ocr_assets"
    )

    print(f"Calibrating {w}x{h} from {len(paths)} screenshot(s)...")
    engine = _rapidocr_engine()
    raw = [
        _extract(_detect(engine, np.asarray(Image.open(p).convert("RGB")), args.top_frac))
        for p in paths
    ]

    # Global x-column assignment: a field missing in one frame leaves a gap rather
    # than shifting the others (which corrupts order-based assignment).
    centers = _column_centers([b[0] for main, _, _ in raw for b in main])
    pitch = (centers[-1] - centers[0]) / (len(centers) - 1) if len(centers) >= 2 else 150
    tol = 0.4 * pitch

    per_frame = []
    for (main, pop, age), p in zip(raw, paths, strict=False):
        f: dict[str, Box] = _assign(main, centers, tol)
        if pop is not None:
            f["population"] = pop
        if age is not None:
            f["age"] = age
        per_frame.append(f)
        print(f"  {p.name}: detected {sorted(f)}")

    fields = _aggregate(per_frame, args.pad)
    found = set(fields)
    expected = {"wood", "food", "gold", "stone", "population", "age"}
    print(f"\nstable fields: {sorted(found)}")
    if expected - found:
        print(
            f"⚠️  MISSING {sorted(expected - found)} — add more/cleaner screenshots, "
            f"or add these boxes to the YAML by hand (see the overlay)."
        )

    cal = {"width": w, "height": h, "template_dir": f"templates/{w}x{h}", "fields": fields}
    cal_path = assets / f"calibration.{w}x{h}.yaml"
    cal_path.write_text(yaml.dump(cal, sort_keys=False))
    preview = assets / f"calibration_preview_{w}x{h}.png"
    _save_overlay(paths[0], fields, preview)
    print(f"\nwrote {cal_path}")
    print(f"wrote {preview}  ← EYEBALL THIS: boxes should bound each number/age")

    if args.template_frame and args.template_values:
        print("\nextracting digit templates...")
        _extract_templates(
            args.template_frame, args.template_values.split(), fields, assets / f"templates/{w}x{h}"
        )

    # Self-check: read every input frame with the new calibration so a bad box is
    # obvious (e.g. a spurious leading digit) without needing labeled fixtures.
    from gameplay_agent.resource_ocr import Calibration, read_resource_bar

    calib = Calibration.from_yaml(cal_path)
    print("\nSelf-check — readings on your screenshots (eyeball vs the actual game):")
    for p in paths:
        readings = read_resource_bar(p.read_bytes(), calib, backend="rapidocr")
        print(f"  {p.name}: {readings}")

    print(
        f"\nNext: (1) eyeball {preview.name} and the self-check above; "
        f"nudge any off box in calibration.{w}x{h}.yaml. "
        f"(2) drop the calibration into resource_ocr_assets/ if you used --out-dir."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
