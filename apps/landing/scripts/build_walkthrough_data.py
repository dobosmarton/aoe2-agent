"""Build the data and textures for the home page walkthrough from one real frame.

Runs the served detector and the agent's HUD reader on a captured 3024x1672 game
frame, then writes:

- src/components/walkthrough/turn-data.json: boxes (normalised 0..1, with the
  category colours of apps/agent/src/overlay.py), HUD readings and the HUD field
  rectangles.
- public/walkthrough/: the frame at two widths and the HUD strip. The score panel
  and the player name are blurred.

scripts/capture-walkthrough-stills.ts then renders the stills and the poster.

The output is committed, so the site build needs no Python. Run from the repo root:

    uv run python apps/landing/scripts/build_walkthrough_data.py \
        --frame <frame.jpg> --labels <frame.json> \
        --model packages/detection/src/inference/models/aoe2_yolo_v12_nms736.onnx

Without a model file the reviewed labels in --labels are used instead, and the
JSON records that with "source": "reviewed".
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

LANDING = Path(__file__).resolve().parent.parent
OUT_JSON = LANDING / "src/components/walkthrough/turn-data.json"
OUT_DIR = LANDING / "public/walkthrough"

# Mirrors the category sets in apps/agent/src/overlay.py.
RESOURCE = {"tree", "gold_mine", "stone_mine", "berry_bush", "relic"}
ANIMAL = {"sheep", "deer", "boar", "wolf", "goose", "goat", "cow", "buffalo", "horse", "rhinoceros"}
BUILDING = {
    "town_center", "house", "lumber_camp", "mining_camp", "mill", "market", "dock", "farm",
    "barracks", "archery_range", "stable", "blacksmith", "siege_workshop", "monastery",
    "castle", "university",
}
DEFENSE = {"gate", "wall", "tower", "wonder", "krepost"}

# Regions with player names, in 3024x1672 pixels: the score panel and the owner
# line of the selection panel.
PRIVATE_REGIONS = [(2400, 1200, 3024, 1395), (1090, 1418, 1345, 1460)]
HUD_STRIP = (0, 0, 1400, 92)


def category(cls: str) -> str:
    if cls in RESOURCE:
        return "resource"
    if cls in ANIMAL:
        return "animal"
    if cls in BUILDING:
        return "building"
    if cls in DEFENSE:
        return "defense"
    return "unit"


def detect(frame: Image.Image, model: Path) -> list[dict]:
    from detection.inference.detector import EntityDetector

    detector = EntityDetector(model_path=str(model), use_sahi=False)
    out = []
    for e in detector.detect(frame):
        x1, y1, x2, y2 = e.bbox
        out.append({"cls": e.class_name, "conf": round(float(e.confidence), 3), "box": (x1, y1, x2, y2)})
    return out


def reviewed(labels: Path) -> list[dict]:
    data = json.loads(labels.read_text())
    out = []
    for shape in data["shapes"]:
        xs = [p[0] for p in shape["points"]]
        ys = [p[1] for p in shape["points"]]
        out.append({"cls": shape["label"], "conf": None, "box": (min(xs), min(ys), max(xs), max(ys))})
    return out


def read_hud(rgb: np.ndarray) -> dict:
    from gameplay_agent.resource_ocr import (
        calibration_for,
        prepare_hud_pixels,
        read_age_pixels,
        read_resource_bar_pixels,
    )

    height, width = rgb.shape[:2]
    calibration = calibration_for(width, height)
    if calibration is None:
        raise SystemExit(f"no HUD calibration for {width}x{height}")
    hud = prepare_hud_pixels(rgb, (width, height))
    readings = dict(read_resource_bar_pixels(hud, calibration))
    readings["age"] = read_age_pixels(hud, calibration)
    keep = ("wood", "food", "gold", "stone", "population", "age", "idle_count", "villagers")
    fields = {
        name: [round(x0 / width, 5), round(y0 / height, 5), round(x1 / width, 5), round(y1 / height, 5)]
        for name, (x0, y0, x1, y1) in calibration.field_rects().items()
        if name in ("wood", "food", "gold", "stone", "population", "age")
    }
    return {"readings": {k: readings[k] for k in keep if k in readings}, "fields": fields}


def redact(frame: Image.Image) -> Image.Image:
    out = frame.copy()
    for box in PRIVATE_REGIONS:
        region = out.crop(box).filter(ImageFilter.GaussianBlur(14))
        out.paste(region, box[:2])
    return out


def save_images(frame: Image.Image) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    w, h = frame.size
    for width in (2048, 1024):
        frame.resize((width, round(h * width / w)), Image.LANCZOS).save(OUT_DIR / f"frame-{width}.webp", quality=82)
    frame.crop(HUD_STRIP).save(OUT_DIR / "hud.webp", quality=90)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frame", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True, help="X-AnyLabeling JSON with reviewed boxes")
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()

    frame = Image.open(args.frame).convert("RGB")
    w, h = frame.size
    if args.model and args.model.exists():
        entities, source = detect(frame, args.model), "detector"
    else:
        entities, source = reviewed(args.labels), "reviewed"
    reviewed_count = len(json.loads(args.labels.read_text())["shapes"])

    counts: Counter[str] = Counter()
    boxes = []
    for e in sorted(entities, key=lambda e: (e["cls"], -(e["conf"] or 0))):
        x1, y1, x2, y2 = e["box"]
        boxes.append({
            "id": f"{e['cls']}_{counts[e['cls']]}",
            "cls": e["cls"],
            "cat": category(e["cls"]),
            "conf": e["conf"],
            "x": round(x1 / w, 5),
            "y": round(y1 / h, 5),
            "w": round((x2 - x1) / w, 5),
            "h": round((y2 - y1) / h, 5),
        })
        counts[e["cls"]] += 1

    data = {
        "frame": {"width": w, "height": h, "run": args.frame.stem},
        "source": source,
        "model": args.model.stem if source == "detector" and args.model else None,
        "reviewed_boxes": reviewed_count,
        "classes": dict(counts.most_common()),
        "entities": boxes,
        "hud": read_hud(np.asarray(frame)),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(data, indent=1) + "\n")
    save_images(redact(frame))
    print(f"{source}: {len(boxes)} boxes {dict(counts.most_common(6))}")
    print("hud:", data["hud"]["readings"])


if __name__ == "__main__":
    main()
