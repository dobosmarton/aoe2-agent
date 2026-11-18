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

This rewrites turn-data.json; run scripts/build_walkthrough_sprites.py with --frame
afterwards to add the village placements (`world`).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias, TypedDict, cast

import numpy as np
from PIL import Image, ImageFilter

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray

LANDING = Path(__file__).resolve().parent.parent
OUT_JSON = LANDING / "src/components/walkthrough/turn-data.json"
OUT_DIR = LANDING / "public/walkthrough"

# Mirrors the category sets in apps/agent/src/overlay.py.
RESOURCE = frozenset({"tree", "gold_mine", "stone_mine", "berry_bush", "relic"})
ANIMAL = frozenset(
    {"sheep", "deer", "boar", "wolf", "goose", "goat", "cow", "buffalo", "horse", "rhinoceros"}
)
BUILDING = frozenset(
    {
        "town_center",
        "house",
        "lumber_camp",
        "mining_camp",
        "mill",
        "market",
        "dock",
        "farm",
        "barracks",
        "archery_range",
        "stable",
        "blacksmith",
        "siege_workshop",
        "monastery",
        "castle",
        "university",
    }
)
DEFENSE = frozenset({"gate", "wall", "tower", "wonder", "krepost"})

# Regions with player names, in 3024x1672 pixels: the score panel and the owner
# line of the selection panel.
PRIVATE_REGIONS = ((2400, 1200, 3024, 1395), (1090, 1418, 1345, 1460))
PRIVATE_BLUR_RADIUS = 14
HUD_STRIP = (0, 0, 1400, 92)
FRAME_WIDTHS = (2048, 1024)
# Decimal places of the normalised (0..1) coordinates in turn-data.json.
COORD_DIGITS = 5
CONFIDENCE_DIGITS = 3

# The HUD fields whose rectangles the page draws, and the readings it shows.
HUD_FIELDS = ("wood", "food", "gold", "stone", "population", "age")
KEPT_READINGS = (*HUD_FIELDS, "idle_count", "villagers")

Category: TypeAlias = Literal["resource", "animal", "building", "defense", "unit"]
Source: TypeAlias = Literal["detector", "reviewed"]
# (x1, y1, x2, y2) in frame pixels.
Box: TypeAlias = tuple[float, float, float, float]


class FrameInfo(TypedDict):
    width: int
    height: int
    run: str


class TurnEntity(TypedDict):
    """One box of turn-data.json, normalised to the frame size."""

    id: str
    cls: str
    cat: Category
    conf: float | None
    x: float
    y: float
    w: float
    h: float


class HudReadings(TypedDict, total=False):
    """The kept HUD readings; a key is present only when it was read."""

    wood: int
    food: int
    gold: int
    stone: int
    population: str
    age: str
    idle_count: int
    villagers: int


class Hud(TypedDict):
    readings: HudReadings
    # Field name -> [x0, y0, x1, y1], normalised to the frame size.
    fields: dict[str, list[float]]


class TurnData(TypedDict):
    """turn-data.json; build_walkthrough_sprites.py adds `world` to it."""

    frame: FrameInfo
    source: Source
    model: str | None
    reviewed_boxes: int
    classes: dict[str, int]
    entities: list[TurnEntity]
    hud: Hud


class LabelShape(TypedDict):
    label: str
    points: list[list[float]]


class LabelFile(TypedDict):
    """The part of an X-AnyLabeling JSON file this script reads."""

    shapes: list[LabelShape]


@dataclass(frozen=True, slots=True)
class Detection:
    cls: str
    conf: float | None
    box: Box


@dataclass(frozen=True, slots=True)
class Options:
    frame: Path
    labels: Path
    model: Path | None


def category(cls: str) -> Category:
    if cls in RESOURCE:
        return "resource"
    if cls in ANIMAL:
        return "animal"
    if cls in BUILDING:
        return "building"
    if cls in DEFENSE:
        return "defense"
    return "unit"


def detect(frame: Image.Image, model: Path) -> list[Detection]:
    from detection.inference.detector import EntityDetector

    detector = EntityDetector(model_path=str(model), use_sahi=False)
    detections: list[Detection] = []
    for entity in detector.detect(frame):
        x1, y1, x2, y2 = entity.bbox
        confidence = round(float(entity.confidence), CONFIDENCE_DIGITS)
        detections.append(Detection(cls=entity.class_name, conf=confidence, box=(x1, y1, x2, y2)))
    return detections


def load_labels(path: Path) -> LabelFile:
    # The file is X-AnyLabeling's own output; this script trusts its shape.
    return cast("LabelFile", json.loads(path.read_text()))


def reviewed(labels: LabelFile) -> list[Detection]:
    return [
        Detection(cls=shape["label"], conf=None, box=_bounding_box(shape["points"]))
        for shape in labels["shapes"]
    ]


def _bounding_box(points: list[list[float]]) -> Box:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (min(xs), min(ys), max(xs), max(ys))


def read_hud(rgb: NDArray[np.uint8]) -> Hud:
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
    readings: dict[str, object] = dict(read_resource_bar_pixels(hud, calibration))
    readings["age"] = read_age_pixels(hud, calibration)
    rects: dict[str, tuple[int, int, int, int]] = calibration.field_rects()
    fields = {
        name: [
            round(x0 / width, COORD_DIGITS),
            round(y0 / height, COORD_DIGITS),
            round(x1 / width, COORD_DIGITS),
            round(y1 / height, COORD_DIGITS),
        ]
        for name, (x0, y0, x1, y1) in rects.items()
        if name in HUD_FIELDS
    }
    return {"readings": _kept_readings(readings), "fields": fields}


def _kept_readings(readings: Mapping[str, object]) -> HudReadings:
    kept = {key: readings[key] for key in KEPT_READINGS if key in readings}
    # resource_ocr returns its readings as dict[str, object]; their keys and value
    # types are the ones its ResourceReadings declares, which HudReadings mirrors.
    return cast("HudReadings", kept)


def redact(frame: Image.Image) -> Image.Image:
    out = frame.copy()
    for box in PRIVATE_REGIONS:
        region = out.crop(box).filter(ImageFilter.GaussianBlur(PRIVATE_BLUR_RADIUS))
        out.paste(region, box[:2])
    return out


def save_images(frame: Image.Image) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    w, h = frame.size
    for width in FRAME_WIDTHS:
        resized = frame.resize((width, round(h * width / w)), Image.Resampling.LANCZOS)
        resized.save(OUT_DIR / f"frame-{width}.webp", quality=82)
    frame.crop(HUD_STRIP).save(OUT_DIR / "hud.webp", quality=90)


def to_entities(detections: list[Detection], frame_size: tuple[int, int]) -> list[TurnEntity]:
    """Normalised boxes sorted by class, most confident first, numbered per class."""
    w, h = frame_size
    numbered: Counter[str] = Counter()
    entities: list[TurnEntity] = []
    for detection in sorted(detections, key=lambda d: (d.cls, -(d.conf or 0))):
        x1, y1, x2, y2 = detection.box
        entities.append(
            {
                "id": f"{detection.cls}_{numbered[detection.cls]}",
                "cls": detection.cls,
                "cat": category(detection.cls),
                "conf": detection.conf,
                "x": round(x1 / w, COORD_DIGITS),
                "y": round(y1 / h, COORD_DIGITS),
                "w": round((x2 - x1) / w, COORD_DIGITS),
                "h": round((y2 - y1) / h, COORD_DIGITS),
            }
        )
        numbered[detection.cls] += 1
    return entities


def parse_options() -> Options:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frame", type=Path, required=True)
    parser.add_argument(
        "--labels", type=Path, required=True, help="X-AnyLabeling JSON with reviewed boxes"
    )
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()
    return Options(frame=args.frame, labels=args.labels, model=args.model)


def main() -> None:
    options = parse_options()
    frame = Image.open(options.frame).convert("RGB")
    labels = load_labels(options.labels)
    model = options.model if options.model is not None and options.model.exists() else None
    source: Source = "detector" if model is not None else "reviewed"
    detections = detect(frame, model) if model is not None else reviewed(labels)

    entities = to_entities(detections, frame.size)
    classes = Counter(entity["cls"] for entity in entities)
    data: TurnData = {
        "frame": {"width": frame.width, "height": frame.height, "run": options.frame.stem},
        "source": source,
        "model": model.stem if model is not None else None,
        "reviewed_boxes": len(labels["shapes"]),
        "classes": dict(classes.most_common()),
        "entities": entities,
        "hud": read_hud(np.asarray(frame)),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(data, indent=1) + "\n")
    save_images(redact(frame))
    print(f"{source}: {len(entities)} boxes {dict(classes.most_common(6))}")
    print("hud:", data["hud"]["readings"])


if __name__ == "__main__":
    main()
