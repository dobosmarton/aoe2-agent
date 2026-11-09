"""Turn annotated game frames into replay sequences with known object identities.

The output uses the `sequence_replay` file format, so generated sequences and recorded
ones run through the same code. Units move as sprites lifted from real frames and
pasted over inpainted terrain. They don't animate, which only recorded footage can fix.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from gameplay_agent.camera_motion import MOBILE_CLASSES
from PIL import Image

Box = tuple[float, float, float, float]

CAPTURE_WIDTH = 3024
CAPTURE_HEIGHT = 1964
CAMERA_MARGIN = 200
FRAME_SECONDS = 0.2
FRAME_JITTER_SECONDS = 0.02
FRAMES = 40  # Long enough to measure target survival at 5 s.
PAN_START_FRAME = 8
PAN_FRAMES = 9
PAN_PIXELS_PER_FRAME = 40
JUMP_FRAME = 13
JUMP_OFFSET = (170, 140)
HUD_TOP_SHARE = 0.05
HUD_BOTTOM_SHARE = 0.76
MIN_VISIBLE_SHARE = 0.5
MOVING_SHARE = 0.5
TURN_FRAMES = (8, 15)
MAX_SPRITE_TILES = 1.5
WOBBLE_PIXELS = 3
SENSOR_NOISE_SIGMA = 2.0
SENSOR_NOISE_TILE = 512
ALPHA_RANGE = 40.0
GAME_SPEED = 1.7
DEFAULT_TILE_WIDTH = 100.0
JPEG_QUALITY = 92

# Footprints in tiles, used to read the zoom level from each frame.
FOOTPRINT_TILES = {"farm": 3, "house": 2, "town_center": 4, "mill": 2, "lumber_camp": 2}
# Walking speeds in tiles per game second. Other mobile classes use the default.
UNIT_TILES_PER_SECOND = {
    "villager": 0.8,
    "sheep": 0.7,
    "scout_line": 1.2,
    "knight_line": 1.35,
    "camel_line": 1.45,
    "cavalry_archer": 1.4,
    "battle_elephant": 0.85,
    "ram": 0.6,
    "mangonel_line": 0.6,
    "monk": 0.7,
}
DEFAULT_TILES_PER_SECOND = 0.96
# Ships can't cross land, and their large holes inpaint badly.
WATER_UNITS = frozenset({"galley", "fire_galley", "fishing_ship", "unique_ship"})
DENSITY_BANDS = (("low", 3), ("typical", 4), ("high", 3), ("peak", 2))
BAND_LIMITS = (("low", 0.33), ("typical", 0.8), ("high", 0.97), ("peak", 1.01))
CAMERA_ORDER = ("stable", "pan", "stable", "jump")


@dataclass(frozen=True, slots=True)
class Annotation:
    box: Box
    label: str


@dataclass(frozen=True, slots=True)
class Layout:
    name: str
    image: Path
    objects: tuple[Annotation, ...]


@dataclass(slots=True)
class _Unit:
    """An object in world coordinates. Static objects keep zero velocity."""

    identifier: str
    label: str
    box: Box
    velocity: tuple[float, float] = (0.0, 0.0)
    sprite: np.ndarray | None = None
    alpha: np.ndarray | None = None
    turns_at: int = 0
    home: Box | None = None


def tile_width(objects: tuple[Annotation, ...]) -> float:
    """Estimate the zoom level from buildings of known size."""
    widths = [
        (item.box[2] - item.box[0]) / FOOTPRINT_TILES[item.label]
        for item in objects
        if item.label in FOOTPRINT_TILES
    ]
    if not widths:
        return DEFAULT_TILE_WIDTH
    return float(np.median(widths))


def pixels_per_second(label: str, tile_px: float) -> float:
    """Screen speed of a unit. A map tile crosses about `width / sqrt(2)` pixels."""
    tiles = UNIT_TILES_PER_SECOND.get(label, DEFAULT_TILES_PER_SECOND)
    return tiles * GAME_SPEED * tile_px / math.sqrt(2)


def _image_index(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for dataset in sorted(root.glob("training_data_*")):
        for directory in sorted(dataset.glob("*/images")):
            for path in sorted(directory.iterdir()):
                stem = path.stem.removeprefix("historical_real_")
                if path.name.startswith("historical_real_") and "__" not in stem:
                    index.setdefault(stem, path)
    return index


def load_layouts(agent_root: Path) -> list[Layout]:
    """Read annotated frames. The batches overlap, so keep each frame's fullest copy."""
    images = _image_index(agent_root / "packages/detection/src")
    best: dict[str, Layout] = {}
    for path in sorted(
        (agent_root / "tmp/cvat_merge").glob("*/annotations/instances_default.json")
    ):
        document = cast("dict[str, list[dict[str, object]]]", json.loads(path.read_text()))
        classes = {item["id"]: cast("str", item["name"]) for item in document["categories"]}
        grouped: dict[object, list[Annotation]] = {}
        for item in document["annotations"]:
            x, y, width, height = cast("list[float]", item["bbox"])
            grouped.setdefault(item["image_id"], []).append(
                Annotation((x, y, x + width, y + height), classes[item["category_id"]])
            )
        for frame in document["images"]:
            stem = Path(cast("str", frame["file_name"])).stem
            objects = grouped.get(frame["id"], [])
            if (
                stem in images
                and objects
                and frame["width"] == CAPTURE_WIDTH
                and frame["height"] == CAPTURE_HEIGHT
                and (stem not in best or len(objects) > len(best[stem].objects))
            ):
                best[stem] = Layout(stem, images[stem], tuple(objects))
    return sorted(best.values(), key=lambda item: (len(item.objects), item.name))


def choose_layouts(layouts: list[Layout], *, quick: bool) -> list[tuple[Layout, str]]:
    """Pick layouts across the real density range, labelled by density band."""
    chosen: list[tuple[Layout, str]] = []
    start = 0.0
    for band, limit in BAND_LIMITS:
        members = [
            item for rank, item in enumerate(layouts) if start <= rank / len(layouts) < limit
        ]
        wanted = dict(DENSITY_BANDS)[band]
        if quick:
            wanted = 1
        step = len(members) / wanted
        chosen += [(members[int(step * (index + 0.5))], band) for index in range(wanted)]
        start = limit
    return chosen


def _full_size_frame(layout: Layout) -> np.ndarray:
    with Image.open(layout.image) as opened:
        image = opened.convert("RGB").resize(
            (CAPTURE_WIDTH, CAPTURE_HEIGHT), Image.Resampling.LANCZOS
        )
    return np.array(image, dtype=np.uint8)


def _int_box(box: Box) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    return (
        max(0, int(x0)),
        max(0, int(y0)),
        min(CAPTURE_WIDTH, math.ceil(x1)),
        min(CAPTURE_HEIGHT, math.ceil(y1)),
    )


def _new_velocity(rng: random.Random, label: str, tile_px: float) -> tuple[float, float]:
    speed = pixels_per_second(label, tile_px)
    heading = rng.uniform(0.0, 2.0 * math.pi)
    return (speed * math.cos(heading), speed * math.sin(heading))


def _is_sprite(item: Annotation, tile_px: float) -> bool:
    """Whether an object can be lifted off the terrain convincingly."""
    width = item.box[2] - item.box[0]
    return (
        item.label in MOBILE_CLASSES
        and item.label not in WATER_UNITS
        and width <= MAX_SPRITE_TILES * tile_px
    )


def _build_units(
    layout: Layout, frame: np.ndarray, rng: random.Random
) -> tuple[list[_Unit], np.ndarray]:
    """Lift moving units off the terrain, leaving an inpainted background."""
    tile_px = tile_width(layout.objects)
    units: list[_Unit] = []
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for index, item in enumerate(layout.objects):
        unit = _Unit(f"gt_{index}", item.label, item.box)
        if _is_sprite(item, tile_px):
            if rng.random() < MOVING_SHARE:
                unit.velocity = _new_velocity(rng, item.label, tile_px)
                unit.turns_at = rng.randint(*TURN_FRAMES)
            else:
                unit.home = item.box
            x0, y0, x1, y1 = _int_box(item.box)
            mask[y0:y1, x0:x1] = 255
        units.append(unit)
    background = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA) if mask.any() else frame.copy()
    for unit in units:
        if unit.velocity == (0.0, 0.0) and unit.home is None:
            continue
        x0, y0, x1, y1 = _int_box(unit.box)
        if x1 <= x0 or y1 <= y0:
            continue
        unit.sprite = frame[y0:y1, x0:x1].copy()
        difference = np.abs(unit.sprite.astype(np.float32) - background[y0:y1, x0:x1])
        unit.alpha = np.clip(difference.mean(axis=2, keepdims=True) / ALPHA_RANGE, 0.0, 1.0)
    return units, background


def _paste(canvas: np.ndarray, unit: _Unit) -> None:
    if unit.sprite is None or unit.alpha is None:
        return
    height, width = unit.sprite.shape[:2]
    left, top = round(unit.box[0]), round(unit.box[1])
    x0, y0 = max(0, left), max(0, top)
    x1, y1 = min(CAPTURE_WIDTH, left + width), min(CAPTURE_HEIGHT, top + height)
    if x1 <= x0 or y1 <= y0:
        return
    sprite = unit.sprite[y0 - top : y1 - top, x0 - left : x1 - left]
    alpha = unit.alpha[y0 - top : y1 - top, x0 - left : x1 - left]
    target = canvas[y0:y1, x0:x1]
    canvas[y0:y1, x0:x1] = (sprite * alpha + target * (1.0 - alpha)).astype(np.uint8)


def _move(unit: _Unit, seconds: float, frame: int, rng: random.Random, tile_px: float) -> None:
    if unit.home is not None:
        # Working units stay put but never hold still, like a gather animation.
        dx = rng.randint(-WOBBLE_PIXELS, WOBBLE_PIXELS)
        dy = rng.randint(-WOBBLE_PIXELS, WOBBLE_PIXELS)
        x0, y0, x1, y1 = unit.home
        unit.box = (x0 + dx, y0 + dy, x1 + dx, y1 + dy)
        return
    if unit.velocity == (0.0, 0.0):
        return
    if frame >= unit.turns_at:
        unit.velocity = _new_velocity(rng, unit.label, tile_px)
        unit.turns_at = frame + rng.randint(*TURN_FRAMES)
    dx, dy = unit.velocity[0] * seconds, unit.velocity[1] * seconds
    x0, y0, x1, y1 = unit.box
    moved = (x0 + dx, y0 + dy, x1 + dx, y1 + dy)
    if moved[0] < 0 or moved[1] < 0 or moved[2] > CAPTURE_WIDTH or moved[3] > CAPTURE_HEIGHT:
        unit.velocity = (-unit.velocity[0], -unit.velocity[1])
        return
    unit.box = moved


def camera_offset(camera: str, frame: int) -> tuple[int, int]:
    """Viewport position inside the world image for one frame."""
    x, y = CAMERA_MARGIN, CAMERA_MARGIN
    if camera == "pan":
        steps = min(PAN_FRAMES, max(0, frame - PAN_START_FRAME))
        return (x - 180 + steps * PAN_PIXELS_PER_FRAME, y)
    if camera == "jump" and frame >= JUMP_FRAME:
        return (x + JUMP_OFFSET[0], y + JUMP_OFFSET[1])
    return (x, y)


def view_box(box: Box, offset: tuple[int, int], width: int, height: int) -> Box | None:
    """World box in viewport coordinates, or None if it is mostly hidden or under the HUD."""
    x0, y0, x1, y1 = (
        box[0] - offset[0],
        box[1] - offset[1],
        box[2] - offset[0],
        box[3] - offset[1],
    )
    left, top = max(0.0, x0), max(height * HUD_TOP_SHARE, y0)
    right, bottom = min(float(width), x1), min(height * HUD_BOTTOM_SHARE, y1)
    if right <= left or bottom <= top:
        return None
    if (right - left) * (bottom - top) < MIN_VISIBLE_SHARE * (x1 - x0) * (y1 - y0):
        return None
    return (left, top, right, bottom)


def _sensor_noise(view: np.ndarray, tile: np.ndarray, rng: random.Random) -> np.ndarray:
    """Vary every frame slightly, or a deterministic detector would see no change at all."""
    shift = (rng.randrange(SENSOR_NOISE_TILE), rng.randrange(SENSOR_NOISE_TILE))
    height, width = view.shape[:2]
    rows = np.arange(height) % SENSOR_NOISE_TILE
    columns = np.arange(width) % SENSOR_NOISE_TILE
    noise = np.roll(tile, shift, axis=(0, 1))[np.ix_(rows, columns)]
    return np.clip(view.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def write_sequence(layout: Layout, band: str, camera: str, directory: Path, seed: int) -> None:
    """Render one sequence. An existing sequence is kept, so detector runs stay cached."""
    if (directory / "scene.json").exists():
        return
    directory.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    world = _full_size_frame(layout)
    units, background = _build_units(layout, world, rng)
    tile_px = tile_width(layout.objects)
    width = CAPTURE_WIDTH - 2 * CAMERA_MARGIN
    height = CAPTURE_HEIGHT - 2 * CAMERA_MARGIN
    hud_top = round(height * HUD_TOP_SHARE)
    hud_bottom = round(height * HUD_BOTTOM_SHARE)
    fixed = world[CAMERA_MARGIN : CAMERA_MARGIN + height, CAMERA_MARGIN : CAMERA_MARGIN + width]
    noise_tile = (
        np.random.default_rng(seed)
        .normal(0.0, SENSOR_NOISE_SIGMA, (SENSOR_NOISE_TILE, SENSOR_NOISE_TILE, 3))
        .astype(np.float32)
    )
    at = 1.0
    frames: list[dict[str, object]] = []
    labels: list[dict[str, object]] = []
    for capture_id in range(FRAMES):
        seconds = FRAME_SECONDS + rng.uniform(-FRAME_JITTER_SECONDS, FRAME_JITTER_SECONDS)
        at += seconds
        for unit in units:
            _move(unit, seconds, capture_id, rng, tile_px)
        canvas = background.copy()
        for unit in units:
            _paste(canvas, unit)
        offset = camera_offset(camera, capture_id)
        view = canvas[offset[1] : offset[1] + height, offset[0] : offset[0] + width].copy()
        view[:hud_top] = fixed[:hud_top]
        view[hud_bottom:] = fixed[hud_bottom:]
        name = f"frame_{capture_id:03d}.jpg"
        noisy = _sensor_noise(view, noise_tile, rng)
        Image.fromarray(noisy).save(directory / name, quality=JPEG_QUALITY)
        jumped = camera == "jump" and capture_id == JUMP_FRAME
        frames.append(
            {
                "capture_id": capture_id,
                "captured_at_monotonic": at,
                "image": name,
                "input_events": ["camera_jump"] if jumped else [],
            }
        )
        visible = ((unit, view_box(unit.box, offset, width, height)) for unit in units)
        labels.append(
            {
                "capture_id": capture_id,
                "objects": [
                    {"track_id": unit.identifier, "class_name": unit.label, "bbox": list(box)}
                    for unit, box in visible
                    if box is not None
                ],
            }
        )
    _jsonl(directory / "sequence.jsonl", frames)
    _jsonl(directory / "labels.jsonl", labels)
    scene = {
        "layout": layout.name,
        "band": band,
        "camera": camera,
        "objects": len(layout.objects),
        "tile_px": tile_px,
        "walking": sum(unit.velocity != (0.0, 0.0) for unit in units),
        "working": sum(unit.home is not None for unit in units),
    }
    (directory / "scene.json").write_text(json.dumps(scene, indent=2) + "\n", encoding="utf-8")


def generate(agent_root: Path, output: Path, *, quick: bool) -> list[Path]:
    """Write every chosen sequence and return their directories."""
    layouts = load_layouts(agent_root)
    directories: list[Path] = []
    for index, (layout, band) in enumerate(choose_layouts(layouts, quick=quick)):
        camera = CAMERA_ORDER[index % len(CAMERA_ORDER)]
        directory = output / f"{index:02d}_{band}_{camera}"
        write_sequence(layout, band, camera, directory, seed=1000 + index)
        directories.append(directory)
    return directories


@dataclass(frozen=True, slots=True)
class _Scripted:
    """One object in a targeted scene, as `(frame) -> box`, with its colour."""

    identifier: str
    label: str
    colour: tuple[int, int, int]
    start: Box
    velocity: tuple[float, float] = (0.0, 0.0)
    hidden: frozenset[int] = frozenset()

    def box(self, frame: int) -> Box:
        dx, dy = self.velocity[0] * frame, self.velocity[1] * frame
        x0, y0, x1, y1 = self.start
        return (x0 + dx, y0 + dy, x1 + dx, y1 + dy)


TARGETED_WIDTH = 960
TARGETED_HEIGHT = 540
TARGETED_FRAMES = 20
DUPLICATE_FRAME = 3
DUPLICATE_OFFSET = 1.0


def _write_targeted(
    directory: Path,
    objects: tuple[_Scripted, ...],
    *,
    duplicate_at: int | None = None,
    jump_at: int | None = None,
) -> None:
    """Write a short scene with scripted detections, so the answer is known exactly."""
    directory.mkdir(parents=True, exist_ok=True)
    backdrop = np.random.default_rng(5).integers(
        60, 140, (TARGETED_HEIGHT, TARGETED_WIDTH, 3), dtype=np.uint8
    )
    frames: list[dict[str, object]] = []
    detections: list[dict[str, object]] = []
    labels: list[dict[str, object]] = []
    for capture_id in range(TARGETED_FRAMES):
        image = backdrop.copy()
        boxes: list[dict[str, object]] = []
        for item in objects:
            box = item.box(capture_id)
            x0, y0, x1, y1 = (round(value) for value in box)
            image[y0:y1, x0:x1] = item.colour
            if capture_id not in item.hidden:
                boxes.append({"class_name": item.label, "bbox": list(box), "confidence": 0.8})
        if capture_id == duplicate_at:
            x0, y0, x1, y1 = objects[0].box(capture_id)
            shifted = [x0 + DUPLICATE_OFFSET, y0, x1 + DUPLICATE_OFFSET, y1]
            boxes.append({"class_name": objects[0].label, "bbox": shifted, "confidence": 0.8})
        name = f"frame_{capture_id:03d}.png"
        Image.fromarray(image).save(directory / name)
        jumped = capture_id == jump_at
        frames.append(
            {
                "capture_id": capture_id,
                "captured_at_monotonic": 1.0 + capture_id * FRAME_SECONDS,
                "image": name,
                "input_events": ["camera_jump"] if jumped else [],
            }
        )
        detections.append({"capture_id": capture_id, "detections": boxes})
        labels.append(
            {
                "capture_id": capture_id,
                "objects": [
                    {
                        "track_id": item.identifier,
                        "class_name": item.label,
                        "bbox": list(item.box(capture_id)),
                    }
                    for item in objects
                ],
            }
        )
    _jsonl(directory / "sequence.jsonl", frames)
    _jsonl(directory / "raw.jsonl", detections)
    _jsonl(directory / "labels.jsonl", labels)
    scene = {
        "layout": directory.name,
        "band": "targeted",
        "camera": "stable",
        "objects": len(objects),
    }
    (directory / "scene.json").write_text(json.dumps(scene, indent=2) + "\n", encoding="utf-8")


def write_targeted(output: Path) -> dict[str, Path]:
    """Scenes with exact expected behaviour. The detector is scripted, so no model runs."""
    still = _Scripted("gt_0", "villager", (220, 40, 40), (200.0, 200.0, 240.0, 260.0))
    other = _Scripted("gt_1", "sheep", (40, 200, 60), (600.0, 300.0, 660.0, 350.0))
    hidden = _Scripted(
        "gt_0",
        "villager",
        (220, 40, 40),
        (200.0, 200.0, 240.0, 260.0),
        hidden=frozenset({6, 7}),
    )
    left = _Scripted("gt_0", "villager", (220, 40, 40), (300.0, 200.0, 340.0, 260.0), (4.0, 0.0))
    right = _Scripted("gt_1", "villager", (40, 60, 220), (380.0, 200.0, 420.0, 260.0), (-4.0, 0.0))
    scenes = {
        "duplicate_box": ((still, other), {"duplicate_at": DUPLICATE_FRAME}),
        "occlusion": ((hidden, other), {}),
        "crossing": ((left, right), {}),
        "camera_jump": ((still, other), {"jump_at": 10}),
    }
    directories: dict[str, Path] = {}
    for name, (objects, options) in scenes.items():
        directory = output / name
        if not (directory / "scene.json").exists():
            _write_targeted(directory, objects, **options)
        directories[name] = directory
    return directories
