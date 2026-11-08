"""Paste game sprites onto reviewed real training frames at the game's own scale.

The v9 synthetic set drew sprites on flat 1280x720 canvases, so objects reached
the detector 2-3x larger than in real 3024x1672 captures, without fog, HUD or
real terrain. Here every canvas is a real training frame: its reviewed labels
are kept, sprites are scaled by (frame width / 3024) x the game zoom, and
placement avoids the HUD, unexplored black, fog-darkened ground and every
labeled box already in the frame.

Only units, animals and buildings are pasted. Real resource labels (trees,
mines, berries) often cover a whole group, which one sprite cannot reproduce.
Sprites load under their fine label (``ibex``, ``elephant``) so each gets its
own share of a scene; the label file gets the class it maps to (``deer``).

Run from the repository root with::

    uv run python -m detection.training.generate_composites \
        --dataset packages/detection/src/training_data_reviewed_20261018 \
        --sprites tmp/sprites_v7 \
        --output packages/detection/src/training_data_composite_20261030 \
        --train 1000 --val 100
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import yaml
from detection.labeling.class_mapping import get_classes_for_cvat, load_label_aliases
from PIL import Image, ImageFilter

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

Box = tuple[float, float, float, float]

# Capture width at which a sprite renders at its native size times the zoom.
NATIVE_WIDTH = 3024
# Real building labels match native sprite sizes (median ratio 0.94-1.08 for
# house, town centre, mill, camps, barracks), so the game zoom is 1.0.
DEFAULT_ZOOM = 1.0
_ZOOM_JITTER = (0.92, 1.08)
# Unit and animal labels are drawn ~9% looser than the sprite (median ratio
# 1.09 over villager, sheep, goat, deer, boar, scout); pad each side to match.
_UNIT_LABEL_PAD = 0.045

# HUD and desktop regions per capture layout, as normalized (x1, y1, x2, y2).
# 3024x1672: native game captures. 1280x831: downscaled macOS screenshots of the
# VM window, which add the menu bar, window title and dock around the game.
HUD_REGIONS: dict[tuple[int, int], tuple[Box, ...]] = {
    (3024, 1672): (
        (0.0, 0.0, 1.0, 0.06),
        (0.0, 0.0, 0.02, 0.12),
        (0.97, 0.0, 1.0, 0.12),
        (0.0, 0.80, 1.0, 1.0),
        (0.80, 0.72, 1.0, 1.0),
    ),
    (1280, 831): (
        (0.0, 0.0, 1.0, 0.125),
        (0.0, 0.0, 0.02, 0.17),
        (0.97, 0.0, 1.0, 0.17),
        (0.0, 0.77, 1.0, 1.0),
        (0.80, 0.64, 1.0, 1.0),
    ),
}

# Ground under a real unit: median luminance 96, 5th percentile 66.
_MIN_GROUND_LUMINANCE = 60.0
_MAX_DARK_FRACTION = 0.10
_DARK_LUMINANCE = 25
_ALPHA_THRESHOLD = 16
# Overlap is measured against the smaller box; units crowd, buildings never do.
_MAX_OVERLAP_EXISTING = 0.15
_MAX_OVERLAP_UNITS = 0.35
_MAX_OVERLAP_BUILDINGS = 0.05

LIVESTOCK = ("sheep", "goat", "cow", "horse", "buffalo")
_LIVESTOCK_WEIGHTS = (0.35, 0.30, 0.15, 0.10, 0.10)
MILITARY = (
    "scout_line",
    "knight_line",
    "camel_line",
    "battle_elephant",
    "archer_line",
    "skirmisher_line",
    "cavalry_archer",
    "hand_cannoneer",
    "militia_line",
    "spearman_line",
    "eagle_line",
    "ram",
    "mangonel_line",
    "scorpion",
    "trebuchet",
    "siege_tower",
    "monk",
    "king",
    "trade_cart",
    "unique_archer",
    "unique_cavalry",
    "unique_infantry",
    "unique_siege",
)
BUILDINGS = (
    "town_center",
    "house",
    "mill",
    "lumber_camp",
    "mining_camp",
    "market",
    "blacksmith",
    "barracks",
    "archery_range",
    "stable",
    "siege_workshop",
    "monastery",
    "castle",
    "tower",
    "university",
    "wonder",
    "krepost",
)

_SPRITE_NAME = re.compile(r"^(?P<cls>[a-z_]+)_\d{2}_f\d+\.png$")


@dataclass(frozen=True, slots=True)
class Background:
    image: Path
    label: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class Group:
    """Instances of one label placed around a shared anchor."""

    label: str
    count: int
    spread: float  # standard deviation of member offsets, in sprite widths


def load_backgrounds(dataset: Path) -> list[Background]:
    """Distinct real training frames whose capture layout is known."""
    manifest = cast(
        "dict[str, list[dict[str, str]]]",
        json.loads((dataset / "dataset_manifest.json").read_text(encoding="utf-8")),
    )
    seen: dict[str, Background] = {}
    for entry in manifest["images"]:
        if entry["split"] != "train" or entry["source"] not in {"real", "historical_real"}:
            continue
        image = dataset / entry["image"]
        with Image.open(image) as opened:
            if opened.size not in HUD_REGIONS:
                continue
        seen.setdefault(
            entry["image_sha256"],
            Background(image, dataset / entry["label"], entry["image_sha256"]),
        )
    if not seen:
        raise ValueError(f"No training frames with a known capture layout in {dataset}")
    return sorted(seen.values(), key=lambda background: background.sha256)


def crop_to_alpha(sprite: Image.Image) -> Image.Image | None:
    """Trim transparent padding so the label box is the visible sprite."""
    alpha = sprite.getchannel("A").point(lambda value: 255 if value > _ALPHA_THRESHOLD else 0)
    bounds = alpha.getbbox()
    return sprite.crop(bounds) if bounds else None


def load_sprites(directory: Path, labels: Sequence[str]) -> dict[str, list[Image.Image]]:
    sprites: dict[str, list[Image.Image]] = {name: [] for name in labels}
    for path in sorted(directory.glob("*.png")):
        match = _SPRITE_NAME.match(path.name)
        if match is None or match["cls"] not in sprites:
            continue
        with Image.open(path) as opened:
            cropped = crop_to_alpha(opened.convert("RGBA"))
        if cropped is not None and min(cropped.size) >= 4:
            sprites[match["cls"]].append(cropped)
    return {name: images for name, images in sprites.items() if images}


def read_labels(label: Path, size: tuple[int, int]) -> list[tuple[int, Box]]:
    width, height = size
    boxes: list[tuple[int, Box]] = []
    for line in label.read_text(encoding="utf-8").splitlines():
        class_id, cx, cy, bw, bh = line.split()
        x, y, w, h = float(cx) * width, float(cy) * height, float(bw) * width, float(bh) * height
        boxes.append((int(class_id), (x - w / 2, y - h / 2, x + w / 2, y + h / 2)))
    return boxes


def overlap(a: Box, b: Box) -> float:
    """Intersection over the smaller box: a small unit inside a big box counts fully."""
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return iw * ih / smaller if smaller > 0 else 1.0


def in_playfield(box: Box, size: tuple[int, int]) -> bool:
    width, height = size
    if box[0] < 0 or box[1] < 0 or box[2] > width or box[3] > height:
        return False
    normalized = (box[0] / width, box[1] / height, box[2] / width, box[3] / height)
    return all(overlap(normalized, region) == 0.0 for region in HUD_REGIONS[size])


def on_visible_ground(luminance: np.ndarray, box: Box) -> bool:
    """Reject unexplored black and fog: units are drawn only on lit ground."""
    patch = luminance[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])]
    if patch.size == 0:
        return False
    return (
        float(patch.mean()) >= _MIN_GROUND_LUMINANCE
        and float((patch < _DARK_LUMINANCE).mean()) <= _MAX_DARK_FRACTION
    )


def plan_groups(rng: random.Random, available: set[str]) -> list[Group]:
    """One scene: villager work groups, a herd, wildlife, an army, a few buildings."""
    groups = [Group("villager", rng.randint(2, 7), 2.5) for _ in range(rng.randint(1, 3))]
    herds = rng.choices(LIVESTOCK, weights=_LIVESTOCK_WEIGHTS, k=2 if rng.random() < 0.3 else 1)
    if rng.random() < 0.85:
        groups += [Group(name, rng.randint(3, 8), 1.5) for name in set(herds)]
    for name, chance, low, high in (
        ("deer", 0.5, 2, 5),
        ("ibex", 0.3, 2, 5),
        ("boar", 0.4, 1, 2),
        # Real captures hold no elephant; keep its look rare instead of 1/3 of boar.
        ("elephant", 0.03, 1, 1),
        ("wolf", 0.1, 1, 2),
        ("rhinoceros", 0.08, 1, 2),
        ("goose", 0.08, 2, 5),
    ):
        if rng.random() < chance:
            groups.append(Group(name, rng.randint(low, high), 1.5))
    if rng.random() < 0.6:
        pool = [name for name in MILITARY if name in available]
        army = rng.sample(pool, k=min(len(pool), rng.randint(1, 3)))
        groups += [Group(name, rng.randint(1, 6), 1.2) for name in army]
    if rng.random() < 0.6:
        pool = [name for name in BUILDINGS if name in available]
        groups += [
            Group(name, 1, 0.0) for name in rng.sample(pool, k=min(len(pool), rng.randint(1, 2)))
        ]
    return [group for group in groups if group.label in available]


@dataclass(slots=True)
class _Placement:
    sprite: Image.Image
    x: int
    y: int
    label: str
    class_id: int
    box: Box


def compose(
    background: Background,
    sprites: dict[str, list[Image.Image]],
    classes: Sequence[str],
    rng: random.Random,
    zoom: float,
    aliases: Mapping[str, str] | None = None,
) -> tuple[Image.Image, list[str], Counter[str]]:
    """Return the composite, its YOLO label lines, and the pasted label counts."""
    aliases = aliases or {}
    with Image.open(background.image) as opened:
        canvas = opened.convert("RGB")
    size = canvas.size
    luminance = np.asarray(canvas.convert("L"), dtype=np.float32)
    existing = read_labels(background.label, size)
    occupied: list[Box] = [box for _, box in existing]
    scale = size[0] / NATIVE_WIDTH * zoom * rng.uniform(*_ZOOM_JITTER)
    placements: list[_Placement] = []

    for group in plan_groups(rng, set(sprites)):
        is_building = group.label in BUILDINGS
        limit = _MAX_OVERLAP_BUILDINGS if is_building else _MAX_OVERLAP_UNITS
        pad = 0.0 if is_building else _UNIT_LABEL_PAD
        class_id = classes.index(aliases.get(group.label, group.label))
        anchor: tuple[float, float] | None = None
        for _ in range(group.count):
            sprite = _scaled(rng.choice(sprites[group.label]), scale, rng)
            if sprite is None:
                continue
            for _ in range(30 if anchor is None else 10):
                if anchor is None:
                    x = rng.randint(0, size[0] - sprite.width)
                    y = rng.randint(0, size[1] - sprite.height)
                else:
                    spread = group.spread * sprite.width
                    x = round(anchor[0] + rng.gauss(0, spread) - sprite.width / 2)
                    y = round(anchor[1] + rng.gauss(0, spread * 0.6) - sprite.height / 2)
                box = _label_box(x, y, sprite.size, pad)
                if (
                    in_playfield(box, size)
                    and on_visible_ground(luminance, box)
                    and all(overlap(box, other) <= _MAX_OVERLAP_EXISTING for other in occupied)
                    and all(overlap(box, other.box) <= limit for other in placements)
                ):
                    anchor = anchor or (x + sprite.width / 2, y + sprite.height / 2)
                    placements.append(_Placement(sprite, x, y, group.label, class_id, box))
                    break

    # Isometric draw order: the lower an object stands, the nearer it is.
    for placement in sorted(placements, key=lambda item: item.box[3]):
        canvas.paste(placement.sprite, (placement.x, placement.y), placement.sprite)
    lines = [_yolo_line(class_id, box, size) for class_id, box in existing]
    lines += [_yolo_line(p.class_id, p.box, size) for p in placements]
    return canvas, lines, Counter(p.label for p in placements)


def _label_box(x: int, y: int, size: tuple[int, int], pad: float) -> Box:
    """The sprite is already cropped to its visible pixels; pad matches label style."""
    dx, dy = size[0] * pad, size[1] * pad
    return (x - dx, y - dy, x + size[0] + dx, y + size[1] + dy)


def _scaled(sprite: Image.Image, scale: float, rng: random.Random) -> Image.Image | None:
    width, height = round(sprite.width * scale), round(sprite.height * scale)
    if min(width, height) < 3:
        return None
    resized = sprite.resize((width, height), Image.Resampling.LANCZOS)
    if rng.random() < 0.5:
        # Soften the cut-out edge; the detector must not key on paste seams.
        resized.putalpha(resized.getchannel("A").filter(ImageFilter.GaussianBlur(0.6)))
    return resized


def _yolo_line(class_id: int, box: Box, size: tuple[int, int]) -> str:
    width, height = size
    x1, y1 = max(0.0, box[0]), max(0.0, box[1])
    x2, y2 = min(float(width), box[2]), min(float(height), box[3])
    return (
        f"{class_id} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
        f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"
    )


def generate(
    dataset: Path,
    sprites_dir: Path,
    output: Path,
    *,
    train: int,
    val: int,
    seed: int = 0,
    zoom: float = DEFAULT_ZOOM,
) -> dict[str, object]:
    """Write ``output/{train,val}`` and a manifest; refuses an existing output."""
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    classes = tuple(c if isinstance(c, str) else c["name"] for c in get_classes_for_cvat())
    backgrounds = load_backgrounds(dataset)
    aliases = load_label_aliases()
    sprites = load_sprites(sprites_dir, (*classes, *aliases))
    order = list(backgrounds)
    random.Random(seed).shuffle(order)
    records: list[dict[str, object]] = []
    pasted: Counter[str] = Counter()
    index = 0
    for split, count in (("train", train), ("val", val)):
        (output / split / "images").mkdir(parents=True)
        (output / split / "labels").mkdir(parents=True)
        for _ in range(count):
            background = order[index % len(order)]
            image, lines, counts = compose(
                background,
                sprites,
                classes,
                random.Random(seed * 1_000_003 + index),
                zoom,
                aliases,
            )
            stem = f"composite_{index:05d}"
            image.save(output / split / "images" / f"{stem}.jpg", quality=95)
            (output / split / "labels" / f"{stem}.txt").write_text(
                "".join(f"{line}\n" for line in lines), encoding="utf-8"
            )
            records.append(
                {
                    "image": f"{split}/images/{stem}.jpg",
                    "background_sha256": background.sha256,
                    "pasted": dict(counts),
                }
            )
            pasted.update(counts)
            index += 1
    (output / "dataset.yaml").write_text(
        yaml.safe_dump(
            {"train": "train/images", "val": "val/images", "names": dict(enumerate(classes))},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    summary: dict[str, object] = {
        "source_dataset_manifest_sha256": hashlib.sha256(
            (dataset / "dataset_manifest.json").read_bytes()
        ).hexdigest(),
        "sprites_dir": str(sprites_dir),
        "sprite_counts": {name: len(images) for name, images in sorted(sprites.items())},
        "backgrounds": len(backgrounds),
        "seed": seed,
        "zoom": zoom,
        "pasted": dict(pasted.most_common()),
        "images": records,
    }
    (output / "composite_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


class _Args(argparse.Namespace):
    dataset: Path
    sprites: Path
    output: Path
    train: int
    val: int
    seed: int
    zoom: float


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path, help="Prepared real dataset")
    parser.add_argument("--sprites", required=True, type=Path, help="Extracted sprite PNGs")
    parser.add_argument("--output", required=True, type=Path, help="New directory; must not exist")
    parser.add_argument("--train", type=int, default=1000)
    parser.add_argument("--val", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--zoom", type=float, default=DEFAULT_ZOOM)
    args = parser.parse_args(namespace=_Args())
    summary = generate(
        args.dataset,
        args.sprites,
        args.output,
        train=args.train,
        val=args.val,
        seed=args.seed,
        zoom=args.zoom,
    )
    print(f"Wrote {args.output}: {args.train} train, {args.val} val; pasted {summary['pasted']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
