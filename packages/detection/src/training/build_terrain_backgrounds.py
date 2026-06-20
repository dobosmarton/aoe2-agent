"""Build background images from AoE2:DE ground-terrain DDS textures.

The game's terrain lives in `resources/_common/terrain/textures/` as seamless,
top-down DDS tiles (grass, dirt, desert, forest floor, ...). This script crops,
scales, jitters, softens, and occasionally blends them into 1280x720 PNGs that
feed `generate_training_data.py --backgrounds`, so synthetic scenes sit on real
game ground instead of flat procedural color.

Pillow reads the DXT1 DDS directly (no extra deps). Pass absolute paths — under
`uv run` the module's cwd is the package dir, so relative paths resolve wrong.

Usage (from the repo root):
    uv run python -m detection.training.build_terrain_backgrounds \
        --terrain-dir "$PWD/game_terrain" \
        --output "$PWD/tmp/terrain_backgrounds" \
        --count 120
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageStat

# Void / non-ground textures to never use as a backdrop.
#   g_bla : black out-of-bounds fill
#   o_*   : DE placeholder terrains — a diamond grid stamped "PLACEHOLDER"
#           (unimplemented/modded terrain slots); never appears in real games.
_EXCLUDE_PREFIXES = ("g_bla", "o_")
# Pure-water textures — excluded by default (land maps); enable with --include-water.
_WATER_PREFIXES = ("g_wt", "g_wtr")
# Below this mean luminance (0-255) a tile reads as void/near-black (g_bla, g_kf1,
# o_mod, ...); legit dark ground (forest/wet rock ~90+) stays.
_MIN_MEAN_LUMINANCE = 20.0

# Weight common open-map terrains higher so backgrounds skew realistic.
# Busy cobblestone/rock is down-weighted hard — it's the harshest backdrop and
# the most likely to swallow sprites (it still appears, just rarely + softened).
_WEIGHT_RULES: list[tuple[tuple[str, ...], float]] = [
    (("g_gr", "g_grs"), 4.0),  # grass — most common
    (("g_des", "g_ds", "g_snd"), 3.0),  # dirt / desert / sand
    (("g_for", "g_fo", "g_fc", "g_fm", "g_underbrush", "g_pal"), 2.0),  # forest floor
    (("g_rd", "g_sr"), 0.6),  # plain roads
    (("g_rc", "g_rck", "g_gravel", "g_rm"), 0.3),  # cobblestone / rock / gravel — busiest
    (("g_ic", "g_ice", "g_sn", "g_sno", "g_snf"), 0.5),  # ice / snow (rarer)
]
_DEFAULT_WEIGHT = 1.0

# Jitter bands (multipliers) — biased below 1.0 to mute the ground behind sprites.
_BRIGHTNESS_RANGE = (0.85, 1.08)
_CONTRAST_RANGE = (0.8, 1.0)
_SATURATION_RANGE = (0.65, 1.0)
_HORIZONTAL_FLIP_PROB = 0.5
# Smoothstep band half-width for the two-terrain blend edge (in projection units).
_BLEND_BAND = 0.4


class WeightedTexture(NamedTuple):
    """A loaded terrain image and its sampling weight."""

    image: Image.Image
    weight: float


@dataclass(frozen=True, slots=True)
class SofteningConfig:
    """Knobs for the realism→soft background spectrum.

    A `soft_fraction` of backgrounds are heavily blurred (muted, so objects always
    pop); the rest get a mild blur matching the game's softer isometric rendering.
    """

    blend_prob: float = 0.35
    zoom_max: float = 1.6
    soft_fraction: float = 0.35
    blur_min: float = 0.6  # mild ("textured") blur band
    blur_max: float = 1.6
    soft_blur_min: float = 5.0  # strong ("muted") blur band
    soft_blur_max: float = 9.0


def _weight_for(stem: str) -> float:
    for prefixes, weight in _WEIGHT_RULES:
        if stem.startswith(prefixes):
            return weight
    return _DEFAULT_WEIGHT


def load_textures(
    terrain_dir: Path, *, include_water: bool, only_water: bool = False
) -> list[WeightedTexture]:
    """Load ground-terrain DDS as weighted RGB textures, skipping void/near-black tiles.

    `only_water` flips the selection to load ONLY the water tiles (`g_wt*`) — used to
    build dedicated water backgrounds for naval/fish scenes.
    """
    textures: list[WeightedTexture] = []
    for path in sorted(terrain_dir.rglob("*.dds")):
        stem = path.stem.lower()
        if stem.startswith(_EXCLUDE_PREFIXES):
            continue
        is_water = stem.startswith(_WATER_PREFIXES)
        if only_water and not is_water:
            continue
        if not only_water and not include_water and is_water:
            continue
        try:
            image = Image.open(path).convert("RGB")
        except OSError as exc:  # unreadable / unsupported DDS variant
            print(f"  skip {path.name}: {type(exc).__name__}: {exc}")
            continue
        mean_luminance = ImageStat.Stat(image.convert("L")).mean[0]
        if mean_luminance < _MIN_MEAN_LUMINANCE:
            print(f"  skip {path.name}: near-black (mean luminance {mean_luminance:.0f})")
            continue
        textures.append(WeightedTexture(image, _weight_for(stem)))
    return textures


def _crop_scaled(
    tex: Image.Image, size: tuple[int, int], rng: random.Random, zoom_max: float
) -> Image.Image:
    """Crop a target-aspect region at a random zoom, then resize to `size`.

    Orientation variety is applied to the SQUARE source texture here (square-in →
    square-out, no black fill) — never to the final non-square frame, which would
    leave black bars.
    """
    quarter_turn = rng.choice([0, 90, 180, 270])
    if quarter_turn == 180 or (quarter_turn and tex.width == tex.height):
        tex = tex.rotate(quarter_turn)
    target_w, target_h = size
    aspect = target_w / target_h
    src_w, src_h = tex.size
    # Lower zoom keeps texture features small/distant-looking; high zoom magnifies
    # individual cobblestones to sprite-size and competes with the objects.
    zoom = rng.uniform(1.0, zoom_max)
    crop_w = min(src_w, int(src_w / zoom))
    crop_h = int(crop_w / aspect)
    if crop_h > src_h:  # texture not tall enough for this aspect at this zoom
        crop_h = src_h
        crop_w = int(crop_h * aspect)
    left = rng.randint(0, src_w - crop_w)
    top = rng.randint(0, src_h - crop_h)
    crop = tex.crop((left, top, left + crop_w, top + crop_h))
    return crop.resize(size, Image.Resampling.LANCZOS)


def _linear_gradient_mask(size: tuple[int, int], rng: random.Random) -> Image.Image:
    """Soft linear gradient (0..255) at a random angle, for terrain transitions."""
    width, height = size
    angle = rng.uniform(0.0, math.pi)
    grid_x, grid_y = np.meshgrid(np.linspace(-1.0, 1.0, width), np.linspace(-1.0, 1.0, height))
    # math.cos/sin return plain floats (numpy's would be untyped `Any`).
    projection = grid_x * math.cos(angle) + grid_y * math.sin(angle)
    # smoothstep across a soft band for a gradual (not hard) edge
    ramp = np.clip((projection + _BLEND_BAND) / (2.0 * _BLEND_BAND), 0.0, 1.0)
    smooth = ramp * ramp * (3.0 - 2.0 * ramp)
    return Image.fromarray((smooth * 255.0).astype(np.uint8), mode="L")


def _jitter(image: Image.Image, rng: random.Random) -> Image.Image:
    """Lighting jitter biased to lower contrast/saturation so the ground recedes
    behind the (colorful, sharp) sprites composited on top later."""
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(*_BRIGHTNESS_RANGE))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(*_CONTRAST_RANGE))
    return ImageEnhance.Color(image).enhance(rng.uniform(*_SATURATION_RANGE))


def _blur_radius(rng: random.Random, config: SofteningConfig) -> float:
    """The realism→soft spectrum: a strong (muted) blur for `soft_fraction` of
    backgrounds, a mild (game-like) blur for the rest."""
    if rng.random() < config.soft_fraction:
        return rng.uniform(config.soft_blur_min, config.soft_blur_max)
    return rng.uniform(config.blur_min, config.blur_max)


def make_background(
    textures: list[WeightedTexture],
    size: tuple[int, int],
    rng: random.Random,
    config: SofteningConfig,
) -> Image.Image:
    images = [t.image for t in textures]
    weights = [t.weight for t in textures]
    base = _crop_scaled(rng.choices(images, weights=weights)[0], size, rng, config.zoom_max)
    if rng.random() < config.blend_prob and len(images) > 1:
        other = _crop_scaled(rng.choices(images, weights=weights)[0], size, rng, config.zoom_max)
        base = Image.composite(other, base, _linear_gradient_mask(size, rng))
    # Orientation variety is handled on the square source in _crop_scaled; here
    # only a horizontal mirror (safe for the rectangular frame).
    if rng.random() < _HORIZONTAL_FLIP_PROB:
        base = base.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    base = _jitter(base, rng)
    return base.filter(ImageFilter.GaussianBlur(_blur_radius(rng, config)))


class _BuildBackgroundsArgs(argparse.Namespace):
    terrain_dir: str
    output: str
    count: int
    size: list[int]
    blend_prob: float
    zoom_max: float
    soft_fraction: float
    blur: float
    soft_blur: float
    include_water: bool
    only_water: bool
    seed: int


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--terrain-dir", required=True, help="Dir with terrain .dds (e.g. game_terrain/)"
    )
    parser.add_argument("--output", required=True, help="Output dir for background PNGs")
    parser.add_argument(
        "--count", type=int, default=120, help="How many backgrounds to build (default: 120)"
    )
    parser.add_argument(
        "--size", type=int, nargs=2, default=[1280, 720], help="WxH (default: 1280 720)"
    )
    parser.add_argument(
        "--blend-prob",
        type=float,
        default=0.35,
        help="Chance of blending two terrains (default: 0.35)",
    )
    parser.add_argument(
        "--zoom-max",
        type=float,
        default=1.6,
        help="Max texture zoom; lower = smaller/distant features (default: 1.6)",
    )
    parser.add_argument(
        "--soft-fraction",
        type=float,
        default=0.35,
        help="Fraction of backgrounds heavily muted so objects pop (default: 0.35)",
    )
    parser.add_argument(
        "--blur",
        type=float,
        default=1.6,
        help="Max blur radius for textured backgrounds (default: 1.6)",
    )
    parser.add_argument(
        "--soft-blur",
        type=float,
        default=9.0,
        help="Max blur radius for the soft/muted fraction (default: 9.0)",
    )
    parser.add_argument(
        "--include-water", action="store_true", help="Include pure-water textures (g_wt*)"
    )
    parser.add_argument(
        "--only-water",
        action="store_true",
        help="Build ONLY water backgrounds (g_wt*) — for naval/fish water scenes",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args(namespace=_BuildBackgroundsArgs())

    terrain_dir = Path(args.terrain_dir).resolve()
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    size = (args.size[0], args.size[1])
    config = SofteningConfig(
        blend_prob=args.blend_prob,
        zoom_max=args.zoom_max,
        soft_fraction=args.soft_fraction,
        blur_max=args.blur,
        soft_blur_max=args.soft_blur,
    )

    print(f"Terrain source: {terrain_dir}")
    textures = load_textures(
        terrain_dir, include_water=args.include_water, only_water=args.only_water
    )
    if not textures:
        print("Error: no usable terrain textures found.")
        return 1
    print(
        f"Loaded {len(textures)} terrain textures. Building {args.count} backgrounds -> {out_dir}"
    )

    rng = random.Random(args.seed)
    for index in range(args.count):
        background = make_background(textures, size, rng, config)
        background.save(out_dir / f"bg_{index:04d}.png")
    print(f"Done: {args.count} background PNGs in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
