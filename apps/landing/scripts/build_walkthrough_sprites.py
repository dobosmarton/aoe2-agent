"""Build the AoE2:DE sprites and terrain textures for the home page village.

Decodes the chosen .sld files from a local game install (sld.py), bakes each frame's
shadow under its graphics, paints the player-colour mask blue (player 1), packs every
frame into one atlas and writes, under public/walkthrough/aoe2/:

- sprites.webp: the atlas
- sprites.json: per sprite, its frames as atlas rects plus the ground anchor (the
  frame's hotspot), in atlas pixels
- terrain textures (grass, dirt, farm, forest floor) as webp

With --frame and --turn-data it also places the village: the recorded frame shows
these same sprites at 1:1, so each detected building and mine is template-matched
against the frame (opaque, non-player-colour pixels only) to find the variant and
the exact screen position of its anchor. Each stand of trees is filled one tree per
tile where the frame shows canopy, with sprites of the frame's crown colours. The
result goes into turn-data.json as `world`.

    uv run python apps/landing/scripts/build_walkthrough_sprites.py \
        --graphics game_graphics --terrain game_terrain/textures/2x \
        --frame <frame.jpg> --turn-data apps/landing/src/components/walkthrough/turn-data.json

The game files themselves are not in the repository; the output is committed.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias, TypedDict, cast

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageFilter

from build_walkthrough_data import TurnData
from sld import LUMA_WEIGHTS, Frame, Layer, luminance, read_sld, tint

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from build_walkthrough_data import Box, TurnEntity

OUT = Path(__file__).resolve().parent.parent / "public/walkthrough/aoe2"
PLAYER_BLUE = (40, 90, 255)
SHADOW_ALPHA = 0.55
# Facings kept for animated units, out of the 16 a unit file stores.
UNIT_FACINGS = 16
FACINGS = (2, 6, 10, 14)
FRAMES_PER_ANIMATION = 12
# Forest in the frame (x0, y0, x1, y1 px) that no detection covers: its edge shows
# below the resource bar and the rest is under it, too hidden to match sprites. The
# lumber camp works it; its lumberjacks are the villagers carrying wood along the top.
HIDDEN_FORESTS = ((590, 100, 1030, 140),)

# A sprite pixel takes part in template matching when it is this opaque and its
# player-colour mask is below PLAYER_MASK_LIMIT.
OPAQUE_ALPHA = 200
PLAYER_MASK_LIMIT = 40
# Some frames of a tree file are empty or a stray leaf: fewer matchable pixels than this.
MIN_STILL_PIXELS = 300
# Below this many matchable pixels a template match means nothing.
MIN_MATCH_PIXELS = 50

ATLAS_WIDTH = 2048
ATLAS_PADDING = 2

TERRAIN_FILES = {
    "grass": "g_gr2",
    "grass_patchy": "g_gr3",
    "dirt": "g_gr5",
    "farm": "g_fm1",
    "forest": "g_for",
}
TERRAIN_SIZE = 512

# Detector class -> prefixes of the sprites that may show it. The detector's "tower"
# class also covers the outpost.
SPRITE_PREFIXES = {
    "town_center": ("town_center",),
    "house": ("house",),
    "mill": ("mill",),
    "lumber_camp": ("lumber_camp",),
    "mining_camp": ("mining_camp",),
    "tower": ("tower", "outpost"),
    "gold_mine": ("gold",),
    "stone_mine": ("stone",),
}
# How far (px) around the detection box the first pass looks for a sprite.
MATCH_REACH = 48
SCORE_DIGITS = 3
# First matches this good, of whole-tile buildings (by id prefix), fix the tile grid.
GRID_ANCHOR_SCORE = 0.95
GRID_ANCHOR_PREFIXES = ("house", "lumber", "mining", "town")
TILE_WIDTHS = range(80, 113, 4)
# Footprint in tiles of the buildings that stand on the tile grid.
FOOTPRINT_TILES = {
    "town_center": 4,
    "house": 2,
    "mill": 2,
    "lumber_camp": 2,
    "mining_camp": 2,
    "tower": 1,
}

FARM_TILES = 3
# Ground distance (tiles) under which two farms would overlap.
FARM_MIN_SPACING = 2.9
# A field's centre sits this share of its detection box's height below the box centre.
FARM_CENTRE_DROP = 0.07

TREE_PREFIXES = ("pine", "oak", "autumn")
# The crown is the top of a tree sprite; the rest is trunk.
CROWN_SHARE = 0.65
# Below this share of the frame height the frame shows the interface, not the map.
INTERFACE_TOP = 0.83
# Share of canopy a tile needs above it and at its foot to hold a tree.
CROWN_CANOPY_MIN = 0.5
FOOT_CANOPY_MIN = 0.4
# Each tree is picked at random from this many sprites closest in crown colour.
TREE_CHOICES = 3

VISIBILITY_CELL = 24
# A cell measures the fog when grass covers more than this share of it.
MIN_GRASS_SHARE = 0.25
# The brightness of fully lit grass, as a percentile of the cells' grass brightness.
LIT_PERCENTILE = 85
MAX_FILL_STEPS = 20
# Cells darker than this are unexplored.
UNEXPLORED_LUMINANCE = 18
VISIBILITY_BLUR = 1.2

# Window (px) of the local brightness and texture measures of forest_mask.
LOCAL_WINDOW = (9, 9)

FloatImage: TypeAlias = NDArray[np.float32]
Point: TypeAlias = tuple[int, int]
# x, y, width, height, anchor x, anchor y, in atlas pixels.
AtlasFrame: TypeAlias = tuple[int, int, int, int, int, int]
# Whole-tile positions: tile corners for even footprints, tile centres for odd ones.
GridAnchors: TypeAlias = Literal["corners", "centres"]


class SpriteAtlas(TypedDict):
    """sprites.json."""

    size: list[int]
    sprites: dict[str, list[AtlasFrame]]


class WorldItem(TypedDict):
    id: str
    sprite: str
    x: int
    y: int
    score: float


class World(TypedDict):
    tile: int
    origin: list[int]
    items: list[WorldItem]
    forests: list[list[int]]


class PlacedTurnData(TurnData):
    world: World


@dataclass(frozen=True, slots=True)
class Sprite:
    image: Image.Image
    anchor: Point
    # For matching against the frame: the graphics layer alone (RGB), the mask of its
    # opaque non-player-colour pixels, and its top-left relative to the anchor.
    rgb: FloatImage
    mask: NDArray[np.uint8]
    offset: Point


SpriteSet: TypeAlias = dict[str, list[Sprite]]


@dataclass(frozen=True, slots=True)
class Match:
    score: float
    x: int
    y: int


@dataclass(frozen=True, slots=True)
class TreeSpot:
    x: int
    y: int
    # Tree sprites, closest crown colour first.
    ranked: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Grid:
    """The game's tile grid on screen: a 2:1 lattice of tile/2 by tile/4 px steps
    around `origin`, a tile corner."""

    origin: Point
    tile: int

    @property
    def step(self) -> tuple[float, float]:
        return self.tile / 2, self.tile / 4

    def to_ground(self, x: float, y: float) -> tuple[float, float]:
        """Screen px -> ground position in tiles."""
        u, v = (x - self.origin[0]) / (self.tile / 2), (y - self.origin[1]) / (self.tile / 4)
        return (u + v) / 2, (v - u) / 2

    def points_near(
        self, guess: tuple[float, float], anchors: GridAnchors, reach_tiles: float
    ) -> Iterator[Point]:
        """Screen anchors of whole-tile positions near `guess`."""
        step_x, step_y = self.step
        u0 = round((guess[0] - self.origin[0]) / step_x)
        v0 = round((guess[1] - self.origin[1]) / step_y)
        reach = int(reach_tiles * 2) + 1
        us = range(u0 - reach, u0 + reach + 1)
        vs = range(v0 - 2 * reach, v0 + 2 * reach + 1)
        return self._lattice(us, vs, anchors)

    def points_around(self, box: Box, anchors: GridAnchors) -> Iterator[Point]:
        """Screen anchors of whole-tile positions over `box`, one step beyond it."""
        step_x, step_y = self.step
        x0, y0, x1, y1 = box
        us = range(int((x0 - self.origin[0]) / step_x) - 1, int((x1 - self.origin[0]) / step_x) + 2)
        vs = range(int((y0 - self.origin[1]) / step_y) - 1, int((y1 - self.origin[1]) / step_y) + 2)
        return self._lattice(us, vs, anchors)

    def _lattice(self, us: range, vs: range, anchors: GridAnchors) -> Iterator[Point]:
        parity = 1 if anchors == "centres" else 0
        step_x, step_y = self.step
        for u in us:
            for v in vs:
                if (u + v) % 2 == parity:
                    yield int(self.origin[0] + u * step_x), int(self.origin[1] + v * step_y)


@dataclass(frozen=True, slots=True)
class Options:
    graphics: Path
    terrain: Path
    frame: Path | None
    turn_data: Path | None


# --- Sprites ---------------------------------------------------------------------


def compose(frame: Frame) -> Sprite | None:
    """Shadow, then graphics, on the union of their bounds; anchor = hotspot."""
    main = frame.main
    if main is None:
        return None
    graphics = tint(frame, PLAYER_BLUE)
    layers = [main] if frame.shadow is None else [main, frame.shadow]
    x0 = min(layer.x1 for layer in layers)
    y0 = min(layer.y1 for layer in layers)
    x1 = max(layer.x1 + layer.pixels.shape[1] for layer in layers)
    y1 = max(layer.y1 + layer.pixels.shape[0] for layer in layers)
    canvas = Image.new("RGBA", (x1 - x0, y1 - y0), (0, 0, 0, 0))
    if frame.shadow is not None:
        shadow = _shadow_image(frame.shadow)
        canvas.alpha_composite(shadow, (frame.shadow.x1 - x0, frame.shadow.y1 - y0))
    canvas.alpha_composite(Image.fromarray(graphics, "RGBA"), (main.x1 - x0, main.y1 - y0))
    bbox = canvas.getbbox()
    if bbox is None:
        return None
    hotspot_x, hotspot_y = frame.hotspot
    return Sprite(
        image=canvas.crop(bbox),
        anchor=(hotspot_x - x0 - bbox[0], hotspot_y - y0 - bbox[1]),
        rgb=graphics[:, :, :3].astype(np.float32),
        mask=_matchable_pixels(frame, graphics),
        offset=(main.x1 - hotspot_x, main.y1 - hotspot_y),
    )


def _shadow_image(shadow: Layer) -> Image.Image:
    rgba = np.zeros((*shadow.pixels.shape, 4), np.uint8)
    rgba[:, :, 3] = (shadow.pixels.astype(np.float32) * SHADOW_ALPHA).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def _matchable_pixels(frame: Frame, graphics: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """1 on the opaque, non-player-colour pixels of the graphics layer."""
    solid = graphics[:, :, 3] > OPAQUE_ALPHA
    if frame.player is not None and frame.player.pixels.shape == solid.shape:
        solid &= frame.player.pixels < PLAYER_MASK_LIMIT
    return solid.astype(np.uint8)


def _decode(graphics: Path, file: str) -> list[Frame]:
    return read_sld(graphics / f"{file}_x1.sld")


def _compose_all(frames: Iterable[Frame]) -> list[Sprite]:
    return [sprite for sprite in map(compose, frames) if sprite is not None]


def still(graphics: Path, file: str, name: str, frames: Sequence[int] = (0,)) -> SpriteSet:
    """One single-frame sprite per usable frame of `frames`, numbered when several."""
    decoded = _decode(graphics, file)
    sprites = [
        sprite
        for sprite in _compose_all(decoded[i] for i in frames)
        if sprite.mask.sum() > MIN_STILL_PIXELS
    ]
    if len(frames) == 1:
        return {name: sprites} if sprites else {}
    return {f"{name}_{number}": [sprite] for number, sprite in enumerate(sprites)}


def animation(graphics: Path, file: str, name: str, facings: Sequence[int] = FACINGS) -> SpriteSet:
    """Per facing, an animation of up to FRAMES_PER_ANIMATION frames."""
    decoded = _decode(graphics, file)
    per_facing = len(decoded) // UNIT_FACINGS
    # Frames spread evenly over the whole cycle, so the loop closes without a jump.
    count = min(FRAMES_PER_ANIMATION, per_facing)
    picks = [round(k * per_facing / count) for k in range(count)]
    return {
        f"{name}_d{facing}": _compose_all(decoded[facing * per_facing + i] for i in picks)
        for facing in facings
    }


def building_sprites(graphics: Path) -> SpriteSet:
    # Magyars build in the Eastern European style, the game's "slav" set; the recorded
    # game is in the Feudal Age.
    return {
        **still(graphics, "b_slav_town_center_age2", "town_center"),
        **still(graphics, "b_slav_house_age2", "house", [0, 1, 2]),
        **still(graphics, "b_slav_mill_age2", "mill"),
        **still(graphics, "b_slav_lumber_camp_age2", "lumber_camp"),
        **still(graphics, "b_slav_mining_camp_age2", "mining_camp"),
        **still(graphics, "b_slav_tower_age2", "tower"),
        # The detector's "tower" class also covers the outpost, a lookout on stilts.
        **still(graphics, "b_dark_outpost_age1", "outpost"),
        **still(graphics, "b_foundation_2x2_rubble", "foundation"),
    }


def nature_sprites(graphics: Path) -> SpriteSet:
    return {
        # Trees of the recorded map's forests: green pines (not the dead, rust or snowy
        # ones), and oaks green, bare and autumn.
        **still(graphics, "n_tree_pine", "pine", [*range(3, 15), 21, 22, 23]),
        **still(graphics, "n_tree_oak", "oak", list(range(12))),
        **still(graphics, "n_tree_autumn_oak", "autumn", list(range(12))),
        **still(graphics, "n_mine_gold", "gold", list(range(0, 7, 2))),
        **still(graphics, "n_mine_stone", "stone", list(range(0, 7, 2))),
        # Ground cover the map is dotted with: grass tufts, bushes and shrubs.
        **still(graphics, "n_plant_grass_green", "tuft", list(range(8))),
        **still(graphics, "n_plant_bush_green", "bush", list(range(8))),
        **still(graphics, "n_plant_shrub_green", "shrub", list(range(8))),
    }


def villager_sprites(graphics: Path, gender: Literal["male", "female"]) -> SpriteSet:
    prefix = f"vil{gender[0]}"
    sprites = {
        **animation(graphics, f"u_vil_{gender}_lumberjack_taskA", f"{prefix}_chop"),
        **animation(graphics, f"u_vil_{gender}_farmer_taskA", f"{prefix}_farm"),
        **animation(graphics, f"u_vil_{gender}_miner_gold_taskA", f"{prefix}_mine"),
        **animation(graphics, f"u_vil_{gender}_builder_taskA", f"{prefix}_build"),
        **animation(graphics, f"u_vil_{gender}_villager_walkA", f"{prefix}_walk", range(16)),
    }
    # Walking back to a drop site with a load, in eight facings. Men have no farmer
    # carry walk in the game files; the forager's basket of food stands in.
    food = "farmer" if gender == "female" else "forager"
    loads = (
        ("wood", "lumberjack"),
        ("gold", "miner_gold"),
        ("stone", "miner_stone"),
        ("food", food),
    )
    for load, file in loads:
        sprites |= animation(
            graphics, f"u_vil_{gender}_{file}_carrywalkA", f"{prefix}_carry_{load}", range(0, 16, 2)
        )
    return sprites


def animal_sprites(graphics: Path) -> SpriteSet:
    return {
        **animation(graphics, "a_hunt_deer_idleA", "deer", [2, 10]),
        **animation(graphics, "a_hunt_boar_idleA", "boar", [2, 10]),
    }


def load_sprites(graphics: Path) -> SpriteSet:
    return {
        **building_sprites(graphics),
        **nature_sprites(graphics),
        **villager_sprites(graphics, "male"),
        **villager_sprites(graphics, "female"),
        **animal_sprites(graphics),
    }


# --- Atlas -----------------------------------------------------------------------


def pack(
    sprites: SpriteSet, width: int = ATLAS_WIDTH, pad: int = ATLAS_PADDING
) -> tuple[Image.Image, dict[str, list[AtlasFrame]]]:
    """Shelf packer, tallest first. Returns the atlas and the frame rects."""
    keys = [(name, i) for name, frames in sprites.items() for i in range(len(frames))]
    keys.sort(key=lambda key: -sprites[key[0]][key[1]].image.height)
    sizes = [sprites[name][i].image.size for name, i in keys]
    positions, height = _shelf_layout(sizes, width, pad)
    placed = dict(zip(keys, positions, strict=True))
    atlas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    for (name, i), position in placed.items():
        atlas.alpha_composite(sprites[name][i].image, position)
    rects = {
        name: [_atlas_frame(sprite, placed[(name, i)]) for i, sprite in enumerate(frames)]
        for name, frames in sprites.items()
    }
    return atlas, rects


def _shelf_layout(sizes: list[tuple[int, int]], width: int, pad: int) -> tuple[list[Point], int]:
    """Top-left of each rect, filling shelves left to right, and the atlas height
    (rounded up to a multiple of 4)."""
    x = y = shelf = 0
    positions: list[Point] = []
    for w, h in sizes:
        if x + w + pad > width:
            x, y, shelf = 0, y + shelf + pad, 0
        positions.append((x, y))
        x += w + pad
        shelf = max(shelf, h)
    return positions, (y + shelf + 3) // 4 * 4


def _atlas_frame(sprite: Sprite, position: Point) -> AtlasFrame:
    x, y = position
    return (x, y, sprite.image.width, sprite.image.height, sprite.anchor[0], sprite.anchor[1])


def write_atlas(sprites: SpriteSet) -> Image.Image:
    atlas, rects = pack(sprites)
    OUT.mkdir(parents=True, exist_ok=True)
    atlas.save(OUT / "sprites.webp", quality=92, method=6)
    manifest: SpriteAtlas = {"size": list(atlas.size), "sprites": rects}
    (OUT / "sprites.json").write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
    return atlas


def write_terrain(terrain: Path) -> None:
    for name, file in TERRAIN_FILES.items():
        image = Image.open(terrain / f"{file}.dds").convert("RGB")
        image = image.resize((TERRAIN_SIZE, TERRAIN_SIZE), Image.Resampling.LANCZOS)
        image.save(OUT / f"{name}.webp", quality=84, method=6)


# --- Template matching -----------------------------------------------------------


def _template_mask(sprite: Sprite) -> FloatImage:
    return np.repeat(sprite.mask[:, :, None], 3, axis=2).astype(np.float32)


def _inside(shape: tuple[int, ...], left: int, top: int, right: int, bottom: int) -> bool:
    """Whether the rect lies within an image of `shape`."""
    return left >= 0 and top >= 0 and right <= shape[1] and bottom <= shape[0]


def match(frame_rgb: FloatImage, sprite: Sprite, guess: tuple[float, float], reach: int) -> Match:
    """Best anchor position for `sprite` within `reach` px of `guess`."""
    height, width = sprite.mask.shape
    offset_x, offset_y = sprite.offset
    left = int(guess[0] + offset_x - reach)
    top = int(guess[1] + offset_y - reach)
    x0, y0 = max(0, left), max(0, top)
    x1 = min(frame_rgb.shape[1], left + width + 2 * reach)
    y1 = min(frame_rgb.shape[0], top + height + 2 * reach)
    region = frame_rgb[y0:y1, x0:x1]
    too_small = region.shape[0] < height or region.shape[1] < width
    if too_small or sprite.mask.sum() < MIN_MATCH_PIXELS:
        return Match(-1.0, int(guess[0]), int(guess[1]))
    scores = cv2.matchTemplate(region, sprite.rgb, cv2.TM_CCORR_NORMED, mask=_template_mask(sprite))
    scores = np.nan_to_num(scores, nan=-1.0, posinf=-1.0, neginf=-1.0)
    _, best, _, at = cv2.minMaxLoc(scores)
    return Match(float(best), x0 + at[0] - offset_x, y0 + at[1] - offset_y)


def match_on_grid(
    frame_rgb: FloatImage,
    sprite: Sprite,
    guess: tuple[float, float],
    grid: Grid,
    anchors: GridAnchors,
) -> Match:
    """The best grid position for `sprite` near `guess`, scored like `match`."""
    height, width = sprite.mask.shape
    mask = _template_mask(sprite)
    best = Match(-1.0, int(guess[0]), int(guess[1]))
    for anchor_x, anchor_y in grid.points_near(guess, anchors, reach_tiles=1.0):
        left, top = anchor_x + sprite.offset[0], anchor_y + sprite.offset[1]
        if not _inside(frame_rgb.shape, left, top, left + width, top + height):
            continue
        window = frame_rgb[top : top + height, left : left + width]
        scores = cv2.matchTemplate(window, sprite.rgb, cv2.TM_CCORR_NORMED, mask=mask)
        score = float(np.nan_to_num(scores[0, 0], nan=-1.0))
        if score > best.score:
            best = Match(score, anchor_x, anchor_y)
    return best


def sprite_names(sprites: SpriteSet, prefix: str) -> list[str]:
    return [name for name in sprites if name == prefix or name.startswith(prefix + "_")]


def pixel_box(entity: TurnEntity, frame_size: Point) -> Box:
    w, h = frame_size
    x, y = entity["x"], entity["y"]
    return (x * w, y * h, (x + entity["w"]) * w, (y + entity["h"]) * h)


def _match_in_box(frame_rgb: FloatImage, sprite: Sprite, box: Box) -> Match:
    # Guess: the graphics' box centred on the detection box.
    mask_h, mask_w = sprite.mask.shape
    guess = (
        (box[0] + box[2]) / 2 - sprite.offset[0] - mask_w / 2,
        (box[1] + box[3]) / 2 - sprite.offset[1] - mask_h / 2,
    )
    return match(frame_rgb, sprite, guess, reach=MATCH_REACH)


def match_buildings(
    frame_rgb: FloatImage, sprites: SpriteSet, entities: list[TurnEntity], frame_size: Point
) -> list[WorldItem]:
    """First pass: for each detected building and mine, the sprite variant that matches
    the frame best around its box, and where."""
    items: list[WorldItem] = []
    for entity in entities:
        prefixes = SPRITE_PREFIXES.get(entity["cls"])
        if prefixes is None:
            continue
        box = pixel_box(entity, frame_size)
        names = [name for prefix in prefixes for name in sprite_names(sprites, prefix)]
        matches = [(name, _match_in_box(frame_rgb, sprites[name][0], box)) for name in names]
        name, best = max(matches, key=lambda candidate: candidate[1].score)
        items.append(_world_item(entity["id"], name, best))
    return items


def _world_item(entity_id: str, sprite: str, found: Match) -> WorldItem:
    return {
        "id": entity_id,
        "sprite": sprite,
        "x": found.x,
        "y": found.y,
        "score": round(found.score, SCORE_DIGITS),
    }


def fit_grid(items: list[WorldItem]) -> Grid:
    """The anchors of whole-tile buildings fall on the game's tile grid. Its pitch is
    the tile width; the Town Center's anchor is the origin."""
    town_center = next(item for item in items if item["id"].startswith("town_center"))
    origin = (town_center["x"], town_center["y"])
    anchors = [
        item
        for item in items
        if item["score"] > GRID_ANCHOR_SCORE and item["id"].split("_")[0] in GRID_ANCHOR_PREFIXES
    ]
    return Grid(origin=origin, tile=grid_pitch(anchors, origin))


def grid_pitch(items: list[WorldItem], origin: Point) -> int:
    """The tile width whose half-tile grid (w/2 by w/4 px) fits the most anchors; on a
    tie, the widest."""

    def fits(width: int) -> int:
        return sum(
            1
            for item in items
            if (item["x"] - origin[0]) % (width // 2) == 0
            and (item["y"] - origin[1]) % (width // 4) == 0
        )

    return max((fits(width), width) for width in TILE_WIDTHS)[1]


def snap_to_grid(
    frame_rgb: FloatImage,
    sprites: SpriteSet,
    items: list[WorldItem],
    entities: list[TurnEntity],
    grid: Grid,
) -> list[WorldItem]:
    """Second pass: buildings may only stand on the tile grid, so re-match each one at
    the grid points near its first match. Mines keep their first match."""
    classes = {entity["id"]: entity["cls"] for entity in entities}
    snapped: list[WorldItem] = []
    for item in items:
        footprint = FOOTPRINT_TILES.get(classes[item["id"]])
        if footprint is None:
            snapped.append(item)
            continue
        sprite = sprites[item["sprite"]][0]
        guess = (item["x"], item["y"])
        found = match_on_grid(frame_rgb, sprite, guess, grid, _anchors_for(footprint))
        snapped.append(_world_item(item["id"], item["sprite"], found))
    return snapped


def _anchors_for(footprint: int) -> GridAnchors:
    return "centres" if footprint % 2 == 1 else "corners"


# --- Farms -----------------------------------------------------------------------


def place_farms(
    frame_rgb: FloatImage, entities: list[TurnEntity], frame_size: Point, grid: Grid
) -> list[WorldItem]:
    """Farms on the tile grid, most confident detection first, none overlapping."""
    # Farm soil is reddish where grass is green.
    soil = frame_rgb[:, :, 0] - frame_rgb[:, :, 1]
    farms = sorted(
        (entity for entity in entities if entity["cls"] == "farm"),
        key=lambda entity: -(entity["conf"] or 0),
    )
    taken: list[Point] = []
    items: list[WorldItem] = []
    for farm in farms:
        spot = place_farm(soil, pixel_box(farm, frame_size), grid, taken)
        if spot is not None:
            taken.append(spot)
            items.append(
                {"id": farm["id"], "sprite": "farm", "x": spot[0], "y": spot[1], "score": 1.0}
            )
    return items


def place_farm(soil: FloatImage, box: Box, grid: Grid, taken: list[Point]) -> Point | None:
    """Grid point whose 3x3-tile diamond covers the most farm soil, without overlapping
    a farm already placed."""
    centre = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2 + (box[3] - box[1]) * FARM_CENTRE_DROP)
    half_w, half_h = FARM_TILES / 2 * grid.tile, FARM_TILES / 4 * grid.tile
    placed = [grid.to_ground(*spot) for spot in taken]
    best: Match | None = None
    for anchor_x, anchor_y in grid.points_near(centre, "centres", reach_tiles=1.5):
        if _overlaps_farm(grid.to_ground(anchor_x, anchor_y), placed):
            continue
        score = _diamond_mean(soil, (anchor_x, anchor_y), half_w, half_h)
        if score is not None and (best is None or score > best.score):
            best = Match(score, anchor_x, anchor_y)
    return None if best is None else (best.x, best.y)


def _overlaps_farm(ground: tuple[float, float], farms: list[tuple[float, float]]) -> bool:
    gx, gz = ground
    return any(
        abs(gx - fx) < FARM_MIN_SPACING and abs(gz - fz) < FARM_MIN_SPACING for fx, fz in farms
    )


def _diamond_mean(values: FloatImage, centre: Point, half_w: float, half_h: float) -> float | None:
    """Mean of `values` inside the diamond around `centre`; None if it leaves the image."""
    cx, cy = centre
    x0, x1 = int(cx - half_w), int(cx + half_w)
    y0, y1 = int(cy - half_h), int(cy + half_h)
    if not _inside(values.shape, x0, y0, x1, y1):
        return None
    ys, xs = np.mgrid[y0:y1, x0:x1]
    inside = np.abs(xs - cx) / half_w + np.abs(ys - cy) / half_h <= 1
    return float(values[y0:y1, x0:x1][inside].mean())


# --- Trees -----------------------------------------------------------------------


# OpenCV's stubs type its results loosely; these keep float32 in, float32 out.
def _blur(image: FloatImage, size: tuple[int, int]) -> FloatImage:
    return np.asarray(cv2.blur(image, size), dtype=np.float32)


def _resize(
    image: FloatImage, size: tuple[int, int], interpolation: int = cv2.INTER_LINEAR
) -> FloatImage:
    return np.asarray(cv2.resize(image, size, interpolation=interpolation), dtype=np.float32)


def forest_mask(frame_rgb: FloatImage) -> FloatImage:
    """1 where the frame shows tree canopy, 0 on open ground.

    Canopy is darker than the grass around it (crowns shade each other) or, for
    autumn oaks, orange and busy where bare dirt is orange and smooth. The grass
    brightness is measured locally because the fog of war dims whole areas."""
    r, g, b = frame_rgb[..., 0], frame_rgb[..., 1], frame_rgb[..., 2]
    lum = luminance(frame_rgb)
    local_lum = _blur(lum, LOCAL_WINDOW)
    grass = ((g > r * 1.05) & (g > b * 1.4) & (lum > 60)).astype(np.float32)
    reference = _local_grass_brightness(lum, grass)
    deviation = np.sqrt(np.maximum(_blur(lum * lum, LOCAL_WINDOW) - local_lum**2, 0))
    orange = (_blur(r, LOCAL_WINDOW) > _blur(g, LOCAL_WINDOW) * 1.05) & (deviation > 22)
    canopy = ((local_lum < 0.62 * reference) | orange).astype(np.float32)
    mask = _blur(canopy, (15, 15)) > 0.4
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    return closed.astype(np.float32)


def _local_grass_brightness(lum: FloatImage, grass: FloatImage) -> FloatImage:
    """Grass brightness around each point: a widening window until it holds grass."""
    h, w = lum.shape[:2]
    small = (w // 8, h // 8)
    small_lum = _resize(lum * grass, small, cv2.INTER_AREA)
    small_cover = _resize(grass, small, cv2.INTER_AREA)
    reference = np.full(small_lum.shape, np.nan, np.float32)
    for k in (25, 51, 101, 201):
        cover = _blur(small_cover, (k, k))
        brightness = _blur(small_lum, (k, k)) / np.maximum(cover, 1e-4)
        unfilled = np.isnan(reference) & (cover > 0.05)
        reference[unfilled] = brightness[unfilled]
    return _resize(np.nan_to_num(reference, nan=140.0), (w, h))


def crown_colour(px: NDArray[np.float32]) -> NDArray[np.float64]:
    """Red and blue relative to green: tells pine, green oak and autumn oak apart
    and, as a ratio, holds up under the fog and the shade inside a forest."""
    px = px.reshape(-1, 3).astype(np.float32) + 1
    return np.array([np.median(px[:, 0] / px[:, 1]), np.median(px[:, 2] / px[:, 1])])


def sprite_crown_colour(sprite: Sprite) -> NDArray[np.float64]:
    crown = sprite.mask.astype(bool)
    crown[int(crown.shape[0] * CROWN_SHARE) :] = False
    return crown_colour(sprite.rgb[crown])


def place_forests(
    frame_rgb: FloatImage,
    sprites: SpriteSet,
    entities: list[TurnEntity],
    frame_size: Point,
    grid: Grid,
) -> list[WorldItem]:
    """Forests: one tree per tile, wherever the frame shows canopy inside a tree box."""
    canopy = forest_mask(frame_rgb)
    crowns = {
        name: sprite_crown_colour(sprites[name][0])
        for prefix in TREE_PREFIXES
        for name in sprite_names(sprites, prefix)
    }
    stands = [entity for entity in entities if entity["cls"] == "tree"]
    items: list[WorldItem] = []
    for seed, stand in enumerate(stands):
        spots = tree_spots(frame_rgb, canopy, crowns, pixel_box(stand, frame_size), grid)
        items.extend(choose_trees(spots, stand["id"], seed))
    return items


def tree_spots(
    frame_rgb: FloatImage,
    canopy: FloatImage,
    crowns: dict[str, NDArray[np.float64]],
    box: Box,
    grid: Grid,
) -> list[TreeSpot]:
    """Fill a tree box the way the game fills a forest: one tree per tile centre,
    wherever the frame shows canopy above the tile and no grass at its foot. Each spot
    ranks the tree sprites by how close their crown colour is to the frame's there."""
    h, w = frame_rgb.shape[:2]
    x0, y0, x1, y1 = box
    spots: list[TreeSpot] = []
    for x, y in grid.points_around(box, "centres"):
        if not (x0 <= x <= x1 and y0 + 30 <= y <= y1 + 10) or y > INTERFACE_TOP * h:
            continue
        crown = (slice(max(0, y - 100), y - 30), slice(max(0, x - 20), min(w, x + 20)))
        foot = (slice(y - 12, min(h, y + 6)), slice(max(0, x - 16), min(w, x + 16)))
        if canopy[crown].mean() < CROWN_CANOPY_MIN or canopy[foot].mean() < FOOT_CANOPY_MIN:
            continue
        spots.append(TreeSpot(x, y, _rank_by_crown(crowns, crown_colour(frame_rgb[crown]))))
    return spots


def _rank_by_crown(
    crowns: dict[str, NDArray[np.float64]], colour: NDArray[np.float64]
) -> tuple[str, ...]:
    def distance(name: str) -> float:
        return float(np.abs(crowns[name] - colour).sum())

    return tuple(sorted(crowns, key=distance))


def choose_trees(spots: list[TreeSpot], entity_id: str, seed: int) -> list[WorldItem]:
    """A sprite per spot, at random among the closest in crown colour. The whole stand
    is either conifer or broadleaf, as the map's forests are; an orange crown in a pine
    stand is still an autumn oak."""
    rnd = random.Random(seed)
    conifer = sum(spot.ranked[0].startswith("pine") for spot in spots) >= len(spots) / 2
    items: list[WorldItem] = []
    for spot in spots:
        if conifer and spot.ranked[0].startswith("autumn"):
            fits = list(spot.ranked)
        else:
            fits = [name for name in spot.ranked if name.startswith("pine") == conifer]
        sprite = rnd.choice(fits[:TREE_CHOICES])
        items.append({"id": entity_id, "sprite": sprite, "x": spot.x, "y": spot.y, "score": 1.0})
    return items


# --- Fog of war ------------------------------------------------------------------


def visibility(frame_rgb: FloatImage, cell: int = VISIBILITY_CELL) -> Image.Image:
    """How lit the map is across the frame, one pixel per `cell` px: 1 where the player
    sees, lower under the fog of war (explored, not in view), 0 where unexplored.

    The fog darkens everything by the same amount, so the brightness of the grass
    alone measures it: per cell, the median brightness of grass-coloured pixels,
    relative to the brightest grass. Cells without grass (buildings, fields) take
    their neighbours' value; near-black cells are unexplored."""
    rows, columns = frame_rgb.shape[0] // cell, frame_rgb.shape[1] // cell
    cells = frame_rgb[: rows * cell, : columns * cell].reshape(rows, cell, columns, cell, 3)
    cells = cells.transpose(0, 2, 1, 3, 4).reshape(rows, columns, -1, 3)
    r, g, b = cells[..., 0], cells[..., 1], cells[..., 2]
    red_weight, green_weight, blue_weight = LUMA_WEIGHTS
    lum = red_weight * r + green_weight * g + blue_weight * b
    grass = (g > r * 1.1) & (g > b * 1.4) & (lum > 20)
    level = _grass_brightness(lum, grass, min_pixels=cell * cell * MIN_GRASS_SHARE)
    lit = np.nanpercentile(level, LIT_PERCENTILE)
    level = np.nan_to_num(_fill_from_neighbours(level), nan=float(lit))
    lit_share = np.clip(level / lit, 0, 1)
    lit_share[np.median(lum, axis=2) < UNEXPLORED_LUMINANCE] = 0
    image = Image.fromarray((lit_share * 255).astype(np.uint8), "L")
    return image.filter(ImageFilter.GaussianBlur(VISIBILITY_BLUR))


def _grass_brightness(lum: FloatImage, grass: NDArray[np.bool_], min_pixels: float) -> FloatImage:
    """Per cell, the median brightness of its grass pixels; NaN with too few of them."""
    rows, columns = grass.shape[:2]
    level = np.full((rows, columns), np.nan, np.float32)
    for y in range(rows):
        for x in range(columns):
            if grass[y, x].sum() > min_pixels:
                level[y, x] = np.median(lum[y, x][grass[y, x]])
    return level


def _fill_from_neighbours(level: FloatImage) -> FloatImage:
    """Fill NaN cells from their neighbours, widening until none are left."""
    filled = level.copy()
    rows, columns = filled.shape
    for _ in range(MAX_FILL_STEPS):
        holes = np.isnan(filled)
        if not holes.any():
            break
        padded = np.pad(filled, 1, constant_values=np.nan)
        stack = np.stack(
            [padded[dy : dy + rows, dx : dx + columns] for dy in range(3) for dx in range(3)]
        )
        filled[holes] = np.nanmean(stack, axis=0)[holes]
    return filled


# --- Placement -------------------------------------------------------------------


def load_turn_data(path: Path) -> TurnData:
    # Written by build_walkthrough_data.py; this script trusts its shape.
    return cast("TurnData", json.loads(path.read_text()))


def unique_spots(items: list[WorldItem]) -> list[WorldItem]:
    """Two detections of one object end up on the same spot: keep one."""
    seen: set[tuple[str, int, int]] = set()
    unique: list[WorldItem] = []
    for item in items:
        spot = (item["sprite"], item["x"], item["y"])
        if spot not in seen:
            seen.add(spot)
            unique.append(item)
    return unique


def place(sprites: SpriteSet, frame_path: Path, turn_path: Path) -> None:
    """Template-match buildings, mines and trees against the frame; write turn['world']."""
    frame_rgb = np.asarray(Image.open(frame_path).convert("RGB"), dtype=np.float32)
    turn = load_turn_data(turn_path)
    frame_size = (turn["frame"]["width"], turn["frame"]["height"])
    entities = turn["entities"]
    matched = match_buildings(frame_rgb, sprites, entities, frame_size)
    grid = fit_grid(matched)
    items = unique_spots(
        [
            *snap_to_grid(frame_rgb, sprites, matched, entities, grid),
            *place_forests(frame_rgb, sprites, entities, frame_size, grid),
            *place_farms(frame_rgb, entities, frame_size, grid),
        ]
    )
    world: World = {
        "tile": grid.tile,
        "origin": list(grid.origin),
        "items": items,
        "forests": [list(forest) for forest in HIDDEN_FORESTS],
    }
    placed: PlacedTurnData = {**turn, "world": world}
    turn_path.write_text(json.dumps(placed, indent=1) + "\n")
    visibility(frame_rgb).save(OUT / "visibility.png")
    buildings = [item for item in items if not item["id"].startswith("tree")]
    mean_score = np.mean([item["score"] for item in buildings])
    print(
        f"placed {len(buildings)} buildings and mines (mean score {mean_score:.3f}), "
        f"{len(items) - len(buildings)} trees"
    )


def parse_options() -> Options:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--graphics", type=Path, required=True, help="game graphics folder with .sld files"
    )
    parser.add_argument(
        "--terrain", type=Path, required=True, help="game terrain textures folder (.dds)"
    )
    parser.add_argument("--frame", type=Path, help="recorded frame to place the village from")
    parser.add_argument(
        "--turn-data", type=Path, help="turn-data.json to write the placements into"
    )
    args = parser.parse_args()
    return Options(
        graphics=args.graphics, terrain=args.terrain, frame=args.frame, turn_data=args.turn_data
    )


def main() -> None:
    options = parse_options()
    sprites = load_sprites(options.graphics)
    atlas = write_atlas(sprites)
    write_terrain(options.terrain)
    if options.frame is not None and options.turn_data is not None:
        place(sprites, options.frame, options.turn_data)
    frame_count = sum(len(frames) for frames in sprites.values())
    print(f"{len(sprites)} sprites, {frame_count} frames, atlas {atlas.size[0]}x{atlas.size[1]}")


if __name__ == "__main__":
    main()
