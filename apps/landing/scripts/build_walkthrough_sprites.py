"""Build the AoE2:DE sprites and terrain textures for the home page village.

Decodes the chosen .sld files from a local game install (sld.py), bakes each frame's
shadow under its graphics, paints the player-colour mask blue (player 1), packs every
frame into one atlas and writes, under public/walkthrough/aoe2/:

- sprites.webp: the atlas
- sprites.json: per sprite, its frames as atlas rects plus the ground anchor (the
  frame's hotspot), in atlas pixels
- terrain textures (grass, dirt, farm, forest floor) as webp

With --frame and --turn-data it also places the village: the recorded frame shows
these same sprites at 1:1, so each detected building, mine and stand of trees is
template-matched against the frame (opaque, non-player-colour pixels only) to find
the variant and the exact screen position of its anchor. The result goes into
turn-data.json as `world`.

    uv run python apps/landing/scripts/build_walkthrough_sprites.py \
        --graphics game_graphics --terrain game_terrain/textures/2x \
        --frame <frame.jpg> --turn-data apps/landing/src/components/walkthrough/turn-data.json

The game files themselves are not in the repository; the output is committed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from sld import Frame, read_sld, tint

OUT = Path(__file__).resolve().parent.parent / "public/walkthrough/aoe2"
PLAYER_BLUE = (40, 90, 255)
SHADOW_ALPHA = 0.55
# Facings kept for animated units, out of the 16 a unit file stores.
FACINGS = [2, 6, 10, 14]
FRAMES_PER_ANIMATION = 12


@dataclass
class Sprite:
    name: str
    image: Image.Image
    anchor: tuple[int, int]
    # For matching against the frame: the graphics layer alone (RGB), the mask of its
    # opaque non-player-colour pixels, and its top-left relative to the anchor.
    rgb: np.ndarray | None = None
    mask: np.ndarray | None = None
    offset: tuple[int, int] = (0, 0)


def compose(frame: Frame) -> Sprite | None:
    """Shadow, then graphics, on the union of their bounds; anchor = hotspot."""
    if frame.main is None:
        return None
    layers = [frame.main] + ([frame.shadow] if frame.shadow is not None else [])
    x0 = min(layer.x1 for layer in layers)
    y0 = min(layer.y1 for layer in layers)
    x1 = max(layer.x1 + layer.pixels.shape[1] for layer in layers)
    y1 = max(layer.y1 + layer.pixels.shape[0] for layer in layers)
    canvas = Image.new("RGBA", (x1 - x0, y1 - y0), (0, 0, 0, 0))
    if frame.shadow is not None:
        sh = np.zeros((*frame.shadow.pixels.shape, 4), np.uint8)
        sh[:, :, 3] = (frame.shadow.pixels.astype(np.float32) * SHADOW_ALPHA).astype(np.uint8)
        canvas.alpha_composite(Image.fromarray(sh, "RGBA"), (frame.shadow.x1 - x0, frame.shadow.y1 - y0))
    canvas.alpha_composite(Image.fromarray(tint(frame, PLAYER_BLUE), "RGBA"), (frame.main.x1 - x0, frame.main.y1 - y0))
    bbox = canvas.getbbox()
    if bbox is None:
        return None
    canvas = canvas.crop(bbox)
    main = tint(frame, PLAYER_BLUE)
    solid = main[:, :, 3] > 200
    if frame.player is not None and frame.player.pixels.shape == solid.shape:
        solid &= frame.player.pixels < 40
    return Sprite(
        "",
        canvas,
        (frame.hotspot[0] - x0 - bbox[0], frame.hotspot[1] - y0 - bbox[1]),
        rgb=main[:, :, :3].astype(np.float32),
        mask=solid.astype(np.uint8),
        offset=(frame.main.x1 - frame.hotspot[0], frame.main.y1 - frame.hotspot[1]),
    )


def is_green(sprite: Sprite) -> bool:
    """True for a living tree: its opaque pixels lean green, not brown or bare."""
    px = np.asarray(sprite.image, dtype=np.float32)
    solid = px[px[:, :, 3] > 200]
    return len(solid) > 0 and solid[:, 1].mean() > solid[:, 0].mean() * 1.08


def still(
    graphics: Path, file: str, name: str, frames: list[int] | None = None, keep=lambda sprite: True
) -> dict[str, list[Sprite]]:
    decoded = read_sld(graphics / f"{file}_x1.sld")
    picks = frames if frames is not None else [0]
    out = {}
    for i in picks:
        sprite = compose(decoded[i])
        if sprite and keep(sprite):
            out[f"{name}_{len(out)}" if len(picks) > 1 else name] = [sprite]
    return out


def animation(graphics: Path, file: str, name: str, facings: list[int] = FACINGS) -> dict[str, list[Sprite]]:
    decoded = read_sld(graphics / f"{file}_x1.sld")
    per = len(decoded) // 16
    # Frames spread evenly over the whole cycle, so the loop closes without a jump.
    count = min(FRAMES_PER_ANIMATION, per)
    picks = [round(k * per / count) for k in range(count)]
    out = {}
    for facing in facings:
        sprites = [compose(decoded[facing * per + i]) for i in picks]
        out[f"{name}_d{facing}"] = [s for s in sprites if s]
    return out


def pack(sprites: dict[str, list[Sprite]], width: int = 2048, pad: int = 2):
    """Shelf packer, tallest first. Returns the atlas and the frame rects."""
    items = [(name, i, s) for name, frames in sprites.items() for i, s in enumerate(frames)]
    items.sort(key=lambda t: -t[2].image.height)
    x = y = shelf = 0
    placed = []
    for name, i, s in items:
        w, h = s.image.size
        if x + w + pad > width:
            x, y, shelf = 0, y + shelf + pad, 0
        placed.append((name, i, s, x, y))
        x += w + pad
        shelf = max(shelf, h)
    height = (y + shelf + 3) // 4 * 4
    atlas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    meta: dict[str, list[list[int]]] = {name: [None] * len(frames) for name, frames in sprites.items()}  # type: ignore[misc]
    for name, i, s, px, py in placed:
        atlas.alpha_composite(s.image, (px, py))
        meta[name][i] = [px, py, s.image.width, s.image.height, s.anchor[0], s.anchor[1]]
    return atlas, meta


def match(frame_rgb: np.ndarray, sprite: Sprite, guess: tuple[float, float], reach: int) -> tuple[float, int, int]:
    """Best anchor position for `sprite` within `reach` px of `guess`: (score, x, y)."""
    import cv2

    assert sprite.rgb is not None and sprite.mask is not None
    th, tw = sprite.mask.shape
    ox, oy = sprite.offset
    left = int(guess[0] + ox - reach)
    top = int(guess[1] + oy - reach)
    x0, y0 = max(0, left), max(0, top)
    x1 = min(frame_rgb.shape[1], left + tw + 2 * reach)
    y1 = min(frame_rgb.shape[0], top + th + 2 * reach)
    region = frame_rgb[y0:y1, x0:x1]
    if region.shape[0] < th or region.shape[1] < tw or sprite.mask.sum() < 50:
        return (-1.0, int(guess[0]), int(guess[1]))
    mask = np.repeat(sprite.mask[:, :, None], 3, axis=2).astype(np.float32)
    scores = cv2.matchTemplate(region, sprite.rgb, cv2.TM_CCORR_NORMED, mask=mask)
    scores = np.nan_to_num(scores, nan=-1.0, posinf=-1.0, neginf=-1.0)
    _, best, _, at = cv2.minMaxLoc(scores)
    return (float(best), x0 + at[0] - ox, y0 + at[1] - oy)


def place(sprites: dict[str, list[Sprite]], frame_path: Path, turn_path: Path) -> None:
    """Template-match buildings, mines and trees against the frame; write turn['world']."""
    frame_rgb = np.asarray(Image.open(frame_path).convert("RGB"), dtype=np.float32)
    turn = json.loads(turn_path.read_text())
    w, h = turn["frame"]["width"], turn["frame"]["height"]
    names = lambda prefix: [n for n in sprites if n == prefix or n.startswith(prefix + "_")]  # noqa: E731
    candidates = {
        "town_center": names("town_center"),
        "house": names("house"),
        "mill": names("mill"),
        "lumber_camp": names("lumber_camp"),
        "mining_camp": names("mining_camp"),
        "tower": names("tower") + names("outpost"),
        "gold_mine": names("gold"),
        "stone_mine": names("stone"),
    }
    world = []
    for e in turn["entities"]:
        box = (e["x"] * w, e["y"] * h, (e["x"] + e["w"]) * w, (e["y"] + e["h"]) * h)
        if e["cls"] in candidates:
            best = None
            for name in candidates[e["cls"]]:
                sprite = sprites[name][0]
                assert sprite.mask is not None
                # Guess: the graphics' box centred on the detection box.
                mh, mw = sprite.mask.shape
                guess = ((box[0] + box[2]) / 2 - sprite.offset[0] - mw / 2, (box[1] + box[3]) / 2 - sprite.offset[1] - mh / 2)
                score, x, y = match(frame_rgb, sprite, guess, reach=48)
                if best is None or score > best[0]:
                    best = (score, x, y, name)
            assert best is not None
            world.append({"id": e["id"], "sprite": best[3], "x": best[1], "y": best[2], "score": round(best[0], 3)})
        elif e["cls"] == "tree":
            world.extend(place_trees(frame_rgb, sprites, names("pine") + names("oak") + names("autumn") + names("birch"), e["id"], box))
    # The anchors of whole-tile buildings fall on the game's tile grid. Its pitch is the
    # tile width; the Town Center's anchor is the origin.
    tc = next(x for x in world if x["id"].startswith("town_center"))
    tile = grid_pitch([x for x in world if x["score"] > 0.95 and x["id"].split("_")[0] in ("house", "lumber", "mining", "town")], tc)
    origin = (tc["x"], tc["y"])
    # Second pass: buildings may only stand on that grid, so re-match each one at the
    # grid points near its first match, and place farms on the grid too.
    footprint = {"town_center": 4, "house": 2, "mill": 2, "lumber_camp": 2, "mining_camp": 2, "tower": 1}
    entities = {e["id"]: e for e in turn["entities"]}
    for item in world:
        cls = entities[item["id"]]["cls"]
        if cls in footprint:
            sprite = sprites[item["sprite"]][0]
            score, x, y = match_on_grid(frame_rgb, sprite, (item["x"], item["y"]), origin, tile, footprint[cls] % 2 == 1)
            item.update(x=x, y=y, score=round(score, 3))
    farms: list[tuple[int, int]] = []
    for e in sorted((e for e in turn["entities"] if e["cls"] == "farm"), key=lambda e: -(e["conf"] or 0)):
        box = (e["x"] * w, e["y"] * h, (e["x"] + e["w"]) * w, (e["y"] + e["h"]) * h)
        spot = place_farm(frame_rgb, box, origin, tile, farms)
        if spot is not None:
            farms.append(spot)
            world.append({"id": e["id"], "sprite": "farm", "x": spot[0], "y": spot[1], "score": 1.0})
    # Two detections of one object end up on the same spot: keep one.
    seen: set[tuple[str, int, int]] = set()
    world = [x for x in world if (k := (x["sprite"], x["x"], x["y"])) not in seen and not seen.add(k)]
    turn["world"] = {"tile": tile, "origin": list(origin), "items": world}
    turn_path.write_text(json.dumps(turn, indent=1) + "\n")
    visibility(frame_rgb).save(OUT / "visibility.png")
    kept = [x for x in world if not x["id"].startswith("tree")]
    print(f"placed {len(kept)} buildings and mines (mean score {np.mean([x['score'] for x in kept]):.3f}), "
          f"{len(world) - len(kept)} trees")


def visibility(frame_rgb: np.ndarray, cell: int = 24) -> Image.Image:
    """How lit the map is across the frame, one pixel per `cell` px: 1 where the player
    sees, lower under the fog of war (explored, not in view), 0 where unexplored.

    The fog darkens everything by the same amount, so the brightness of the grass
    alone measures it: per cell, the median brightness of grass-coloured pixels,
    relative to the brightest grass. Cells without grass (buildings, fields) take
    their neighbours' value; near-black cells are unexplored."""
    from PIL import ImageFilter

    h, w = frame_rgb.shape[0] // cell, frame_rgb.shape[1] // cell
    rgb = frame_rgb[: h * cell, : w * cell].reshape(h, cell, w, cell, 3).transpose(0, 2, 1, 3, 4).reshape(h, w, -1, 3)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    lum = 0.3 * r + 0.59 * g + 0.11 * b
    grass = (g > r * 1.1) & (g > b * 1.4) & (lum > 20)
    level = np.full((h, w), np.nan, np.float32)
    for y in range(h):
        for x in range(w):
            if grass[y, x].sum() > cell * cell * 0.25:
                level[y, x] = np.median(lum[y, x][grass[y, x]])
    lit = np.nanpercentile(level, 85)
    # Fill cells without grass from their neighbours, widening until none are left.
    for _ in range(20):
        holes = np.isnan(level)
        if not holes.any():
            break
        padded = np.pad(level, 1, constant_values=np.nan)
        stack = np.stack([padded[dy : dy + h, dx : dx + w] for dy in range(3) for dx in range(3)])
        level[holes] = np.nanmean(stack, axis=0)[holes]
    level = np.nan_to_num(level, nan=lit)
    vis = np.clip(level / lit, 0, 1)
    vis[np.median(lum, axis=2) < 18] = 0
    image = Image.fromarray((vis * 255).astype(np.uint8), "L")
    return image.filter(ImageFilter.GaussianBlur(1.2))


def grid_points(guess: tuple[float, float], origin: tuple[int, int], tile: int, odd: bool, reach_tiles: float):
    """Screen anchors of whole-tile positions near `guess`: tile corners for even
    footprints, tile centres for odd ones (a 2:1 grid of tile/2 by tile/4 px)."""
    hx, hy = tile / 2, tile / 4
    u0 = round((guess[0] - origin[0]) / hx)
    v0 = round((guess[1] - origin[1]) / hy)
    r = int(reach_tiles * 2) + 1
    for u in range(u0 - r, u0 + r + 1):
        for v in range(v0 - 2 * r, v0 + 2 * r + 1):
            if (u + v) % 2 == (1 if odd else 0):
                yield int(origin[0] + u * hx), int(origin[1] + v * hy)


def match_on_grid(frame_rgb, sprite: Sprite, guess, origin, tile: int, odd: bool) -> tuple[float, int, int]:
    """The best grid position for `sprite` near `guess`, scored like `match`."""
    import cv2

    assert sprite.rgb is not None and sprite.mask is not None
    th, tw = sprite.mask.shape
    mask = np.repeat(sprite.mask[:, :, None], 3, axis=2).astype(np.float32)
    best = (-1.0, int(guess[0]), int(guess[1]))
    for ax, ay in grid_points(guess, origin, tile, odd, 1.0):
        left, top = ax + sprite.offset[0], ay + sprite.offset[1]
        if left < 0 or top < 0 or left + tw > frame_rgb.shape[1] or top + th > frame_rgb.shape[0]:
            continue
        window = frame_rgb[top : top + th, left : left + tw]
        score = float(np.nan_to_num(cv2.matchTemplate(window, sprite.rgb, cv2.TM_CCORR_NORMED, mask=mask)[0, 0], nan=-1.0))
        if score > best[0]:
            best = (score, ax, ay)
    return best


def place_farm(frame_rgb, box, origin, tile: int, taken: list[tuple[int, int]]) -> tuple[int, int] | None:
    """Grid point whose 3x3-tile diamond covers the most farm soil (reddish pixels,
    where grass is green), without overlapping a farm already placed."""

    def ground(x: float, y: float) -> tuple[float, float]:
        u, v = (x - origin[0]) / (tile / 2), (y - origin[1]) / (tile / 4)
        return (u + v) / 2, (v - u) / 2

    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2 + (box[3] - box[1]) * 0.07
    half_w, half_h = 1.5 * tile, 0.75 * tile
    soil = frame_rgb[:, :, 0] - frame_rgb[:, :, 1]
    best: tuple[float, int, int] | None = None
    placed = [ground(*t) for t in taken]
    for ax, ay in grid_points((cx, cy), origin, tile, True, 1.5):
        gx, gz = ground(ax, ay)
        if any(abs(gx - px) < 2.9 and abs(gz - pz) < 2.9 for px, pz in placed):
            continue
        x0, x1 = int(ax - half_w), int(ax + half_w)
        y0, y1 = int(ay - half_h), int(ay + half_h)
        if x0 < 0 or y0 < 0 or x1 > soil.shape[1] or y1 > soil.shape[0]:
            continue
        ys, xs = np.mgrid[y0:y1, x0:x1]
        inside = np.abs(xs - ax) / half_w + np.abs(ys - ay) / half_h <= 1
        score = float(soil[y0:y1, x0:x1][inside].mean())
        if best is None or score > best[0]:
            best = (score, ax, ay)
    return None if best is None else (best[1], best[2])


def grid_pitch(items: list[dict], origin: dict) -> int:
    """The tile width whose half-tile grid (w/2 by w/4 px) fits the most anchors."""
    best = (0, 96)
    for width in range(80, 113, 4):
        fits = sum(
            1
            for x in items
            if (x["x"] - origin["x"]) % (width // 2) == 0 and (x["y"] - origin["y"]) % (width // 4) == 0
        )
        best = max(best, (fits, width))
    return best[1]


def place_trees(frame_rgb: np.ndarray, sprites, tree_names: list[str], entity_id: str, box) -> list[dict]:
    """Greedily pick the best-matching tree sprites inside a forest box."""
    import cv2

    x0, y0, x1, y1 = (int(v) for v in box)
    region = frame_rgb[max(0, y0 - 40) : y1 + 10, max(0, x0 - 20) : x1 + 20]
    rx, ry = max(0, x0 - 20), max(0, y0 - 40)
    peaks = []
    for name in tree_names:
        sprite = sprites[name][0]
        assert sprite.mask is not None and sprite.rgb is not None
        th, tw = sprite.mask.shape
        if region.shape[0] < th or region.shape[1] < tw:
            continue
        mask = np.repeat(sprite.mask[:, :, None], 3, axis=2).astype(np.float32)
        scores = np.nan_to_num(cv2.matchTemplate(region, sprite.rgb, cv2.TM_CCORR_NORMED, mask=mask), nan=-1.0)
        for _ in range(25):
            _, best, _, at = cv2.minMaxLoc(scores)
            if best < 0.86:
                break
            ax = rx + at[0] - sprite.offset[0]
            ay = ry + at[1] - sprite.offset[1]
            peaks.append((best, ax, ay, name))
            cv2.rectangle(scores, (at[0] - 12, at[1] - 12), (at[0] + 12, at[1] + 12), -1.0, -1)
    peaks.sort(key=lambda p: -p[0])
    chosen: list[tuple[float, float, float, str]] = []
    for p in peaks:
        if x0 - 10 <= p[1] <= x1 + 10 and y0 <= p[2] <= y1 + 10 and all(abs(p[1] - q[1]) > 16 or abs(p[2] - q[2]) > 10 for q in chosen):
            chosen.append(p)
        if len(chosen) >= 60:
            break
    return [{"id": entity_id, "sprite": n, "x": x, "y": y, "score": round(s, 3)} for s, x, y, n in chosen]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--graphics", type=Path, required=True, help="game graphics folder with .sld files")
    parser.add_argument("--terrain", type=Path, required=True, help="game terrain textures folder (.dds)")
    parser.add_argument("--frame", type=Path, help="recorded frame to place the village from")
    parser.add_argument("--turn-data", type=Path, help="turn-data.json to write the placements into")
    args = parser.parse_args()
    g = args.graphics

    sprites: dict[str, list[Sprite]] = {}
    # Magyars build in the Eastern European style, the game's "slav" set; the recorded
    # game is in the Feudal Age.
    sprites |= still(g, "b_slav_town_center_age2", "town_center")
    sprites |= still(g, "b_slav_house_age2", "house", [0, 1, 2])
    sprites |= still(g, "b_slav_mill_age2", "mill")
    sprites |= still(g, "b_slav_lumber_camp_age2", "lumber_camp")
    sprites |= still(g, "b_slav_mining_camp_age2", "mining_camp")
    sprites |= still(g, "b_slav_tower_age2", "tower")
    # The detector's "tower" class also covers the outpost, a lookout on stilts.
    sprites |= still(g, "b_dark_outpost_age1", "outpost")
    sprites |= still(g, "b_foundation_2x2_rubble", "foundation")
    # Living trees for the forests: matching picks among them where the frame shows
    # trees; the green ones also fill stands that matching leaves empty.
    sprites |= still(g, "n_tree_pine", "pine", list(range(27)), keep=is_green)
    sprites |= still(g, "n_tree_oak", "oak", list(range(12)), keep=is_green)
    sprites |= still(g, "n_tree_autumn_oak", "autumn", list(range(12)))
    sprites |= still(g, "n_tree_birch", "birch", list(range(12)), keep=is_green)
    sprites |= still(g, "n_mine_gold", "gold", list(range(0, 7, 2)))
    sprites |= still(g, "n_mine_stone", "stone", list(range(0, 7, 2)))
    # Ground cover the map is dotted with: grass tufts, bushes and shrubs.
    sprites |= still(g, "n_plant_grass_green", "tuft", list(range(8)))
    sprites |= still(g, "n_plant_bush_green", "bush", list(range(8)))
    sprites |= still(g, "n_plant_shrub_green", "shrub", list(range(8)))
    for gender in ("male", "female"):
        g_ = gender[0]
        sprites |= animation(g, f"u_vil_{gender}_lumberjack_taskA", f"vil{g_}_chop")
        sprites |= animation(g, f"u_vil_{gender}_farmer_taskA", f"vil{g_}_farm")
        sprites |= animation(g, f"u_vil_{gender}_miner_gold_taskA", f"vil{g_}_mine")
        sprites |= animation(g, f"u_vil_{gender}_builder_taskA", f"vil{g_}_build")
        sprites |= animation(g, f"u_vil_{gender}_villager_walkA", f"vil{g_}_walk", list(range(16)))
    sprites |= animation(g, "a_hunt_deer_idleA", "deer", [2, 10])
    sprites |= animation(g, "a_hunt_boar_idleA", "boar", [2, 10])

    atlas, meta = pack(sprites)
    OUT.mkdir(parents=True, exist_ok=True)
    atlas.save(OUT / "sprites.webp", quality=92, method=6)
    (OUT / "sprites.json").write_text(json.dumps({"size": list(atlas.size), "sprites": meta}, separators=(",", ":")) + "\n")

    terrain = {
        "grass": "g_gr2",
        "grass_patchy": "g_gr3",
        "dirt": "g_gr5",
        "farm": "g_fm1",
        "forest": "g_for",
    }
    for name, file in terrain.items():
        image = Image.open(args.terrain / f"{file}.dds").convert("RGB").resize((512, 512), Image.LANCZOS)
        image.save(OUT / f"{name}.webp", quality=84, method=6)

    if args.frame and args.turn_data:
        place(sprites, args.frame, args.turn_data)

    frames = sum(len(v) for v in sprites.values())
    print(f"{len(sprites)} sprites, {frames} frames, atlas {atlas.size[0]}x{atlas.size[1]}")


if __name__ == "__main__":
    main()
