"""Build the AoE2:DE sprites and terrain textures for the home page village.

Decodes the chosen .sld files from a local game install (sld.py), bakes each frame's
shadow under its graphics, paints the player-colour mask blue (player 1), packs every
frame into one atlas and writes, under public/walkthrough/aoe2/:

- sprites.webp: the atlas
- sprites.json: per sprite, its frames as atlas rects plus the ground anchor (the
  frame's hotspot), in atlas pixels
- terrain textures (grass, dirt, farm, forest floor) as webp

    uv run python apps/landing/scripts/build_walkthrough_sprites.py \
        --graphics game_graphics --terrain game_terrain/textures/2x

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
    return Sprite("", canvas, (frame.hotspot[0] - x0 - bbox[0], frame.hotspot[1] - y0 - bbox[1]))


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
    step = max(1, per // FRAMES_PER_ANIMATION)
    out = {}
    for facing in facings:
        sprites = [compose(decoded[facing * per + i]) for i in range(0, per, step)][:FRAMES_PER_ANIMATION]
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--graphics", type=Path, required=True, help="game graphics folder with .sld files")
    parser.add_argument("--terrain", type=Path, required=True, help="game terrain textures folder (.dds)")
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
    sprites |= still(g, "b_foundation_2x2_rubble", "foundation")
    sprites |= still(g, "n_tree_pine", "pine", list(range(27)), keep=is_green)
    sprites |= still(g, "n_tree_oak", "oak", list(range(12)), keep=is_green)
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

    frames = sum(len(v) for v in sprites.values())
    print(f"{len(sprites)} sprites, {frames} frames, atlas {atlas.size[0]}x{atlas.size[1]}")


if __name__ == "__main__":
    main()
