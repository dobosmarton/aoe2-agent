"""Decode AoE2:DE .sld sprite files: main graphics, shadow and player-colour mask.

Layout (as documented by the openage project): a 16-byte file header, then per frame
a 12-byte header (canvas size, hotspot, layer flags) and one length-prefixed layer per
flag bit. Main and shadow layers carry their own bounds; the damage and player-colour
masks reuse the main layer's bounds. Each layer is a list of (skip, draw) block counts
over the 4x4 blocks of its bounds, then the drawn blocks: BC1 for colour, BC4 for the
masks. A layer whose flag has bit 0x80 set draws over the previous frame's layer
instead of starting empty, which is how animations store their frames.

packages/detection/src/extraction/sld_extractor.py decodes the main layer only, for
training data; this module adds the layers and frame deltas the home page needs.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MAIN, SHADOW, UNKNOWN, DAMAGE, PLAYER = 0x01, 0x02, 0x04, 0x08, 0x10


@dataclass
class Layer:
    x1: int
    y1: int
    pixels: np.ndarray  # (h, w, 4) uint8 for colour, (h, w) uint8 for masks


@dataclass
class Frame:
    canvas: tuple[int, int]
    hotspot: tuple[int, int]
    main: Layer | None = None
    shadow: Layer | None = None
    player: Layer | None = None
    extra: dict[str, object] = field(default_factory=dict)


def _bc1(blocks: np.ndarray) -> np.ndarray:
    """(n, 8) BC1 blocks -> (n, 16, 4) RGBA."""
    c = blocks[:, :4].copy().view("<u2").astype(np.int32)
    c0, c1 = c[:, 0], c[:, 1]

    def rgb(v: np.ndarray) -> np.ndarray:
        r = (v >> 11) & 31
        g = (v >> 5) & 63
        b = v & 31
        return np.stack([(r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)], axis=1)

    a, b = rgb(c0), rgb(c1)
    four = (c0 > c1)[:, None]
    p2 = np.where(four, (2 * a + b) // 3, (a + b) // 2)
    p3 = np.where(four, (a + 2 * b) // 3, 0)
    palette = np.stack([a, b, p2, p3], axis=1)  # (n, 4, 3)
    alpha = np.full((len(blocks), 4), 255, dtype=np.int32)
    alpha[:, 3] = np.where(four[:, 0], 255, 0)
    bits = blocks[:, 4:8].copy().view("<u4")[:, 0]
    idx = (bits[:, None] >> (np.arange(16) * 2)) & 3
    rgb_px = np.take_along_axis(palette, idx[:, :, None].repeat(3, 2), axis=1)
    a_px = np.take_along_axis(alpha, idx, axis=1)
    return np.concatenate([rgb_px, a_px[:, :, None]], axis=2).astype(np.uint8)


def _bc4(blocks: np.ndarray) -> np.ndarray:
    """(n, 8) BC4 blocks -> (n, 16) single channel."""
    a0 = blocks[:, 0].astype(np.int32)
    a1 = blocks[:, 1].astype(np.int32)
    eight = (a0 > a1)[:, None]
    i = np.arange(8)[None, :]
    p8 = ((8 - i) * a0[:, None] + (i - 1) * a1[:, None]) // 7
    p6 = ((6 - i) * a0[:, None] + (i - 1) * a1[:, None]) // 5
    p6 = np.where(i == 6, 0, np.where(i == 7, 255, p6))
    palette = np.where(eight, p8, p6)
    palette[:, 0] = a0
    palette[:, 1] = a1
    bits = np.zeros(len(blocks), dtype=np.uint64)
    for k in range(6):
        bits |= blocks[:, 2 + k].astype(np.uint64) << np.uint64(8 * k)
    idx = ((bits[:, None] >> (np.arange(16, dtype=np.uint64) * np.uint64(3))) & np.uint64(7)).astype(np.int64)
    return np.take_along_axis(palette, idx, axis=1).astype(np.uint8)


def _rebase(prev: Layer | None, x1: int, y1: int, width: int, height: int) -> np.ndarray | None:
    """The previous frame's layer moved into this layer's bounds (canvas coordinates)."""
    if prev is None:
        return None
    src = prev.pixels
    out = np.zeros((height, width, *src.shape[2:]), dtype=np.uint8)
    dx, dy = prev.x1 - x1, prev.y1 - y1
    sx0, sy0 = max(0, -dx), max(0, -dy)
    tx0, ty0 = max(0, dx), max(0, dy)
    w = min(src.shape[1] - sx0, width - tx0)
    h = min(src.shape[0] - sy0, height - ty0)
    if w > 0 and h > 0:
        out[ty0 : ty0 + h, tx0 : tx0 + w] = src[sy0 : sy0 + h, sx0 : sx0 + w]
    return out


def _draw(
    data: bytes,
    pos: int,
    width: int,
    height: int,
    decode,
    channels: int,
    base: np.ndarray | None,
) -> tuple[np.ndarray, int]:
    """Decode a command list and its blocks into an image, over `base` if given."""
    (count,) = struct.unpack_from("<H", data, pos)
    pos += 2
    commands = np.frombuffer(data, dtype=np.uint8, count=count * 2, offset=pos).reshape(-1, 2)
    pos += count * 2
    drawn = int(commands[:, 1].sum())
    blocks = np.frombuffer(data, dtype=np.uint8, count=drawn * 8, offset=pos).reshape(-1, 8)
    decoded = decode(blocks) if drawn else np.zeros((0, 16, channels) if channels > 1 else (0, 16), np.uint8)

    bw = (width + 3) // 4
    bh = (height + 3) // 4
    shape = (bh * 4, bw * 4, channels) if channels > 1 else (bh * 4, bw * 4)
    image = np.zeros(shape, dtype=np.uint8)
    if base is not None:
        image[:height, :width] = base
    # Block indices that get drawn, in order.
    targets = []
    cursor = 0
    for skip, draw in commands:
        cursor += int(skip)
        targets.extend(range(cursor, cursor + int(draw)))
        cursor += int(draw)
    for n, t in enumerate(targets[: len(decoded)]):
        by, bx = divmod(t, bw)
        if by >= bh:
            break
        tile = decoded[n].reshape(4, 4, channels) if channels > 1 else decoded[n].reshape(4, 4)
        image[by * 4 : by * 4 + 4, bx * 4 : bx * 4 + 4] = tile
    return image[:height, :width], pos


def read_sld(path: Path, wanted: set[int] | None = None) -> list[Frame]:
    """Decode frames of an .sld file. With `wanted`, frames outside it are decoded only
    as far as later delta frames need them, and returned as None-free placeholders."""
    data = path.read_bytes()
    signature, _version, num_frames = struct.unpack_from("<4sHH", data, 0)
    if signature != b"SLDX":
        raise ValueError(f"{path.name}: not an SLD file")
    pos = 16
    frames: list[Frame] = []
    prev_main: Layer | None = None
    prev_shadow: Layer | None = None
    prev_player: Layer | None = None
    for _ in range(num_frames):
        cw, ch, hx, hy, flags, _u, _index = struct.unpack_from("<4H2BH", data, pos)
        pos += 12
        frame = Frame(canvas=(cw, ch), hotspot=(hx, hy))
        main_bounds: tuple[int, int, int, int] | None = None
        for bit in (MAIN, SHADOW, UNKNOWN, DAMAGE, PLAYER):
            if not flags & bit:
                continue
            (length,) = struct.unpack_from("<I", data, pos)
            start = pos + 4
            nxt = pos + ((length + 3) & ~3)
            if bit in (MAIN, SHADOW):
                x1, y1, x2, y2, lflag, _ = struct.unpack_from("<4H2B", data, start)
                w, h = x2 - x1, y2 - y1
                if bit == MAIN:
                    main_bounds = (x1, y1, w, h)
                    base = _rebase(prev_main, x1, y1, w, h) if lflag & 0x80 else None
                    img, _ = _draw(data, start + 10, w, h, _bc1, 4, base)
                    frame.main = prev_main = Layer(x1, y1, img)
                else:
                    base = _rebase(prev_shadow, x1, y1, w, h) if lflag & 0x80 else None
                    img, _ = _draw(data, start + 10, w, h, _bc4, 1, base)
                    frame.shadow = prev_shadow = Layer(x1, y1, img)
            elif bit == PLAYER and main_bounds is not None:
                lflag = data[start]
                x1, y1, w, h = main_bounds
                base = _rebase(prev_player, x1, y1, w, h) if lflag & 0x80 else None
                img, _ = _draw(data, start + 2, w, h, _bc4, 1, base)
                frame.player = prev_player = Layer(x1, y1, img)
            pos = nxt
        frames.append(frame)
    return frames


def tint(frame: Frame, colour: tuple[int, int, int]) -> np.ndarray:
    """The main layer with the player-colour mask painted in `colour` (RGBA, uint8)."""
    assert frame.main is not None
    rgba = frame.main.pixels.astype(np.float32)
    if frame.player is not None and frame.player.pixels.shape == rgba.shape[:2]:
        mask = frame.player.pixels.astype(np.float32)[:, :, None] / 255.0
        lum = (rgba[:, :, :3] @ np.array([0.3, 0.59, 0.11], np.float32))[:, :, None] / 255.0
        painted = np.clip(np.array(colour, np.float32)[None, None, :] * (0.35 + 1.1 * lum), 0, 255)
        rgba[:, :, :3] = rgba[:, :, :3] * (1 - mask) + painted * mask
    return rgba.astype(np.uint8)
