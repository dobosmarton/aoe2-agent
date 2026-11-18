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
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

# Layer flag bits of a frame header, in the order their layers follow it.
MAIN, SHADOW, UNKNOWN, DAMAGE, PLAYER = 0x01, 0x02, 0x04, 0x08, 0x10
LAYER_BITS = (MAIN, SHADOW, UNKNOWN, DAMAGE, PLAYER)

SIGNATURE = b"SLDX"
FILE_HEADER_SIZE = 16
FRAME_HEADER_SIZE = 12
# x1, y1, x2, y2, flag, unused: the header of a layer with its own bounds.
BOUNDED_LAYER_HEADER_SIZE = 10
# flag, unused: the header of a mask layer that reuses the main layer's bounds.
MASK_LAYER_HEADER_SIZE = 2
LENGTH_PREFIX_SIZE = 4
# Layer flag bit: draw over the previous frame's layer instead of an empty one.
DELTA_FLAG = 0x80

BLOCK_SIDE = 4
BLOCK_PIXELS = BLOCK_SIDE * BLOCK_SIDE
BLOCK_BYTES = 8

# Perceived brightness of an RGB pixel.
LUMA_WEIGHTS = (0.3, 0.59, 0.11)
# The player colour is painted with the shading of the pixels under the mask.
TINT_BASE, TINT_GAIN = 0.35, 1.1

Pixels: TypeAlias = NDArray[np.uint8]
Blocks: TypeAlias = NDArray[np.uint8]
Palette: TypeAlias = NDArray[np.int32]


@dataclass(frozen=True, slots=True)
class Layer:
    x1: int
    y1: int
    pixels: Pixels  # (h, w, 4) uint8 for colour, (h, w) uint8 for masks


@dataclass(frozen=True, slots=True)
class Frame:
    canvas: tuple[int, int]
    hotspot: tuple[int, int]
    main: Layer | None = None
    shadow: Layer | None = None
    player: Layer | None = None


@dataclass(frozen=True, slots=True)
class _Bounds:
    x1: int
    y1: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class _Codec:
    """A block compression format and the shape of one decoded pixel."""

    decode: Callable[[Blocks], Pixels]
    pixel_shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _LastLayers:
    """The most recent layer of each kind, which a delta layer draws over."""

    main: Layer | None = None
    shadow: Layer | None = None
    player: Layer | None = None

    def after(self, frame: Frame) -> _LastLayers:
        return _LastLayers(
            main=_latest(frame.main, self.main),
            shadow=_latest(frame.shadow, self.shadow),
            player=_latest(frame.player, self.player),
        )


def _latest(current: Layer | None, previous: Layer | None) -> Layer | None:
    return previous if current is None else current


def luminance(rgb: NDArray[np.float32]) -> NDArray[np.float32]:
    """Per-pixel brightness of an (..., 3) float32 RGB array."""
    weighted: NDArray[np.float32] = rgb @ np.array(LUMA_WEIGHTS, np.float32)
    return weighted


def _rgb565_to_rgb888(packed: NDArray[np.int32]) -> Palette:
    red = (packed >> 11) & 31
    green = (packed >> 5) & 63
    blue = packed & 31
    channels = [(red << 3) | (red >> 2), (green << 2) | (green >> 4), (blue << 3) | (blue >> 2)]
    return np.stack(channels, axis=1)


def _bc1(blocks: Blocks) -> Pixels:
    """(n, 8) BC1 blocks -> (n, 16, 4) RGBA."""
    endpoints = blocks[:, :4].copy().view("<u2").astype(np.int32)
    colour0, colour1 = endpoints[:, 0], endpoints[:, 1]
    rgb0, rgb1 = _rgb565_to_rgb888(colour0), _rgb565_to_rgb888(colour1)
    # colour0 > colour1 selects four opaque colours; otherwise three and transparent.
    four_colours = (colour0 > colour1)[:, None]
    third = np.where(four_colours, (2 * rgb0 + rgb1) // 3, (rgb0 + rgb1) // 2)
    fourth = np.where(four_colours, (rgb0 + 2 * rgb1) // 3, 0)
    palette = np.stack([rgb0, rgb1, third, fourth], axis=1)  # (n, 4, 3)
    alpha = np.full((len(blocks), 4), 255, dtype=np.int32)
    alpha[:, 3] = np.where(four_colours[:, 0], 255, 0)
    bits = blocks[:, 4:8].copy().view("<u4")[:, 0]
    indices = (bits[:, None] >> (np.arange(BLOCK_PIXELS) * 2)) & 3
    rgb_pixels = np.take_along_axis(palette, indices[:, :, None].repeat(3, 2), axis=1)
    alpha_pixels = np.take_along_axis(alpha, indices, axis=1)
    return np.concatenate([rgb_pixels, alpha_pixels[:, :, None]], axis=2).astype(np.uint8)


def _bc4(blocks: Blocks) -> Pixels:
    """(n, 8) BC4 blocks -> (n, 16) single channel."""
    value0 = blocks[:, 0].astype(np.int32)
    value1 = blocks[:, 1].astype(np.int32)
    # value0 > value1 selects eight interpolated levels; otherwise six plus 0 and 255.
    eight_levels = (value0 > value1)[:, None]
    level = np.arange(8)[None, :]
    interpolated8 = ((8 - level) * value0[:, None] + (level - 1) * value1[:, None]) // 7
    interpolated6 = ((6 - level) * value0[:, None] + (level - 1) * value1[:, None]) // 5
    interpolated6 = np.where(level == 6, 0, np.where(level == 7, 255, interpolated6))
    palette = np.where(eight_levels, interpolated8, interpolated6)
    palette[:, 0] = value0
    palette[:, 1] = value1
    bits = np.zeros(len(blocks), dtype=np.uint64)
    for byte in range(6):
        bits |= blocks[:, 2 + byte].astype(np.uint64) << np.uint64(8 * byte)
    shifts = np.arange(BLOCK_PIXELS, dtype=np.uint64) * np.uint64(3)
    indices = ((bits[:, None] >> shifts) & np.uint64(7)).astype(np.int64)
    return np.take_along_axis(palette, indices, axis=1).astype(np.uint8)


BC1 = _Codec(_bc1, (4,))
BC4 = _Codec(_bc4, ())


def _rebase(previous: Layer | None, bounds: _Bounds) -> Pixels | None:
    """The previous frame's layer moved into this layer's bounds (canvas coordinates)."""
    if previous is None:
        return None
    source = previous.pixels
    out = np.zeros((bounds.height, bounds.width, *source.shape[2:]), dtype=np.uint8)
    dx, dy = previous.x1 - bounds.x1, previous.y1 - bounds.y1
    source_x, source_y = max(0, -dx), max(0, -dy)
    target_x, target_y = max(0, dx), max(0, dy)
    overlap_w = min(source.shape[1] - source_x, bounds.width - target_x)
    overlap_h = min(source.shape[0] - source_y, bounds.height - target_y)
    if overlap_w > 0 and overlap_h > 0:
        out[target_y : target_y + overlap_h, target_x : target_x + overlap_w] = source[
            source_y : source_y + overlap_h, source_x : source_x + overlap_w
        ]
    return out


def _block_targets(commands: NDArray[np.uint8]) -> list[int]:
    """Indices of the blocks that get drawn, in order: each command skips a run of
    blocks, then draws one."""
    runs: list[list[int]] = commands.tolist()
    targets: list[int] = []
    cursor = 0
    for skip, draw in runs:
        cursor += skip
        targets.extend(range(cursor, cursor + draw))
        cursor += draw
    return targets


def _draw(data: bytes, pos: int, bounds: _Bounds, codec: _Codec, base: Pixels | None) -> Pixels:
    """Decode a command list and its blocks into an image, over `base` if given."""
    (count,) = struct.unpack_from("<H", data, pos)
    pos += 2
    commands = np.frombuffer(data, dtype=np.uint8, count=count * 2, offset=pos).reshape(-1, 2)
    pos += count * 2
    drawn = int(commands[:, 1].sum())
    blocks = np.frombuffer(data, dtype=np.uint8, count=drawn * BLOCK_BYTES, offset=pos).reshape(
        -1, BLOCK_BYTES
    )
    decoded = (
        codec.decode(blocks) if drawn else np.zeros((0, BLOCK_PIXELS, *codec.pixel_shape), np.uint8)
    )

    blocks_wide = (bounds.width + BLOCK_SIDE - 1) // BLOCK_SIDE
    blocks_high = (bounds.height + BLOCK_SIDE - 1) // BLOCK_SIDE
    image_shape = (blocks_high * BLOCK_SIDE, blocks_wide * BLOCK_SIDE, *codec.pixel_shape)
    image = np.zeros(image_shape, dtype=np.uint8)
    if base is not None:
        image[: bounds.height, : bounds.width] = base
    for block, target in zip(decoded, _block_targets(commands), strict=False):
        row, column = divmod(target, blocks_wide)
        if row >= blocks_high:
            break
        top, left = row * BLOCK_SIDE, column * BLOCK_SIDE
        tile = block.reshape(BLOCK_SIDE, BLOCK_SIDE, *codec.pixel_shape)
        image[top : top + BLOCK_SIDE, left : left + BLOCK_SIDE] = tile
    return image[: bounds.height, : bounds.width]


def _read_bounded_layer(
    data: bytes, start: int, previous: Layer | None, codec: _Codec
) -> tuple[Layer, _Bounds]:
    """A main or shadow layer, which carries its own bounds."""
    x1, y1, x2, y2, layer_flag, _ = struct.unpack_from("<4H2B", data, start)
    bounds = _Bounds(x1, y1, x2 - x1, y2 - y1)
    base = _rebase(previous, bounds) if layer_flag & DELTA_FLAG else None
    pixels = _draw(data, start + BOUNDED_LAYER_HEADER_SIZE, bounds, codec, base)
    return Layer(x1, y1, pixels), bounds


def _read_mask_layer(data: bytes, start: int, previous: Layer | None, bounds: _Bounds) -> Layer:
    """A player-colour mask, drawn in the main layer's bounds."""
    layer_flag = data[start]
    base = _rebase(previous, bounds) if layer_flag & DELTA_FLAG else None
    pixels = _draw(data, start + MASK_LAYER_HEADER_SIZE, bounds, BC4, base)
    return Layer(bounds.x1, bounds.y1, pixels)


def _read_frame(data: bytes, pos: int, last: _LastLayers) -> tuple[Frame, int]:
    """The frame whose header starts at `pos`, and the position after its layers."""
    canvas_w, canvas_h, hotspot_x, hotspot_y, flags, _unknown, _index = struct.unpack_from(
        "<4H2BH", data, pos
    )
    pos += FRAME_HEADER_SIZE
    main: Layer | None = None
    shadow: Layer | None = None
    player: Layer | None = None
    main_bounds: _Bounds | None = None
    for bit in LAYER_BITS:
        if not flags & bit:
            continue
        (length,) = struct.unpack_from("<I", data, pos)
        start = pos + LENGTH_PREFIX_SIZE
        if bit == MAIN:
            main, main_bounds = _read_bounded_layer(data, start, last.main, BC1)
        elif bit == SHADOW:
            shadow, _ = _read_bounded_layer(data, start, last.shadow, BC4)
        elif bit == PLAYER and main_bounds is not None:
            player = _read_mask_layer(data, start, last.player, main_bounds)
        pos += _word_aligned(length)
    frame = Frame(
        canvas=(canvas_w, canvas_h),
        hotspot=(hotspot_x, hotspot_y),
        main=main,
        shadow=shadow,
        player=player,
    )
    return frame, pos


def _word_aligned(length: int) -> int:
    return (length + 3) & ~3


def read_sld(path: Path) -> list[Frame]:
    """Decode every frame of an .sld file, delta layers drawn over their predecessors."""
    data = path.read_bytes()
    signature, _version, num_frames = struct.unpack_from("<4sHH", data, 0)
    if signature != SIGNATURE:
        raise ValueError(f"{path.name}: not an SLD file")
    pos = FILE_HEADER_SIZE
    frames: list[Frame] = []
    last = _LastLayers()
    for _ in range(num_frames):
        frame, pos = _read_frame(data, pos, last)
        frames.append(frame)
        last = last.after(frame)
    return frames


def tint(frame: Frame, colour: tuple[int, int, int]) -> Pixels:
    """The main layer with the player-colour mask painted in `colour` (RGBA, uint8)."""
    if frame.main is None:
        raise ValueError("frame has no main layer to tint")
    rgba = frame.main.pixels.astype(np.float32)
    if frame.player is not None and frame.player.pixels.shape == rgba.shape[:2]:
        mask = frame.player.pixels.astype(np.float32)[:, :, None] / 255.0
        brightness = luminance(rgba[:, :, :3])[:, :, None] / 255.0
        shading = TINT_BASE + TINT_GAIN * brightness
        painted = np.clip(np.array(colour, np.float32)[None, None, :] * shading, 0, 255)
        rgba[:, :, :3] = rgba[:, :, :3] * (1 - mask) + painted * mask
    return rgba.astype(np.uint8)
