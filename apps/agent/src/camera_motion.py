"""Estimate a small camera pan from two native game-window captures.

Only an agreed translation of textured terrain is trusted. A rejected estimate
is a view discontinuity: callers must forget screen-space object history.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from collections.abc import Sequence

MOBILE_CLASSES = frozenset(
    {
        "villager",
        "trade_cart",
        "fishing_ship",
        "sheep",
        "deer",
        "boar",
        "wolf",
        "goose",
        "militia_line",
        "spearman_line",
        "eagle_line",
        "archer_line",
        "skirmisher_line",
        "cavalry_archer",
        "hand_cannoneer",
        "scout_line",
        "knight_line",
        "camel_line",
        "battle_elephant",
        "ram",
        "mangonel_line",
        "scorpion",
        "trebuchet",
        "siege_tower",
        "unique_archer",
        "unique_cavalry",
        "unique_infantry",
        "unique_siege",
        "monk",
        "king",
        "galley",
        "fire_galley",
        "unique_ship",
    }
)


@dataclass(frozen=True, slots=True)
class MotionLimits:
    analysis_width: int = 320
    analysis_height: int = 180
    min_agreeing_regions: int = 3
    min_overlap: float = 0.60
    max_region_disagreement: float = 2.0
    min_texture: float = 0.018
    max_aligned_error: float = 0.115
    moving_mask_padding: int = 8
    cursor_mask_radius: int = 18


@dataclass(frozen=True, slots=True)
class CameraMotion:
    kind: Literal["stable", "pan", "reset"]
    shift: tuple[float, float] = (0.0, 0.0)  # content displacement, full pixels
    reason: str = ""


def cursor_exclusions(
    cursor: tuple[int, int] | None, limits: MotionLimits
) -> tuple[tuple[float, float, float, float], ...]:
    if cursor is None:
        return ()
    x, y = cursor
    radius = limits.cursor_mask_radius
    return ((x - radius, y - radius, x + radius, y + radius),)


def analysis_pixels(rgb: bytes, width: int, height: int, limits: MotionLimits) -> np.ndarray:
    """Return a small grayscale view of the *native* pixels, before JPEG."""
    image = Image.frombytes("RGB", (width, height), rgb)
    return (
        np.asarray(
            image.resize(
                (limits.analysis_width, limits.analysis_height), Image.Resampling.BILINEAR
            ).convert("L"),
            dtype=np.float32,
        )
        / 255.0
    )


def _shift(previous: np.ndarray, current: np.ndarray, excluded: np.ndarray) -> tuple[float, float]:
    """Phase correlation of one terrain region; positive means content moved right/down."""
    # NumPy's installed stubs expose ndarray arithmetic and shape as Any.
    old_mean = cast("float", previous[~excluded].mean())
    new_mean = cast("float", current[~excluded].mean())
    old = cast("np.ndarray", np.where(excluded, old_mean, previous) - old_mean)
    new = cast("np.ndarray", np.where(excluded, new_mean, current) - new_mean)
    shape = cast("tuple[int, int]", old.shape)
    spectrum = np.fft.fft2(new) * np.conj(np.fft.fft2(old))
    spectrum /= np.maximum(np.abs(spectrum), 1e-9)
    peak_y, peak_x = cast(
        "tuple[int, int]",
        np.unravel_index(np.argmax(np.abs(np.fft.ifft2(spectrum))), shape),
    )
    return (
        float(peak_x if peak_x <= shape[1] // 2 else peak_x - shape[1]),
        float(peak_y if peak_y <= shape[0] // 2 else peak_y - shape[0]),
    )


def _aligned_error(
    previous: np.ndarray, current: np.ndarray, excluded: np.ndarray, dx: int, dy: int
) -> float:
    height, width = cast("tuple[int, int]", previous.shape)
    old_x = slice(max(0, -dx), min(width, width - dx))
    old_y = slice(max(0, -dy), min(height, height - dy))
    new_x = slice(max(0, dx), min(width, width + dx))
    new_y = slice(max(0, dy), min(height, height + dy))
    valid = ~(excluded[old_y, old_x] | excluded[new_y, new_x])
    if not bool(valid.any()):
        return float("inf")
    return float(np.mean(np.abs(previous[old_y, old_x] - current[new_y, new_x])[valid]))


def _excluded_pixels(
    shape: tuple[int, int],
    full_size: tuple[int, int],
    boxes: Sequence[tuple[float, float, float, float]],
    padding: int,
) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.bool_)
    for x0, y0, x1, y1 in boxes:
        left = max(0, int((x0 - padding) * width / full_size[0]))
        top = max(0, int((y0 - padding) * height / full_size[1]))
        right = min(width, int((x1 + padding) * width / full_size[0]) + 1)
        bottom = min(height, int((y1 + padding) * height / full_size[1]) + 1)
        mask[top:bottom, left:right] = True
    return mask


def estimate_motion(
    previous: np.ndarray | None,
    current: np.ndarray,
    *,
    full_size: tuple[int, int],
    limits: MotionLimits = MotionLimits(),
    excluded_boxes: Sequence[tuple[float, float, float, float]] = (),
) -> CameraMotion:
    if previous is None or previous.shape != current.shape:
        return CameraMotion("reset", reason="first_or_resized_frame")
    # Avoid stationary HUD, scoreboard, minimap and selection panel. Moving
    # units/cursor are sparse within these broad terrain regions.
    height, width = cast("tuple[int, int]", current.shape)
    excluded = _excluded_pixels(
        (height, width), full_size, excluded_boxes, limits.moving_mask_padding
    )
    top, bottom = int(height * 0.15), int(height * 0.76)
    left, right = int(width * 0.05), int(width * 0.76)
    old_view = previous[top:bottom, left:right]
    new_view = current[top:bottom, left:right]
    view_mask = excluded[top:bottom, left:right]
    if not bool((~view_mask).any()):
        return CameraMotion("reset", reason="terrain_fully_masked")
    if float(np.mean(np.abs(old_view - new_view)[~view_mask])) < 0.012:
        return CameraMotion("stable")

    motions: list[tuple[float, float]] = []
    view_height, view_width = cast("tuple[int, int]", old_view.shape)
    for row in range(2):
        for col in range(3):
            y0, y1 = row * view_height // 2, (row + 1) * view_height // 2
            x0, x1 = col * view_width // 3, (col + 1) * view_width // 3
            old_region = old_view[y0:y1, x0:x1]
            new_region = new_view[y0:y1, x0:x1]
            region_mask = view_mask[y0:y1, x0:x1]
            valid = ~region_mask
            if cast("float", valid.mean()) < limits.min_overlap:
                continue
            old_texture = cast("float", old_region[valid].std())
            new_texture = cast("float", new_region[valid].std())
            if min(old_texture, new_texture) >= limits.min_texture:
                motions.append(_shift(old_region, new_region, region_mask))
    if len(motions) < limits.min_agreeing_regions:
        return CameraMotion("reset", reason="insufficient_texture")
    dx = float(np.median([item[0] for item in motions]))
    dy = float(np.median([item[1] for item in motions]))
    agreeing = sum(
        abs(x - dx) <= limits.max_region_disagreement
        and abs(y - dy) <= limits.max_region_disagreement
        for x, y in motions
    )
    if agreeing < limits.min_agreeing_regions:
        return CameraMotion("reset", reason="inconsistent_regions")
    ix, iy = round(dx), round(dy)
    overlap = max(0, view_width - abs(ix)) * max(0, view_height - abs(iy))
    overlap /= view_width * view_height
    if overlap < limits.min_overlap:
        return CameraMotion("reset", reason="insufficient_overlap")
    if _aligned_error(old_view, new_view, view_mask, ix, iy) > limits.max_aligned_error:
        return CameraMotion("reset", reason="alignment_residual")
    if ix == 0 and iy == 0:
        return CameraMotion("stable")
    return CameraMotion("pan", (dx * full_size[0] / width, dy * full_size[1] / height))


__all__ = [
    "MOBILE_CLASSES",
    "CameraMotion",
    "MotionLimits",
    "analysis_pixels",
    "cursor_exclusions",
    "estimate_motion",
]
