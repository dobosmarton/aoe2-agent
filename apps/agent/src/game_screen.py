"""Whether the game is showing a menu or popup over play.

AoE2 dims everything behind its in-game menus. The resource bar and the bottom
panel are fixed HUD artwork, so their brightness doesn't depend on the map: on the
Mac at 3024x1898 they measured 66 and 150-156 across eight scenes, and 13 and 31
with the Main Menu open. Input sent then lands on the menu instead (the screen's
centre is its Load Game button), and Escape doesn't close it.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np

Frame: TypeAlias = np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]

HUD_BAR_ROWS = 80
HUD_BAR_WIDTH = 0.426  # Share of the frame's width the resource bar covers.
PANEL_ROWS = 258  # The bottom panel, counted up from the frame's bottom edge.
PANEL_COLUMNS = (0.142, 0.463)
# Halfway between dimmed and normal, in mean RGB brightness.
HUD_BAR_LIT = 40.0
PANEL_LIT = 90.0


def hud_brightness(rgb: Frame) -> tuple[float, float]:
    """Mean brightness of the resource bar and of the bottom panel."""
    height, width, _ = rgb.shape
    bar = rgb[:HUD_BAR_ROWS, : round(width * HUD_BAR_WIDTH)]
    panel = rgb[
        max(0, height - PANEL_ROWS) :,
        round(width * PANEL_COLUMNS[0]) : round(width * PANEL_COLUMNS[1]),
    ]
    return float(np.mean(bar)), float(np.mean(panel))


def popup_open(rgb: Frame) -> bool:
    """True when the game is dimmed behind a menu or popup."""
    bar, panel = hud_brightness(rgb)
    return bar < HUD_BAR_LIT and panel < PANEL_LIT


__all__ = ["hud_brightness", "popup_open"]
