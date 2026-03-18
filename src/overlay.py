"""Live detection overlay for AoE2 agent.

Draws YOLO bounding boxes on a transparent, click-through window
positioned over the game. Windows-only (uses WS_EX_TRANSPARENT).

Usage:
    overlay = DetectionOverlay()
    overlay.show(entities, window_rect)  # draw boxes
    overlay.hide()                       # hide before screenshot
    overlay.close()                      # cleanup
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from detection.inference.detector import DetectedEntity

logger = logging.getLogger(__name__)

# Category colors (RGB) — matches detection/labeling/prelabel.py
_RESOURCE_CLASSES = {"tree", "gold_mine", "stone_mine", "berry_bush", "relic"}
_ANIMAL_CLASSES = {"sheep", "deer", "boar", "wolf", "goose"}
_BUILDING_CLASSES = {
    "town_center", "house", "lumber_camp", "mining_camp", "mill", "market",
    "dock", "farm", "barracks", "archery_range", "stable", "blacksmith",
    "siege_workshop", "monastery", "castle", "university",
}
_DEFENSE_CLASSES = {"gate", "wall", "tower", "wonder", "krepost"}


def _get_color(class_name: str) -> str:
    """Get tkinter-compatible hex color for a class name."""
    if class_name in _RESOURCE_CLASSES:
        return "#228B22"   # green
    elif class_name in _ANIMAL_CLASSES:
        return "#FFA500"   # orange
    elif class_name in _BUILDING_CLASSES:
        return "#4169E1"   # blue
    elif class_name in _DEFENSE_CLASSES:
        return "#9400D3"   # purple
    else:
        return "#DC143C"   # red (units, military)


class DetectionOverlay:
    """Transparent click-through overlay that shows YOLO detections."""

    OVERLAY_TITLE = "AoE2 Detection Overlay"

    def __init__(self):
        import tkinter as tk

        self._root = tk.Tk()
        self._root.title(self.OVERLAY_TITLE)
        self._root.overrideredirect(True)        # no window decorations
        self._root.attributes("-topmost", True)   # always on top
        self._root.configure(bg="black")

        # Full-screen canvas with black background (black = transparent via color key)
        self._canvas = tk.Canvas(
            self._root, bg="black", highlightthickness=0, bd=0,
        )
        self._canvas.pack(fill=tk.BOTH, expand=True)

        # Start hidden until first show() call
        self._root.withdraw()
        self._visible = False

        # Apply Windows click-through transparency
        self._root.update_idletasks()
        self._make_click_through()

        logger.info("Detection overlay initialized")

    def _make_click_through(self):
        """Set WS_EX_TRANSPARENT + WS_EX_LAYERED so clicks pass through."""
        if sys.platform != "win32":
            logger.warning("Overlay click-through only works on Windows")
            return

        try:
            import ctypes

            # Get the HWND for the overlay window
            hwnd = ctypes.windll.user32.FindWindowW(None, self.OVERLAY_TITLE)
            if not hwnd:
                logger.warning("Could not find overlay HWND")
                return

            GWL_EXSTYLE = -20
            WS_EX_LAYERED = 0x00080000
            WS_EX_TRANSPARENT = 0x00000020
            LWA_COLORKEY = 0x00000001

            # Add layered + transparent extended styles
            ex_style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            ctypes.windll.user32.SetWindowLongW(
                hwnd, GWL_EXSTYLE, ex_style | WS_EX_LAYERED | WS_EX_TRANSPARENT
            )

            # Set black (0x00000000) as the transparent color key
            ctypes.windll.user32.SetLayeredWindowAttributes(
                hwnd, 0x00000000, 0, LWA_COLORKEY
            )

            logger.info("Overlay click-through enabled")
        except Exception as e:
            logger.warning("Failed to set click-through: %s", e)

    def hide(self):
        """Hide overlay (call before screenshot capture)."""
        if self._visible:
            self._root.withdraw()
            self._root.update_idletasks()
            self._visible = False

    def show(
        self,
        entities: list[DetectedEntity],
        window_rect: tuple[int, int, int, int] | None,
    ):
        """Draw detection boxes and show the overlay.

        Args:
            entities: List of DetectedEntity from YOLO detection
            window_rect: Game window (left, top, width, height) in screen coords
        """
        if not window_rect:
            return

        left, top, width, height = window_rect

        # Reposition overlay to match game window
        self._root.geometry(f"{width}x{height}+{left}+{top}")

        # Clear previous drawings
        self._canvas.delete("all")

        # Draw each entity
        for entity in entities:
            x1, y1, x2, y2 = entity.bbox
            color = _get_color(entity.class_name)

            # Bounding box
            self._canvas.create_rectangle(
                x1, y1, x2, y2,
                outline=color, width=2,
            )

            # Label text
            label = f"{entity.class_name} {entity.confidence:.0%}"
            # Text background (small filled rect behind text)
            self._canvas.create_rectangle(
                x1, y1 - 16, x1 + len(label) * 7, y1 - 1,
                fill=color, outline=color,
            )
            self._canvas.create_text(
                x1 + 2, y1 - 9,
                text=label, fill="white", anchor="w",
                font=("Consolas", 9),
            )

        # Show overlay
        if not self._visible:
            self._root.deiconify()
            self._visible = True

        self._root.update_idletasks()
        self._root.update()

    def close(self):
        """Destroy the overlay window."""
        try:
            self._root.destroy()
        except Exception:
            pass
