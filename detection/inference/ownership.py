"""Ownership classifier for AoE2 detected entities.

Uses pixel color analysis on the screenshot region around each detected
military unit to determine whether it belongs to the player (blue) or enemy.

In AoE2:DE, Player 1 is always blue. Units and health bars are tinted
in the player's color, making RGB blue-dominance a reliable ownership signal.
"""

from __future__ import annotations

import io
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    pass

import structlog

log = structlog.get_logger()


class Owner(Enum):
    OWN = "own"
    ENEMY = "enemy"
    UNKNOWN = "unknown"


# --- Blue-dominance thresholds (RGB space) ---
# A pixel is "blue" (Player 1 color) if:
#   B > _BLUE_MIN  AND  B > R * _DOMINANCE  AND  B > G * _DOMINANCE
_BLUE_MIN = 120
_DOMINANCE = 1.5

# A unit is "own" if blue_ratio exceeds this threshold.
# Own blue units typically show 15-30% blue pixels in health bar region,
# background/terrain shows < 1%.
_OWN_THRESHOLD = 0.04

# Health bar sampling region (pixels relative to bbox top y1)
_HB_ABOVE = 12  # how far above bbox top to start
_HB_HEIGHT = 14  # total height of the health bar sample strip

# Minimum region size to attempt classification (pixels)
_MIN_REGION_PX = 9  # 3x3


def _blue_ratio(region: np.ndarray) -> float:
    """Fraction of blue-dominant pixels in an RGB region."""
    if region.size == 0 or region.shape[0] < 1 or region.shape[1] < 1:
        return 0.0

    r = region[:, :, 0].astype(np.int16)
    g = region[:, :, 1].astype(np.int16)
    b = region[:, :, 2].astype(np.int16)

    blue_mask = (b > _BLUE_MIN) & (b > r * _DOMINANCE) & (b > g * _DOMINANCE)
    return float(np.count_nonzero(blue_mask)) / blue_mask.size


def classify_entity(
    img_array: np.ndarray,
    bbox: tuple[float, float, float, float],
) -> tuple[Owner, float]:
    """Classify ownership of a single entity by checking blue pixel ratio.

    Samples two regions:
      1. Health bar zone — narrow band above/at the bounding box top
      2. Unit body zone — top 30% of the bounding box (livery/clothing)

    Args:
        img_array: Screenshot as HxWx3 uint8 numpy array (RGB).
        bbox: (x1, y1, x2, y2) bounding box coordinates.

    Returns:
        (Owner, blue_ratio) tuple.
    """
    h, w = img_array.shape[:2]
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

    # Clamp to image bounds
    x1 = max(0, min(x1, w - 1))
    x2 = max(x1 + 1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(y1 + 1, min(y2, h))

    best_ratio = 0.0

    # Region 1: Health bar zone (above / at top of bbox)
    hb_y1 = max(0, y1 - _HB_ABOVE)
    hb_y2 = min(h, y1 + _HB_HEIGHT - _HB_ABOVE)
    hb_region = img_array[hb_y1:hb_y2, x1:x2]
    if hb_region.size >= _MIN_REGION_PX * 3:
        best_ratio = max(best_ratio, _blue_ratio(hb_region))

    # Region 2: Unit body top 30% (livery/clothing color)
    body_height = int((y2 - y1) * 0.3)
    if body_height >= 3:
        body_region = img_array[y1 : y1 + body_height, x1:x2]
        if body_region.size >= _MIN_REGION_PX * 3:
            best_ratio = max(best_ratio, _blue_ratio(body_region))

    if best_ratio >= _OWN_THRESHOLD:
        return Owner.OWN, best_ratio
    elif best_ratio < 0.01 and (x2 - x1) < 10:
        # Too small to tell
        return Owner.UNKNOWN, best_ratio
    else:
        return Owner.ENEMY, best_ratio


def classify_entities(
    screenshot_bytes: bytes,
    entities: list,
    threat_classes: frozenset[str],
) -> dict[str, tuple[Owner, float]]:
    """Classify ownership for military entities in a screenshot.

    Only processes entities whose class_name is in threat_classes.
    Opens the screenshot once and reuses the numpy array.

    Args:
        screenshot_bytes: JPEG bytes of the full screenshot.
        entities: List of DetectedEntity objects.
        threat_classes: Set of military class names to classify.

    Returns:
        Dict mapping entity_id -> (Owner, blue_ratio).
    """
    # Open image once
    img = Image.open(io.BytesIO(screenshot_bytes)).convert("RGB")
    img_array = np.array(img)

    results: dict[str, tuple[Owner, float]] = {}

    for entity in entities:
        cls = entity.class_name if hasattr(entity, "class_name") else entity.get("class", "")
        if cls not in threat_classes:
            continue

        eid = entity.id if hasattr(entity, "id") else entity.get("id", "unknown")
        bbox = entity.bbox if hasattr(entity, "bbox") else entity.get("bbox", (0, 0, 0, 0))

        owner, ratio = classify_entity(img_array, bbox)
        results[eid] = (owner, ratio)
        log.debug(
            "ownership_classified",
            entity_id=eid,
            cls=cls,
            owner=owner.value,
            blue_ratio=f"{ratio:.3f}",
        )

    return results
