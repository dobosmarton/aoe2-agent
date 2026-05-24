"""Per-class confidence thresholds for entity detection.

Single source of truth used by both the detection server (server/app.py)
and the local detector (detection/inference/detector.py).
"""

from __future__ import annotations

# Lower thresholds for small or hard-to-detect entities
CLASS_THRESHOLDS: dict[str, float] = {
    "sheep": 0.20,
    "deer": 0.20,
    "berry_bush": 0.25,
    "villager": 0.25,
    "relic": 0.20,
}

DEFAULT_CONFIDENCE: float = 0.35


def get_threshold(class_name: str) -> float:
    """Return the confidence threshold for a given class name."""
    return CLASS_THRESHOLDS.get(class_name, DEFAULT_CONFIDENCE)
