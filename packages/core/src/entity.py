"""DetectedEntity dataclass — the schema YOLO inference emits and synthetic
perception projects into.

Lives in core because both `detection` (real inference) and `evaluation`
(synth render) produce it, and `gameplay_agent` consumes it. Putting it
here breaks the would-be cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DetectedEntity:
    """Represents a detected game entity."""

    id: str  # Unique ID: e.g., "sheep_0", "villager_1"
    class_name: str  # Entity class: "sheep", "villager", "tc"
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    center: tuple[float, float]  # Center point (x, y)
    confidence: float  # Detection confidence 0-1
    area: float = field(default=0)  # Bounding box area in pixels

    def to_dict(self) -> dict:
        """Convert to dictionary for LLM context."""
        return {
            "id": self.id,
            "class": self.class_name,
            "bbox": list(self.bbox),
            "center": self.center,
            "confidence": self.confidence,
        }
