"""The live appearance signature matches the pinned tracker's, number for number."""

from __future__ import annotations

import numpy as np
import pytest
from gameplay_agent.appearance import appearance, prepare_detections
from gameplay_agent.entity_snapshot import EntitySnapshot
from gameplay_agent.temporal_entities import _appearance
from gameplay_agent.temporal_entities import prepare_detections as pinned_prepare


def _frame() -> np.ndarray:
    rgb = np.random.default_rng(7).integers(0, 256, (190, 300, 3), dtype=np.uint8)
    # Read-only, like a capture's pixels.
    return np.frombuffer(rgb.tobytes(), dtype=np.uint8).reshape(rgb.shape)


@pytest.mark.parametrize(
    "box",
    [
        (10, 20, 60, 90),
        (0, 0, 300, 190),  # The whole frame.
        (-15, -5, 40, 30),  # Partly off the frame.
        (250, 150, 400, 260),
        (120, 80, 121, 81),  # One pixel.
        (50, 50, 50, 80),  # Empty.
    ],
)
def test_the_signature_matches_the_pinned_tracker(box: tuple[float, float, float, float]) -> None:
    frame = _frame()
    assert appearance(frame, box) == pytest.approx(_appearance(frame, box), abs=1e-9)


def test_prepared_detections_match_and_skip_boxless_entities() -> None:
    frame = _frame()
    entities = [
        EntitySnapshot("a", "tree", (10, 20, 60, 90), (35, 55), 0.9),
        EntitySnapshot("b", "house", None, (0, 0), 0.9),
    ]
    fast, pinned = prepare_detections(entities, frame), pinned_prepare(entities, frame)
    assert [item.entity for item in fast] == [item.entity for item in pinned]
    assert fast[0].appearance == pytest.approx(pinned[0].appearance, abs=1e-9)
