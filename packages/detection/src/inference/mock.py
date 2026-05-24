"""Synthetic detection generator for tests / dev without YOLO weights.

Two entry points, each doing one thing:

- `mock_detect(screenshot, id_factory)` — frozen Dark-Age fixture
  (town center, 2-4 sheep, 3 villagers, 1 scout) sized to the screenshot.
  Used by the real-game tier's mock fallback and existing tests.
- `mock_detect_from_world(screenshot, id_factory, world_state)` — projects a
  `WorldState` (from the synthetic-arena tier) to detections at the
  screenshot's dimensions. Phase 1 of the synthetic-arena buildout.

The IDs assigned here are placeholders — they're reassigned downstream by
`_assign_persistent_ids` or the Kalman tracker — but we still need a callable
that mints them, so the caller passes one in.

`random.seed(42)` in `mock_detect` makes the same screenshot dimensions
produce the same detections; existing tests rely on this determinism.

NOTE (follow-up): the global `random.seed(42)` mutates process-wide RNG state
— a side effect. `mock_detect_from_world` uses a local RNG inside
`evaluation.world_sim.render` instead. The legacy path should migrate when
its test surface allows.
"""

from __future__ import annotations

import io
import random
from typing import TYPE_CHECKING

from evaluation.world_sim import WorldState, render
from PIL import Image

if TYPE_CHECKING:
    from collections.abc import Callable

    from .detector import DetectedEntity


_DEFAULT_FALLBACK_WIDTH = 1920
_DEFAULT_FALLBACK_HEIGHT = 1080


def _extract_screenshot_dims(screenshot: bytes | Image.Image) -> tuple[int, int]:
    if isinstance(screenshot, bytes):
        image = Image.open(io.BytesIO(screenshot))
        return image.size
    if hasattr(screenshot, "size"):
        return screenshot.size
    return _DEFAULT_FALLBACK_WIDTH, _DEFAULT_FALLBACK_HEIGHT


def mock_detect(
    screenshot: bytes | Image.Image,
    id_factory: Callable[[str], str],
) -> list[DetectedEntity]:
    """Generate plausible Dark-Age detections sized to the screenshot dimensions.

    `id_factory(class_name)` is called once per entity to mint its placeholder
    ID. Returned entities are sorted by class name, then confidence (desc).
    """
    # Deferred import — `detector.py` imports this module, so a top-level
    # `from .detector import DetectedEntity` would create a cycle.
    from .detector import DetectedEntity

    width, height = _extract_screenshot_dims(screenshot)
    random.seed(42)

    entities: list[DetectedEntity] = []

    tc_x = width * 0.5 + random.uniform(-100, 100)
    tc_y = height * 0.5 + random.uniform(-50, 50)
    entities.append(
        DetectedEntity(
            id=id_factory("town_center"),
            class_name="town_center",
            bbox=(tc_x - 80, tc_y - 60, tc_x + 80, tc_y + 60),
            center=(tc_x, tc_y),
            confidence=0.95,
            area=160 * 120,
        )
    )

    for _ in range(random.randint(2, 4)):
        sheep_x = tc_x + random.uniform(-200, 200)
        sheep_y = tc_y + random.uniform(-150, 150)
        entities.append(
            DetectedEntity(
                id=id_factory("sheep"),
                class_name="sheep",
                bbox=(sheep_x - 15, sheep_y - 10, sheep_x + 15, sheep_y + 10),
                center=(sheep_x, sheep_y),
                confidence=random.uniform(0.7, 0.95),
                area=30 * 20,
            )
        )

    for _ in range(3):
        vill_x = tc_x + random.uniform(-150, 150)
        vill_y = tc_y + random.uniform(-100, 100)
        entities.append(
            DetectedEntity(
                id=id_factory("villager"),
                class_name="villager",
                bbox=(vill_x - 12, vill_y - 20, vill_x + 12, vill_y + 5),
                center=(vill_x, vill_y),
                confidence=random.uniform(0.75, 0.92),
                area=24 * 25,
            )
        )

    scout_x = random.uniform(100, width - 100)
    scout_y = random.uniform(100, height - 100)
    entities.append(
        DetectedEntity(
            id=id_factory("scout_line"),
            class_name="scout_line",
            bbox=(scout_x - 15, scout_y - 18, scout_x + 15, scout_y + 8),
            center=(scout_x, scout_y),
            confidence=0.88,
            area=30 * 26,
        )
    )

    entities.sort(key=lambda e: (e.class_name, -e.confidence))

    return entities


def mock_detect_from_world(
    screenshot: bytes | Image.Image,
    id_factory: Callable[[str], str],
    world_state: WorldState,
) -> list[DetectedEntity]:
    """Project a `WorldState` to detections at the screenshot's dimensions.

    Synthetic-arena tier entry point. Unlike `mock_detect`, output depends on
    `world_state` and is fully deterministic per (state, dims, seed) — no
    global RNG mutation.
    """
    width, height = _extract_screenshot_dims(screenshot)
    return render(world_state, width, height, id_factory)
