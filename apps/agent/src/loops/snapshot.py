"""What the three loops hand each other: one frame, one slot to put it in.

The loops are asyncio tasks in one thread, so a slot needs no lock. The
discipline that replaces one: the writer builds a whole new frame and swaps the
reference, so a reader never sees a half-built one. The frame is replaced whole,
not deep-frozen — `entities` and `hud_readings` still hold plain dicts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from ..resource_ocr import ResourceReadings
from ..turn_timing import elapsed_ms


@dataclass(frozen=True, slots=True)
class Perception:
    """One frame, shared by reference across the three loops.

    `captured_at` and `age_ms` mirror `policy.state.PolicyState`, so a frame and
    the state built from it answer the freshness question the same way.
    """

    screenshot: bytes = b""
    width: int = 0
    height: int = 0
    entities: tuple[dict[str, object], ...] = ()
    entity_summary: str = ""
    hud_readings: ResourceReadings = field(default_factory=ResourceReadings)
    alarm: bool = False
    tick: int = 0
    captured_at: float = field(default_factory=time.monotonic)

    @property
    def age_ms(self) -> float:
        """Milliseconds since this frame was captured."""
        return elapsed_ms(self.captured_at)


T = TypeVar("T")  # PEP 695 syntax needs 3.12; this repo targets 3.11.


class Slot(Generic[T]):
    """One value, replaced whole. Empty until the first `put`."""

    __slots__ = ("_value",)

    def __init__(self, value: T | None = None) -> None:
        self._value = value

    def put(self, value: T) -> None:
        self._value = value

    def get(self) -> T | None:
        """The current value, or None. The reader must tolerate an empty slot:
        the act loop starts before the first frame exists."""
        return self._value


__all__ = ["Perception", "Slot"]
