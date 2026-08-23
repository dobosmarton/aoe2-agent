"""A frame source and an actuator with no game behind them.

The offline coverage the three clocks would otherwise lack. Phase 5.3 replaces
these with `world_sim`.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from gameplay_agent.executor import ActionResult, _as_dict
from gameplay_agent.loops.snapshot import Perception
from gameplay_agent.loops.source import Sighting

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from detection.inference.ownership import Owner
    from gameplay_agent.models import Action
    from gameplay_agent.turn_timing import TickTimings


class FakeSource:
    """Serves canned frames, then repeats the last one forever."""

    def __init__(
        self,
        frames: Sequence[Perception] | None = None,
        ownership: Mapping[str, tuple[Owner, float]] | None = None,
    ) -> None:
        self.frames = list(frames or [])
        self.ownership = dict(ownership or {})
        self.captures = 0
        self.closed = False

    async def capture(self, tick: int, timings: TickTimings) -> Sighting:
        with timings.phase("capture"):
            self.captures += 1
        index = min(self.captures - 1, len(self.frames) - 1)
        frame = self.frames[index] if self.frames else Perception()
        # The frame id is this pass's, not the canned frame's: `after` and the
        # act log both key on it.
        return Sighting(frame=replace(frame, tick=tick), ownership=self.ownership)

    def close(self) -> None:
        self.closed = True


class FakeActuator:
    """Records every batch instead of pressing a key."""

    def __init__(self, *, succeed: bool = True) -> None:
        self.batches: list[list[dict[str, object]]] = []
        self.succeed = succeed

    async def execute(self, actions: Sequence[Action | dict[str, object]]) -> list[ActionResult]:
        self.batches.append([_as_dict(action) for action in actions])
        return [ActionResult(self.succeed, "ok") for _ in actions]

    @property
    def actions(self) -> list[dict[str, object]]:
        """Every action across every batch, in order."""
        return [action for batch in self.batches for action in batch]


__all__ = ["FakeActuator", "FakeSource"]
