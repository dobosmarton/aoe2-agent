"""A frame source and an actuator with no game behind them.

The offline coverage the three clocks would otherwise lack. Phase 5.3 replaces
these with `world_sim`.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from gameplay_agent.executor import ActionResult, as_dict
from gameplay_agent.loops.snapshot import Perception
from gameplay_agent.loops.source import Sighting
from gameplay_agent.policy.advice import PolicyAdvice, readonly_probabilities

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from detection.inference.ownership import Owner
    from gameplay_agent.loops.source import Actuator, FrameSource
    from gameplay_agent.models import Action
    from gameplay_agent.policy.advice import PolicyRequest
    from gameplay_agent.providers.policy import PolicyAdvisor
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
        self.batches.append([as_dict(action) for action in actions])
        return [ActionResult(self.succeed, "ok") for _ in actions]

    @property
    def actions(self) -> list[dict[str, object]]:
        """Every action across every batch, in order."""
        return [action for batch in self.batches for action in batch]


class FakePolicyAdvisor:
    """Always chooses the safe no-spend candidate without calling a service."""

    def __init__(self) -> None:
        self.requests: list[PolicyRequest] = []
        self.closed = False

    async def advise(self, request: PolicyRequest) -> PolicyAdvice:
        self.requests.append(request)
        return PolicyAdvice(
            source_tick=request.source_tick,
            source_captured_at=request.source_captured_at,
            model="fake-system-one",
            action_choice="wait",
            action_confidence=1.0,
            action_probabilities=readonly_probabilities({"wait": 1.0}),
            allocation_focus="balanced",
            allocation_confidence=1.0,
            allocation_probabilities=readonly_probabilities({"balanced": 1.0}),
        )

    async def aclose(self) -> None:
        self.closed = True


if TYPE_CHECKING:
    # Not duck-typed by luck: basedpyright fails here if a fake drifts from the
    # Protocol the clocks are written against.
    _source: FrameSource = FakeSource()
    _actuator: Actuator = FakeActuator()
    _policy_advisor: PolicyAdvisor = FakePolicyAdvisor()


__all__ = ["FakeActuator", "FakePolicyAdvisor", "FakeSource"]
