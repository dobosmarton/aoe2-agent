"""Immutable policy judgments and their small mutable handoff store."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

from .candidates import find_candidate

if TYPE_CHECKING:
    from .allocation import AllocationFocus
    from .candidates import ActionCandidate, CandidateId
    from .state import PolicyState

AdviceStatus = Literal[
    "applicable",
    "consumed",
    "missing",
    "stale",
    "low_confidence",
    "candidate_unavailable",
]


@dataclass(frozen=True, slots=True)
class PolicyGoal:
    """Only the goal fields relevant to a bounded policy judgment."""

    name: str
    metric: str
    target: str
    priority: int
    progress: float


@dataclass(frozen=True, slots=True)
class PolicyRequest:
    """A frame-local policy question prepared for an external advisor."""

    source_tick: int
    source_captured_at: float
    state: PolicyState
    candidates: tuple[ActionCandidate, ...]
    goals: tuple[PolicyGoal, ...]


@dataclass(frozen=True, slots=True)
class PolicyAdvice:
    """Two bounded TypeSafe judgments tied to their source frame."""

    source_tick: int
    source_captured_at: float
    created_at: float
    model: str
    candidate_ids: frozenset[CandidateId]
    action_choice: CandidateId
    action_confidence: float
    action_probabilities: MappingProxyType[str, float]
    allocation_focus: AllocationFocus
    allocation_confidence: float
    allocation_probabilities: MappingProxyType[str, float]
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class AdviceResolution:
    """What the synchronous actor may safely use from cached advice."""

    status: AdviceStatus
    action: ActionCandidate | None = None
    allocation_focus: AllocationFocus | None = None
    advice: PolicyAdvice | None = None

    @property
    def should_fallback(self) -> bool:
        return self.status in {
            "missing",
            "stale",
            "low_confidence",
            "candidate_unavailable",
        }


class PolicyAdviceStore:
    """Single-event-loop handoff from the async advisor to the actor."""

    __slots__ = ("_consumed_tick", "_latest")

    def __init__(self) -> None:
        self._latest: PolicyAdvice | None = None
        self._consumed_tick: int | None = None

    def publish(self, advice: PolicyAdvice) -> None:
        """Atomically replace the whole judgment; older results cannot win."""
        latest = self._latest
        if latest is None or advice.source_tick >= latest.source_tick:
            self._latest = advice

    def latest(self) -> PolicyAdvice | None:
        return self._latest

    def inspect(
        self,
        candidates: tuple[ActionCandidate, ...],
        *,
        now: float,
        ttl_seconds: float,
        minimum_confidence: float,
    ) -> AdviceResolution:
        """Resolve cached advice without consuming its one-shot action."""
        advice = self._latest
        if advice is None:
            return AdviceResolution(status="missing")
        if now - advice.source_captured_at > ttl_seconds:
            return AdviceResolution(status="stale", advice=advice)

        action = find_candidate(candidates, advice.action_choice)
        if action is None:
            return AdviceResolution(status="candidate_unavailable", advice=advice)
        if advice.action_confidence < minimum_confidence:
            return AdviceResolution(status="low_confidence", advice=advice)

        allocation_focus = (
            advice.allocation_focus if advice.allocation_confidence >= minimum_confidence else None
        )
        status: AdviceStatus = (
            "consumed" if self._consumed_tick == advice.source_tick else "applicable"
        )
        return AdviceResolution(
            status=status,
            action=None if status == "consumed" else action,
            allocation_focus=allocation_focus,
            advice=advice,
        )

    def consume(
        self,
        candidates: tuple[ActionCandidate, ...],
        *,
        now: float,
        ttl_seconds: float,
        minimum_confidence: float,
    ) -> AdviceResolution:
        """Resolve advice and consume its spend action at most once."""
        resolution = self.inspect(
            candidates,
            now=now,
            ttl_seconds=ttl_seconds,
            minimum_confidence=minimum_confidence,
        )
        if resolution.status == "applicable" and resolution.advice is not None:
            self._consumed_tick = resolution.advice.source_tick
        return resolution


def readonly_probabilities(values: dict[str, float]) -> MappingProxyType[str, float]:
    """Copy untrusted response data into an immutable internal mapping."""
    return MappingProxyType(dict(values))


__all__ = [
    "AdviceResolution",
    "AdviceStatus",
    "PolicyAdvice",
    "PolicyAdviceStore",
    "PolicyGoal",
    "PolicyRequest",
    "readonly_probabilities",
]
