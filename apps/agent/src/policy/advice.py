"""Immutable requests and judgments for bounded policy evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .allocation import AllocationFocus
    from .candidates import ActionCandidate, CandidateId
    from .state import PolicyState


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
    source_input_revision: int = 0
    goal_revision: int = 0
    recent_failures: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PolicyAdvice:
    """Two bounded TypeSafe judgments tied to their source frame."""

    source_tick: int
    source_captured_at: float
    model: str
    action_choice: CandidateId
    action_confidence: float
    action_probabilities: MappingProxyType[str, float]
    allocation_focus: AllocationFocus
    allocation_confidence: float
    allocation_probabilities: MappingProxyType[str, float]
    input_tokens: int | None = None
    output_tokens: int | None = None


def readonly_probabilities(values: dict[str, float]) -> MappingProxyType[str, float]:
    """Copy untrusted response data into an immutable internal mapping."""
    return MappingProxyType(dict(values))


__all__ = [
    "PolicyAdvice",
    "PolicyGoal",
    "PolicyRequest",
    "readonly_probabilities",
]
