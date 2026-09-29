"""Bounded economy obligations around the dynamic action selector."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .allocation import Allocation, for_state, next_kind

if TYPE_CHECKING:
    from collections.abc import Callable

    from .candidates import ActionCandidate
    from .state import PolicyState

_SERVICE_DEADLINE_SECONDS = 5.0
_IDLE_WORKER = "idle_worker"
_TC_PRODUCTION = "tc_production"


def _obligation_for(action_id: str) -> str | None:
    if action_id.startswith("assign_"):
        return _IDLE_WORKER
    if action_id == "queue_villager":
        return _TC_PRODUCTION
    return None


def essential_intentions(
    state: PolicyState,
    candidates: tuple[ActionCandidate, ...],
    allocation: Allocation | None,
) -> tuple[str, ...]:
    """Named actions needed for basic growth; feasibility stays in the catalog."""
    available = {candidate.id for candidate in candidates}
    due: list[str] = []
    if state.idle_present and not state.assignment_pending:
        preferred = next_kind(for_state(state, allocation), state.villager_jobs)
        for kind in (preferred, "food", "wood", "gold", "stone"):
            action_id = f"assign_{kind}"
            if action_id in available:
                due.append(action_id)
                break
    if state.pending_villagers == 0 and "queue_villager" in available:
        due.append("queue_villager")
    return tuple(due)


@dataclass(slots=True)
class AgentController:
    """Remember when eligible essential work first became observable."""

    clock: Callable[[], float] = time.monotonic
    first_seen: dict[str, float] = field(default_factory=dict)

    def overdue(
        self,
        state: PolicyState,
        candidates: tuple[ActionCandidate, ...],
        allocation: Allocation | None,
        *,
        now: float | None = None,
    ) -> ActionCandidate | None:
        instant = self.clock() if now is None else now
        essential = essential_intentions(state, candidates, allocation)
        observed = set()
        if not state.assignment_pending and (
            state.idle_present is True
            or (state.idle_present is None and _IDLE_WORKER in self.first_seen)
        ):
            observed.add(_IDLE_WORKER)
        if state.pending_villagers == 0 and (
            state.population_known or _TC_PRODUCTION in self.first_seen
        ):
            observed.add(_TC_PRODUCTION)
        self.first_seen = {key: stamp for key, stamp in self.first_seen.items() if key in observed}
        for obligation in observed:
            self.first_seen.setdefault(obligation, min(state.captured_at, instant))
        due: list[tuple[ActionCandidate, float]] = []
        for candidate in candidates:
            obligation = _obligation_for(candidate.id)
            if (
                candidate.id in essential
                and obligation is not None
                and instant - self.first_seen[obligation] >= _SERVICE_DEADLINE_SECONDS
            ):
                due.append((candidate, self.first_seen[obligation]))
        if not due:
            return None
        # Oldest evidence wins; the declared order breaks simultaneous ties.
        order = {action_id: index for index, action_id in enumerate(essential)}
        return min(due, key=lambda item: (item[1], order[item[0].id]))[0]

    def record_attempt(self, action_id: str) -> None:
        """A handled action receives a new deadline if it stays eligible."""
        obligation = _obligation_for(action_id)
        if obligation is not None:
            self.first_seen.pop(obligation, None)


__all__ = ["AgentController", "essential_intentions"]
