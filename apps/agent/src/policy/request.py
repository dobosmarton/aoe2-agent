"""Pure construction of frame-local policy requests."""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
from typing import TYPE_CHECKING

from ..entity_utils import extract_attrs
from ..executor import pending_placement_counts
from ..villager_roles import gather_counts, infer_jobs, job_counts
from .advice import PolicyGoal, PolicyRequest
from .candidates import feasible_candidates
from .state import PolicyState, from_game_state

if TYPE_CHECKING:
    from ..loops.context import LoopContext
    from ..loops.snapshot import Perception


def policy_request(ctx: LoopContext, frame: Perception) -> PolicyRequest:
    """Build the narrow, immutable question for one exact perception frame."""
    state = state_for_frame(ctx, frame)
    goals = tuple(
        PolicyGoal(
            name=goal.name,
            metric=goal.metric,
            target=str(goal.target),
            priority=goal.priority,
            progress=goal.progress,
        )
        for goal in ctx.goal_manager.active_goals
    )
    return build_policy_request(
        source_tick=frame.tick,
        source_captured_at=frame.captured_at,
        state=state,
        goals=goals,
        source_input_revision=frame.input_revision,
        goal_revision=ctx.goal_manager.revision,
        recent_failures=tuple(ctx.ledger.recent_failures) if ctx.ledger is not None else (),
    )


def build_policy_request(
    *,
    source_tick: int,
    source_captured_at: float,
    state: PolicyState,
    goals: tuple[PolicyGoal, ...],
    source_input_revision: int = 0,
    goal_revision: int = 0,
    recent_failures: tuple[str, ...] = (),
) -> PolicyRequest:
    """Build a request from immutable domain values without side effects."""
    return PolicyRequest(
        source_tick=source_tick,
        source_captured_at=source_captured_at,
        state=state,
        candidates=feasible_candidates(state),
        goals=goals,
        source_input_revision=source_input_revision,
        goal_revision=goal_revision,
        recent_failures=recent_failures,
    )


def state_for_frame(ctx: LoopContext, frame: Perception) -> PolicyState:
    """Create the actor and advisor's common state from one perception."""
    entities = list(frame.entities)
    jobs = gather_counts(job_counts(infer_jobs(entities))) if entities else {}
    visible_classes = frozenset(extract_attrs(entity).class_name for entity in entities)
    if frame.world is not None:
        return replace(
            frame.world,
            villager_jobs=MappingProxyType(dict(jobs)),
            visible_classes=visible_classes,
            spatial_valid=frame.spatial_valid,
        )
    state = from_game_state(
        ctx.memory.game_state,
        captured_at=frame.captured_at,
        villager_jobs=jobs,
        pending_buildings=frozenset(pending_placement_counts()),
    )
    return replace(state, visible_classes=visible_classes, spatial_valid=frame.spatial_valid)


__all__ = ["build_policy_request", "policy_request", "state_for_frame"]
