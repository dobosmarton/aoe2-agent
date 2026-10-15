"""Per-turn context and response bookkeeping for deliberate model turns.

Owns the four pieces the game loop runs once per iteration:

  - `get_ground_commands`: hardcoded actions that run alongside the LLM call
    (zoom on turn 1). Pure function of the iteration number.
  - `build_llm_context`: stitches memory + goals + entities into the prompt.
  - `record_llm_turn`: parses the LLM's response, strips the
    `[applied: ...]` memory-attribution prefix, snapshots state for reward
    computation, and returns `(actions, game_end_reason | None)`.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import structlog

from .entity_utils import extract_attrs
from .executor import (
    CASTLE_PREREQ_COUNT,
    FEUDAL_PREREQ_CLASSES,
    blocked_actions,
    confirmed_buildings,
    pending_placement_counts,
    sighted_buildings,
)
from .memory import GameState
from .villager_roles import infer_jobs, job_counts

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .goal_logger import GoalLogger
    from .goals import GoalManager
    from .memory import AgentMemory
    from .policy.state import PolicyState
    from .providers.base import LLMResult

log = structlog.stdlib.get_logger()


# Per-turn cross-game memory attribution. The LLM is instructed (in core.md)
# to prefix its `reasoning` with `[applied: title1, title2]` when a memory
# rule directly drove its decision this turn. We strip the prefix before the
# reasoning is stored in working_memory so the tag doesn't pollute future
# turns' context.
_APPLIED_RE = re.compile(r"^\s*\[applied:\s*([^\]]+)\]", re.IGNORECASE)

# Consecutive failed executor turns before raising the outage alarm (T-533).
# One-off API blips shouldn't shout; a systemic outage (run 12: 90 in a row,
# the schema-grammar 400) must. Alarm fires once on the transition through the
# threshold so a long outage produces one loud line, not one per turn.
_EXECUTOR_OUTAGE_STREAK = 3


def _extract_applied_memories(
    reasoning: str, loaded_titles: set[str]
) -> tuple[list[str], list[str], str]:
    """Parse the `[applied: ...]` prefix from a reasoning string.

    Returns (known_titles, unknown_titles, cleaned_reasoning):
      - known_titles: titles that match `loaded_titles` (counted in metrics)
      - unknown_titles: titles the LLM made up — logged as a warning
      - cleaned_reasoning: reasoning without the prefix
    """
    m = _APPLIED_RE.match(reasoning or "")
    if not m:
        return [], [], reasoning
    raw = [t.strip() for t in m.group(1).split(",") if t.strip()]
    known = [t for t in raw if t in loaded_titles]
    unknown = [t for t in raw if t not in loaded_titles]
    cleaned = reasoning[m.end() :].lstrip()
    return known, unknown, cleaned


INITIAL_ZOOM_CLICKS = 5


def get_ground_commands(iteration: int) -> list[dict]:
    """Return hardcoded actions injected BEFORE LLM actions each turn."""
    if iteration != 1:
        return []
    return [
        {
            "type": "scroll",
            "clicks": INITIAL_ZOOM_CLICKS,
            "intent": "Zoom in for better object detection",
        },
        {"type": "press", "key": ",", "intent": "Select scout (ground cmd)"},
        {"type": "press", "key": "g", "intent": "Auto Scout (ground cmd)"},
    ]


def build_llm_context(
    memory: AgentMemory,
    goal_manager: GoalManager,
    entity_summary: str,
    detected_entities: list[object] | None = None,
    observed_state: PolicyState | None = None,
) -> str:
    """Assemble the full text context for the executor LLM."""
    context = memory.get_context_for_llm(observed_state)

    resource_context = goal_manager.get_resource_context() if observed_state is None else ""
    if resource_context:
        context = resource_context + "\n\n" + context

    goal_context = goal_manager.get_context_for_llm()
    if goal_context:
        context = goal_context + "\n\n" + context

    if entity_summary:
        entity_context = (
            "\n## Detected Entities (from YOLO)\n"
            "Use target_class or target_id to interact with these:\n" + entity_summary + "\n"
        )
        entity_context += _villager_jobs_line(detected_entities)
        entity_context += known_buildings_line(detected_entities)
        entity_context += castle_gate_line(memory.game_state.current_age)
        entity_context += blocked_actions_line()
        context = entity_context + "\n" + context

    return context


def _villager_jobs_line(detected_entities: list[object] | None) -> str:
    """One-line villager-by-job breakdown so the LLM can rebalance the economy.

    Inferred from villager↔resource proximity (a single YOLO `villager` class can't
    say who is on wood). Empty when no villagers are visible; zero-count kinds are
    omitted to keep the line short.
    """
    if not detected_entities:
        return ""
    counts = job_counts(infer_jobs(detected_entities))
    working = {kind: n for kind, n in counts.items() if n}
    if not working:
        return ""
    breakdown = " ".join(f"{kind}={n}" for kind, n in working.items())
    return f"Villagers by job (approx, from proximity): {breakdown}\n"


def blocked_actions_line() -> str:
    """One line of what the executor will refuse this turn.

    A FACT beside the other two, not a rule: the agent chooses what to do with
    it. Run 2026_08_22_1 spent 94 of its 138 refused actions on state it could
    not see.
    """
    blocked = blocked_actions()
    if not blocked:
        return ""
    return "Currently refused: " + " ".join(blocked) + "\n"


def castle_gate_line(age: str) -> str:
    """One line of the Castle Age's building requirement, as a count.

    A FACT, not an instruction: the prompt already explains what the gate means,
    and every tier can read a number. Run 2026_08_21_2 pressed the age-up key 10
    times with the resources banked and zero qualifying buildings standing.
    Silent outside Feudal, where the gate does not apply yet.
    """
    if not age.startswith("Feudal"):
        return ""
    have = sorted(confirmed_buildings() & FEUDAL_PREREQ_CLASSES)
    names = " ".join(have) if have else "none"
    return (
        f"Feudal-Age buildings: {len(have)}/{CASTLE_PREREQ_COUNT} ({names}) — "
        f"the Castle Age needs {CASTLE_PREREQ_COUNT} of "
        f"{', '.join(sorted(FEUDAL_PREREQ_CLASSES))}\n"
    )


def _class_counts(entities: Sequence[object]) -> dict[str, int]:
    """Count visible entities by class for the known-building context."""
    counts: dict[str, int] = {}
    for entity in entities:
        class_name = extract_attrs(entity).class_name
        counts[class_name] = counts.get(class_name, 0) + 1
    return counts


def known_buildings_line(detected_entities: list[object] | None) -> str:
    """One-line owned-building ledger so the LLM stops re-building what it has.

    Only PURCHASE-confirmed classes count as owned — detection can't vouch for
    ownership (F-29/F-36: phantom mills) — with this frame's detected count
    floored at 1: a confirmed building that's off-screen still exists. Pending
    placements (awaiting wood-delta settlement) are shown so a build already
    in flight isn't re-ordered; persistent detections nothing proved are
    flagged unverified so the LLM doesn't mistake them for owned.
    """
    confirmed = confirmed_buildings()
    pending = pending_placement_counts()
    unverified = sighted_buildings() - confirmed
    if not confirmed and not pending and not unverified:
        return ""
    segments: list[str] = []
    if confirmed:
        detected = _class_counts(detected_entities or [])
        segments.append(" ".join(f"{c}={max(detected.get(c, 0), 1)}" for c in sorted(confirmed)))
    if pending:
        segments.append(
            "(pending: " + " ".join(f"{c}={n}" for c, n in sorted(pending.items())) + ")"
        )
    if unverified:
        segments.append("(unverified sightings, NOT owned: " + " ".join(sorted(unverified)) + ")")
    return "Known buildings: " + " ".join(segments) + "\n"


def record_llm_turn(
    response: LLMResult,
    memory: AgentMemory,
    goal_manager: GoalManager,
    iteration: int,
    goal_logger: GoalLogger,
) -> list[dict[str, object]]:
    """Parse one LLM answer and book the turn it belongs to.

    Every metric that scales with LLM turns hangs off `create_turn` here —
    `turn_count`, the goal reward, and the saved trace. Returns its actions.
    """
    reasoning = response.get("reasoning", "")
    observations = response.get("observations", {})
    actions = response.get("actions", [])

    # Executor-health accounting (T-533). response["error"] is True only when
    # every LLM path failed and the turn is a safe-wait no-op; count it and
    # alarm on a sustained outage so a dead-executor run (run 12: 90 grammar
    # 400s, still accepted=true) is loud in the log and in llm_error_rate.
    errored = bool(response.get("error", False))
    streak = memory.record_llm_outcome(errored=errored)
    if errored and streak == _EXECUTOR_OUTAGE_STREAK:
        log.error(
            "executor_outage",
            iteration=iteration,
            consecutive_failures=streak,
            detail=reasoning[:200],
            hint="the tactical LLM is failing; routine TypeSafe policy remains active",
        )

    loaded = set(memory.memories_loaded)
    known_titles, unknown_titles, reasoning = _extract_applied_memories(reasoning, loaded)
    if known_titles:
        memory.record_memories_applied(known_titles)
        log.info("memories_applied", iteration=iteration, titles=known_titles)
    if unknown_titles:
        log.warning(
            "memories_applied_unknown",
            iteration=iteration,
            titles=unknown_titles,
            loaded=sorted(loaded),
        )

    log.info(
        "llm_response",
        iteration=iteration,
        reasoning=reasoning[:100] + "..." if len(reasoning) > 100 else reasoning,
        action_count=len(actions),
    )

    prev_state = GameState(
        resources=dict(memory.game_state.resources),
        population=memory.game_state.population,
        population_cap=memory.game_state.population_cap,
        current_age=memory.game_state.current_age,
    )
    turn = memory.create_turn(reasoning=reasoning, actions=actions, observations=observations)

    goal_manager.evaluate_progress(memory.game_state, iteration)
    reward = goal_manager.compute_turn_reward(prev_state, memory.game_state)
    turn.reward = reward.get("total", 0.0)
    goal_logger.log_progress(iteration, goal_manager.active_goals, reward)

    for goal in goal_manager.completed_goals:
        if goal.created_turn != iteration:
            goal_logger.log_goal_completed(iteration, goal)

    if reward["total"] != 0:
        log.info("turn_reward", iteration=iteration, **reward)

    return actions


__all__ = [
    "INITIAL_ZOOM_CLICKS",
    "blocked_actions_line",
    "build_llm_context",
    "castle_gate_line",
    "get_ground_commands",
    "known_buildings_line",
    "record_llm_turn",
]
