"""Per-turn decisions: context build, response parsing, action execution.

Owns the four pieces the game loop runs once per iteration:

  - `_get_ground_commands`: hardcoded actions that run alongside the LLM call
    (zoom on turn 1). Pure function of the iteration number.
  - `_build_llm_context`: stitches memory + goals + entities into the prompt
    string the executor sees. Mirrors `evaluation.context_builder._build_context`.
  - `_process_response`: parses the LLM's response, strips the
    `[applied: ...]` memory-attribution prefix, snapshots state for reward
    computation, and returns `(actions, game_end_reason | None)`.
  - `_execute_turn_actions`: runs the validated actions through the executor
    and records success/failure feedback into memory, with a hardcoded
    fallback when the LLM returns no actions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import structlog

from .entity_utils import extract_attrs
from .executor import (
    CAMERA_KEYS,
    build_steps,
    clear_detected_entities,
    confirmed_buildings,
    execute_actions,
    get_detected_entities,
    get_rescan_fn,
    pending_placement_counts,
    sighted_buildings,
)
from .memory import GameState
from .models import validate_actions
from .villager_roles import infer_jobs, job_counts

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .executor import ActionResult
    from .goal_logger import GoalLogger
    from .goals import GoalManager
    from .memory import AgentMemory
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


def _get_ground_commands(iteration: int) -> list[dict]:
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


# Age-dependent population thresholds for maintenance villager queuing.
# Beyond these caps, stop queuing to save food for age-up research.
def _build_llm_context(
    memory: AgentMemory,
    goal_manager: GoalManager,
    entity_summary: str,
    detected_entities: list[object] | None = None,
) -> str:
    """Assemble the full text context for the executor LLM."""
    context = memory.get_context_for_llm()

    resource_context = goal_manager.get_resource_context()
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


def _process_response(
    response: LLMResult,
    memory: AgentMemory,
    goal_manager: GoalManager,
    iteration: int,
    goal_logger: GoalLogger,
    time_budget: float | None,
) -> tuple[list[dict[str, object]], str | None]:
    """Parse LLM response, update memory/goals, check for game-over.

    Returns (actions, game_end_reason). game_end_reason is None if game continues.
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
            hint="every LLM path is failing; the reactive tier alone cannot build a mill",
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

    game_state = observations.get("game_state", "playing") if observations else "playing"
    if game_state in ("victory", "defeat"):
        memory.game_end_reason = game_state
        log.info("game_over_detected", result=game_state, iteration=iteration)
        return actions, game_state

    if time_budget and memory.get_game_duration_seconds() >= time_budget:
        memory.game_end_reason = "timeout"
        log.info("time_budget_reached", seconds=time_budget, iteration=iteration)
        return actions, "timeout"

    return actions, None


# ---------------------------------------------------------------------------
# Action-effect verification (R1)
# ---------------------------------------------------------------------------

# Building classes whose appearance confirms a successful build/place action.
_BUILDING_CLASSES: frozenset[str] = frozenset(
    {
        "town_center",
        "house",
        "lumber_camp",
        "mining_camp",
        "mill",
        "market",
        "dock",
        "farm",
        "barracks",
        "archery_range",
        "stable",
        "blacksmith",
        "siege_workshop",
        "monastery",
        "castle",
        "university",
        "gate",
        "wall",
        "tower",
        "wonder",
        "krepost",
    }
)


@dataclass(frozen=True, slots=True)
class _Expectation:
    """The observable effect an action should produce, if any."""

    kind: Literal["new_building", "selection_change", "none"]
    detail: str


def _expectation_for(action: dict) -> _Expectation:
    """Map an action to its verifiable effect (bounded and honest)."""
    a_type = str(action.get("type", ""))
    intent = str(action.get("intent", "")).lower()
    if a_type == "build" or (
        a_type in ("click", "right_click") and ("build" in intent or "place" in intent)
    ):
        return _Expectation("new_building", "build/place should add a building")
    if a_type == "press":
        key = str(action.get("key", "")).lower()
        if action.get("rescan") or key in CAMERA_KEYS:
            return _Expectation("selection_change", f"press {key} should change the view")
    return _Expectation("none", "")


def _any_entity_expectation(actions: list) -> bool:
    """True if any action expects an entity/view change worth a rescan."""
    return any(_expectation_for(a).kind != "none" for a in actions if isinstance(a, dict))


def _class_counts(entities: Sequence[object]) -> dict[str, int]:
    """Count detected entities by class (robust to bbox jitter)."""
    counts: dict[str, int] = {}
    for e in entities:
        cls = extract_attrs(e).class_name
        counts[cls] = counts.get(cls, 0) + 1
    return counts


def _new_buildings(before: list[dict], after: list[dict]) -> list[str]:
    """Building classes whose count increased from before → after."""
    bc, ac = _class_counts(before), _class_counts(after)
    return sorted(cls for cls in _BUILDING_CLASSES if ac.get(cls, 0) > bc.get(cls, 0))


def _failed_lines(actions: list, results: list[ActionResult]) -> list[str]:
    """Verification lines for actions the executor reported as failed."""
    lines: list[str] = []
    for action, result in zip(actions, results, strict=True):
        if not result.success:
            a_intent = action.get("intent", "") if isinstance(action, dict) else ""
            a_type = action.get("type", "") if isinstance(action, dict) else ""
            lines.append(f"- FAILED {a_type}: {a_intent} — {result.detail}")
    return lines


def _build_verification(
    actions: list,
    results: list[ActionResult],
    before_entities: list[dict],
    after_entities: list[dict],
) -> str:
    """Combine executor failures with positive/negative effect checks.

    Emits the exact phrase "no visible change" on unmet expectations so the
    stuck-loop detector in AgentMemory.get_context_for_llm picks it up.
    """
    lines = _failed_lines(actions, results)
    kinds = {_expectation_for(a).kind for a in actions if isinstance(a, dict)}
    if "new_building" in kinds:
        built = _new_buildings(before_entities, after_entities)
        if built:
            lines.append(f"- CONFIRMED built: {', '.join(built)}")
        else:
            lines.append("- no visible change: build produced no new building")
    counts_unchanged = _class_counts(before_entities) == _class_counts(after_entities)
    if "selection_change" in kinds and counts_unchanged:
        lines.append("- no visible change: view unchanged after camera action")
    return "\n".join(lines)


def _fallback_actions(memory: AgentMemory) -> list[dict[str, object]]:
    """Pick the actions to run on a turn where the LLM returned none.

    While housed, queuing a villager fails (no population room), so build a house
    to unfreeze growth — the executor chooses the placement, since the text-only
    model can't see open ground. Otherwise nudge production: go to the Town
    Center, queue a villager, and select an idle one.
    """
    state = memory.game_state
    is_housed = state.population_cap > 0 and state.population >= state.population_cap
    if is_housed:
        return build_steps("q", "Place house to raise pop cap (fallback build)")
    return [
        {"type": "queue_villager", "intent": "Queue villager (fallback)"},
        {"type": "press", "key": ".", "rescan": True, "intent": "Select idle villager (fallback)"},
    ]


async def _execute_turn_actions(
    actions: list,
    iteration: int,
    memory: AgentMemory,
    reasoning: str,
) -> None:
    """Execute LLM actions or fallback actions.

    Ground commands (zoom, scout) are handled separately in the main loop
    to ensure they always run, even in agentic tool loop mode. After
    entity-affecting actions, re-detect and record positive/negative
    verification (R1).
    """
    if actions:
        verify = _any_entity_expectation(actions)
        before_entities = list(get_detected_entities()) if verify else []

        results = await execute_actions(actions)
        success_count = sum(1 for r in results if r.success)
        memory.record_action_results(success_count, len(actions))
        log.info(
            "actions_executed", iteration=iteration, total=len(actions), successful=success_count
        )

        if verify:
            rescan = get_rescan_fn()
            if rescan is not None:
                await rescan()
            verification = _build_verification(
                actions, results, before_entities, list(get_detected_entities())
            )
        else:
            verification = "\n".join(_failed_lines(actions, results))
        if verification:
            memory.set_last_verification(verification)
    else:
        log.warning("no_actions_fallback", iteration=iteration, reasoning=reasoning[:200])
        fallback_actions = validate_actions(_fallback_actions(memory))
        if fallback_actions:
            fb_results = await execute_actions(fallback_actions)
            fb_success = sum(1 for r in fb_results if r.success)
            memory.record_action_results(fb_success, len(fallback_actions))

    clear_detected_entities()
