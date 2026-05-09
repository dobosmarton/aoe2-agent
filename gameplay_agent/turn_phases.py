"""Per-turn decisions: context build, response parsing, action execution.

Owns the four pieces the game loop runs once per iteration:

  - `_get_ground_commands` / `_get_maintenance_actions`: hardcoded actions that
    run alongside the LLM call (zoom on turn 1, queue villagers while the LLM
    thinks). Pure functions of game state.
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
from typing import TYPE_CHECKING

import structlog

from .executor import clear_detected_entities, execute_actions
from .memory import GameState
from .models import validate_actions

if TYPE_CHECKING:
    from .goal_logger import GoalLogger
    from .goals import GoalManager
    from .memory import AgentMemory
    from .providers.base import LLMResult

log = structlog.get_logger()


# Per-turn cross-game memory attribution. The LLM is instructed (in core.md)
# to prefix its `reasoning` with `[applied: title1, title2]` when a memory
# rule directly drove its decision this turn. We strip the prefix before the
# reasoning is stored in working_memory so the tag doesn't pollute future
# turns' context.
_APPLIED_RE = re.compile(r"^\s*\[applied:\s*([^\]]+)\]", re.IGNORECASE)


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
_MAINTENANCE_POP_CAP: dict[str, int] = {
    "Dark Age": 22,
    "Feudal Age": 35,
}


def _get_maintenance_actions(memory: AgentMemory) -> list[dict]:
    """Safe hotkey actions to execute while the LLM call is in-flight."""
    pop = memory.game_state.population
    pop_cap = memory.game_state.population_cap
    age_cap = _MAINTENANCE_POP_CAP.get(memory.game_state.current_age, pop_cap)
    if pop < min(pop_cap, age_cap):
        return [
            {"type": "press", "key": "h", "intent": "Select TC (maintenance)"},
            {"type": "press", "key": "q", "intent": "Queue villager (maintenance)"},
        ]
    return []


def _build_llm_context(
    memory: AgentMemory,
    goal_manager: GoalManager,
    entity_summary: str,
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
        context = entity_context + "\n" + context

    return context


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


async def _execute_turn_actions(
    actions: list,
    iteration: int,
    memory: AgentMemory,
    reasoning: str,
) -> None:
    """Execute LLM actions or fallback actions.

    Ground commands (zoom, scout) are handled separately in the main loop
    to ensure they always run, even in agentic tool loop mode.
    """
    if actions:
        results = await execute_actions(actions)
        success_count = sum(1 for r in results if r.success)
        memory.record_action_results(success_count, len(actions))
        log.info(
            "actions_executed", iteration=iteration, total=len(actions), successful=success_count
        )

        verification_lines = []
        for action, result in zip(actions, results, strict=True):
            if not result.success:
                a_intent = action.get("intent", "") if isinstance(action, dict) else ""
                a_type = action.get("type", "") if isinstance(action, dict) else ""
                verification_lines.append(f"- FAILED {a_type}: {a_intent} — {result.detail}")
        if verification_lines:
            memory.set_last_verification("\n".join(verification_lines))
    else:
        log.warning("no_actions_fallback", iteration=iteration, reasoning=reasoning[:200])
        fallback = [
            {"type": "press", "key": "h", "intent": "Go to TC (fallback)"},
            {"type": "press", "key": "q", "intent": "Queue villager (fallback)"},
            {
                "type": "press",
                "key": ".",
                "rescan": True,
                "intent": "Select idle villager (fallback)",
            },
        ]
        fallback_actions = validate_actions(fallback)
        if fallback_actions:
            fb_results = await execute_actions(fallback_actions)
            fb_success = sum(1 for r in fb_results if r.success)
            memory.record_action_results(fb_success, len(fallback_actions))

    clear_detected_entities()
