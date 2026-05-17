"""Synthetic game loop — drives a WorldState through the executor LLM without
screenshots, real detection, or pyautogui.

Phase 2 of the synthetic-arena buildout. The loop is the foundation for
Phase 6 (multi-process racing). Each turn:

    invoke(state) → (actions, reasoning, cost)
    state = apply_action(state, action) for action in actions
    state = tick(state)

`invoke` is dependency-injected so tests pass a stub. The real driver wraps
the Anthropic SDK provider with a callable that builds context from the
WorldState and returns the parsed executor output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from evaluation.world_sim import WorldState, apply_action, tick

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


_COST_DECIMAL_PLACES = 4


# Action dict values are typed `object` rather than `Any` so any consumer that
# reads a specific field (e.g. `action["building_key"]`) must narrow with
# isinstance — keys vary by action type, so the dict shape is genuinely
# polymorphic. PEP 695 `type` aliases would tighten this further but require
# Python 3.12; this project targets 3.11.


@dataclass(frozen=True, slots=True)
class SynthTurn:
    """Audit record for one iteration of the synth loop."""

    turn_num: int
    state_before: WorldState
    actions: list[dict[str, object]]
    reasoning: str
    cost_usd: float
    state_after: WorldState


@dataclass(frozen=True, slots=True)
class SynthLoopResult:
    final_state: WorldState
    turns: tuple[SynthTurn, ...]
    total_cost_usd: float


async def synth_game_loop(
    invoke: Callable[[WorldState], Awaitable[tuple[list[dict[str, object]], str, float]]],
    initial_state: WorldState,
    max_iterations: int,
) -> SynthLoopResult:
    """Run `max_iterations` synthetic turns and return the audit trail.

    Each turn applies every action the executor returns, then ticks the world
    (gather, complete villagers, advance age). State is captured before and
    after the turn so callers can diff or replay.
    """
    state = initial_state
    turns: list[SynthTurn] = []
    total_cost = 0.0

    for turn_num in range(1, max_iterations + 1):
        state_before = state
        actions, reasoning, cost = await invoke(state)
        total_cost += cost
        for action in actions:
            state = apply_action(state, action)
        state = tick(state)
        turns.append(
            SynthTurn(
                turn_num=turn_num,
                state_before=state_before,
                actions=actions,
                reasoning=reasoning,
                cost_usd=cost,
                state_after=state,
            )
        )

    return SynthLoopResult(
        final_state=state,
        turns=tuple(turns),
        total_cost_usd=round(total_cost, _COST_DECIMAL_PLACES),
    )
