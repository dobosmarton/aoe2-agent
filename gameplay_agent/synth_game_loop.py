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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from evaluation.world_sim import WorldState, apply_action, tick

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


_COST_DECIMAL_PLACES = 4


@dataclass
class SynthTurn:
    """Frozen audit record for one iteration of the synth loop."""

    turn_num: int
    state_before: WorldState
    actions: list[dict]
    reasoning: str
    cost_usd: float
    state_after: WorldState


@dataclass
class SynthLoopResult:
    final_state: WorldState
    turns: list[SynthTurn] = field(default_factory=list)
    total_cost_usd: float = 0.0


async def synth_game_loop(
    invoke: Callable[[WorldState], Awaitable[tuple[list[dict], str, float]]],
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
        turns=turns,
        total_cost_usd=round(total_cost, _COST_DECIMAL_PLACES),
    )
