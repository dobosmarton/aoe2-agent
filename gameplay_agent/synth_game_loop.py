"""Synthetic game loop — drives a WorldState through the executor LLM without
screenshots, real detection, or pyautogui.

Phase 2 of the synthetic-arena buildout (extended by Phase 4 to emit events).
The loop is the foundation for Phase 6 (multi-process racing). Each turn:

    invoke(state) → (actions, reasoning, cost)
    state = apply_action(state, action) for action in actions
    state = tick(state)

`invoke` is dependency-injected so tests pass a stub. The real driver wraps
the Anthropic SDK provider with a callable that builds context from the
WorldState and returns the parsed executor output.

Each turn emits a sequence of events to the injected `sink` (default
NullEventSink). Phase 4 emits 6 of the 9 design-doc §C event kinds:
turn_start, llm_prompt, llm_response, action, action_result, metric.
observation/world_mutation/fork are intentionally not emitted here —
they need data outside the loop's scope (entity rendering / operator
hooks / forking, respectively).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from evaluation.event_log import (
    ActionPayload,
    ActionResultPayload,
    Event,
    LlmPromptPayload,
    LlmResponsePayload,
    MetricPayload,
    NullEventSink,
    TurnStartPayload,
)
from evaluation.world_sim import WorldState, apply_action, tick

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from evaluation.event_log import EventSink, Payload


_COST_DECIMAL_PLACES = 4
_NULL_SINK = NullEventSink()


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
    run_id: str
    agent_id: str


def _summarize_state(state: WorldState) -> str:
    return (
        f"turn={state.turn} age={state.age} food={int(state.food)} "
        f"wood={int(state.wood)} pop={state.population}/{state.pop_cap} "
        f"buildings={len(state.buildings)}"
    )


def _emit(sink: EventSink, run_id: str, agent_id: str, t: int, payload: Payload) -> None:
    sink.emit(
        Event(
            run_id=run_id,
            agent_id=agent_id,
            t=t,
            payload=payload,
            ts=datetime.now(UTC),
        )
    )


async def synth_game_loop(
    invoke: Callable[[WorldState], Awaitable[tuple[list[dict[str, object]], str, float]]],
    initial_state: WorldState,
    max_iterations: int,
    sink: EventSink = _NULL_SINK,
) -> SynthLoopResult:
    """Run `max_iterations` synthetic turns and return the audit trail.

    Each turn applies every action the executor returns, then ticks the world
    (gather, complete villagers, advance age). State is captured before and
    after the turn so callers can diff or replay.

    Events are emitted to `sink` (default: NullEventSink — no persistence).
    Provide a DuckDBEventSink to persist; both share the EventSink protocol.
    """
    run_id = uuid.uuid4().hex
    agent_id = uuid.uuid4().hex
    state = initial_state
    turns: list[SynthTurn] = []
    total_cost = 0.0

    for turn_num in range(1, max_iterations + 1):
        state_before = state
        _emit(sink, run_id, agent_id, turn_num, TurnStartPayload(turn_num=turn_num))
        _emit(
            sink,
            run_id,
            agent_id,
            turn_num,
            LlmPromptPayload(state_summary=_summarize_state(state_before)),
        )

        actions, reasoning, cost = await invoke(state)
        total_cost += cost
        _emit(
            sink,
            run_id,
            agent_id,
            turn_num,
            LlmResponsePayload(actions=actions, reasoning=reasoning, cost_usd=cost),
        )

        for index, action in enumerate(actions):
            _emit(
                sink, run_id, agent_id, turn_num, ActionPayload(index_in_turn=index, action=action)
            )
            state_before_action = state
            state = apply_action(state, action)
            action_type = action.get("type", "")
            _emit(
                sink,
                run_id,
                agent_id,
                turn_num,
                ActionResultPayload(
                    index_in_turn=index,
                    action_type=str(action_type),
                    state_changed=state != state_before_action,
                ),
            )

        state = tick(state)
        _emit(
            sink,
            run_id,
            agent_id,
            turn_num,
            MetricPayload(name="cost_usd", value=cost),
        )

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
        run_id=run_id,
        agent_id=agent_id,
    )
