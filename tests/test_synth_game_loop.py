"""Unit tests for gameplay_agent/synth_game_loop.py.

Uses a recording stub for `invoke` — no LLM calls, no API key required.
All tests are offline and deterministic.
"""

from __future__ import annotations

import asyncio

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state(**kwargs):
    from evaluation.world_sim import WorldState

    defaults = {
        "food": 200.0,
        "wood": 150.0,
        "gold": 0.0,
        "stone": 0.0,
        "population": 8,
        "pop_cap": 25,
        "age": "Dark Age",
        "buildings": [],
        "villager_queue": [],
        "age_up_ticks_remaining": 0,
        "turn": 0,
    }
    defaults.update(kwargs)
    return WorldState(**defaults)


class _RecordingStub:
    """Async callable mimicking the `invoke` contract. Records every state it
    was called with and returns canned responses in order."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.received_states = []

    async def __call__(self, state):
        self.received_states.append(state)
        return self.responses.pop(0)


def _run(coro):
    return asyncio.run(coro)


def _empty_turn(cost=0.0):
    return ([], "", cost)


# ---------------------------------------------------------------------------
# synth_game_loop — Phase 2
# ---------------------------------------------------------------------------


def test_synth_game_loop_runs_requested_iterations():
    from gameplay_agent.synth_game_loop import synth_game_loop

    invoke = _RecordingStub([_empty_turn() for _ in range(5)])
    result = _run(synth_game_loop(invoke, _state(), max_iterations=5))
    assert len(result.turns) == 5


def test_synth_game_loop_zero_iterations_returns_initial_state():
    from gameplay_agent.synth_game_loop import synth_game_loop

    initial = _state(turn=7)
    invoke = _RecordingStub([])
    result = _run(synth_game_loop(invoke, initial, max_iterations=0))
    assert result.final_state == initial


def test_synth_game_loop_advances_turn_counter():
    from gameplay_agent.synth_game_loop import synth_game_loop

    invoke = _RecordingStub([_empty_turn() for _ in range(3)])
    result = _run(synth_game_loop(invoke, _state(turn=0), max_iterations=3))
    assert result.final_state.turn == 3


def test_synth_game_loop_applies_actions_to_state():
    from gameplay_agent.synth_game_loop import synth_game_loop

    invoke = _RecordingStub([([{"type": "queue_villager"}], "queueing", 0.0)])
    result = _run(synth_game_loop(invoke, _state(food=200.0), max_iterations=1))
    # Action deducts 50 food; tick adds 20 food gather. Net: 200 - 50 + 20 = 170.
    assert result.final_state.food == 170.0


def test_synth_game_loop_invokes_with_current_state_each_turn():
    from gameplay_agent.synth_game_loop import synth_game_loop

    invoke = _RecordingStub([_empty_turn() for _ in range(3)])
    _run(synth_game_loop(invoke, _state(turn=0), max_iterations=3))
    turn_numbers_seen = [s.turn for s in invoke.received_states]
    assert turn_numbers_seen == [0, 1, 2]


def test_synth_game_loop_accumulates_cost():
    from gameplay_agent.synth_game_loop import synth_game_loop

    invoke = _RecordingStub(
        [_empty_turn(cost=0.01), _empty_turn(cost=0.02), _empty_turn(cost=0.03)]
    )
    result = _run(synth_game_loop(invoke, _state(), max_iterations=3))
    assert result.total_cost_usd == 0.06


def test_synth_game_loop_rounds_total_cost_to_four_decimals():
    from gameplay_agent.synth_game_loop import synth_game_loop

    invoke = _RecordingStub([_empty_turn(cost=0.00001), _empty_turn(cost=0.00001)])
    result = _run(synth_game_loop(invoke, _state(), max_iterations=2))
    assert result.total_cost_usd == 0.0


def test_synth_game_loop_captures_per_turn_actions():
    from gameplay_agent.synth_game_loop import synth_game_loop

    invoke = _RecordingStub([([{"type": "queue_villager"}], "vill", 0.0)])
    result = _run(synth_game_loop(invoke, _state(food=200.0), max_iterations=1))
    assert result.turns[0].actions == [{"type": "queue_villager"}]


def test_synth_game_loop_captures_per_turn_reasoning():
    from gameplay_agent.synth_game_loop import synth_game_loop

    invoke = _RecordingStub([([], "go forth and multiply", 0.0)])
    result = _run(synth_game_loop(invoke, _state(), max_iterations=1))
    assert result.turns[0].reasoning == "go forth and multiply"


def test_synth_game_loop_state_before_matches_state_at_invoke():
    from gameplay_agent.synth_game_loop import synth_game_loop

    initial = _state(turn=0, food=200.0)
    invoke = _RecordingStub([([{"type": "queue_villager"}], "", 0.0)])
    result = _run(synth_game_loop(invoke, initial, max_iterations=1))
    assert result.turns[0].state_before == initial


def test_synth_game_loop_state_after_reflects_tick():
    from gameplay_agent.synth_game_loop import synth_game_loop

    # Empty turn: tick advances turn counter from 5 to 6.
    invoke = _RecordingStub([_empty_turn()])
    result = _run(synth_game_loop(invoke, _state(turn=5), max_iterations=1))
    assert result.turns[0].state_after.turn == 6


@pytest.mark.parametrize(
    "action_type", ["click", "right_click", "drag", "wait", "scroll", "detect"]
)
def test_synth_game_loop_accepts_mouse_keyboard_primitives_without_crashing(action_type):
    from gameplay_agent.synth_game_loop import synth_game_loop

    invoke = _RecordingStub([([{"type": action_type}], "", 0.0)])
    result = _run(synth_game_loop(invoke, _state(), max_iterations=1))
    assert len(result.turns) == 1


# ---------------------------------------------------------------------------
# Event emission (Phase 4)
# ---------------------------------------------------------------------------


class _RecordingSink:
    """EventSink implementation that captures every emit in order."""

    def __init__(self):
        self.events = []

    def emit(self, event):
        self.events.append(event)


def test_synth_game_loop_result_includes_run_id():
    from gameplay_agent.synth_game_loop import synth_game_loop

    invoke = _RecordingStub([_empty_turn()])
    result = _run(synth_game_loop(invoke, _state(), max_iterations=1))
    assert len(result.run_id) == 32  # uuid4().hex is 32 hex chars


def test_synth_game_loop_result_run_id_differs_across_runs():
    from gameplay_agent.synth_game_loop import synth_game_loop

    invoke_a = _RecordingStub([_empty_turn()])
    invoke_b = _RecordingStub([_empty_turn()])
    a = _run(synth_game_loop(invoke_a, _state(), max_iterations=1))
    b = _run(synth_game_loop(invoke_b, _state(), max_iterations=1))
    assert a.run_id != b.run_id


def test_synth_game_loop_emits_turn_start_per_turn():
    from gameplay_agent.synth_game_loop import synth_game_loop

    sink = _RecordingSink()
    invoke = _RecordingStub([_empty_turn() for _ in range(3)])
    _run(synth_game_loop(invoke, _state(), max_iterations=3, sink=sink))
    turn_starts = [e for e in sink.events if e.payload.kind == "turn_start"]
    assert len(turn_starts) == 3


def test_synth_game_loop_emits_llm_prompt_and_response_per_turn():
    from gameplay_agent.synth_game_loop import synth_game_loop

    sink = _RecordingSink()
    invoke = _RecordingStub([_empty_turn(cost=0.01)])
    _run(synth_game_loop(invoke, _state(), max_iterations=1, sink=sink))
    kinds = [e.payload.kind for e in sink.events]
    assert "llm_prompt" in kinds and "llm_response" in kinds


def test_synth_game_loop_emits_one_action_event_per_action():
    from gameplay_agent.synth_game_loop import synth_game_loop

    sink = _RecordingSink()
    actions = [{"type": "queue_villager"}, {"type": "build", "building_key": "q"}]
    invoke = _RecordingStub([(actions, "do stuff", 0.0)])
    _run(synth_game_loop(invoke, _state(food=200.0, wood=150.0), max_iterations=1, sink=sink))
    action_events = [e for e in sink.events if e.payload.kind == "action"]
    assert len(action_events) == 2


def test_synth_game_loop_emits_action_result_with_state_changed_flag():
    from gameplay_agent.synth_game_loop import synth_game_loop

    sink = _RecordingSink()
    invoke = _RecordingStub([([{"type": "queue_villager"}], "", 0.0)])
    _run(synth_game_loop(invoke, _state(food=200.0), max_iterations=1, sink=sink))
    result_events = [e for e in sink.events if e.payload.kind == "action_result"]
    assert result_events[0].payload.state_changed is True


def test_synth_game_loop_emits_action_result_state_unchanged_for_noop():
    from gameplay_agent.synth_game_loop import synth_game_loop

    sink = _RecordingSink()
    invoke = _RecordingStub([([{"type": "wait"}], "", 0.0)])
    _run(synth_game_loop(invoke, _state(), max_iterations=1, sink=sink))
    result_events = [e for e in sink.events if e.payload.kind == "action_result"]
    assert result_events[0].payload.state_changed is False


def test_synth_game_loop_emits_metric_per_turn():
    from gameplay_agent.synth_game_loop import synth_game_loop

    sink = _RecordingSink()
    invoke = _RecordingStub([_empty_turn(cost=0.01), _empty_turn(cost=0.02)])
    _run(synth_game_loop(invoke, _state(), max_iterations=2, sink=sink))
    metric_events = [e for e in sink.events if e.payload.kind == "metric"]
    assert len(metric_events) == 2


def test_synth_game_loop_tags_all_events_with_same_run_id():
    from gameplay_agent.synth_game_loop import synth_game_loop

    sink = _RecordingSink()
    invoke = _RecordingStub([_empty_turn() for _ in range(2)])
    result = _run(synth_game_loop(invoke, _state(), max_iterations=2, sink=sink))
    run_ids = {e.run_id for e in sink.events}
    assert run_ids == {result.run_id}


def test_synth_game_loop_default_sink_emits_nothing():
    """No sink argument = NullEventSink default; should not raise."""
    from gameplay_agent.synth_game_loop import synth_game_loop

    invoke = _RecordingStub([_empty_turn()])
    # No sink argument; should run with the NullEventSink default.
    result = _run(synth_game_loop(invoke, _state(), max_iterations=1))
    assert result.turns[0].turn_num == 1
