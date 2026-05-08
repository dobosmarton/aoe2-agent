"""Unit tests for gameplay_agent/goals.py.

Layers:
  1. Goal dataclass — defaults + invariants.
  2. GoalManager.set_goals — replace + progress preservation by name.
  3. evaluate_progress + _compute_goal_progress — per-metric math,
     completion, deadline failure.
  4. compute_turn_reward — resource / pop / age deltas weighted.
  5. Context-rendering helpers (get_context_for_llm, get_state_snapshot,
     get_goals_summary, get_resource_context).
  6. Resource-readings cache + memory integration.
  7. Alarm system (check_alarm) — confidence gate, threshold,
     ownership-fallback, emergency-goal injection.
"""

from __future__ import annotations

from dataclasses import dataclass

from gameplay_agent.goals import (
    ALARM_CONFIDENCE_GATE,
    Goal,
    GoalManager,
)
from gameplay_agent.memory import AgentMemory, GameState

# ---------------------------------------------------------------------------
# Test infra
# ---------------------------------------------------------------------------


@dataclass
class _FakeEntity:
    id: str
    class_name: str
    center: tuple[float, float]
    confidence: float


def _g(
    name: str,
    metric: str = "food",
    target: object = 100,
    priority: int = 5,
    created_turn: int = 0,
    deadline_turns: int | None = None,
    type_: str = "local",
) -> Goal:
    return Goal(
        name=name,
        type=type_,  # type: ignore[arg-type]
        metric=metric,
        target=target,
        priority=priority,
        created_turn=created_turn,
        deadline_turns=deadline_turns,
    )


# ---------------------------------------------------------------------------
# Layer 1 — Goal dataclass
# ---------------------------------------------------------------------------


def test_goal_defaults() -> None:
    g = Goal(name="x", type="local", metric="food", target=200, priority=5, created_turn=0)
    assert g.progress == 0.0
    assert g.completed is False
    assert g.failed is False
    assert g.deadline_turns is None


# ---------------------------------------------------------------------------
# Layer 2 — set_goals: replace + progress preservation
# ---------------------------------------------------------------------------


def test_set_goals_replaces_previous() -> None:
    m = GoalManager()
    m.set_goals([_g("a"), _g("b")])
    m.set_goals([_g("c")])
    assert [g.name for g in m.active_goals] == ["c"]


def test_set_goals_preserves_progress_for_same_name() -> None:
    """When the strategist resends a goal with the same name, the progress
    should not reset to 0 — old progress carries forward."""
    m = GoalManager()
    g1 = _g("queue_villagers", target=10)
    g1.progress = 0.6
    m.active_goals = [g1]
    m.set_goals([_g("queue_villagers", target=10), _g("new_goal")])
    by_name = {g.name: g for g in m.active_goals}
    assert by_name["queue_villagers"].progress == 0.6
    assert by_name["new_goal"].progress == 0.0


def test_set_goals_drops_completed_goals() -> None:
    m = GoalManager()
    g_done = _g("done")
    g_done.completed = True
    m.active_goals = [g_done]
    m.set_goals([_g("done"), _g("new")])
    # Both 'done' (preserved-completed) and 'new' arrive; completed is filtered out
    names = [g.name for g in m.active_goals]
    assert names == ["new"]


# ---------------------------------------------------------------------------
# Layer 3 — evaluate_progress + _compute_goal_progress
# ---------------------------------------------------------------------------


def test_progress_population_target_partial() -> None:
    m = GoalManager()
    m.active_goals = [_g("pop", metric="population", target=10)]
    state = GameState(population=4)
    m.evaluate_progress(state, turn=1)
    assert m.active_goals[0].progress == 0.4


def test_progress_food_target_full_completes_and_moves_to_completed() -> None:
    m = GoalManager()
    m.active_goals = [_g("food", metric="food", target=100)]
    state = GameState(resources={"food": 200, "wood": 0, "gold": 0, "stone": 0})
    m.evaluate_progress(state, turn=1)
    assert m.active_goals == []  # moved out of active
    assert len(m.completed_goals) == 1
    assert m.completed_goals[0].progress == 1.0
    assert m.completed_goals[0].completed is True


def test_progress_age_metric_uses_age_scores() -> None:
    m = GoalManager()
    m.active_goals = [_g("age", metric="age", target="Castle Age")]
    state = GameState(current_age="Feudal Age")
    m.evaluate_progress(state, turn=1)
    # AGE_SCORES: Dark=0, Feudal=0.33, Castle=0.66 → 0.33/0.66 = 0.5
    assert abs(m.active_goals[0].progress - 0.5) < 0.01


def test_progress_unknown_metric_zero() -> None:
    m = GoalManager()
    m.active_goals = [_g("x", metric="diplomacy", target=100)]
    state = GameState()
    m.evaluate_progress(state, turn=1)
    assert m.active_goals[0].progress == 0.0


def test_progress_zero_target_yields_zero() -> None:
    """Don't divide by zero — target of 0 should produce 0 progress."""
    m = GoalManager()
    m.active_goals = [_g("x", metric="food", target=0)]
    state = GameState(resources={"food": 100, "wood": 0, "gold": 0, "stone": 0})
    m.evaluate_progress(state, turn=1)
    assert m.active_goals[0].progress == 0.0


def test_progress_non_numeric_target_for_numeric_metric_zero() -> None:
    """A string target on 'food' should fall back to 0 rather than crash."""
    m = GoalManager()
    m.active_goals = [_g("x", metric="food", target="lots")]
    state = GameState(resources={"food": 200, "wood": 0, "gold": 0, "stone": 0})
    m.evaluate_progress(state, turn=1)
    assert m.active_goals[0].progress == 0.0


def test_evaluate_progress_clamps_above_one() -> None:
    """Population well over target shouldn't show >100% progress."""
    m = GoalManager()
    m.active_goals = [_g("pop", metric="population", target=10)]
    state = GameState(population=50)
    m.evaluate_progress(state, turn=1)
    # Goal completes → moved out of active; check completed list
    assert m.completed_goals[0].progress == 1.0


def test_deadline_expiration_marks_failed() -> None:
    m = GoalManager()
    m.active_goals = [_g("late", metric="food", target=999, deadline_turns=5, created_turn=0)]
    state = GameState(resources={"food": 0, "wood": 0, "gold": 0, "stone": 0})
    m.evaluate_progress(state, turn=10)  # 10 - 0 > 5 → failed
    assert m.active_goals == []  # dropped from active


# ---------------------------------------------------------------------------
# Layer 4 — compute_turn_reward
# ---------------------------------------------------------------------------


def test_compute_turn_reward_resource_delta() -> None:
    m = GoalManager()
    prev = GameState(resources={"food": 100, "wood": 0, "gold": 0, "stone": 0})
    curr = GameState(resources={"food": 200, "wood": 0, "gold": 0, "stone": 0})
    reward = m.compute_turn_reward(prev, curr)
    # 100 / RESOURCE_REWARD_DIVISOR (1000) = 0.1
    assert reward["food"] == 0.1


def test_compute_turn_reward_population_factor() -> None:
    m = GoalManager()
    prev = GameState(population=4)
    curr = GameState(population=6)
    reward = m.compute_turn_reward(prev, curr)
    # delta=2 * POPULATION_REWARD_FACTOR (0.05) = 0.1
    assert reward["population"] == 0.1


def test_compute_turn_reward_age_advance() -> None:
    m = GoalManager()
    prev = GameState(current_age="Dark Age")
    curr = GameState(current_age="Feudal Age")
    reward = m.compute_turn_reward(prev, curr)
    # AGE_SCORES delta: 0.33 - 0.0 = 0.33
    assert abs(reward["age"] - 0.33) < 0.01


def test_compute_turn_reward_total_sums_components() -> None:
    m = GoalManager()
    prev = GameState(resources={"food": 0, "wood": 0, "gold": 0, "stone": 0}, population=0)
    curr = GameState(resources={"food": 100, "wood": 0, "gold": 0, "stone": 0}, population=2)
    reward = m.compute_turn_reward(prev, curr)
    # food: 0.1, pop: 0.1, age: 0 → total: 0.2
    assert reward["total"] == 0.2


# ---------------------------------------------------------------------------
# Layer 5 — context rendering
# ---------------------------------------------------------------------------


def test_context_for_llm_empty_returns_empty() -> None:
    assert GoalManager().get_context_for_llm() == ""


def test_context_for_llm_sorts_high_priority_first() -> None:
    m = GoalManager()
    m.active_goals = [_g("low", priority=2), _g("high", priority=9), _g("med", priority=5)]
    out = m.get_context_for_llm()
    high_idx = out.find("high:")
    med_idx = out.find("med:")
    low_idx = out.find("low:")
    assert high_idx < med_idx < low_idx


def test_context_for_llm_priority_labels() -> None:
    m = GoalManager()
    m.active_goals = [_g("h", priority=9), _g("m", priority=6), _g("l", priority=2)]
    out = m.get_context_for_llm()
    assert "[HIGH]" in out
    assert "[MED]" in out
    assert "[LOW]" in out


def test_context_for_llm_truncates_to_top_5() -> None:
    m = GoalManager()
    m.active_goals = [_g(f"g{i}", priority=10 - i) for i in range(10)]
    out = m.get_context_for_llm()
    # MAX_DISPLAY_GOALS = 5
    assert "g0:" in out
    assert "g4:" in out
    assert "g5:" not in out


def test_context_for_llm_includes_recent_completed() -> None:
    m = GoalManager()
    m.active_goals = [_g("active")]
    m.completed_goals = [_g("done_old"), _g("done_new")]
    out = m.get_context_for_llm()
    assert "[DONE] done_old" in out
    assert "[DONE] done_new" in out


def test_context_for_llm_string_target_renders_as_pct_only() -> None:
    """Goals with string targets (e.g. age) get a '50%' progress, not '50/100'."""
    m = GoalManager()
    g = _g("advance", metric="age", target="Feudal Age")
    g.progress = 0.5
    m.active_goals = [g]
    out = m.get_context_for_llm()
    assert "50%" in out
    assert "/Feudal Age" not in out  # not rendered as fraction


def test_state_snapshot_serializable_dict() -> None:
    m = GoalManager()
    state = GameState(
        resources={"food": 100, "wood": 50, "gold": 0, "stone": 0},
        population=8,
        population_cap=15,
        current_age="Dark Age",
    )
    snap = m.get_state_snapshot(state)
    assert snap["resources"]["food"] == 100
    assert snap["population"] == 8
    assert snap["population_cap"] == 15
    assert snap["age"] == "Dark Age"


def test_goals_summary_empty_string() -> None:
    assert GoalManager().get_goals_summary() == "No goals yet."


def test_goals_summary_renders_active_and_recent_completed() -> None:
    m = GoalManager()
    m.active_goals = [_g("a", priority=8)]
    m.completed_goals = [_g("done1"), _g("done2")]
    out = m.get_goals_summary()
    assert "local/a P8" in out
    assert "COMPLETED: done1" in out
    assert "COMPLETED: done2" in out


# ---------------------------------------------------------------------------
# Layer 6 — resource readings cache + memory integration
# ---------------------------------------------------------------------------


def test_update_resource_readings_caches_dict() -> None:
    m = GoalManager()
    m.update_resource_readings({"food": 200, "wood": 100})
    assert m._resource_readings["food"] == 200


def test_update_resource_readings_empty_input_noop() -> None:
    m = GoalManager()
    m.update_resource_readings({})
    assert m._resource_readings == {}


def test_update_resource_readings_propagates_to_memory() -> None:
    m = GoalManager()
    mem = AgentMemory()
    m.update_resource_readings(
        {
            "food": 200,
            "wood": 150,
            "gold": 0,
            "stone": 0,
            "population": "8/15",
            "age": "Feudal Age",
        },
        memory=mem,
    )
    assert mem.game_state.resources["food"] == 200
    assert mem.game_state.population == 8
    assert mem.game_state.population_cap == 15
    assert mem.game_state.current_age == "Feudal Age"


def test_get_resource_context_empty_returns_empty() -> None:
    assert GoalManager().get_resource_context() == ""


def test_get_resource_context_renders_status_block() -> None:
    m = GoalManager()
    m.update_resource_readings(
        {"food": 200, "wood": 150, "gold": 50, "stone": 25, "population": "8/15", "age": "Dark Age"}
    )
    out = m.get_resource_context()
    assert "Food: 200" in out
    assert "Population: 8/15" in out
    assert "Age: Dark Age" in out


# ---------------------------------------------------------------------------
# Layer 7 — alarm system
# ---------------------------------------------------------------------------


def test_check_alarm_no_threats_returns_false() -> None:
    m = GoalManager()
    assert m.check_alarm([]) is False
    assert m._alarm_active is False


def test_check_alarm_low_confidence_filtered_out() -> None:
    m = GoalManager()
    entities = [
        _FakeEntity(
            id="x", class_name="archer_line", center=(0, 0), confidence=ALARM_CONFIDENCE_GATE - 0.1
        )
    ]
    assert m.check_alarm(entities) is False  # below confidence gate


def test_check_alarm_below_threshold_no_alarm() -> None:
    """One spearman is not enough — the TC's auto-arrows handle a stray unit
    (commit 97b4f06 documented this threshold)."""
    m = GoalManager()
    entities = [_FakeEntity(id="x", class_name="spearman_line", center=(0, 0), confidence=0.9)]
    assert m.check_alarm(entities) is False
    assert m._alarm_active is False


def test_check_alarm_three_threats_triggers_and_injects_emergency_goal() -> None:
    m = GoalManager()
    entities = [
        _FakeEntity(id=f"e{i}", class_name="archer_line", center=(0, 0), confidence=0.9)
        for i in range(3)
    ]
    assert m.check_alarm(entities) is True
    assert m._alarm_active is True
    # Emergency "Defend base" goal pushed to front
    assert m.active_goals[0].name == "Defend base"
    assert m.active_goals[0].priority == 10


def test_check_alarm_does_not_double_inject_emergency_goal() -> None:
    """If 'Defend base' is already active, a second alarm shouldn't add another."""
    m = GoalManager()
    entities = [
        _FakeEntity(id=f"e{i}", class_name="archer_line", center=(0, 0), confidence=0.9)
        for i in range(3)
    ]
    m.check_alarm(entities)
    m.check_alarm(entities)  # second call
    defend_count = sum(1 for g in m.active_goals if g.name == "Defend base")
    assert defend_count == 1


def test_check_alarm_resets_when_threats_disappear() -> None:
    m = GoalManager()
    entities = [
        _FakeEntity(id=f"e{i}", class_name="archer_line", center=(0, 0), confidence=0.9)
        for i in range(3)
    ]
    m.check_alarm(entities)
    assert m._alarm_active is True
    m.check_alarm([])  # threats gone
    assert m._alarm_active is False


def test_check_alarm_non_threat_class_ignored() -> None:
    """A villager (non-military) doesn't count regardless of confidence."""
    m = GoalManager()
    entities = [
        _FakeEntity(id=f"v{i}", class_name="villager", center=(0, 0), confidence=0.99)
        for i in range(10)
    ]
    assert m.check_alarm(entities) is False
