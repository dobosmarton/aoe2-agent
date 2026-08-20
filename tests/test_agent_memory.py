"""Unit tests for gameplay_agent/memory.py.

Layers:
  1. Turn / GameState dataclass defaults.
  2. AgentMemory init + working-memory deque.
  3. add_turn — counter, timer, total_actions, food tracking.
  4. update_from_observations — resources, population (with peak), idle/attack
     flags, intentional skip of executor-reported `age`.
  5. update_age — strategist-authoritative source, monotonic highest_age.
  6. get_context_for_llm — game-state block, recent turns, stuck-loop banner.
  7. record_memories_applied + memories_loaded plumbing.
  8. get_metrics_snapshot — full keyset, success rate edge cases.
  9. reset — full reset state.
"""

from __future__ import annotations

import time

from gameplay_agent.memory import (
    AGE_SCORES,
    INITIAL_POPULATION,
    INITIAL_POPULATION_CAP,
    INITIAL_RESOURCES,
    AgentMemory,
    GameState,
    Turn,
)
from gameplay_agent.turn_timing import ACT_LOOP, TURN_LOOP, LatencyRecorder

# ---------------------------------------------------------------------------
# Layer 1 — dataclass defaults
# ---------------------------------------------------------------------------


def test_turn_defaults() -> None:
    t = Turn(iteration=1, timestamp="t", reasoning="r", actions=[])
    assert t.observed_resources is None
    assert t.observed_events == []
    assert t.verification == ""
    assert t.goal_progress == {}
    assert t.reward == 0.0


def test_game_state_defaults_match_dark_age() -> None:
    s = GameState()
    assert s.resources == INITIAL_RESOURCES
    assert s.population == INITIAL_POPULATION
    assert s.population_cap == INITIAL_POPULATION_CAP
    assert s.current_age == "Dark Age"
    assert s.idle_tc is False
    assert s.under_attack is False


def test_game_state_resources_default_factory_independent() -> None:
    """Two GameState instances should not share the same resources dict."""
    a = GameState()
    b = GameState()
    a.resources["food"] = 999
    assert b.resources["food"] == INITIAL_RESOURCES["food"]


# ---------------------------------------------------------------------------
# Layer 2 — AgentMemory init
# ---------------------------------------------------------------------------


def test_agent_memory_default_init() -> None:
    m = AgentMemory()
    assert m.turn_count == 0
    assert len(m.working_memory) == 0
    assert m.working_memory.maxlen == 10
    assert m.episode_summary == ""
    assert m.highest_age == "Dark Age"
    assert m.game_start_time is None


def test_agent_memory_custom_working_size() -> None:
    m = AgentMemory(working_memory_size=3)
    assert m.working_memory.maxlen == 3


def test_working_memory_drops_oldest_at_capacity() -> None:
    m = AgentMemory(working_memory_size=2)
    for i in range(5):
        m.add_turn(Turn(iteration=i, timestamp="t", reasoning="r", actions=[]))
    assert len(m.working_memory) == 2
    iters = [t.iteration for t in m.working_memory]
    assert iters == [3, 4]  # only the last 2 retained


# ---------------------------------------------------------------------------
# Layer 3 — add_turn
# ---------------------------------------------------------------------------


def test_add_turn_increments_count_and_starts_timer() -> None:
    m = AgentMemory()
    m.add_turn(Turn(iteration=1, timestamp="t", reasoning="r", actions=[{"type": "press"}]))
    assert m.turn_count == 1
    assert m.game_start_time is not None
    assert m.total_actions == 1


def test_add_turn_does_not_reset_timer_after_first() -> None:
    m = AgentMemory()
    m.add_turn(Turn(iteration=1, timestamp="t", reasoning="r", actions=[]))
    first_time = m.game_start_time
    time.sleep(0.005)
    m.add_turn(Turn(iteration=2, timestamp="t", reasoning="r", actions=[]))
    assert m.game_start_time == first_time


def test_add_turn_food_does_not_feed_income_counter() -> None:
    """Turn observations (LLM-echoed) must never count as gathered food —
    only OCR frames via record_food_reading do (T-601)."""
    m = AgentMemory()
    m.add_turn(
        Turn(
            iteration=1, timestamp="t", reasoning="r", actions=[], observed_resources={"food": 200}
        )
    )
    m.add_turn(
        Turn(iteration=2, timestamp="t", reasoning="r", actions=[], observed_resources={"food": 50})
    )  # later, smaller
    assert m.total_food_gathered == 0


# ---------------------------------------------------------------------------
# Layer 4 — update_from_observations
# ---------------------------------------------------------------------------


def test_update_from_observations_empty_dict_noop() -> None:
    m = AgentMemory()
    m.update_from_observations({})
    assert m.game_state.resources == INITIAL_RESOURCES


def test_update_from_observations_resources_merged() -> None:
    m = AgentMemory()
    m.update_from_observations({"resources": {"food": 500}})
    assert m.game_state.resources["food"] == 500
    # other resources unchanged
    assert m.game_state.resources["wood"] == INITIAL_RESOURCES["wood"]


def test_update_from_observations_never_counts_food_income() -> None:
    m = AgentMemory()
    m.update_from_observations({"resources": {"food": 300}})
    m.update_from_observations({"resources": {"food": 100}})
    assert m.total_food_gathered == 0  # income comes from record_food_reading only


def test_update_from_observations_food_handles_non_numeric() -> None:
    """A garbled food reading shouldn't crash the update."""
    m = AgentMemory()
    m.update_from_observations({"resources": {"food": "garbage"}})
    assert m.total_food_gathered == 0


def test_update_from_observations_population_parses_n_over_cap() -> None:
    m = AgentMemory()
    m.update_from_observations({"population": "8/15"})
    assert m.game_state.population == 8
    assert m.game_state.population_cap == 15
    assert m.peak_population == 8


def test_update_from_observations_population_tracks_peak() -> None:
    m = AgentMemory()
    m.update_from_observations({"population": "10/15"})
    m.update_from_observations({"population": "6/15"})  # drop after losses
    assert m.peak_population == 10


def test_update_from_observations_malformed_population_ignored() -> None:
    m = AgentMemory()
    m.update_from_observations({"population": "abc/def"})
    assert m.game_state.population == INITIAL_POPULATION  # unchanged


def test_update_from_observations_does_not_set_age() -> None:
    """Executor-reported age is intentionally NOT used (hallucinations)."""
    m = AgentMemory()
    m.update_from_observations({"age": "Castle Age"})
    assert m.game_state.current_age == "Dark Age"  # unchanged


def test_update_from_observations_idle_tc_and_under_attack() -> None:
    m = AgentMemory()
    m.update_from_observations({"idle_tc": True, "under_attack": True})
    assert m.game_state.idle_tc is True
    assert m.game_state.under_attack is True


# ---------------------------------------------------------------------------
# Layer 5 — update_age (strategist-authoritative)
# ---------------------------------------------------------------------------


def test_update_age_sets_current_and_highest() -> None:
    m = AgentMemory()
    m.update_age("Feudal Age")
    assert m.game_state.current_age == "Feudal Age"
    assert m.highest_age == "Feudal Age"


def test_update_age_highest_is_monotonic() -> None:
    """If somehow the strategist reads back to a lower age, highest stays."""
    m = AgentMemory()
    m.update_age("Castle Age")
    m.update_age("Feudal Age")  # regression
    assert m.game_state.current_age == "Feudal Age"  # current follows reading
    assert m.highest_age == "Castle Age"  # peak preserved


def test_update_age_empty_string_noop() -> None:
    m = AgentMemory()
    m.update_age("")
    assert m.game_state.current_age == "Dark Age"


# ---------------------------------------------------------------------------
# Layer 6 — get_context_for_llm
# ---------------------------------------------------------------------------


def test_context_includes_game_state() -> None:
    m = AgentMemory()
    out = m.get_context_for_llm()
    assert "Current Game State" in out
    assert f"Food={INITIAL_RESOURCES['food']}" in out


def test_context_includes_episode_summary_when_set() -> None:
    m = AgentMemory()
    m.episode_summary = "the agent advanced to Feudal Age"
    out = m.get_context_for_llm()
    assert "Previous Events Summary" in out
    assert "advanced to Feudal Age" in out


def test_context_includes_recent_turns() -> None:
    m = AgentMemory()
    m.add_turn(
        Turn(iteration=1, timestamp="t", reasoning="hello", actions=[{"type": "press", "key": "h"}])
    )
    out = m.get_context_for_llm()
    assert "Recent Decisions" in out
    assert "Turn 1" in out
    assert "press(h)" in out


def test_context_stuck_loop_warning_after_3_failures() -> None:
    """3 consecutive failed verifications trigger the warning banner."""
    m = AgentMemory()
    for i in range(3):
        m.add_turn(Turn(iteration=i + 1, timestamp="t", reasoning="r", actions=[]))
        m.set_last_verification("FAILED — no visible change")
    out = m.get_context_for_llm()
    assert "WARNING" in out
    assert "completely different approach" in out


def test_context_no_warning_for_one_failure() -> None:
    m = AgentMemory()
    m.add_turn(Turn(iteration=1, timestamp="t", reasoning="r", actions=[]))
    m.set_last_verification("FAILED")
    out = m.get_context_for_llm()
    assert "WARNING" not in out


def test_format_game_state_renders_housed_flag() -> None:
    m = AgentMemory()
    m.update_from_observations({"population": "5/5"})  # at cap
    out = m.get_context_for_llm()
    assert "HOUSED" in out


# ---------------------------------------------------------------------------
# Layer 7 — record_memories_applied
# ---------------------------------------------------------------------------


def test_record_memories_applied_increments_per_title() -> None:
    m = AgentMemory()
    m.record_memories_applied(["a", "b"])
    m.record_memories_applied(["a"])
    assert m.memories_applied_count == {"a": 2, "b": 1}


def test_record_memories_applied_empty_list_noop() -> None:
    m = AgentMemory()
    m.record_memories_applied([])
    assert m.memories_applied_count == {}


def test_set_last_verification_attaches_to_latest_turn() -> None:
    m = AgentMemory()
    m.add_turn(Turn(iteration=1, timestamp="t", reasoning="r", actions=[]))
    m.set_last_verification("ok")
    assert m.working_memory[-1].verification == "ok"


def test_set_last_verification_noop_on_empty_working_memory() -> None:
    """Should not crash if no turn has been added yet."""
    m = AgentMemory()
    m.set_last_verification("ok")  # no turn — should silently no-op


def test_record_action_results_accumulates_success() -> None:
    m = AgentMemory()
    m.record_action_results(3, 5)
    m.record_action_results(2, 4)
    assert m.successful_actions == 5


# ---------------------------------------------------------------------------
# Layer 8 — get_metrics_snapshot
# ---------------------------------------------------------------------------


def test_metrics_snapshot_keys() -> None:
    """The snapshot is the contract autoresearch consumes — pin its shape."""
    m = AgentMemory()
    snap = m.get_metrics_snapshot()
    expected_keys = {
        "survival_time",
        "peak_population",
        "highest_age",
        "age_score",
        "total_food_gathered",
        "total_actions",
        "successful_actions",
        "action_success_rate",
        "turn_count",
        "game_end_reason",
        "memories_loaded",
        "memories_used",
        "executed_actions",
        "llm_calls",
        "llm_errors",
        "llm_error_rate",
        # Age timings (plan 2.1).
        "feudal_time_s",
        "castle_time_s",
        # Latency (plan 0.3, plan 3).
        "turn_latency_p50_ms",
        "turn_latency_p90_ms",
        "turn_latency_max_ms",
        "phase_latency_p50_ms",
        "act_latency_p95_ms",
        "perceive_latency_p50_ms",
        "loop_arch",
    }
    assert set(snap.keys()) == expected_keys


def test_metrics_snapshot_latency_is_zero_without_a_recorder() -> None:
    """The scenario and synth paths attach no recorder, and must still report."""
    snap = AgentMemory().get_metrics_snapshot()
    assert snap["turn_latency_p50_ms"] == 0.0


def test_metrics_snapshot_phase_map_is_empty_without_a_recorder() -> None:
    assert AgentMemory().get_metrics_snapshot()["phase_latency_p50_ms"] == {}


def test_metrics_snapshot_reports_recorded_turn_latency() -> None:
    """An attached recorder surfaces in the snapshot."""
    memory = AgentMemory()
    memory.latency = _recorder_with_ocr_turns()
    assert memory.get_metrics_snapshot()["turn_latency_p50_ms"] > 0.0


def test_metrics_snapshot_reports_recorded_phase_names() -> None:
    memory = AgentMemory()
    memory.latency = _recorder_with_ocr_turns()
    assert set(memory.get_metrics_snapshot()["phase_latency_p50_ms"]) == {"ocr"}


def _recorder_with_ocr_turns(turns: int = 3) -> LatencyRecorder:
    """A recorder holding `turns` timed turns with one measurable `ocr` phase."""
    recorder = LatencyRecorder()
    for iteration in range(turns):
        with recorder.tick(TURN_LOOP, iteration) as tick, tick.phase("ocr"):
            time.sleep(0.002)  # an empty phase rounds to 0.0 ms
    return recorder


def test_metrics_snapshot_success_rate_zero_total_no_div_zero() -> None:
    snap = AgentMemory().get_metrics_snapshot()
    assert snap["action_success_rate"] == 0.0


def test_metrics_snapshot_success_rate_correct() -> None:
    m = AgentMemory()
    m.add_turn(
        Turn(
            iteration=1,
            timestamp="t",
            reasoning="r",
            actions=[{"type": "press"}, {"type": "press"}],
        )
    )
    m.record_action_results(1, 2)
    snap = m.get_metrics_snapshot()
    assert snap["action_success_rate"] == 0.5


def test_metrics_snapshot_age_score_uses_age_scores_table() -> None:
    m = AgentMemory()
    m.update_age("Feudal Age")
    snap = m.get_metrics_snapshot()
    assert snap["age_score"] == AGE_SCORES["Feudal Age"]


def test_llm_error_rate_counts_failed_executor_turns() -> None:
    """llm_error_rate is failed executor turns / total — the metric that would
    have flagged run 12's dead executor (90/95 errors, still accepted)."""
    m = AgentMemory()
    for errored in (False, True, True, False):  # 2 of 4 failed
        m.record_llm_outcome(errored=errored)
    snap = m.get_metrics_snapshot()
    assert snap["llm_calls"] == 4
    assert snap["llm_errors"] == 2
    assert snap["llm_error_rate"] == 0.5


def test_llm_error_rate_zero_when_no_calls() -> None:
    assert AgentMemory().get_metrics_snapshot()["llm_error_rate"] == 0.0


def test_record_llm_outcome_streak_resets_on_success() -> None:
    """The returned streak counts CONSECUTIVE failures and a success zeroes it —
    this is what the game loop alarms on."""
    m = AgentMemory()
    assert m.record_llm_outcome(errored=True) == 1
    assert m.record_llm_outcome(errored=True) == 2
    assert m.record_llm_outcome(errored=False) == 0  # success breaks the streak
    assert m.record_llm_outcome(errored=True) == 1  # counts up again from zero


def test_metrics_snapshot_survival_time_from_first_turn() -> None:
    m = AgentMemory()
    m.add_turn(Turn(iteration=1, timestamp="t", reasoning="r", actions=[]))
    time.sleep(0.005)
    snap = m.get_metrics_snapshot()
    assert snap["survival_time"] > 0


def test_metrics_snapshot_no_turns_zero_survival() -> None:
    snap = AgentMemory().get_metrics_snapshot()
    assert snap["survival_time"] == 0.0


# ---------------------------------------------------------------------------
# Layer 9 — reset + create_turn
# ---------------------------------------------------------------------------


def test_reset_clears_all_state() -> None:
    m = AgentMemory()
    m.add_turn(Turn(iteration=1, timestamp="t", reasoning="r", actions=[{"type": "press"}]))
    m.update_age("Feudal Age")
    m.record_memories_applied(["a"])
    m.peak_population = 20
    m.game_end_reason = "victory"

    m.reset()

    assert m.turn_count == 0
    assert len(m.working_memory) == 0
    assert m.game_state.current_age == "Dark Age"
    assert m.highest_age == "Dark Age"
    assert m.peak_population == 0
    assert m.total_actions == 0
    assert m.successful_actions == 0
    assert m.game_start_time is None
    assert m.game_end_reason == ""
    assert m.memories_applied_count == {}


def test_create_turn_attaches_to_working_memory_and_increments() -> None:
    m = AgentMemory()
    turn = m.create_turn(
        reasoning="hello",
        actions=[{"type": "press", "key": "h"}],
        observations={"resources": {"food": 250}},
    )
    assert turn.iteration == 1
    assert m.turn_count == 1
    assert m.working_memory[-1] is turn
    # Observations also flowed through update_from_observations
    assert m.game_state.resources["food"] == 250


def test_create_turn_handles_no_observations() -> None:
    m = AgentMemory()
    turn = m.create_turn(reasoning="r", actions=[])
    assert turn.observed_resources is None
    assert turn.observed_events == []


# ---------------------------------------------------------------------------
# Honest metrics (T-601)
# ---------------------------------------------------------------------------


def test_action_success_rate_uses_executed_denominator():
    """Fallback/composite executions never enter turn.actions, so the old
    successful/total ratio exceeded 1.0 (runs 1 and 3: 2.38, 1.54)."""
    memory = AgentMemory()
    memory.create_turn(reasoning="r", actions=[])  # planned 0
    memory.record_action_results(3, 3)  # but 3 fallback actions executed
    snap = memory.get_metrics_snapshot()
    assert snap["executed_actions"] == 3
    assert snap["action_success_rate"] == 1.0  # 3/3, never > 1 by construction
    memory.record_action_results(0, 2)
    assert memory.get_metrics_snapshot()["action_success_rate"] == 0.6  # 3/5


def test_food_gathered_sums_positive_ocr_deltas():
    memory = AgentMemory()
    memory.record_food_reading(200)  # baseline (starting stock — not income)
    memory.record_food_reading(150)  # spent on a villager — not income
    memory.record_food_reading(210)  # +60 gathered
    memory.record_food_reading(240)  # +30 gathered
    assert memory.total_food_gathered == 90


def test_food_gathered_drops_ocr_glitch_jumps():
    memory = AgentMemory()
    memory.record_food_reading(40)
    memory.record_food_reading(900)  # misread — beyond the sanity cap, dropped
    memory.record_food_reading(60)  # below the glitch baseline — no count
    assert memory.total_food_gathered == 0


def test_llm_observations_do_not_count_as_gathered_food():
    memory = AgentMemory()
    memory.record_food_reading(100)
    # LLM-echoed observations update state but never the income counter.
    memory.update_from_observations({"resources": {"food": 9999}})
    memory.create_turn(reasoning="r", actions=[], observations={"resources": {"food": 5000}})
    assert memory.total_food_gathered == 0


def test_reset_clears_metric_state():
    memory = AgentMemory()
    memory.record_food_reading(100)
    memory.record_food_reading(200)
    memory.record_action_results(1, 2)
    memory.reset()
    assert memory.total_food_gathered == 0 and memory.executed_actions == 0
    memory.record_food_reading(300)  # fresh baseline, not a +100 delta
    assert memory.total_food_gathered == 0


def test_loop_arch_reads_turn_before_the_cutover() -> None:
    """The single-tick loop and the act loop do not measure the same thing."""
    memory = AgentMemory()
    memory.latency = _recorder_with_ocr_turns()
    assert memory.get_metrics_snapshot()["loop_arch"] == "turn"


def test_loop_arch_reads_clocks_once_the_act_loop_records() -> None:
    """Presence, not duration: a fast act tick still rounds to 0.0 ms."""
    recorder = LatencyRecorder()
    with recorder.tick(ACT_LOOP, 0):
        pass
    memory = AgentMemory()
    memory.latency = recorder
    assert memory.get_metrics_snapshot()["loop_arch"] == "clocks"
