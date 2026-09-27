"""Regressions for economic effects that input acceptance cannot establish."""

from __future__ import annotations

import asyncio

import pytest
from gameplay_agent import executor as ex
from gameplay_agent.memory import AgentMemory


@pytest.fixture
def ledger() -> ex.ActionLedger:
    value = ex.ActionLedger()
    token = ex.bind_ledger(value)
    try:
        yield value
    finally:
        ex.unbind_ledger(token)


def test_purchase_and_villager_delivery_in_same_observation(ledger: ex.ActionLedger) -> None:
    ex.observe_hud(4, 5, {"food": 200}, villagers=3)
    pending = ex._note_pending_training("villager")
    assert pending is not None
    ledger.note_input()

    # A 20-food net fall includes a 50-food purchase and 30 food gathered.
    ex.observe_hud(5, 5, {"food": 180}, villagers=4, input_revision=ledger.input_revision)

    assert ledger.pending_training == []
    assert ledger.queued_training == []
    assert ledger.food_gathered == 30
    assert ledger.villagers_ordered == 4
    assert [
        outcome.status
        for outcome in ledger.outcomes
        if outcome.operation_id == pending.operation_id
    ] == ["pending", "purchased", "confirmed"]


def test_partial_net_drop_without_delivery_is_pending_then_uncertain(
    ledger: ex.ActionLedger, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [100.0]
    monkeypatch.setattr(ex, "_now", lambda: clock[0])
    ex.observe_hud(4, 5, {"food": 200}, villagers=3)
    pending = ex._note_pending_training("villager")
    assert pending is not None
    ledger.note_input()
    ex.observe_hud(4, 5, {"food": 180}, villagers=3, input_revision=ledger.input_revision)
    assert ledger.pending_training == [pending]
    assert ledger.reservations()["food"] == 50
    assert ledger.food_gathered == 0

    clock[0] += ex._RESEARCH_SETTLE_SECONDS + 1
    ex.observe_hud(4, 5, {"food": 180}, villagers=3, input_revision=ledger.input_revision)
    assert ledger.pending_training == [pending]
    assert pending.operation_id in ledger.uncertain_operations
    memory = AgentMemory()
    memory.action_ledger = ledger
    metrics = memory.get_metrics_snapshot()
    assert metrics["score_valid"] is False
    assert metrics["action_success_rate"] == 0.0


def test_run_end_marks_unsettled_purchase_unverifiable(ledger: ex.ActionLedger) -> None:
    ex.observe_hud(4, 5, {"food": 200}, villagers=3)
    pending = ex._note_pending_training("villager")
    assert pending is not None
    ledger.note_input()

    ledger.finalize_unverified()

    assert ledger.pending_training == [pending]
    assert pending.operation_id in ledger.uncertain_operations
    memory = AgentMemory()
    memory.action_ledger = ledger
    assert memory.get_metrics_snapshot()["unverifiable_action_ids"] == [pending.operation_id]
    ledger.finalize_unverified()  # finalization is idempotent
    assert sum(outcome.status == "uncertain" for outcome in ledger.outcomes) == 1


def test_failed_named_action_counts_as_attempt_not_success(ledger: ex.ActionLedger) -> None:
    ledger.record_outcome(
        ex.ActionOutcome(ledger.new_operation_id(), "queue_villager", "failed", "TC not selected")
    )
    memory = AgentMemory()
    memory.action_ledger = ledger
    metrics = memory.get_metrics_snapshot()
    assert metrics["executed_actions"] == 1
    assert metrics["successful_actions"] == 0
    assert metrics["action_success_rate"] == 0.0


def test_navigation_requires_new_hud_baseline_before_spending(ledger: ex.ActionLedger) -> None:
    ex.observe_hud(4, 5, {"food": 200, "wood": 200}, villagers=3)
    ledger.note_input()  # navigation happened; pre-navigation HUD is stale
    assert ex._note_pending_training("villager") is None
    assert ex._note_pending_placement("q") is None
    ex.observe_hud(4, 5, {"food": 200, "wood": 200}, villagers=3)
    assert ex._note_pending_training("villager") is not None


def test_military_population_does_not_deliver_villager(ledger: ex.ActionLedger) -> None:
    ex.observe_hud(4, 10, {"food": 200}, villagers=3)
    pending = ex._note_pending_training("villager")
    assert pending is not None
    ledger.note_input()
    ex.observe_hud(5, 10, {"food": 150}, villagers=3, input_revision=ledger.input_revision)
    assert len(ledger.queued_training) == 1
    assert pending.operation_id not in ledger.confirmed_economic_ids


def test_walking_worker_is_not_a_confirmed_assignment(ledger: ex.ActionLedger) -> None:
    ex.observe_hud(
        4,
        5,
        {"food": 200},
        villagers=3,
        worker_counts={"food": 0},
        idle_present=True,
        idle_count=1,
    )
    ledger.note_input()
    pending = ex._PendingAssignment(
        operation_id=ledger.new_operation_id(),
        resource="food",
        target_id="sheep",
        idle_count_before=1,
        workers_before=0,
        noted_at_snapshot=ledger.snapshot_count,
        command_revision=ledger.input_revision,
        settle_deadline=ex._now() + 12,
    )
    ledger.pending_assignment = pending
    ledger.record_outcome(
        ex.ActionOutcome(pending.operation_id, "assign_food", "pending", "issued")
    )
    ex.observe_hud(
        4,
        5,
        {"food": 200},
        villagers=3,
        worker_counts={"food": 0},
        idle_present=False,
        idle_count=0,
        input_revision=ledger.input_revision,
    )
    assert ledger.pending_assignment is pending
    ledger.note_input()  # unrelated TC input cannot erase this assignment
    ex.observe_hud(
        4,
        5,
        {"food": 200},
        villagers=3,
        worker_counts={"food": 1},
        idle_present=False,
        idle_count=0,
        input_revision=ledger.input_revision,
    )
    assert ledger.pending_assignment is None
    assert pending.operation_id in ledger.confirmed_economic_ids


def test_unverified_tc_selection_never_presses_purchase_key(
    ledger: ex.ActionLedger, monkeypatch: pytest.MonkeyPatch
) -> None:
    pressed: list[str] = []

    class Input:
        def press(self, key: str) -> None:
            pressed.append(key)

        def moveTo(self, *_args: object) -> None:
            pass

    async def refresh() -> bool:
        return True  # refresh arrived but could not read the selected panel

    monkeypatch.setattr(ex, "pyautogui", Input())
    monkeypatch.setattr(ex, "get_game_window_rect", lambda: None)
    monkeypatch.setattr(ex, "RESCAN_SETTLE_DELAY", 0.0)
    monkeypatch.setattr(ex.config, "action_delay", 0.0)
    ex.set_rescan_fn(refresh)
    ex.observe_hud(4, 5, {"food": 200}, villagers=3)
    result = asyncio.run(ex.execute_action({"type": "queue_villager", "intent": "grow"}))
    assert not result.success
    assert "selection unverified" in result.detail
    assert pressed == ["h"]
    assert not ledger.pending_training


def test_intent_text_cannot_retarget_coordinates(
    ledger: ex.ActionLedger, monkeypatch: pytest.MonkeyPatch
) -> None:
    clicked: list[tuple[int, int]] = []

    class Input:
        def rightClick(self, x: int, y: int) -> None:
            clicked.append((x, y))

    monkeypatch.setattr(ex, "pyautogui", Input())
    monkeypatch.setattr(ex, "get_game_window_rect", lambda: (0, 0, 1920, 1080))
    ex.set_detected_entities([{"id": "sheep", "class": "sheep", "center": (900, 600)}])
    result = asyncio.run(ex._handle_right_click({"x": 700, "y": 600}, "move to sheep"))
    assert result.success
    assert clicked == [(700, 600)]


def test_bound_target_movement_aborts_click(ledger: ex.ActionLedger) -> None:
    ex.set_detected_entities([{"id": "sheep", "class": "sheep", "center": (905, 600)}])
    error, point = ex._resolve_coords(
        {
            "target_id": "sheep",
            "expected_class": "sheep",
            "expected_coords": (900, 600),
            "spatial_revision": ledger.input_revision,
        }
    )
    assert "moved" in error
    assert point is None
