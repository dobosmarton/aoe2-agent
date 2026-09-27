"""Regressions for the observed-facts → named-action → settlement path."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from detection.inference.ownership import Owner
from gameplay_agent import executor
from gameplay_agent.goal_logger import GoalLogger
from gameplay_agent.goals import GoalManager
from gameplay_agent.loops.context import LoopContext
from gameplay_agent.loops.perceive import perceive_once
from gameplay_agent.loops.snapshot import Perception
from gameplay_agent.memory import AgentMemory
from gameplay_agent.policy.advice import PolicyGoal
from gameplay_agent.policy.allocation import Allocation, focused
from gameplay_agent.policy.candidates import eligible, feasible_candidates
from gameplay_agent.policy.catalog import BY_ID
from gameplay_agent.policy.fallback import select_fallback
from gameplay_agent.policy.state import PolicyState
from gameplay_agent.scenario_runner import run_scenario_async

from tests.loop_fakes import FakeActuator, FakeSource


@pytest.fixture
def ledger() -> executor.ActionLedger:
    active = executor.ActionLedger()
    token = executor.bind_ledger(active)
    try:
        yield active
    finally:
        executor.clear_detected_entities()
        executor.unbind_ledger(token)


def test_purchase_baseline_survives_a_pre_spend_observation(
    ledger: executor.ActionLedger,
) -> None:
    executor.observe_hud(4, 10, {"wood": 250}, idle_present=True, input_revision=0)
    pending = executor._note_pending_placement("w")
    assert pending is not None
    assert (pending.wood_before, pending.spend_revision) == (250, 1)
    assert ledger.reservations()["wood"] == 100

    # OCR finishes for an older screenshot after the operation was registered.
    executor.observe_hud(4, 10, {"wood": 150}, idle_present=True, input_revision=0)
    assert ledger.pending_placements == [pending]
    assert ledger.reservations()["wood"] == 100

    ledger.note_input()  # the placement click
    executor.observe_hud(4, 10, {"wood": 150}, idle_present=True, input_revision=1)
    assert ledger.pending_placements == []
    assert ledger.reservations()["wood"] == 0
    assert "mill" in ledger.building_purchases
    assert "mill" not in ledger.buildings_confirmed
    assert sum(outcome.status == "purchased" for outcome in ledger.outcomes) == 1

    executor.observe_hud(4, 10, {"wood": 150}, idle_present=True, input_revision=1)
    assert sum(outcome.status == "purchased" for outcome in ledger.outcomes) == 1
    assert "mill" not in ledger.buildings_confirmed
    executor.record_observed_buildings([("mill_new", "mill")])
    assert "mill" in ledger.buildings_confirmed
    assert sum(outcome.status == "confirmed" for outcome in ledger.outcomes) == 1


def test_new_building_observation_completes_a_paid_placement(
    ledger: executor.ActionLedger, tmp_path: Path
) -> None:
    executor.observe_hud(4, 10, {"wood": 250}, input_revision=0)
    pending = executor._note_pending_placement("w")
    assert pending is not None
    ledger.note_input()
    source = FakeSource(
        [
            Perception(
                entities=({"id": "mill_1", "class": "mill", "center": (700, 500)},),
                hud_readings={"wood": 150, "population": "4/10", "age": "Dark Age"},
                input_revision=1,
            )
        ],
        ownership={"mill_1": (Owner.ENEMY, 0.95)},
    )
    ctx = LoopContext(
        memory=AgentMemory(),
        goal_manager=GoalManager(),
        goal_logger=GoalLogger(tmp_path),
        source=source,
        actuator=FakeActuator(),
        ledger=ledger,
    )

    asyncio.run(perceive_once(ctx, 1))
    assert "mill" in ledger.building_purchases
    assert "mill" not in ledger.buildings_confirmed
    # The current classifier does not label buildings; absence of an enemy
    # label must still permit a new matching entity to complete the purchase.
    source.ownership = {}
    asyncio.run(perceive_once(ctx, 2))

    assert ctx.frames.latest() is not None
    assert "mill" in ctx.frames.latest().world.buildings_seen
    assert "mill" not in ledger.building_purchases
    assert [outcome.status for outcome in ledger.outcomes] == [
        "pending",
        "purchased",
        "confirmed",
    ]


def test_existing_building_cannot_complete_a_new_purchase(
    ledger: executor.ActionLedger,
) -> None:
    executor.set_detected_entities([{"id": "mill_old", "class": "mill", "center": (500, 500)}])
    executor.observe_hud(4, 10, {"wood": 200}, input_revision=0)
    pending = executor._note_pending_placement("w")
    assert pending is not None
    ledger.note_input()
    executor.observe_hud(4, 10, {"wood": 100}, input_revision=1)

    executor.record_observed_buildings([("mill_old", "mill")])
    assert "mill" in ledger.building_purchases
    assert "mill" not in ledger.buildings_confirmed
    executor.record_observed_buildings([("mill_new", "mill")])
    assert "mill" in ledger.buildings_confirmed


def test_pending_house_suppresses_duplicates_until_failure_expires(
    ledger: executor.ActionLedger, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = 1000.0
    monkeypatch.setattr(executor, "_now", lambda: now)
    executor.observe_hud(4, 5, {"wood": 200}, idle_present=True)
    house = BY_ID["build_house"]
    assert eligible(house, executor.ledger_policy_state())
    pending = executor._note_pending_placement("q")
    assert pending is not None
    assert not eligible(house, executor.ledger_policy_state())
    assert executor.build_rejection("q") is not None

    executor.observe_hud(4, 5, {"wood": 210}, idle_present=True)
    assert ledger.pending_placements == [pending]  # inconclusive, not failed yet
    now = pending.settle_deadline + 1
    executor.observe_hud(4, 5, {"wood": 220}, idle_present=True)
    assert ledger.pending_placements == [pending]
    assert ledger.reservations()["wood"] == 25
    assert ledger.outcomes[-1].status == "uncertain"
    assert not eligible(house, executor.ledger_policy_state())
    executor.observe_hud(7, 10, {"wood": 220}, idle_present=True)
    assert eligible(house, executor.ledger_policy_state())


def test_interrupted_purchase_releases_only_when_no_input_was_issued(
    ledger: executor.ActionLedger,
) -> None:
    executor.observe_hud(4, 10, {"food": 1000, "gold": 100}, idle_present=True)
    before = executor._note_pending_research("feudal_age", executor._TECHS["feudal_age"])
    assert before is not None
    ledger.record_interrupted_purchase(before.operation_id, "feudal_age", 0, "cancelled")
    assert ledger.pending_research == []
    assert ledger.reservations().get("food", 0) == 0
    assert ledger.outcomes[-1].status == "cancelled"

    after = executor._note_pending_research("feudal_age", executor._TECHS["feudal_age"])
    assert after is not None
    ledger.note_input()
    ledger.record_interrupted_purchase(after.operation_id, "feudal_age", 0, "cancelled")
    assert ledger.pending_research == [after]
    assert ledger.reservations()["food"] == 500
    assert ledger.outcomes[-1].status == "uncertain"


def test_catalog_advertisements_and_executor_preflight_agree(
    ledger: executor.ActionLedger,
) -> None:
    executor.observe_hud(
        20,
        100,
        {"food": 5000, "wood": 5000, "gold": 5000, "stone": 5000},
        idle_present=True,
    )
    executor.observe_age("Dark Age")
    ledger.buildings_confirmed.update({"mill", "lumber_camp", "mining_camp", "barracks"})
    executor.set_detected_entities(
        [
            {"id": "v", "class": "villager", "center": (500, 400)},
            {"id": "t", "class": "tree", "center": (600, 400)},
            {"id": "s", "class": "sheep", "center": (550, 430)},
            {"id": "g", "class": "gold_mine", "center": (650, 410)},
        ]
    )
    state = executor.ledger_policy_state()
    advertised = feasible_candidates(state)
    assert {"advance_to_feudal", "build_farm", "assign_wood"} <= {
        candidate.id for candidate in advertised
    }
    for candidate in advertised:
        spec = BY_ID[candidate.id]
        if spec.kind == "build":
            assert executor.build_rejection(spec.key, menu=spec.menu) is None, candidate.id
        elif spec.kind == "research":
            assert executor.research_rejection(spec.subject) is None, candidate.id
        elif spec.kind == "train":
            assert eligible(spec, executor.ledger_policy_state()), candidate.id


def test_missing_age_reading_does_not_advertise_age_research(
    ledger: executor.ActionLedger,
) -> None:
    executor.observe_hud(20, 100, {"food": 1000, "wood": 500}, idle_present=True)
    ledger.buildings_confirmed.update({"mill", "lumber_camp"})
    assert not ledger.age_known
    assert "advance_to_feudal" not in {
        candidate.id for candidate in feasible_candidates(executor.ledger_policy_state())
    }
    executor.observe_age("Dark Age")
    assert "advance_to_feudal" in {
        candidate.id for candidate in feasible_candidates(executor.ledger_policy_state())
    }


def test_paid_unit_keeps_population_slot_until_delivery(
    ledger: executor.ActionLedger,
) -> None:
    executor.observe_hud(9, 10, {"food": 200, "gold": 200}, input_revision=0)
    executor.observe_age("Castle Age")
    ledger.buildings_confirmed.add("stable")
    knight = BY_ID["train_knight"]
    assert eligible(knight, executor.ledger_policy_state())

    pending = executor._note_pending_training("knight")
    assert pending is not None
    ledger.note_input()
    executor.observe_hud(9, 10, {"food": 140, "gold": 125}, input_revision=1)
    assert ledger.pending_training == []
    assert len(ledger.queued_training) == 1
    assert ledger.reservations().get("food", 0) == 0
    assert not eligible(knight, executor.ledger_policy_state())

    executor.observe_hud(
        9,
        10,
        {"food": 140, "gold": 125},
        population_known=False,
        input_revision=1,
    )
    assert len(ledger.queued_training) == 1
    executor.observe_hud(10, 10, {"food": 140, "gold": 125}, input_revision=1)
    assert ledger.queued_training == []
    assert [outcome.status for outcome in ledger.outcomes] == [
        "pending",
        "purchased",
        "confirmed",
    ]


def test_food_shortage_without_a_mill_sends_idle_worker_to_wood() -> None:
    state = PolicyState(
        food=0,
        wood=40,
        population=8,
        population_cap=20,
        idle_present=True,
        visible_classes=frozenset({"tree"}),
    )
    choice = select_fallback(feasible_candidates(state), state, focused("Dark Age", "food"))
    assert choice.id == "assign_wood"


def test_empty_workforce_respects_wood_only_allocation() -> None:
    state = PolicyState(
        food=100,
        wood=100,
        idle_present=True,
        visible_classes=frozenset({"tree", "sheep"}),
        villager_jobs={},
    )
    choice = select_fallback(feasible_candidates(state), state, Allocation(targets={"wood": 1}))
    assert choice.id == "assign_wood"


def test_unreadable_food_worker_count_does_not_force_a_food_goal() -> None:
    state = PolicyState(
        food=100,
        wood=100,
        idle_present=True,
        visible_classes=frozenset({"tree", "sheep"}),
        villager_jobs={},
    )
    goal = PolicyGoal("Establish food workers", "food_workers", "2", 10, 0.5)
    choice = select_fallback(
        feasible_candidates(state), state, Allocation(targets={"wood": 1}), (goal,)
    )
    assert choice.id == "assign_wood"


def test_production_scenario_reaches_imperial_and_trains_a_unit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(executor, "BUILD_SETTLE_DELAY", 0.0)
    monkeypatch.setattr(executor, "BUILD_RETRY_DELAY", 0.0)
    monkeypatch.setattr(executor, "RESCAN_SETTLE_DELAY", 0.0)
    monkeypatch.setattr(executor.config, "action_delay", 0.0)
    fixture = (
        Path(__file__).resolve().parents[1]
        / "apps/agent/src/scenarios/production/castle_imperial_army.yaml"
    )
    result = asyncio.run(run_scenario_async(fixture))
    assert result.passed, result.failures
    assert "advance_to_castle" in result.actions
    assert "advance_to_imperial" in result.actions
    assert "train_knight" in result.actions
    assert "press:z" in result.inputs


def test_production_scenario_does_not_apply_unearned_consequences(tmp_path: Path) -> None:
    fixture = tmp_path / "invalid.yaml"
    fixture.write_text(
        """initial:
  resources: {food: 500, wood: 500, gold: 500, stone: 0}
  population: 4
  population_cap: 10
  age: Dark Age
steps:
  - action: train_knight
    after: {age: Imperial Age}
""",
        encoding="utf-8",
    )
    result = asyncio.run(run_scenario_async(fixture))
    assert not result.passed
    assert result.observed_age == "Dark Age"
    assert "not eligible" in result.failures[0]
