"""Unit tests for evaluation/world_sim.py.

All tests are offline (no LLM, no API key). Tests cover:
  1. WorldState initialisation from fixture inputs
  2. Individual action effects (queue_villager, build, press z)
  3. Tick mechanics (resource gathering, villager completion, age advancement)
  4. apply_actions dispatching
  5. state_to_fixture_inputs round-trip
  6. evaluate_end_state assertions
"""

from __future__ import annotations

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


# ---------------------------------------------------------------------------
# init_from_fixture
# ---------------------------------------------------------------------------


def test_init_from_fixture_parses_population():
    from evaluation.world_sim import init_from_fixture

    inputs = {
        "age": "Dark Age",
        "resources": {"food": 300, "wood": 250, "gold": 0, "stone": 0, "population": "12/20"},
        "detected_entities": [],
    }
    state = init_from_fixture(inputs)
    assert state.population == 12
    assert state.pop_cap == 20
    assert state.food == 300.0
    assert state.wood == 250.0


def test_init_from_fixture_seeds_buildings_from_entities():
    from evaluation.world_sim import init_from_fixture

    inputs = {
        "age": "Dark Age",
        "resources": {"food": 200, "wood": 200, "gold": 0, "stone": 0, "population": "8/25"},
        "detected_entities": [
            {"class": "mill", "x": 800, "y": 600},
            {"class": "lumber_camp", "x": 1000, "y": 600},
            {"class": "villager", "x": 500, "y": 500},
        ],
    }
    state = init_from_fixture(inputs)
    assert "mill" in state.buildings
    assert "lumber_camp" in state.buildings
    assert "villager" not in state.buildings  # non-buildings excluded


# ---------------------------------------------------------------------------
# queue_villager
# ---------------------------------------------------------------------------


def test_queue_villager_deducts_50_food():
    from evaluation.world_sim import _apply_queue_villager

    state = _state(food=200.0)
    new_state = _apply_queue_villager(state)
    assert new_state.food == 150.0
    assert len(new_state.villager_queue) == 1
    assert new_state.villager_queue[0] == 3  # 3-tick countdown


def test_queue_villager_noop_when_food_insufficient():
    from evaluation.world_sim import _apply_queue_villager

    state = _state(food=30.0)
    new_state = _apply_queue_villager(state)
    assert new_state.food == 30.0
    assert new_state.villager_queue == []


def test_multiple_queue_villager_accumulates():
    from evaluation.world_sim import _apply_queue_villager

    state = _state(food=200.0)
    state = _apply_queue_villager(state)
    state = _apply_queue_villager(state)
    assert state.food == 100.0
    assert len(state.villager_queue) == 2


# ---------------------------------------------------------------------------
# Villager completion via tick
# ---------------------------------------------------------------------------


def test_villager_completes_after_3_ticks():
    from evaluation.world_sim import _apply_queue_villager, tick

    state = _state(food=200.0, population=8)
    state = _apply_queue_villager(state)
    assert state.population == 8
    state = tick(state)  # countdown 3→2
    state = tick(state)  # countdown 2→1
    state = tick(state)  # countdown 1→0, villager completes
    assert state.population == 9
    assert state.villager_queue == []


def test_two_villagers_both_complete():
    from evaluation.world_sim import _apply_queue_villager, tick

    state = _state(food=200.0, population=8)
    state = _apply_queue_villager(state)
    state = _apply_queue_villager(state)
    for _ in range(3):
        state = tick(state)
    assert state.population == 10


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def test_build_house_deducts_25_wood_and_expands_pop_cap():
    from evaluation.world_sim import _apply_build

    state = _state(wood=100.0, pop_cap=25, buildings=[])
    new_state = _apply_build(state, "q")  # q = house
    assert new_state.wood == 75.0
    assert new_state.pop_cap == 30
    assert "house" in new_state.buildings


def test_build_mill_deducts_100_wood():
    from evaluation.world_sim import _apply_build

    state = _state(wood=150.0, buildings=[])
    new_state = _apply_build(state, "w")  # w = mill
    assert new_state.wood == 50.0
    assert "mill" in new_state.buildings


def test_build_noop_when_wood_insufficient():
    from evaluation.world_sim import _apply_build

    state = _state(wood=20.0, buildings=[])
    new_state = _apply_build(state, "q")  # house costs 25
    assert new_state.wood == 20.0
    assert new_state.buildings == []


def test_build_noop_on_unknown_key():
    from evaluation.world_sim import _apply_build

    state = _state(wood=200.0, buildings=[])
    new_state = _apply_build(state, "x")
    assert new_state.buildings == []


# ---------------------------------------------------------------------------
# Age up
# ---------------------------------------------------------------------------


def test_age_up_starts_6_tick_timer_when_prereqs_met():
    from evaluation.world_sim import _apply_age_up

    state = _state(
        food=600.0,
        population=22,
        age="Dark Age",
        buildings=["mill", "lumber_camp"],
    )
    new_state = _apply_age_up(state)
    assert new_state.age_up_ticks_remaining == 6
    assert new_state.food == 100.0


def test_age_up_noop_when_food_insufficient():
    from evaluation.world_sim import _apply_age_up

    state = _state(food=400.0, population=22, age="Dark Age", buildings=["mill", "lumber_camp"])
    new_state = _apply_age_up(state)
    assert new_state.age_up_ticks_remaining == 0


def test_age_up_noop_when_population_insufficient():
    from evaluation.world_sim import _apply_age_up

    state = _state(food=600.0, population=18, age="Dark Age", buildings=["mill", "lumber_camp"])
    new_state = _apply_age_up(state)
    assert new_state.age_up_ticks_remaining == 0


def test_age_up_noop_when_prereq_buildings_missing():
    from evaluation.world_sim import _apply_age_up

    state = _state(
        food=600.0, population=22, age="Dark Age", buildings=["mill"]
    )  # missing lumber_camp
    new_state = _apply_age_up(state)
    assert new_state.age_up_ticks_remaining == 0


def test_age_up_noop_when_already_in_progress():
    from evaluation.world_sim import _apply_age_up

    state = _state(
        food=600.0,
        population=22,
        age="Dark Age",
        buildings=["mill", "lumber_camp"],
        age_up_ticks_remaining=3,
    )
    new_state = _apply_age_up(state)
    assert new_state.age_up_ticks_remaining == 3  # unchanged


def test_age_advances_to_feudal_after_6_ticks():
    from evaluation.world_sim import _apply_age_up, tick

    state = _state(food=600.0, population=22, age="Dark Age", buildings=["mill", "lumber_camp"])
    state = _apply_age_up(state)
    for _ in range(6):
        state = tick(state)
    assert state.age == "Feudal Age"
    assert state.age_up_ticks_remaining == 0


# ---------------------------------------------------------------------------
# Resource gather tick
# ---------------------------------------------------------------------------


def test_tick_adds_gather_rates():
    from evaluation.world_sim import FOOD_GATHER_RATE, WOOD_GATHER_RATE, tick

    state = _state(food=100.0, wood=50.0)
    new_state = tick(state)
    assert new_state.food == 100.0 + FOOD_GATHER_RATE
    assert new_state.wood == 50.0 + WOOD_GATHER_RATE


def test_tick_increments_turn():
    from evaluation.world_sim import tick

    state = _state(turn=4)
    new_state = tick(state)
    assert new_state.turn == 5


# ---------------------------------------------------------------------------
# apply_actions dispatch
# ---------------------------------------------------------------------------


def test_apply_actions_dispatches_queue_villager():
    from evaluation.world_sim import apply_actions

    state = _state(food=200.0)
    actions = [{"type": "queue_villager"}]
    new_state = apply_actions(state, actions)
    assert new_state.food == 150.0


def test_apply_actions_dispatches_build():
    from evaluation.world_sim import apply_actions

    state = _state(wood=150.0, buildings=[])
    actions = [{"type": "build", "building_key": "q"}]
    new_state = apply_actions(state, actions)
    assert "house" in new_state.buildings


def test_apply_actions_dispatches_press_z():
    from evaluation.world_sim import apply_actions

    state = _state(food=600.0, population=22, age="Dark Age", buildings=["mill", "lumber_camp"])
    actions = [{"type": "press", "key": "z"}]
    new_state = apply_actions(state, actions)
    assert new_state.age_up_ticks_remaining == 6


def test_apply_actions_ignores_unrelated_presses():
    from evaluation.world_sim import apply_actions

    state = _state(food=200.0)
    actions = [
        {"type": "press", "key": "h"},
        {"type": "right_click", "target_class": "sheep"},
        {"type": "send_villager", "target_class": "tree"},
    ]
    new_state = apply_actions(state, actions)
    # Only food should be unchanged (no queue_villager, no build)
    assert new_state.food == 200.0
    assert new_state.buildings == []


# ---------------------------------------------------------------------------
# state_to_fixture_inputs round-trip
# ---------------------------------------------------------------------------


def test_state_to_fixture_inputs_encodes_population():
    from evaluation.world_sim import state_to_fixture_inputs

    state = _state(population=12, pop_cap=20, food=300.0, wood=250.0, gold=50.0, age="Feudal Age")
    base = {"resources": {"gold": 0, "stone": 100}}
    result = state_to_fixture_inputs(state, base)
    assert result["age"] == "Feudal Age"
    assert result["resources"]["population"] == "12/20"
    assert result["resources"]["food"] == 300
    assert result["resources"]["wood"] == 250
    assert result["resources"]["gold"] == 50  # world state gold takes precedence


# ---------------------------------------------------------------------------
# evaluate_end_state
# ---------------------------------------------------------------------------


def test_evaluate_end_state_passes_on_age_match():
    from evaluation.world_sim import evaluate_end_state

    state = _state(age="Feudal Age", turn=8)
    failures = evaluate_end_state({"age": "Feudal Age"}, state)
    assert failures == []


def test_evaluate_end_state_fails_on_age_mismatch():
    from evaluation.world_sim import evaluate_end_state

    state = _state(age="Dark Age", turn=15)
    failures = evaluate_end_state({"age": "Feudal Age"}, state)
    assert any("end_state FAILED" in f for f in failures)
    assert any("Dark Age" in f for f in failures)


def test_evaluate_end_state_numeric_uses_gte_semantics():
    from evaluation.world_sim import evaluate_end_state

    state = _state(population=18, turn=10)
    assert evaluate_end_state({"population": 15}, state) == []
    assert evaluate_end_state({"population": 18}, state) == []
    failures = evaluate_end_state({"population": 20}, state)
    assert any("end_state FAILED" in f for f in failures)


def test_evaluate_end_state_unknown_field_is_failure():
    from evaluation.world_sim import evaluate_end_state

    state = _state(turn=5)
    failures = evaluate_end_state({"nonexistent_field": 42}, state)
    assert any("unknown WorldState field" in f for f in failures)


# ---------------------------------------------------------------------------
# render() — WorldState → list[DetectedEntity] projection (Phase 1)
# ---------------------------------------------------------------------------


def test_render_emits_exactly_one_town_center():
    from evaluation.world_sim import render

    entities = render(_state(population=0, buildings=[]))
    town_centers = [e for e in entities if e.class_name == "town_center"]
    assert len(town_centers) == 1


def test_render_villager_count_equals_population():
    from evaluation.world_sim import render

    entities = render(_state(population=5, buildings=[]))
    villager_count = sum(1 for e in entities if e.class_name == "villager")
    assert villager_count == 5


def test_render_ignores_villager_queue():
    from evaluation.world_sim import render

    entities = render(_state(population=5, villager_queue=[3, 3, 3], buildings=[]))
    villager_count = sum(1 for e in entities if e.class_name == "villager")
    assert villager_count == 5


def test_render_emits_one_entity_per_building():
    from evaluation.world_sim import render

    entities = render(_state(population=0, buildings=["mill", "lumber_camp", "house"]))
    building_class_names = {"mill", "lumber_camp", "house"}
    matched = [e for e in entities if e.class_name in building_class_names]
    assert len(matched) == 3


def test_render_preserves_building_class_names():
    from evaluation.world_sim import render

    entities = render(_state(population=0, buildings=["mill", "lumber_camp", "house"]))
    rendered_classes = {e.class_name for e in entities if e.class_name != "town_center"}
    assert rendered_classes == {"mill", "lumber_camp", "house"}


def test_render_empty_state_emits_only_town_center():
    from evaluation.world_sim import render

    entities = render(_state(population=0, buildings=[]))
    assert len(entities) == 1


def test_render_is_deterministic():
    from evaluation.world_sim import render

    state = _state(population=4, buildings=["mill"])
    assert render(state, 1920, 1080) == render(state, 1920, 1080)


def test_render_seed_overrides_turn():
    from evaluation.world_sim import render

    state = _state(population=4, buildings=[], turn=7)
    tc_a = next(e for e in render(state, seed=1) if e.class_name == "town_center")
    tc_b = next(e for e in render(state, seed=2) if e.class_name == "town_center")
    assert tc_a.center != tc_b.center


@pytest.mark.parametrize("width,height", [(1280, 720), (1920, 1080)])
def test_render_bboxes_are_within_image_dims(width, height):
    # Projection assumes realistic screenshot dims (≥ 720p). Smaller canvases
    # are out of contract because the building-grid offset would overflow.
    from evaluation.world_sim import render

    entities = render(_state(population=3, buildings=["mill"]), width, height)
    out_of_bounds = [
        e
        for e in entities
        if not (0 <= e.bbox[0] <= width and 0 <= e.bbox[2] <= width)
        or not (0 <= e.bbox[1] <= height and 0 <= e.bbox[3] <= height)
    ]
    assert out_of_bounds == []


def test_render_building_positions_stable_across_appends():
    from evaluation.world_sim import render

    state_two = _state(population=0, buildings=["mill", "lumber_camp"])
    state_three = _state(population=0, buildings=["mill", "lumber_camp", "house"])
    mill_two = next(e for e in render(state_two) if e.class_name == "mill")
    mill_three = next(e for e in render(state_three) if e.class_name == "mill")
    assert mill_two.center == mill_three.center


def test_render_confidence_is_ground_truth():
    from evaluation.world_sim import render

    entities = render(_state(population=3, buildings=["mill", "house"]))
    confidences = {e.confidence for e in entities}
    assert confidences == {1.0}


def test_render_all_ids_are_unique():
    from evaluation.world_sim import render

    entities = render(_state(population=5, buildings=["mill", "lumber_camp", "house"]))
    ids = [e.id for e in entities]
    assert len(set(ids)) == len(ids)


# ---------------------------------------------------------------------------
# apply_action — single-action dispatch covering all 7 emitted types (Phase 2)
# ---------------------------------------------------------------------------


def test_apply_action_queue_villager_deducts_food():
    from evaluation.world_sim import apply_action

    new_state = apply_action(_state(food=200.0), {"type": "queue_villager"})
    assert new_state.food == 150.0


def test_apply_action_build_adds_house():
    from evaluation.world_sim import apply_action

    new_state = apply_action(
        _state(wood=150.0, buildings=[]), {"type": "build", "building_key": "q"}
    )
    assert "house" in new_state.buildings


def test_apply_action_press_z_starts_age_up():
    from evaluation.world_sim import apply_action

    state = _state(food=600.0, population=22, age="Dark Age", buildings=["mill", "lumber_camp"])
    new_state = apply_action(state, {"type": "press", "key": "z"})
    assert new_state.age_up_ticks_remaining == 6


def test_apply_action_press_non_z_is_noop():
    from evaluation.world_sim import apply_action

    state = _state(food=200.0)
    new_state = apply_action(state, {"type": "press", "key": "b"})
    assert new_state == state


@pytest.mark.parametrize(
    "action_type", ["click", "right_click", "drag", "wait", "scroll", "detect"]
)
def test_apply_action_mouse_keyboard_primitives_are_noop(action_type):
    from evaluation.world_sim import apply_action

    state = _state(food=200.0, wood=150.0, buildings=["mill"])
    new_state = apply_action(state, {"type": action_type})
    assert new_state == state


def test_apply_action_unknown_type_is_noop():
    from evaluation.world_sim import apply_action

    state = _state(food=200.0)
    new_state = apply_action(state, {"type": "this_action_does_not_exist"})
    assert new_state == state


def test_apply_action_missing_type_is_noop():
    from evaluation.world_sim import apply_action

    state = _state(food=200.0)
    new_state = apply_action(state, {})
    assert new_state == state
