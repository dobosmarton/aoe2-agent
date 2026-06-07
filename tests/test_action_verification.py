"""Unit tests for R1 action-effect verification in gameplay_agent/turn_phases.py.

Pure helpers over fake entity dicts + one integration test proving the output
feeds the existing stuck-loop detector. No pyautogui / executor calls.
"""

from __future__ import annotations

from gameplay_agent.executor import ActionResult
from gameplay_agent.turn_phases import (
    _any_entity_expectation,
    _build_verification,
    _expectation_for,
    _new_buildings,
)


def _ent(cls: str) -> dict:
    return {"class": cls, "id": f"{cls}_0", "center": (0, 0), "confidence": 0.9}


# ---------------------------------------------------------------------------
# _expectation_for
# ---------------------------------------------------------------------------


def test_build_composite_is_new_building() -> None:
    assert _expectation_for({"type": "build", "intent": "Build house"}).kind == "new_building"


def test_place_intent_click_is_new_building() -> None:
    assert _expectation_for({"type": "click", "intent": "place lumber camp"}).kind == "new_building"


def test_gather_order_is_none() -> None:
    assert _expectation_for({"type": "right_click", "intent": "gather from sheep"}).kind == "none"
    assert _expectation_for({"type": "scroll", "clicks": 5}).kind == "none"


def test_camera_press_is_selection_change() -> None:
    assert _expectation_for({"type": "press", "key": "h"}).kind == "selection_change"
    assert (
        _expectation_for({"type": "press", "key": "x", "rescan": True}).kind == "selection_change"
    )


def test_plain_press_is_none() -> None:
    assert _expectation_for({"type": "press", "key": "q"}).kind == "none"


def test_any_entity_expectation() -> None:
    assert _any_entity_expectation([{"type": "build", "intent": "house"}])
    assert not _any_entity_expectation([{"type": "right_click", "intent": "gather"}])


# ---------------------------------------------------------------------------
# _new_buildings / _build_verification
# ---------------------------------------------------------------------------


def test_new_buildings_detects_increase() -> None:
    before = [_ent("villager")]
    after = [_ent("villager"), _ent("house")]
    assert _new_buildings(before, after) == ["house"]


def test_verification_confirms_new_building() -> None:
    actions = [{"type": "build", "intent": "Build house"}]
    results = [ActionResult(True, "ok")]
    out = _build_verification(
        actions, results, [_ent("villager")], [_ent("villager"), _ent("house")]
    )
    assert "CONFIRMED built: house" in out


def test_verification_reports_no_visible_change_on_failed_build() -> None:
    actions = [{"type": "build", "intent": "Build house"}]
    results = [ActionResult(True, "ok")]
    out = _build_verification(actions, results, [_ent("villager")], [_ent("villager")])
    assert "no visible change" in out


def test_verification_includes_failed_actions() -> None:
    actions = [{"type": "click", "intent": "send villager"}]
    results = [ActionResult(False, "target_id not found")]
    out = _build_verification(actions, results, [], [])
    assert "FAILED click" in out


def test_selection_change_no_visible_change_when_counts_identical() -> None:
    actions = [{"type": "press", "key": "h", "intent": "select TC"}]
    results = [ActionResult(True, "ok")]
    same = [_ent("town_center"), _ent("villager")]
    out = _build_verification(actions, results, same, list(same))
    assert "no visible change: view unchanged" in out


def test_selection_change_silent_when_view_changed() -> None:
    actions = [{"type": "press", "key": "h", "intent": "select TC"}]
    results = [ActionResult(True, "ok")]
    before = [_ent("town_center")]
    after = [_ent("town_center"), _ent("sheep")]  # view scrolled, new entity
    out = _build_verification(actions, results, before, after)
    assert "no visible change" not in out


# ---------------------------------------------------------------------------
# Integration: "no visible change" drives the existing stuck-loop detector
# ---------------------------------------------------------------------------


def test_no_visible_change_feeds_stuck_loop() -> None:
    from gameplay_agent.memory import AgentMemory

    mem = AgentMemory()
    for _ in range(3):
        mem.create_turn(reasoning="trying", actions=[], observations={})
        mem.set_last_verification("- no visible change: build produced no new building")
    ctx = mem.get_context_for_llm()
    assert "NO EFFECT" in ctx
