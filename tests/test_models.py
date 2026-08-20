"""Unit tests for gameplay_agent/models.py — Pydantic action validation.

Layers:
  1. Per-action validation (ClickAction, RightClickAction, PressAction,
     DragAction, WaitAction, ScrollAction, DetectAction).
  2. PointTargetAction's coords-or-target invariant.
  3. PressAction's key validator (single chars, special keys, function keys).
  4. validate_action / validate_actions dispatcher.
  5. LLMResponse's salvage-valid-actions field_validator.
  6. Observations.
"""

from __future__ import annotations

import pytest
from gameplay_agent.models import (
    BuildAction,
    ClickAction,
    DetectAction,
    DragAction,
    LLMResponse,
    Observations,
    PressAction,
    RightClickAction,
    ScrollAction,
    WaitAction,
    validate_action,
    validate_actions,
)
from gameplay_agent.providers.strategist import StrategistResponse
from pydantic import BaseModel, ValidationError

# ---------------------------------------------------------------------------
# Layer 1 — per-action validation (happy paths)
# ---------------------------------------------------------------------------


def test_click_with_coords_validates() -> None:
    a = ClickAction(type="click", x=100, y=200, intent="ok")
    assert a.x == 100
    assert a.y == 200


def test_right_click_with_target_id_validates() -> None:
    a = RightClickAction(type="right_click", target_id="sheep_0", intent="gather")
    assert a.target_id == "sheep_0"
    assert a.x is None


def test_right_click_with_target_class_validates() -> None:
    a = RightClickAction(type="right_click", target_class="tree", intent="chop")
    assert a.target_class == "tree"


def test_press_simple_key() -> None:
    a = PressAction(type="press", key="h", intent="TC")
    assert a.key == "h"
    assert a.modifiers == []
    assert a.rescan is False


def test_drag_validates() -> None:
    a = DragAction(type="drag", start_x=10, start_y=20, end_x=50, end_y=60, intent="select")
    assert a.end_x == 50


def test_wait_validates() -> None:
    a = WaitAction(type="wait", ms=500, intent="settle")
    assert a.ms == 500


def test_scroll_with_coords_validates() -> None:
    a = ScrollAction(type="scroll", clicks=3, x=100, y=200, intent="zoom in")
    assert a.clicks == 3
    assert a.x == 100


def test_scroll_without_coords_validates() -> None:
    a = ScrollAction(type="scroll", clicks=-2, intent="zoom out")
    assert a.clicks == -2
    assert a.x is None


def test_detect_validates() -> None:
    a = DetectAction(type="detect", intent="full scan")
    assert a.type == "detect"


def test_build_validates_with_building_key() -> None:
    a = BuildAction(type="build", building_key="w", intent="build mill")
    assert a.building_key == "w"
    assert a.type == "build"


def test_build_requires_building_key() -> None:
    with pytest.raises(ValidationError):
        BuildAction.model_validate({"type": "build", "intent": "build something"})


def test_validate_action_dispatches_build() -> None:
    a = validate_action({"type": "build", "building_key": "q", "intent": "house"})
    assert isinstance(a, BuildAction)
    assert a.building_key == "q"


# ---------------------------------------------------------------------------
# Layer 2 — PointTargetAction's coords-or-target invariant
# ---------------------------------------------------------------------------


def test_click_without_coords_target_or_class_raises() -> None:
    with pytest.raises(ValidationError, match="Must provide"):
        ClickAction(type="click", intent="nowhere")


def test_click_with_only_x_no_y_raises() -> None:
    with pytest.raises(ValidationError, match="Must provide"):
        ClickAction(type="click", x=100, intent="incomplete")


def test_right_click_negative_coord_rejected() -> None:
    with pytest.raises(ValidationError):
        RightClickAction(type="right_click", x=-5, y=10, intent="bad")


def test_right_click_oversized_coord_rejected() -> None:
    """Coord ranges go up to 7680 wide / 4320 tall (8K)."""
    with pytest.raises(ValidationError):
        RightClickAction(type="right_click", x=10000, y=10, intent="bad")


# ---------------------------------------------------------------------------
# Layer 3 — PressAction.validate_key
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["a", "z", "1", ".", ",", "/", "?"])
def test_press_single_character_keys(key: str) -> None:
    a = PressAction(type="press", key=key, intent="x")
    assert a.key == key


@pytest.mark.parametrize(
    "key",
    [
        "enter",
        "return",
        "space",
        "tab",
        "backspace",
        "delete",
        "up",
        "down",
        "left",
        "right",
        "ctrl",
        "alt",
        "shift",
    ],
)
def test_press_special_keys_normalized_to_lower(key: str) -> None:
    a = PressAction(type="press", key=key.upper(), intent="x")
    assert a.key == key  # lowercased


@pytest.mark.parametrize("key", ["f1", "f5", "f12"])
def test_press_function_keys_accepted(key: str) -> None:
    a = PressAction(type="press", key=key, intent="x")
    assert a.key == key


@pytest.mark.parametrize("key", ["escape", "ESC", "f10", "f3"])
def test_press_menu_pausing_keys_rejected(key: str) -> None:
    # Escape with nothing to cancel opens the game menu and pauses the game
    # (run 8, F-32); F10 is the menu, F3 is pause. 'h' clears UI state instead.
    with pytest.raises(ValidationError, match="opens the game menu"):
        PressAction(type="press", key=key, intent="x")


def test_press_invalid_key_rejected() -> None:
    with pytest.raises(ValidationError, match="Invalid key"):
        PressAction(type="press", key="hellokitty", intent="bad")


def test_press_with_modifiers_and_rescan() -> None:
    a = PressAction(type="press", key="1", modifiers=["ctrl"], rescan=True, intent="select group")
    assert a.modifiers == ["ctrl"]
    assert a.rescan is True


def test_press_empty_key_rejected() -> None:
    with pytest.raises(ValidationError):
        PressAction(type="press", key="", intent="x")


# ---------------------------------------------------------------------------
# Layer 4 — validate_action / validate_actions
# ---------------------------------------------------------------------------


def test_validate_action_returns_typed_instance() -> None:
    result = validate_action({"type": "press", "key": "h", "intent": "TC"})
    assert isinstance(result, PressAction)
    assert result.key == "h"


def test_validate_action_dispatches_by_type() -> None:
    click = validate_action({"type": "click", "x": 10, "y": 20, "intent": "x"})
    drag = validate_action({"type": "drag", "start_x": 1, "start_y": 2, "end_x": 3, "end_y": 4})
    assert isinstance(click, ClickAction)
    assert isinstance(drag, DragAction)


def test_validate_action_unknown_type_returns_none() -> None:
    assert validate_action({"type": "fly_to_moon", "intent": "win"}) is None


def test_validate_action_missing_type_returns_none() -> None:
    assert validate_action({"x": 10, "y": 20}) is None


def test_validate_action_invalid_dict_returns_none() -> None:
    """Action dict missing required field returns None, not raises."""
    assert validate_action({"type": "press"}) is None  # missing `key`


def test_validate_action_invalid_coord_range_returns_none() -> None:
    assert validate_action({"type": "click", "x": -1, "y": 0, "intent": "x"}) is None


def test_validate_actions_filters_invalid_silently() -> None:
    raw = [
        {"type": "press", "key": "h", "intent": "TC"},
        {"type": "press"},  # invalid — no key
        {"type": "fly_to_moon"},  # unknown type
        {"type": "click", "x": 10, "y": 20, "intent": "x"},
    ]
    result = validate_actions(raw)
    assert len(result) == 2
    assert isinstance(result[0], PressAction)
    assert isinstance(result[1], ClickAction)


def test_validate_actions_empty_list() -> None:
    assert validate_actions([]) == []


# ---------------------------------------------------------------------------
# Layer 5 — LLMResponse.salvage_valid_actions
# ---------------------------------------------------------------------------


def test_llm_response_drops_invalid_actions_keeps_valid() -> None:
    """A single bad action shouldn't fail the whole LLMResponse — see the
    field_validator's docstring (referencing messages.parse() failure modes)."""
    data = {
        "actions": [
            {"type": "press", "key": "h", "intent": "TC"},
            {"type": "right_click"},  # invalid — no coords/target
            {"type": "click", "x": 10, "y": 20, "intent": "x"},
        ],
        "reasoning": "ok",
    }
    resp = LLMResponse.model_validate(data)
    assert len(resp.actions) == 2  # the right_click was dropped
    assert resp.reasoning == "ok"


def test_llm_response_with_no_actions() -> None:
    resp = LLMResponse.model_validate({"actions": [], "reasoning": "thinking"})
    assert resp.actions == []
    assert resp.reasoning == "thinking"


def test_llm_response_default_observations() -> None:
    assert LLMResponse().observations.game_state == "playing"


def test_llm_response_passes_through_already_validated_actions() -> None:
    """If the input is a Pydantic Action instance (not dict), keep it."""
    pre_validated = PressAction(type="press", key="h", intent="x")
    resp = LLMResponse.model_validate({"actions": [pre_validated]})
    assert len(resp.actions) == 1


# ---------------------------------------------------------------------------
# Layer 6 — Observations
# ---------------------------------------------------------------------------


def test_observations_defaults() -> None:
    obs = Observations()
    assert obs.game_state == "playing"
    assert obs.idle_tc is False
    assert obs.events == []


def test_observations_full_payload() -> None:
    obs = Observations(
        population="8/15",
        age="Dark Age",
        idle_tc=True,
        under_attack=False,
        game_state="playing",
        events=["villager_idle"],
    )
    assert obs.population == "8/15"
    assert obs.idle_tc is True


def test_observations_invalid_game_state_rejected() -> None:
    with pytest.raises(ValidationError):
        Observations(game_state="paused")


def test_click_action_preserves_building_key() -> None:
    """The place-click's building_key must survive validation — it's what lets
    the executor verify the placement landed (it was silently dropped before)."""
    action = validate_action(
        {"type": "click", "x": 10, "y": 20, "building_key": "q", "intent": "Place building"}
    )
    assert action is not None
    assert action.model_dump()["building_key"] == "q"


# ---------------------------------------------------------------------------
# Layer 7 — grammar-size regression guard (F-40)
# ---------------------------------------------------------------------------
# The single-shot executor path feeds LLMResponse's JSON schema to Anthropic
# structured output as a constrained-decoding grammar. Bounded integers
# (Field(ge=, le=)) emit minimum/maximum, each of which compiles to a large
# numeric automaton; 22 such bounds across the Action union pushed the compiled
# grammar over Anthropic's size limit and 400'd every executor turn for a whole
# game (run 12). Ranges are now enforced by field_validators, which validate but
# emit no schema bounds. These tests fail the moment a Field(ge=/le=) creeps
# back — catching the regression here instead of on a burned VM run.


def test_action_schema_has_no_numeric_bounds() -> None:
    """No minimum/maximum anywhere in the single-shot schema — they are the
    constrained-decoding heavyweight that blew the grammar limit (F-40)."""
    import json

    schema_text = json.dumps(LLMResponse.model_json_schema())
    assert '"minimum"' not in schema_text, "a Field(ge=) bound regressed — see F-40"
    assert '"maximum"' not in schema_text, "a Field(le=) bound regressed — see F-40"


def test_coordinate_ranges_still_enforced_by_validator() -> None:
    """Dropping schema bounds must NOT drop enforcement — validators still reject
    out-of-range coords/drags/waits (the contract the bounds used to hold)."""
    with pytest.raises(ValidationError):
        DragAction(type="drag", start_x=1, start_y=2, end_x=99999, end_y=4, intent="oob")
    with pytest.raises(ValidationError):
        WaitAction(type="wait", ms=999999, intent="too long")
    # and valid values still pass through unchanged
    assert DragAction(type="drag", start_x=1, start_y=2, end_x=3, end_y=4).end_x == 3
    assert WaitAction(type="wait", ms=1000).ms == 1000


# ---------------------------------------------------------------------------
# Open objects — the other schema shape that 400s every call
# ---------------------------------------------------------------------------
#
# OpenAI strict structured outputs requires every object to declare `properties`.
# A `dict[str, int]` field emits `{"type":"object","additionalProperties":{...}}`
# with none, and the server rejects the whole request. Two such fields cost a
# whole game on 2026-08-20: llm_calls=1, llm_errors=1, llm_error_rate=1.0.
# Same reasoning as the bounds tests above — catch it here, not on a burned run.


def _open_objects(model: type[BaseModel]) -> list[str]:
    """Names of object properties that declare no `properties` of their own."""
    schema = model.model_json_schema()
    found: list[str] = []
    for owner, node in [("", schema), *schema.get("$defs", {}).items()]:
        for name, prop in node.get("properties", {}).items():
            if isinstance(prop, dict) and prop.get("type") == "object" and "properties" not in prop:
                found.append(f"{owner}.{name}" if owner else name)
    return found


def test_llm_response_declares_no_open_objects() -> None:
    assert _open_objects(LLMResponse) == []


def test_strategist_response_declares_no_open_objects() -> None:
    assert _open_objects(StrategistResponse) == []
