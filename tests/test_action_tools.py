"""Schema validation tests for `_ACTION_TOOLS`.

Anthropic's tool-use API rejects requests with malformed schemas at runtime —
these tests catch field/shape regressions without paying for an API call.

The contract:
  - Each tool is a dict with `name`, `description`, `input_schema` keys.
  - `input_schema` is a JSON-Schema-shaped object with type/properties/required.
  - Tool names are unique (the agentic loop dispatches on name).
  - Composite tools (build, send_villager, queue_villager) only require
    `intent` — extra fields are optional. Standard tools require all the
    coordinates/keys they actually need to execute.
"""

from __future__ import annotations

import pytest
from gameplay_agent.providers.action_tools import _ACTION_TOOLS, _click_schema

# ---------------------------------------------------------------------------
# _click_schema helper
# ---------------------------------------------------------------------------


def test_click_schema_returns_object_type():
    schema = _click_schema("test")
    assert schema["type"] == "object"


def test_click_schema_requires_x_y_intent():
    schema = _click_schema("test")
    assert set(schema["required"]) == {"x", "y", "intent"}


def test_click_schema_target_class_is_optional():
    """target_class is a useful escape hatch but mustn't be forced."""
    schema = _click_schema("test")
    assert "target_class" in schema["properties"]
    assert "target_class" not in schema["required"]


def test_click_schema_disallows_extra_properties():
    schema = _click_schema("test")
    assert schema["additionalProperties"] is False


def test_click_schema_uses_passed_description_for_intent():
    schema = _click_schema("Why this click happens")
    assert schema["properties"]["intent"]["description"] == "Why this click happens"


# ---------------------------------------------------------------------------
# _ACTION_TOOLS structural contract
# ---------------------------------------------------------------------------


def test_action_tools_is_non_empty_list():
    assert isinstance(_ACTION_TOOLS, list)
    assert len(_ACTION_TOOLS) > 0


def test_every_tool_has_required_top_level_fields():
    for tool in _ACTION_TOOLS:
        assert {"name", "description", "input_schema"} <= tool.keys(), tool


def test_tool_names_are_unique():
    names = [t["name"] for t in _ACTION_TOOLS]
    assert len(names) == len(set(names))


def test_expected_tool_set_present():
    """The 8 standard + 3 composite tools the agent depends on."""
    names = {t["name"] for t in _ACTION_TOOLS}
    assert {
        "click",
        "right_click",
        "press",
        "drag",
        "wait",
        "scroll",
        "detect",
        "build",
        "send_villager",
        "queue_villager",
    } <= names


def test_every_input_schema_is_object():
    for tool in _ACTION_TOOLS:
        assert tool["input_schema"]["type"] == "object", tool["name"]


def test_every_input_schema_has_properties():
    for tool in _ACTION_TOOLS:
        assert "properties" in tool["input_schema"], tool["name"]
        assert isinstance(tool["input_schema"]["properties"], dict)


def test_every_input_schema_has_required_list():
    for tool in _ACTION_TOOLS:
        required = tool["input_schema"].get("required", [])
        assert isinstance(required, list), tool["name"]


def test_every_required_field_appears_in_properties():
    """A `required` field that isn't in `properties` rejects every call."""
    for tool in _ACTION_TOOLS:
        properties = tool["input_schema"]["properties"]
        for field in tool["input_schema"].get("required", []):
            assert field in properties, (
                f"{tool['name']}: required field {field!r} not in properties"
            )


def test_every_input_schema_blocks_additional_properties():
    """Anthropic flags loose schemas as low-quality — be strict everywhere."""
    for tool in _ACTION_TOOLS:
        assert tool["input_schema"].get("additionalProperties") is False, tool["name"]


def test_every_tool_requires_intent():
    """`intent` is required by every tool so the agent always explains itself."""
    for tool in _ACTION_TOOLS:
        assert "intent" in tool["input_schema"]["required"], tool["name"]


# ---------------------------------------------------------------------------
# Per-tool spot checks (catches accidental field deletions)
# ---------------------------------------------------------------------------


def _by_name(name: str) -> dict:
    for tool in _ACTION_TOOLS:
        if tool["name"] == name:
            return tool
    raise AssertionError(f"tool {name!r} not in _ACTION_TOOLS")


@pytest.mark.parametrize(
    ("tool_name", "must_require"),
    [
        ("click", {"x", "y", "intent"}),
        ("right_click", {"x", "y", "intent"}),
        ("press", {"key", "intent"}),
        ("drag", {"start_x", "start_y", "end_x", "end_y", "intent"}),
        ("wait", {"ms", "intent"}),
        ("scroll", {"clicks", "intent"}),
        ("detect", {"intent"}),
        # build takes no coordinates: the executor places on open ground after
        # the camera settles (F-33); send composites require target_class for
        # the same reason — pre-jump x/y land on the wrong terrain.
        ("build", {"building_key", "intent"}),
        ("send_villager", {"target_class", "intent"}),
        ("send_all_idle", {"target_class", "intent"}),
        ("queue_villager", {"intent"}),
    ],
)
def test_tool_required_fields(tool_name: str, must_require: set):
    tool = _by_name(tool_name)
    assert set(tool["input_schema"]["required"]) == must_require


def test_press_supports_modifiers_and_rescan():
    """Press is the most overloaded tool; its optional fields must remain reachable."""
    tool = _by_name("press")
    properties = tool["input_schema"]["properties"]
    assert "modifiers" in properties
    assert properties["modifiers"]["type"] == "array"
    assert "rescan" in properties
    assert properties["rescan"]["type"] == "boolean"


def test_send_villager_has_target_class_for_resource_targeting():
    """send_villager's value comes from being able to right-click a resource by class."""
    tool = _by_name("send_villager")
    assert "target_class" in tool["input_schema"]["properties"]
