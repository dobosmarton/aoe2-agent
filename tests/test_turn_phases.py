"""Unit tests for `gameplay_agent.turn_phases`.

These cover the pure (or near-pure) helpers that drive a single iteration:
applied-memory parsing, hardcoded ground/maintenance commands, LLM context
assembly. The async/state-mutating pieces (`_process_response`,
`_execute_turn_actions`) are exercised indirectly via the evaluation runner;
direct tests for them would need extensive AgentMemory + GoalManager mocking
without much marginal coverage gain.
"""

from __future__ import annotations

import pytest
from gameplay_agent.memory import AgentMemory
from gameplay_agent.turn_phases import (
    INITIAL_ZOOM_CLICKS,
    _build_llm_context,
    _extract_applied_memories,
    _get_ground_commands,
)

# ---------------------------------------------------------------------------
# _extract_applied_memories
# ---------------------------------------------------------------------------


def test_extract_applied_memories_no_tag_returns_unchanged():
    reasoning = "I'm queuing a villager because food is low."
    known, unknown, cleaned = _extract_applied_memories(reasoning, loaded_titles=set())
    assert known == []
    assert unknown == []
    assert cleaned == reasoning


def test_extract_applied_memories_empty_reasoning_returns_empty():
    known, unknown, cleaned = _extract_applied_memories("", loaded_titles={"x"})
    assert known == []
    assert unknown == []
    assert cleaned == ""


def test_extract_applied_memories_known_title_split():
    loaded = {"build_house_at_pop_4", "scout_with_g"}
    reasoning = "[applied: build_house_at_pop_4] Building a house now."
    known, unknown, cleaned = _extract_applied_memories(reasoning, loaded_titles=loaded)
    assert known == ["build_house_at_pop_4"]
    assert unknown == []
    assert cleaned == "Building a house now."


def test_extract_applied_memories_strips_leading_whitespace():
    """The tag may have leading whitespace; cleaned reasoning starts at the first non-ws char."""
    reasoning = "   [applied: rule1]   Then I do something."
    known, _unknown, cleaned = _extract_applied_memories(reasoning, loaded_titles={"rule1"})
    assert known == ["rule1"]
    assert cleaned == "Then I do something."


def test_extract_applied_memories_separates_known_and_unknown():
    """Titles the LLM hallucinated are returned in `unknown` for warning logs."""
    loaded = {"real_rule"}
    reasoning = "[applied: real_rule, fake_rule] Doing things."
    known, unknown, cleaned = _extract_applied_memories(reasoning, loaded_titles=loaded)
    assert known == ["real_rule"]
    assert unknown == ["fake_rule"]
    assert cleaned == "Doing things."


def test_extract_applied_memories_handles_extra_spaces():
    loaded = {"a", "b"}
    reasoning = "[applied:  a  ,  b ] rest"
    known, _, _ = _extract_applied_memories(reasoning, loaded_titles=loaded)
    assert known == ["a", "b"]


def test_extract_applied_memories_case_insensitive_prefix():
    """The `[applied:` regex is IGNORECASE so capitalization variants still parse."""
    reasoning = "[Applied: rule1] doing thing"
    known, _, cleaned = _extract_applied_memories(reasoning, loaded_titles={"rule1"})
    assert known == ["rule1"]
    assert cleaned == "doing thing"


def test_extract_applied_memories_only_anchored_at_start():
    """The tag must be at the start; `[applied:` mid-string is left alone."""
    reasoning = "Some text. [applied: rule1] more text."
    known, _, cleaned = _extract_applied_memories(reasoning, loaded_titles={"rule1"})
    assert known == []
    assert cleaned == reasoning


def test_extract_applied_memories_drops_empty_tokens():
    """Trailing commas / double commas don't produce empty titles."""
    loaded = {"rule1"}
    reasoning = "[applied: rule1,, ] body"
    known, _, _ = _extract_applied_memories(reasoning, loaded_titles=loaded)
    assert known == ["rule1"]


# ---------------------------------------------------------------------------
# _get_ground_commands
# ---------------------------------------------------------------------------


def test_ground_commands_only_run_on_first_iteration():
    """The zoom-in + auto-scout sequence belongs at game start, not every turn."""
    cmds = _get_ground_commands(iteration=1)
    assert len(cmds) > 0


@pytest.mark.parametrize("iteration", [2, 5, 100])
def test_ground_commands_empty_after_first_iteration(iteration: int):
    assert _get_ground_commands(iteration) == []


def test_ground_commands_first_iter_zooms_in():
    cmds = _get_ground_commands(iteration=1)
    types = [c["type"] for c in cmds]
    assert "scroll" in types


def test_ground_commands_first_iter_uses_initial_zoom_constant():
    """Constant change should propagate into the emitted scroll action."""
    cmds = _get_ground_commands(iteration=1)
    scroll = next(c for c in cmds if c["type"] == "scroll")
    assert scroll["clicks"] == INITIAL_ZOOM_CLICKS


def test_ground_commands_first_iter_includes_auto_scout():
    """Auto-scout (g hotkey) is the second purpose of the ground sequence."""
    cmds = _get_ground_commands(iteration=1)
    keys = [c.get("key") for c in cmds if c["type"] == "press"]
    assert "g" in keys


# ---------------------------------------------------------------------------
# _build_llm_context
# ---------------------------------------------------------------------------


class _FakeGoalManager:
    """Minimal stub matching the slice of GoalManager that _build_llm_context uses."""

    def __init__(self, goal_text: str = "", resource_text: str = "") -> None:
        self._goal_text = goal_text
        self._resource_text = resource_text

    def get_resource_context(self) -> str:
        return self._resource_text

    def get_context_for_llm(self) -> str:
        return self._goal_text


def test_build_llm_context_includes_memory_context_alone():
    memory = AgentMemory()
    gm = _FakeGoalManager()
    context = _build_llm_context(memory, gm, entity_summary="")
    assert isinstance(context, str)


def test_build_llm_context_prepends_resource_block():
    memory = AgentMemory()
    gm = _FakeGoalManager(resource_text="## Resource Status\n- Food: 200")
    context = _build_llm_context(memory, gm, entity_summary="")
    assert context.startswith("## Resource Status")


def test_build_llm_context_prepends_goals_above_resources():
    """Goals come before resources so the LLM reads its objectives first."""
    memory = AgentMemory()
    gm = _FakeGoalManager(goal_text="## Goals\n- gather food", resource_text="## Res\n- 200 food")
    context = _build_llm_context(memory, gm, entity_summary="")
    goal_pos = context.find("## Goals")
    res_pos = context.find("## Res")
    assert 0 <= goal_pos < res_pos


def test_build_llm_context_includes_entity_summary_block():
    """When entities are present, the YOLO entity block leads the context."""
    memory = AgentMemory()
    gm = _FakeGoalManager()
    summary = "  sheep_0: sheep at (100,100) [90%]"
    context = _build_llm_context(memory, gm, entity_summary=summary)
    assert "Detected Entities (from YOLO)" in context
    assert "sheep_0: sheep" in context


def test_build_llm_context_omits_entity_block_when_empty():
    memory = AgentMemory()
    gm = _FakeGoalManager()
    context = _build_llm_context(memory, gm, entity_summary="")
    assert "Detected Entities" not in context


def test_build_llm_context_omits_goal_block_when_empty():
    memory = AgentMemory()
    gm = _FakeGoalManager(goal_text="", resource_text="## Res\n- food")
    context = _build_llm_context(memory, gm, entity_summary="")
    # Goal text is empty so it shouldn't be glued in (which would add a stray separator)
    # The resource section is still present.
    assert "## Res" in context
