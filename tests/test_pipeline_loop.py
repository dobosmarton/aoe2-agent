"""Tests for S6 turn pipelining (RTC-style) in gameplay_agent/game_loop.py.

Pure helpers are tested directly; two integration tests drive the real
`game_loop` with mocked window/screenshot/detection seams (no LLM, no pyautogui,
no API key) to assert the one-turn-lag behavior and the disabled==baseline path.
"""

from __future__ import annotations

import asyncio

import gameplay_agent.game_loop as gl
from gameplay_agent.executor import clear_detected_entities, set_detected_entities
from gameplay_agent.providers.base import LLMResult
from gameplay_agent.providers.claude import ClaudeProvider


def _run(coro):
    return asyncio.run(coro)


def _press(key):
    return {"type": "press", "key": key}


# ---------------------------------------------------------------------------
# _commit_head / _revalidate_against_fresh
# ---------------------------------------------------------------------------


def test_commit_head_keeps_only_leading_actions(monkeypatch):
    monkeypatch.setattr(gl.config, "pipeline_commit_max", 2)
    actions = [_press("a"), _press("b"), _press("c"), _press("d")]
    assert gl._commit_head(actions) == [_press("a"), _press("b")]


def test_commit_head_handles_short_lists(monkeypatch):
    monkeypatch.setattr(gl.config, "pipeline_commit_max", 2)
    assert gl._commit_head([_press("a")]) == [_press("a")]
    assert gl._commit_head([]) == []


def test_revalidate_drops_unresolved_targets():
    set_detected_entities([{"id": "tc1", "class": "town_center", "center": (10, 10)}])
    try:
        actions = [
            _press("q"),  # no target → always kept
            {"type": "right_click", "target_class": "town_center"},  # resolves → kept
            {"type": "right_click", "target_class": "dragon"},  # gone → dropped
            {"type": "click", "target_id": "ghost"},  # gone → dropped
        ]
        kept = gl._revalidate_against_fresh(actions)
        assert kept == [_press("q"), {"type": "right_click", "target_class": "town_center"}]
    finally:
        clear_detected_entities()


# ---------------------------------------------------------------------------
# _should_pipeline
# ---------------------------------------------------------------------------


def test_should_pipeline_true_for_routine_single_shot():
    provider = ClaudeProvider(api_key="test", use_dynamic_context=False)
    assert gl._should_pipeline("economy turn, gather food", provider) is True


def test_should_pipeline_false_for_combat():
    provider = ClaudeProvider(api_key="test", use_dynamic_context=False)
    assert gl._should_pipeline("enemy spotted — under attack: true", provider) is False


def test_should_pipeline_false_for_non_claude_provider():
    class _Other:
        pass

    assert gl._should_pipeline("routine", _Other()) is False


# ---------------------------------------------------------------------------
# _execute_or_record
# ---------------------------------------------------------------------------


def test_execute_or_record_executes_when_not_preexecuted(monkeypatch):
    executed = []

    async def _rec(actions, iteration, memory, reasoning):
        executed.append(actions)

    monkeypatch.setattr(gl, "_execute_turn_actions", _rec)
    memory = gl.AgentMemory()
    _run(gl._execute_or_record(LLMResult(reasoning="r"), [_press("q")], memory, 1))
    assert executed == [[_press("q")]]


def test_execute_or_record_records_when_preexecuted(monkeypatch):
    memory = gl.AgentMemory()
    calls = []
    monkeypatch.setattr(memory, "record_action_results", lambda s, t: calls.append((s, t)))
    payload = LLMResult(actions_already_executed=True, success_count=2)
    _run(gl._execute_or_record(payload, [{}, {}, {}], memory, 1))
    assert calls == [(2, 3)]


# ---------------------------------------------------------------------------
# _drain_pending
# ---------------------------------------------------------------------------


def test_drain_pending_returns_game_end_without_executing(monkeypatch, tmp_path):
    executed = []

    async def _rec(actions, iteration, memory, reasoning):
        executed.append(actions)

    monkeypatch.setattr(gl, "_execute_turn_actions", _rec)
    monkeypatch.setattr(gl, "_process_response", lambda resp, *a, **k: ([], "defeat"))

    async def _scenario():
        async def _plan():
            return LLMResult(actions=[_press("a")], reasoning="r")

        plan = gl._PendingPlan(task=asyncio.ensure_future(_plan()), iteration=1)
        memory = gl.AgentMemory()
        logger = gl.GoalLogger(tmp_path)
        return await gl._drain_pending(plan, memory, gl.GoalManager(), 2, logger, None)

    assert _run(_scenario()) == "defeat"
    assert executed == []  # game over → no actions executed


# ---------------------------------------------------------------------------
# Integration: drive the real game_loop with mocked seams
# ---------------------------------------------------------------------------


class _FakePipelineProvider(ClaudeProvider):
    """Records each context it is asked for and returns 5 tagged press actions
    per turn so tests can trace which turn's plan executed when."""

    def __init__(self):
        super().__init__(api_key="test", use_dynamic_context=False)
        self.contexts = []
        self._turn = 0

    async def get_actions(self, context, width=1920, height=1080):
        self._turn += 1
        self.contexts.append(context)
        actions = [_press(f"t{self._turn}a{i}") for i in range(5)]
        return LLMResult(actions=actions, reasoning=f"turn{self._turn}")


def _patch_loop_seams(monkeypatch, tmp_path, executed, context_text):
    async def _capture(*_a, **_k):
        return (b"x", 100, 100)

    async def _noop_execute(*_a, **_k):
        return []

    async def _record_exec(actions, iteration, memory, reasoning):
        executed.append([a.get("key") for a in actions])

    monkeypatch.setattr(gl, "is_game_running", lambda: True)
    monkeypatch.setattr(gl, "ensure_game_focused", lambda: True)
    monkeypatch.setattr(gl, "_init_frame_differ", lambda: None)
    monkeypatch.setattr(gl, "_capture_screenshot", _capture)
    monkeypatch.setattr(gl, "_classify_entities", lambda *_a, **_k: ("", {}))
    monkeypatch.setattr(gl, "_maybe_launch_strategist", lambda *_a, **_k: None)
    monkeypatch.setattr(gl, "_build_llm_context", lambda *_a, **_k: context_text)
    monkeypatch.setattr(
        gl, "_process_response", lambda resp, *a, **k: (list(resp.get("actions", [])), None)
    )
    monkeypatch.setattr(gl, "execute_actions", _noop_execute)
    monkeypatch.setattr(gl, "_execute_turn_actions", _record_exec)
    monkeypatch.setattr(gl.config, "log_dir", tmp_path)
    monkeypatch.setattr(gl.config, "save_screenshots", False)


def test_pipeline_executes_previous_plan_head_next_turn(monkeypatch, tmp_path):
    executed = []
    _patch_loop_seams(monkeypatch, tmp_path, executed, "routine economy turn")
    monkeypatch.setattr(gl.config, "pipeline_commit_max", 2)

    provider = _FakePipelineProvider()
    _run(gl.game_loop(provider, max_iterations=2, use_detection=False, use_overlay=False))

    # Turn 1 launches plan-1 (nothing executed yet). Turn 2 drains plan-1's
    # revalidated head (2 of 5) and launches plan-2, which is left pending and
    # cancelled in finally. So only plan-1's 2-action head executes.
    assert executed == [["t1a0", "t1a1"]]
    assert len(provider.contexts) == 2  # both turns pre-launched a plan
