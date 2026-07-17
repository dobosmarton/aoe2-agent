"""Unit tests for gameplay_agent/providers/claude.py — the executor provider.

The Anthropic client is replaced with a recorder so tests run offline: no
network, no API key. Async methods are driven via `asyncio.run` (the repo does
not use pytest-asyncio). Prompt loading is stubbed by pre-setting `_core_prompt`,
which short-circuits `_load_prompts()` so no disk/memory I/O happens.

Sections:
  S1 — executor `effort` forwarded to the API call.
  S3 — cache_control breakpoints on the system age block and the tool loop.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest
from gameplay_agent.providers.claude import ClaudeProvider

if TYPE_CHECKING:
    from collections.abc import Awaitable


def _run(coro: Awaitable[object]) -> object:
    """Drive a coroutine to completion in a fresh event loop."""
    return asyncio.run(coro)


def _usage(cache_read: int = 0) -> SimpleNamespace:
    """A minimal Anthropic `usage` stand-in with the fields _call_api reads."""
    return SimpleNamespace(
        input_tokens=1,
        output_tokens=1,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=0,
    )


def _end_turn_response(cache_read: int = 0) -> SimpleNamespace:
    """A response that ends the tool loop immediately (no tool_use)."""
    return SimpleNamespace(content=[], stop_reason="end_turn", usage=_usage(cache_read))


def _install_create(provider: ClaudeProvider, *responses: object) -> AsyncMock:
    """Point the client's messages.create at an AsyncMock yielding `responses`."""
    create = AsyncMock(side_effect=list(responses))
    provider.client = SimpleNamespace(messages=SimpleNamespace(create=create))
    return create


@pytest.fixture
def provider() -> ClaudeProvider:
    """A ClaudeProvider with stubbed prompts and no game-knowledge DB."""
    p = ClaudeProvider(api_key="test", use_dynamic_context=False)
    # Non-None _core_prompt short-circuits _load_prompts() → no I/O.
    p._core_prompt = "SYSTEM"
    p._age_prompts = {"dark": "DARK", "feudal": "FEUDAL"}
    return p


# ---------------------------------------------------------------------------
# S1 — effort forwarded to the executor API call
# ---------------------------------------------------------------------------


def test_call_api_forwards_default_effort(provider: ClaudeProvider) -> None:
    create = _install_create(provider, _end_turn_response())
    _run(provider._call_api([{"type": "text", "text": "hi"}]))
    assert create.call_args.kwargs["output_config"] == {"effort": "low"}


def test_call_api_forwards_configured_effort(
    provider: ClaudeProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gameplay_agent.providers import claude

    monkeypatch.setattr(claude.config, "executor_effort", "medium")
    create = _install_create(provider, _end_turn_response())
    _run(provider._call_api([{"type": "text", "text": "hi"}]))
    assert create.call_args.kwargs["output_config"] == {"effort": "medium"}


# ---------------------------------------------------------------------------
# Single-shot zero-action retry (T-506)
# ---------------------------------------------------------------------------


def _parse_response(actions: list[dict], reasoning: str = "r") -> SimpleNamespace:
    from gameplay_agent.models import LLMResponse

    return SimpleNamespace(
        usage=_usage(), parsed_output=LLMResponse(actions=actions, reasoning=reasoning)
    )


def _install_parse(provider: ClaudeProvider, *responses: object) -> AsyncMock:
    parse = AsyncMock(side_effect=list(responses))
    provider.client = SimpleNamespace(messages=SimpleNamespace(parse=parse))
    return parse


def test_single_shot_retries_once_on_zero_actions(provider: ClaudeProvider) -> None:
    parse = _install_parse(
        provider,
        _parse_response([], reasoning="plan narrated, nothing emitted"),
        _parse_response([{"type": "press", "key": "h"}]),
    )
    out = _run(provider._call_single_shot([{"type": "text", "text": "ctx"}]))
    assert parse.await_count == 2
    assert out["actions"] and out["actions"][0]["key"] == "h"
    retry_messages = parse.await_args.kwargs["messages"]
    assert "zero actions" in retry_messages[-1]["content"]  # the nudge was sent


def test_single_shot_no_retry_when_actions_present(provider: ClaudeProvider) -> None:
    parse = _install_parse(provider, _parse_response([{"type": "press", "key": "q"}]))
    _run(provider._call_single_shot([{"type": "text", "text": "ctx"}]))
    assert parse.await_count == 1


def test_single_shot_gives_up_after_one_retry(provider: ClaudeProvider) -> None:
    parse = _install_parse(provider, _parse_response([]), _parse_response([]))
    out = _run(provider._call_single_shot([{"type": "text", "text": "ctx"}]))
    assert parse.await_count == 2  # bounded: no infinite nudging
    assert out["actions"] == []  # game loop's fallback handles it from here


# ---------------------------------------------------------------------------
# Single-shot 400 → tool-loop fallback (T-533; de-risks the F-40 grammar fix)
# ---------------------------------------------------------------------------


def _bad_request(message: str) -> object:
    """An anthropic.BadRequestError with a minimal (non-network) 400 response."""
    import anthropic
    import httpx

    response = httpx.Response(400, request=httpx.Request("POST", "http://test"))
    return anthropic.BadRequestError(message, response=response, body=None)


def test_single_shot_falls_back_to_tool_loop_on_400(provider: ClaudeProvider) -> None:
    """A 400 on the structured-output path (e.g. "compiled grammar is too
    large") must retry THIS turn via the tool loop, not burn it (run 12, F-40)."""
    parse = AsyncMock(side_effect=_bad_request("compiled grammar is too large"))
    create = AsyncMock(side_effect=[_end_turn_response()])
    provider.client = SimpleNamespace(messages=SimpleNamespace(parse=parse, create=create))

    out = _run(provider._single_shot_or_tool_loop([{"type": "text", "text": "ctx"}], "Dark Age"))

    assert parse.await_count == 1  # single-shot attempted
    assert create.await_count == 1  # then fell through to the tool loop
    assert out["actions_already_executed"] is True  # tool-loop serialization, not a wait no-op


def test_single_shot_non_400_error_propagates(provider: ClaudeProvider) -> None:
    """Only 400s fall back — other API errors surface to get_actions' handler."""
    import anthropic

    parse = AsyncMock(side_effect=anthropic.APITimeoutError(request=None))  # type: ignore[arg-type]
    create = AsyncMock(side_effect=[_end_turn_response()])
    provider.client = SimpleNamespace(messages=SimpleNamespace(parse=parse, create=create))

    with pytest.raises(anthropic.APIError):
        _run(provider._single_shot_or_tool_loop([{"type": "text", "text": "ctx"}], "Dark Age"))
    assert create.await_count == 0  # no tool-loop fallback for non-400s


# ---------------------------------------------------------------------------
# S3 — prompt caching across the tool loop
# ---------------------------------------------------------------------------


def test_age_block_is_cached(provider: ClaudeProvider) -> None:
    blocks = provider.get_system_prompt("Feudal Age")
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}  # core block
    assert blocks[1]["text"] == "FEUDAL"
    assert blocks[1]["cache_control"] == {"type": "ephemeral"}  # age block


def test_moving_breakpoint_marks_last_and_strips_previous() -> None:
    messages: list[dict] = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "a", "cache_control": {"type": "ephemeral"}}],
        },
        {"role": "user", "content": [{"type": "text", "text": "b"}]},
    ]
    ClaudeProvider._apply_moving_cache_breakpoint(messages)
    assert "cache_control" not in messages[0]["content"][0]
    assert messages[1]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_moving_breakpoint_ignores_non_dict_blocks() -> None:
    # Assistant turns hold SDK content objects, not dicts — must not crash.
    messages: list[dict] = [
        {"role": "assistant", "content": [SimpleNamespace(type="tool_use")]},
        {"role": "user", "content": [{"type": "text", "text": "b"}]},
    ]
    ClaudeProvider._apply_moving_cache_breakpoint(messages)
    assert messages[1]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_create_call_caches_last_user_block(provider: ClaudeProvider) -> None:
    create = _install_create(provider, _end_turn_response())
    _run(provider._call_api([{"type": "text", "text": "hi"}]))
    sent = create.call_args.kwargs["messages"]
    assert sent[-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}


def test_cache_read_tokens_accumulate(
    provider: ClaudeProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Round 1 asks for a tool, round 2 ends the turn; cache reads sum across both.
    tool_block = SimpleNamespace(type="tool_use", id="t1", name="press", input={})
    first = SimpleNamespace(content=[tool_block], stop_reason="tool_use", usage=_usage(40))
    _install_create(provider, first, _end_turn_response(60))

    async def _fake_exec(_block: object) -> tuple[dict, dict]:
        return ({"type": "press"}, {"tool_use_id": "t1", "content": '{"success": true}'})

    monkeypatch.setattr(provider, "_execute_tool_call", _fake_exec)
    _run(provider._call_api([{"type": "text", "text": "hi"}]))
    assert provider._total_cache_read_tokens == 100


# ---------------------------------------------------------------------------
# S4 — two-mode executor routing + single-shot serialization
# ---------------------------------------------------------------------------


def test_routine_turn_routes_to_single_shot(
    provider: ClaudeProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    single = AsyncMock(return_value={"actions": [], "actions_already_executed": False})
    tool = AsyncMock()
    monkeypatch.setattr(provider, "_call_single_shot", single)
    monkeypatch.setattr(provider, "_call_api", tool)

    out = _run(provider.get_actions("Food=200, Wood=150. TC Idle: True"))

    single.assert_awaited_once()
    tool.assert_not_awaited()
    assert out["actions_already_executed"] is False


def test_under_attack_routes_to_tool_loop(
    provider: ClaudeProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    single = AsyncMock()
    tool = AsyncMock(return_value=SimpleNamespace(reasoning="ok"))
    monkeypatch.setattr(provider, "_call_single_shot", single)
    monkeypatch.setattr(provider, "_call_api", tool)
    monkeypatch.setattr(provider, "_serialize_response", lambda _r: {"path": "tool_loop"})

    out = _run(provider.get_actions("Under Attack: True"))

    tool.assert_awaited_once()
    single.assert_not_awaited()
    assert out == {"path": "tool_loop"}


def test_serialize_single_shot_is_not_pre_executed() -> None:
    from gameplay_agent.models import LLMResponse

    resp = LLMResponse(actions=[{"type": "press", "key": "h"}])
    out = ClaudeProvider._serialize_single_shot(resp)
    assert out["actions_already_executed"] is False
    assert len(out["actions"]) == 1


def test_serialize_response_is_pre_executed() -> None:
    from gameplay_agent.models import LLMResponse

    resp = LLMResponse(actions=[{"type": "press", "key": "h"}])
    out = ClaudeProvider._serialize_response(resp)
    assert out["actions_already_executed"] is True


# ---------------------------------------------------------------------------
# reassign_villager composite: jump-to-camp → pick worker → build → place
# ---------------------------------------------------------------------------


def _allow_farm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Satisfy the farm build gate: a mill has been seen this game."""
    from gameplay_agent import executor as ex

    monkeypatch.setattr(ex._build_gates, "buildings_confirmed", {"mill"})


def test_reassign_villager_sequences_camp_select_build(
    provider: ClaudeProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gameplay_agent.providers import claude as claude_mod

    steps: list[dict] = []

    async def _record(action: dict) -> object:
        steps.append(action)
        return SimpleNamespace(success=True, detail="ok")

    # After the Ctrl-Z rescan the fresh view has a wood villager on a tree.
    entities = [
        {"class": "tree", "id": "tree_0", "center": (300, 300), "confidence": 0.9},
        {"class": "villager", "id": "villager_0", "center": (310, 305), "confidence": 0.9},
    ]
    monkeypatch.setattr(claude_mod, "execute_action", _record)
    monkeypatch.setattr(claude_mod, "get_detected_entities", lambda: entities)
    monkeypatch.setattr(claude_mod, "_tracker_velocities", lambda: {})
    monkeypatch.setattr(provider, "_entity_snapshot", lambda: [])
    _allow_farm(monkeypatch)  # a mill has been seen → the farm gate passes

    block = SimpleNamespace(
        id="tu1",
        name="reassign_villager",
        input={"from_job": "wood", "building_key": "a", "intent": "need food"},
    )
    action_dict, _result = _run(provider._execute_reassign_villager(block))

    assert action_dict["type"] == "reassign_villager"
    kinds = [(s["type"], s.get("key"), s.get("modifiers")) for s in steps]
    # 1) Ctrl-Z to the lumber camp (rescan), 2) click the wood villager,
    # 3) q (econ menu), 4) a (Farm), 5) place with building_key.
    assert kinds[0] == ("press", "z", ["ctrl"]) and steps[0]["rescan"] is True
    assert steps[1]["type"] == "click" and (steps[1]["x"], steps[1]["y"]) == (310, 305)
    assert kinds[2] == ("press", "q", None)
    assert kinds[3] == ("press", "a", None)
    assert steps[4]["type"] == "click" and steps[4]["building_key"] == "a"
    assert steps[4]["auto_placement"] is True  # placement resolved at click time (F-33)


def test_reassign_villager_rejected_when_farm_gate_fails(
    provider: ClaudeProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No mill seen → the whole reassign composite is rejected before the camera
    jump, and the reason reaches the LLM as the tool result."""
    from gameplay_agent import executor as ex
    from gameplay_agent.providers import claude as claude_mod

    steps: list[dict] = []

    async def _record(action: dict) -> object:
        steps.append(action)
        return SimpleNamespace(success=True, detail="ok")

    monkeypatch.setattr(claude_mod, "execute_action", _record)
    monkeypatch.setattr(provider, "_entity_snapshot", lambda: [])
    monkeypatch.setattr(ex._build_gates, "buildings_confirmed", set())

    block = SimpleNamespace(
        id="tu8",
        name="reassign_villager",
        input={"from_job": "wood", "building_key": "a", "intent": "need food"},
    )
    action_dict, result = _run(provider._execute_reassign_villager(block))

    assert action_dict["type"] == "reassign_villager"
    assert steps == []  # rejected before the camera jump
    assert "mill" in str(result)


def test_reassign_villager_falls_back_to_villager_class(
    provider: ClaudeProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gameplay_agent.providers import claude as claude_mod

    steps: list[dict] = []

    async def _record(action: dict) -> object:
        steps.append(action)
        return SimpleNamespace(success=True, detail="ok")

    monkeypatch.setattr(claude_mod, "execute_action", _record)
    monkeypatch.setattr(claude_mod, "get_detected_entities", lambda: [])  # no worker found
    monkeypatch.setattr(claude_mod, "_tracker_velocities", lambda: {})
    monkeypatch.setattr(provider, "_entity_snapshot", lambda: [])
    _allow_farm(monkeypatch)  # a mill has been seen → the farm gate passes

    block = SimpleNamespace(
        id="tu2",
        name="reassign_villager",
        input={"from_job": "gold", "building_key": "a", "intent": "farm"},
    )
    _run(provider._execute_reassign_villager(block))

    assert steps[0]["key"] == "g" and steps[0]["modifiers"] == ["ctrl"]  # Ctrl-G mining camp
    # Selection falls back to nearest villager by class when the job model finds none.
    assert steps[1]["type"] == "click" and steps[1]["target_class"] == "villager"


# ---------------------------------------------------------------------------
# Composite step-list characterization: the exact step dicts are the contract
# (guards the shared-helper refactor against silent behavior drift)
# ---------------------------------------------------------------------------


@pytest.fixture
def recorded_steps(provider: ClaudeProvider, monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Record every step a composite executes; entity snapshot stubbed empty."""
    from gameplay_agent.providers import claude as claude_mod

    steps: list[dict] = []

    async def _record(action: dict) -> object:
        steps.append(action)
        return SimpleNamespace(success=True, detail="ok")

    monkeypatch.setattr(claude_mod, "execute_action", _record)
    monkeypatch.setattr(provider, "_entity_snapshot", lambda: [])
    return steps


def test_send_villager_step_list_verbatim(
    provider: ClaudeProvider, recorded_steps: list[dict]
) -> None:
    block = SimpleNamespace(
        id="tu3", name="send_villager", input={"target_class": "tree", "intent": "chop"}
    )
    action_dict, _result = _run(provider._execute_send_villager(block))
    assert action_dict == {"type": "send_villager", "target_class": "tree", "intent": "chop"}
    assert recorded_steps == [
        {"type": "press", "key": ".", "rescan": True, "intent": "Select idle villager (chop)"},
        {"type": "right_click", "intent": "chop", "target_class": "tree"},
    ]


def test_send_all_idle_step_list_verbatim(
    provider: ClaudeProvider, recorded_steps: list[dict]
) -> None:
    block = SimpleNamespace(
        id="tu4", name="send_all_idle", input={"target_class": "tree", "intent": "regroup"}
    )
    action_dict, _result = _run(provider._execute_send_all_idle(block))
    assert action_dict == {"type": "send_all_idle", "target_class": "tree", "intent": "regroup"}
    assert recorded_steps == [
        {
            "type": "press",
            "key": ".",
            "modifiers": ["shift"],
            "rescan": True,
            "intent": "Select ALL idle villagers (regroup)",
        },
        {"type": "right_click", "intent": "regroup", "target_class": "tree"},
    ]


@pytest.mark.parametrize("composite", ["send_villager", "send_all_idle"])
def test_send_composites_refuse_raw_coordinates(
    provider: ClaudeProvider, recorded_steps: list[dict], composite: str
) -> None:
    """F-33: the '.' select re-centers the camera, so literal x/y computed from
    the pre-jump frame land on arbitrary terrain — the composite fails with a
    teaching detail instead of spending keystrokes."""
    block = SimpleNamespace(id="tu5", name=composite, input={"x": 100, "y": 200, "intent": "go"})
    handler = getattr(provider, f"_execute_{composite}")
    action_dict, result = _run(handler(block))
    assert action_dict["type"] == composite
    assert recorded_steps == []  # refused before any keystroke
    assert "target_class" in str(result)


def test_build_house_rejected_by_headroom_gate(
    provider: ClaudeProvider, recorded_steps: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    from gameplay_agent import executor as ex

    monkeypatch.setattr(ex._build_gates, "population", (10, 30))  # 20 headroom
    block = SimpleNamespace(id="tu6", name="build", input={"building_key": "q", "intent": "house"})
    action_dict, result = _run(provider._execute_build(block))
    assert action_dict == {"type": "build", "building_key": "q", "intent": "house"}
    assert recorded_steps == []  # gate fired before any step executed
    assert "headroom" in str(result)  # the reason reaches the LLM as the tool result


def test_build_house_allowed_near_cap(
    provider: ClaudeProvider, recorded_steps: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    from gameplay_agent import executor as ex

    monkeypatch.setattr(ex._build_gates, "population", (28, 30))  # housed-adjacent
    block = SimpleNamespace(id="tu7", name="build", input={"building_key": "q", "intent": "house"})
    _run(provider._execute_build(block))
    assert recorded_steps  # steps ran — the gate let it through


def test_queue_villager_step_list_verbatim(
    provider: ClaudeProvider, recorded_steps: list[dict]
) -> None:
    block = SimpleNamespace(id="tu5", name="queue_villager", input={"intent": "more vils"})
    action_dict, _result = _run(provider._execute_queue_villager(block))
    assert action_dict == {"type": "queue_villager", "intent": "more vils"}
    # One first-class action, not raw h+q presses — the executor's order
    # ledger gate must see (and count) every LLM-initiated queue (F-38).
    assert recorded_steps == [{"type": "queue_villager", "intent": "more vils"}]
