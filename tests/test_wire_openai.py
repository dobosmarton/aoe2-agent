"""Unit tests for gameplay_agent/providers/wire_openai.py.

The OpenAI SDK client is replaced with a recorder so tests run offline: no
network, no API key. Async methods are driven via `asyncio.run` (the repo does
not use pytest-asyncio).
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest
from gameplay_agent.models import Observations
from gameplay_agent.providers.base import (
    AssistantTurn,
    ChatRequest,
    SystemBlock,
    TokenUsage,
    ToolCall,
    ToolOutcome,
    ToolResultsTurn,
    UserTurn,
)
from gameplay_agent.providers.executor_provider import ExecutorProvider
from gameplay_agent.providers.wire_openai import OpenAIWire

if TYPE_CHECKING:
    from collections.abc import Awaitable


def _run(coro: Awaitable[object]) -> object:
    return asyncio.run(coro)


@pytest.fixture
def wire() -> OpenAIWire:
    return OpenAIWire(model="gpt-5.6-luna", api_key="test", base_url="http://test/v1")


def _usage(prompt: int = 100, completion: int = 10, cached: int | None = 0) -> SimpleNamespace:
    """A minimal `CompletionUsage` stand-in; `details` is always present."""
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
    )


def _completion(
    *,
    content: str = "",
    tool_calls: list[SimpleNamespace] | None = None,
    finish_reason: str = "stop",
    usage: SimpleNamespace | None = None,
) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=usage or _usage())


def _sdk_tool_call(call_id: str, name: str, arguments: str) -> SimpleNamespace:
    """The SDK's tool-call shape: arguments arrive as a JSON *string*."""
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


# ---------------------------------------------------------------------------
# S1 — rendering
# ---------------------------------------------------------------------------


def test_system_blocks_collapse_into_one_ordered_message(wire: OpenAIWire) -> None:
    """Prefix caching keys on order, so the stable core must stay first."""
    rendered = wire._render_system((SystemBlock("CORE", True), SystemBlock("AGE", True)))
    assert len(rendered) == 1
    assert rendered[0]["role"] == "system"
    assert rendered[0]["content"] == "CORE\n\nAGE"


def test_each_tool_outcome_gets_its_own_message(wire: OpenAIWire) -> None:
    """Anthropic batches results into one user turn; this API pairs by id."""
    messages = wire._render_turns(
        (
            ToolResultsTurn(
                outcomes=(
                    ToolOutcome("call_a", True, "ok"),
                    ToolOutcome("call_b", False, "nope"),
                )
            ),
        )
    )
    assert [m["tool_call_id"] for m in messages] == ["call_a", "call_b"]
    assert all(m["role"] == "tool" for m in messages)


def test_assistant_tool_calls_serialize_arguments_as_json_string(wire: OpenAIWire) -> None:
    messages = wire._render_turns(
        (AssistantTurn(text="doing it", tool_calls=(ToolCall("c1", "press", {"key": "h"}),)),)
    )
    call = messages[0]["tool_calls"][0]
    assert call["id"] == "c1"
    assert call["function"]["name"] == "press"
    assert json.loads(call["function"]["arguments"]) == {"key": "h"}


def test_assistant_without_text_sends_null_content(wire: OpenAIWire) -> None:
    """A tool-only assistant turn carries no prose; the field must be null."""
    messages = wire._render_turns((AssistantTurn(tool_calls=(ToolCall("c1", "press", {}),)),))
    assert messages[0]["content"] is None


# ---------------------------------------------------------------------------
# S2 — usage normalisation
# ---------------------------------------------------------------------------


def test_cached_tokens_are_subtracted_from_prompt_tokens(wire: OpenAIWire) -> None:
    """`prompt_tokens` includes cached ones here but not on Anthropic.

    Both wires must mean the same thing or the shared pricing table double-counts
    the cached prefix at full input price.
    """
    usage = wire._usage_of(_completion(usage=_usage(prompt=1000, completion=50, cached=800)))
    # cache_write stays 0: automatic prefix caching has no write charge.
    assert usage == TokenUsage(input_tokens=200, output_tokens=50, cache_read_tokens=800)


def test_usage_survives_a_null_cached_count(wire: OpenAIWire) -> None:
    """`cached_tokens` is `int | None` — a null must read as zero, not crash."""
    usage = wire._usage_of(_completion(usage=_usage(prompt=7, completion=3, cached=None)))
    assert usage == TokenUsage(input_tokens=7, output_tokens=3)


# ---------------------------------------------------------------------------
# S3 — tool-call parsing
# ---------------------------------------------------------------------------


def test_malformed_arguments_keep_the_call_with_empty_args(wire: OpenAIWire) -> None:
    """Dropping the call would strand its tool_call_id and 400 the next request."""
    message = SimpleNamespace(tool_calls=[_sdk_tool_call("c1", "press", "{not json")])
    calls = wire._parse_tool_calls(message)
    assert len(calls) == 1
    assert calls[0].id == "c1"
    assert calls[0].arguments == {}


def test_tool_calls_ignored_when_finish_reason_is_not_tool_calls(wire: OpenAIWire) -> None:
    """`finish_reason` is the authority, mirroring Anthropic's `stop_reason`."""
    wire.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(
                    return_value=_completion(
                        content="just thinking",
                        tool_calls=[_sdk_tool_call("c1", "press", "{}")],
                        finish_reason="stop",
                    )
                )
            )
        )
    )
    request = ChatRequest(system=(), turns=(UserTurn("hi"),), max_tokens=64, temperature=0.0)
    result = _run(wire.tool_turn(request, []))
    assert result.tool_calls == ()
    assert result.wants_more_tools is False


def test_tool_turn_sends_no_reasoning_effort(wire: OpenAIWire) -> None:
    """Function tools plus reasoning_effort are unsupported on this endpoint —
    the pair 400'd all 18 tool-loop turns of run 2026_08_21_2."""
    create = AsyncMock(return_value=_completion())
    wire.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    request = ChatRequest(system=(), turns=(UserTurn("hi"),), max_tokens=64, temperature=None)
    _run(wire.tool_turn(request, []))
    assert create.await_args.kwargs["reasoning_effort"] == "none"


def test_structured_output_still_sends_the_configured_effort(wire: OpenAIWire) -> None:
    """Only the TOOL path is affected — this one carries no tools."""
    message = SimpleNamespace(refusal=None, parsed=Observations())
    parse = AsyncMock(
        return_value=SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=_usage())
    )
    wire.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse)))
    request = ChatRequest(
        system=(), turns=(UserTurn("hi"),), max_tokens=64, temperature=None, effort="high"
    )
    _run(wire.parse_structured(request, Observations))
    assert parse.await_args.kwargs["reasoning_effort"] == "high"


def test_schema_size_fallback_never_fires_on_this_wire(wire: OpenAIWire) -> None:
    """No compiled-grammar cap here, so the F-40 fallback stays Anthropic-only."""
    assert wire.is_schema_too_large(ValueError("compiled grammar is too large")) is False


# ---------------------------------------------------------------------------
# S4 — the executor end-to-end through this wire
# ---------------------------------------------------------------------------


def test_executor_runs_its_tool_loop_through_the_openai_wire(
    wire: OpenAIWire, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point of the seam: same game logic, different vendor."""
    provider = ExecutorProvider(api_key="test", use_dynamic_context=False, wire=wire)
    provider._core_prompt = "SYSTEM"
    provider._age_prompts = {"dark": "DARK"}

    create = AsyncMock(
        side_effect=[
            _completion(
                content="pressing",
                tool_calls=[_sdk_tool_call("c1", "press", '{"key": "h"}')],
                finish_reason="tool_calls",
                usage=_usage(prompt=500, completion=20, cached=400),
            ),
            _completion(content="done", usage=_usage(prompt=600, completion=30, cached=500)),
        ]
    )
    wire.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    async def _fake_exec(_call: object) -> tuple[dict, ToolOutcome]:
        return ({"type": "press", "key": "h"}, ToolOutcome("c1", True, "ok"))

    monkeypatch.setattr(provider, "_execute_tool_call", _fake_exec)
    _run(provider._call_api([{"type": "text", "text": "ctx"}]))

    assert create.await_count == 2
    # 400 + 500 cached reads accumulated through the neutral TokenUsage.
    assert provider._usage.cache_read_tokens == 900
    assert provider._usage.input_tokens == 200  # (500-400) + (600-500)

    # Round 2 must replay round 1 as a proper assistant turn plus its tool result.
    replayed = create.await_args.kwargs["messages"]
    roles = [m["role"] for m in replayed]
    assert roles == ["system", "user", "assistant", "tool"]
    assert replayed[-1]["tool_call_id"] == "c1"


def test_temperature_is_omitted_when_unset() -> None:
    """Sending 0 returns `unsupported_value` on gpt-5.6; 88 of 88 calls 400'd."""
    import openai
    from gameplay_agent.providers.wire_openai import _temperature

    assert _temperature(None) is openai.NOT_GIVEN


def test_temperature_is_forwarded_when_set() -> None:
    """An operator who sets it on a model that supports it still gets it."""
    from gameplay_agent.providers.wire_openai import _temperature

    assert _temperature(0.0) == 0.0
