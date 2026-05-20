"""Synth-arena invoke callables (Phase 6).

Each racing variant needs its own `invoke` function that accepts a WorldState
and returns (actions, reasoning, cost_usd). Two builders are provided:

  `build_synth_invoke(profile, api_key)` — real Anthropic API, one call per
      turn, simple JSON-array response format (no agentic tool-use loop).

  `build_mock_invoke(responses)` — deterministic stub that cycles through
      canned responses; no API key required. Used by offline tests and
      `just arena-smoke`.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, TypeAlias

from anthropic import AsyncAnthropic
from pydantic import TypeAdapter, ValidationError

from arena.prompts import get_prompt

if TYPE_CHECKING:
    from arena.config_profile import ConfigProfile
    from evaluation.world_sim import WorldState


# ---------------------------------------------------------------------------
# Public type alias — shared with race.py and tests.
# ---------------------------------------------------------------------------

InvokeFn: TypeAlias = Callable[
    ["WorldState"],
    Awaitable[tuple[list[dict[str, object]], str, float]],
]

# ---------------------------------------------------------------------------
# Prompt — variants live in arena/prompts.py and are looked up per profile.
# ---------------------------------------------------------------------------


def _state_to_prompt(state: WorldState) -> str:
    buildings = ", ".join(state.buildings) if state.buildings else "none"
    queue = len(state.villager_queue)
    age_up = (
        "not started"
        if state.age_up_ticks_remaining == 0
        else f"{state.age_up_ticks_remaining} turns remaining"
    )
    return (
        f"food={int(state.food)} wood={int(state.wood)} "
        f"gold={int(state.gold)} stone={int(state.stone)} "
        f"pop={state.population}/{state.pop_cap} age={state.age} "
        f"buildings=[{buildings}] queued_villagers={queue} age_up={age_up}"
    )


# ---------------------------------------------------------------------------
# Response parsing — uses Pydantic to avoid Any at the JSON boundary.
# ---------------------------------------------------------------------------

_ACTION_LIST_ADAPTER: TypeAdapter[list[dict[str, object]]] = TypeAdapter(list[dict[str, object]])


def _parse_actions(text: str) -> list[dict[str, object]]:
    """Find the first JSON array in `text` and validate it as action dicts."""
    cleaned = text.strip()
    try:
        return _ACTION_LIST_ADAPTER.validate_json(cleaned)
    except ValidationError:
        pass
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if match is None:
        return []
    try:
        return _ACTION_LIST_ADAPTER.validate_json(match.group())
    except ValidationError:
        return []


# ---------------------------------------------------------------------------
# Cost computation
# ---------------------------------------------------------------------------

# Pricing in USD per 1 million tokens (input, output).
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5-20251001": (0.80, 4.00),
    "claude-haiku-4-5": (0.80, 4.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-7": (15.00, 75.00),
}


def _compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = _MODEL_PRICING.get(model, (0.0, 0.0))
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


# ---------------------------------------------------------------------------
# Real Claude invoke builder
# ---------------------------------------------------------------------------


async def _call_claude(
    client: AsyncAnthropic,
    model: str,
    temperature: float,
    system_prompt: str,
    prompt: str,
) -> tuple[list[dict[str, object]], str, float]:
    from anthropic.types import TextBlock

    response = await client.messages.create(
        model=model,
        max_tokens=256,
        temperature=temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": prompt}],  # pyright: ignore[reportArgumentType]
    )

    text = next(
        (block.text for block in response.content if isinstance(block, TextBlock)),
        "",
    )
    cost = _compute_cost(model, response.usage.input_tokens, response.usage.output_tokens)
    return _parse_actions(text), text, cost


def build_synth_invoke(profile: ConfigProfile, api_key: str) -> InvokeFn:
    """Build an invoke callable backed by the real Anthropic API.

    One API call per turn: sends a WorldState summary, parses the JSON-array
    response into actions. Each call uses `profile.model` and
    `profile.temperature`, isolated from the global `gameplay_agent.config`
    singleton.
    """
    client = AsyncAnthropic(api_key=api_key)
    model = profile.model
    temperature = profile.temperature
    system_prompt = get_prompt(profile.prompt_variant)

    async def invoke(
        state: WorldState,
    ) -> tuple[list[dict[str, object]], str, float]:
        return await _call_claude(
            client, model, temperature, system_prompt, _state_to_prompt(state)
        )

    return invoke


# ---------------------------------------------------------------------------
# Mock invoke builder (offline / CI)
# ---------------------------------------------------------------------------

_EMPTY_RESPONSE: tuple[list[dict[str, object]], str, float] = ([], "no-op", 0.0)


def build_mock_invoke(
    responses: list[tuple[list[dict[str, object]], str, float]] | None = None,
) -> InvokeFn:
    """Deterministic stub cycling through `responses`. No API key required.

    Cycles indefinitely: when the list is exhausted it wraps around. Defaults
    to a single no-op response so callers don't need to specify anything for
    basic smoke tests.
    """
    _responses = list(responses) if responses is not None else [_EMPTY_RESPONSE]
    state = {"index": 0}

    async def invoke(
        _state: WorldState,
    ) -> tuple[list[dict[str, object]], str, float]:
        result = _responses[state["index"] % len(_responses)]
        state["index"] += 1
        return result

    return invoke


# ---------------------------------------------------------------------------
# JSON parsing test helper (module-private, exposed for direct unit tests)
# ---------------------------------------------------------------------------

_parse_actions_for_test = _parse_actions
