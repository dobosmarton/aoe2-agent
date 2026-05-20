"""Async racing controller for the synthetic arena (Phase 6).

`race(config, api_key, initial_state, sink)` runs all profiles concurrently
via `asyncio.gather` — each variant gets its own independently-built
AsyncAnthropic client, so they never share API state.

`race_with_mock(config, initial_state, sink)` uses `build_mock_invoke()` for
fully offline runs (CI, smoke tests, `just arena-smoke`).

Both functions delegate to `_race_with_factory`, the private implementation
that accepts any `Callable[[ConfigProfile], InvokeFn]` strategy.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from arena.invoke import InvokeFn, build_mock_invoke, build_synth_invoke
from evaluation.event_log import NullEventSink
from gameplay_agent.synth_game_loop import SynthLoopResult, synth_game_loop

if TYPE_CHECKING:
    from collections.abc import Callable

    from arena.config_profile import ConfigProfile, RaceConfig
    from evaluation.event_log import EventSink
    from evaluation.world_sim import WorldState


_NULL_SINK = NullEventSink()


@dataclass(frozen=True, slots=True)
class VariantResult:
    """Outcome of one profile variant after `turns` synthetic turns."""

    profile: ConfigProfile
    loop_result: SynthLoopResult


async def _race_with_factory(
    config: RaceConfig,
    invoke_factory: Callable[[ConfigProfile], InvokeFn],
    initial_state: WorldState,
    sink: EventSink,
) -> list[VariantResult]:
    tasks = [
        synth_game_loop(
            invoke=invoke_factory(p),
            initial_state=initial_state,
            max_iterations=config.turns,
            sink=sink,
        )
        for p in config.profiles
    ]
    loop_results = await asyncio.gather(*tasks)
    return [
        VariantResult(profile=p, loop_result=r)
        for p, r in zip(config.profiles, loop_results, strict=True)
    ]


async def race(
    config: RaceConfig,
    api_key: str,
    initial_state: WorldState,
    sink: EventSink = _NULL_SINK,
) -> list[VariantResult]:
    """Run all profiles concurrently with real Claude API calls.

    Each variant gets its own `AsyncAnthropic` client built from `api_key`.
    All variants share `initial_state` as their starting position.
    """
    return await _race_with_factory(
        config,
        lambda p: build_synth_invoke(p, api_key),
        initial_state,
        sink,
    )


async def race_with_mock(
    config: RaceConfig,
    initial_state: WorldState,
    sink: EventSink = _NULL_SINK,
) -> list[VariantResult]:
    """Run all profiles concurrently with the deterministic mock invoke.

    No API key required — suitable for CI, offline tests, and `just arena-smoke`.
    """
    return await _race_with_factory(
        config,
        lambda _: build_mock_invoke(),
        initial_state,
        sink,
    )
