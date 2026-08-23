"""Unit tests for loops/context.py — the state the 3 clocks share."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from gameplay_agent.goal_logger import GoalLogger
from gameplay_agent.goals import GoalManager
from gameplay_agent.loops.context import LoopContext
from gameplay_agent.memory import AgentMemory

from tests.loop_fakes import FakeActuator, FakeSource

if TYPE_CHECKING:
    from collections.abc import Awaitable


def _run(coro: Awaitable[object]) -> object:
    """Drive a coroutine to completion in a fresh event loop."""
    return asyncio.run(coro)


def _context(tmp_path) -> LoopContext:
    """A context with no game, no LLM and no network behind it."""
    return LoopContext(
        memory=AgentMemory(),
        goal_manager=GoalManager(),
        goal_logger=GoalLogger(tmp_path),
        source=FakeSource(),
        actuator=FakeActuator(),
    )


# Conformance to `FrameSource` / `Actuator` is checked where `_context` builds
# the LoopContext: basedpyright rejects a fake that drifts from the protocol.


def test_a_fresh_context_is_not_stopping(tmp_path) -> None:
    assert _context(tmp_path).stopping is False


def test_the_first_stop_reason_wins(tmp_path) -> None:
    """Three loops can each notice the end. The first names it; a later one
    would overwrite the cause with a consequence of it."""
    ctx = _context(tmp_path)
    ctx.request_stop("victory")
    ctx.request_stop("game_not_found")
    assert ctx.memory.game_end_reason == "victory"


def test_stopping_releases_every_waiter(tmp_path) -> None:
    """One event, so a loop parked on it leaves without being cancelled."""

    async def drive() -> bool:
        ctx = _context(tmp_path)
        waiter = asyncio.create_task(ctx.stop.wait())
        await asyncio.sleep(0)
        ctx.request_stop("interrupted")
        await asyncio.wait_for(waiter, timeout=1.0)
        return waiter.done()

    assert _run(drive()) is True
