"""Unit tests for loops/snapshot.py — the frame and the pipe the 3 clocks share."""

from __future__ import annotations

import asyncio
import time
from dataclasses import FrozenInstanceError
from typing import TYPE_CHECKING

import pytest
from gameplay_agent.loops.snapshot import FramePipe, Perception

if TYPE_CHECKING:
    from collections.abc import Awaitable

# ---------------------------------------------------------------------------
# Perception
# ---------------------------------------------------------------------------


def test_a_fresh_frame_is_almost_zero_milliseconds_old() -> None:
    assert Perception().age_ms < 100.0


def test_age_grows_with_the_clock() -> None:
    """The act loop skips a stale rule on this number, so it must move."""
    frame = Perception()
    time.sleep(0.01)
    assert frame.age_ms >= 10.0


def test_a_frame_is_frozen() -> None:
    """Three loops share one frame by reference; a writable field would tear."""
    frame = Perception()
    with pytest.raises(FrozenInstanceError):
        frame.alarm = True  # type: ignore[misc]


def test_a_frame_starts_with_no_hud_reading() -> None:
    """ResourceReadings is total=False, so "not read yet" is the empty dict."""
    assert Perception().hud_readings == {}


# ---------------------------------------------------------------------------
# FramePipe
# ---------------------------------------------------------------------------


def _run(coro: Awaitable[object]) -> object:
    """Drive a coroutine to completion in a fresh event loop."""
    return asyncio.run(coro)


def test_an_empty_pipe_reads_none() -> None:
    assert FramePipe().latest() is None


def test_latest_returns_the_newest_frame() -> None:
    pipe = FramePipe()
    first, second = Perception(tick=1), Perception(tick=2)
    pipe.put(first)
    pipe.put(second)
    assert pipe.latest() is second


def test_after_returns_a_frame_that_already_arrived() -> None:
    """No wait when the pipe already holds something newer."""
    pipe = FramePipe()
    pipe.put(Perception(tick=1, captured_at=100.0))
    assert _run(pipe.after(50.0)).tick == 1


def test_after_waits_for_a_newer_frame() -> None:
    """The idle-dispatch handshake: the act loop pressed '.', the camera jumped,
    and the frame it holds is now useless."""
    pipe = FramePipe()
    pipe.put(Perception(tick=1, captured_at=100.0))

    async def drive() -> Perception:
        async def publish() -> None:
            await asyncio.sleep(0)
            pipe.put(Perception(tick=2, captured_at=200.0))

        waiter = asyncio.create_task(pipe.after(100.0))
        await publish()
        return await waiter

    assert _run(drive()).tick == 2


def test_after_does_not_miss_a_frame_that_lands_mid_check() -> None:
    """The clear-then-check ordering: a put between the two must still wake it."""
    pipe = FramePipe()

    async def drive() -> Perception:
        waiter = asyncio.create_task(pipe.after(0.0))
        await asyncio.sleep(0)  # let it reach the wait
        pipe.put(Perception(tick=7, captured_at=1.0))
        return await waiter

    assert _run(drive()).tick == 7


def test_wait_for_due_holds_the_cadence() -> None:
    """No urgent request: the perceive loop waits out its interval."""

    async def drive() -> float:
        pipe = FramePipe()
        started = time.monotonic()
        await pipe.wait_for_due(0.05)
        return time.monotonic() - started

    assert _run(drive()) >= 0.04


def test_request_now_cuts_the_wait_short() -> None:
    """An act tick that needs a fresh frame must not wait out a full interval."""

    async def drive() -> float:
        pipe = FramePipe()
        pipe.request_now()
        started = time.monotonic()
        await pipe.wait_for_due(5.0)
        return time.monotonic() - started

    assert _run(drive()) < 1.0


def test_an_urgent_request_is_consumed_once() -> None:
    """One request buys one early wake-up, not every wake-up after it."""

    async def drive() -> float:
        pipe = FramePipe()
        pipe.request_now()
        await pipe.wait_for_due(5.0)
        started = time.monotonic()
        await pipe.wait_for_due(0.05)
        return time.monotonic() - started

    assert _run(drive()) >= 0.04
