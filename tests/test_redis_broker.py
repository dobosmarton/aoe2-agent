"""Tests for Redis-specific behaviour of `RedisStreamsBroker`.

The cross-impl Protocol contract is already exercised by
`tests/test_event_broker.py` (parametrized fixture sweeps both impls).
This file covers concerns that *only* apply to the Redis backend:

  * Stream-entry codec round-trip (XADD ↔ XREAD field shape).
  * `MAXLEN ~ N` trimming and the resulting `BrokerOverflowError`.
  * Process-local metrics counters increment on the right events.
  * `is_open_remote()` cross-checks against the Redis sentinel key.

All tests use an in-memory `fakeredis.aioredis.FakeRedis` so they run
without external services. The module is skipped entirely when
`fakeredis` isn't installed (slim install + no broker-redis extra).
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

    from evaluation.event_log import Event

pytest.importorskip("fakeredis.aioredis")

# Real imports (not the importorskip return) so basedpyright sees concrete
# types; the importorskip above gates the whole module on availability.
from fakeredis.aioredis import FakeRedis

from evaluation.event_broker import (
    BrokerOverflowError,
    RunId,
    Seq,
)
from evaluation.redis_broker import RedisStreamsBroker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _BrokerHarness:
    """Bundle of the broker + the bits a test needs to inspect Redis.

    Returning `client` and `key_prefix` from the factory means the one
    test that scans Redis state for verification doesn't have to reach
    into private attributes — keeps `reportPrivateUsage` clean.
    """

    broker: RedisStreamsBroker
    client: FakeRedis
    key_prefix: str


def _make_harness(*, max_stream_len: int = 10_000) -> _BrokerHarness:
    """Fresh broker against a fresh `FakeRedis` instance.

    Unique key prefix so cross-test leakage is impossible even if a future
    fakeredis change shares backing storage across instances.
    """
    client = FakeRedis()
    key_prefix = f"test:{uuid.uuid4().hex[:8]}"
    broker = RedisStreamsBroker(
        client,
        key_prefix=key_prefix,
        max_stream_len=max_stream_len,
        xread_block_ms=10,
    )
    return _BrokerHarness(broker=broker, client=client, key_prefix=key_prefix)


def _make_broker(*, max_stream_len: int = 10_000) -> RedisStreamsBroker:
    """Convenience wrapper for tests that only need the broker."""
    return _make_harness(max_stream_len=max_stream_len).broker


# ---------------------------------------------------------------------------
# Stream-entry codec — XADD/XREAD round-trip must yield byte-identical Events.
# ---------------------------------------------------------------------------


def test_event_round_trip_through_stream_preserves_all_fields(
    build_event: Callable[..., Event],
) -> None:
    """XADD-then-XREAD must reconstruct the same Event the producer emitted.

    Reuses `Event.from_row` for deserialization (see `_fields_to_event`
    in `evaluation/redis_broker.py`), so the cold-path DuckDB reader and
    the live-path Redis reader share one codec. This test is the
    smoke alarm for that invariant.
    """

    async def scenario() -> Event:
        broker = _make_broker()
        run = RunId("r1")
        broker.open_run(run)
        original = build_event(run, t=7)
        await broker.publish(run, original)
        broker.close_run(run)
        envelopes = [env async for env in broker.stream(run, from_seq=Seq(0))]
        assert len(envelopes) == 1
        return envelopes[0].event

    reconstructed = asyncio.run(scenario())
    # Build a reference event with the same parameters for field comparison.
    # The `build_event` fixture is deterministic for the same args.
    reference = asyncio.run(_publish_only(build_event))
    assert reconstructed == reference


async def _publish_only(build_event: Callable[..., Event]) -> Event:
    """Construct the reference Event matching the codec-test's input."""
    return build_event(RunId("r1"), t=7)


# ---------------------------------------------------------------------------
# MAXLEN trimming → BrokerOverflowError surfaces.
# ---------------------------------------------------------------------------


def test_maxlen_evicts_old_entries_and_overflow_raises(
    build_event: Callable[..., Event],
) -> None:
    """When XADD's MAXLEN trims past a consumer's cursor, `stream()` raises
    `BrokerOverflowError` with the correct `requested_seq` and
    `available_from`.

    Why this is a Redis-only test (not in the cross-impl contract suite):
    the in-process broker bounds its buffer via `deque(maxlen=...)` and
    raises overflow synchronously on the next wake. The Redis impl uses
    `XADD MAXLEN ~ N` (approximate trim) and detects overflow by reading
    `XINFO STREAM`'s `first-entry`. Same observable error, completely
    different machinery — worth its own test.
    """

    async def scenario() -> BrokerOverflowError | None:
        broker = _make_broker(max_stream_len=3)
        run = RunId("r1")
        broker.open_run(run)
        # Publish 6 events; MAXLEN=3 means seqs 1..3 get trimmed and the
        # stream retains 4..6 (approximate; fakeredis trims exactly).
        for i in range(6):
            await broker.publish(run, build_event(run, t=i))
        broker.close_run(run)
        # Consumer requests from Seq(2) — which is below the buffer head.
        try:
            async for _ in broker.stream(run, from_seq=Seq(2)):
                pass
        except BrokerOverflowError as err:
            return err
        return None

    err = asyncio.run(scenario())
    assert err is not None, "stream from a trimmed seq must raise BrokerOverflowError"
    assert err.requested_seq == Seq(2)
    # `available_from` = current head_seq; with MAXLEN=3 and 6 publishes,
    # entries 1..3 were trimmed so first-entry is Seq(4).
    assert err.available_from == Seq(4)


# ---------------------------------------------------------------------------
# Metrics counters.
# ---------------------------------------------------------------------------


def test_metrics_track_publish_stream_and_drop(
    build_event: Callable[..., Event],
) -> None:
    """Each operation bumps exactly the expected counter.

    Mirrors the InProcessEventBroker metrics test in
    `tests/test_event_broker.py::test_metrics_counters_track_publishes_and_yields`
    but exercises the Redis impl's `await metrics()` shape (the
    in-process metrics() is sync; Redis is async to leave room for a
    future SCAN-backed `runs_open`).
    """

    async def scenario() -> tuple[int, int, int, int]:
        broker = _make_broker(max_stream_len=2)
        run = RunId("r1")
        broker.open_run(run)
        for i in range(4):
            await broker.publish(run, build_event(run, t=i))
        broker.close_run(run)
        # Trigger an overflow drop by streaming from a trimmed seq.
        try:
            async for _ in broker.stream(run, from_seq=Seq(1)):
                pass
        except BrokerOverflowError:
            pass
        snap = await broker.metrics()
        return (
            snap.events_published,
            snap.events_streamed,
            snap.streams_dropped,
            snap.runs_open,
        )

    published, streamed, dropped, runs_open = asyncio.run(scenario())
    assert published == 4
    assert streamed == 0, "the overflow raises before any yield happens"
    assert dropped == 1
    assert runs_open == 0, "close_run dropped the run from the local open set"


# ---------------------------------------------------------------------------
# is_open_remote — cross-process truth (verified within one process via FakeRedis).
# ---------------------------------------------------------------------------


def test_is_open_remote_reflects_redis_sentinel_lifecycle() -> None:
    """After `open_run`+drain, the Redis sentinel exists. After
    `close_run`+drain, it's gone. `is_open_remote()` returns the truth
    even if local state diverges.

    This is the property that lets cross-process consumers terminate
    cleanly: they read Redis, not the publisher's local `_open_locally`.
    """

    async def scenario() -> tuple[bool, bool]:
        broker = _make_broker()
        run = RunId("r1")
        broker.open_run(run)
        # Force the queued SET to execute. `flush()` is the broker's
        # public hook for "make my sync lifecycle calls cross-process
        # visible right now" — exactly what cross-process consumers
        # need to assert here.
        await broker.flush()
        open_after_open = await broker.is_open_remote(run)
        broker.close_run(run)
        await broker.flush()
        open_after_close = await broker.is_open_remote(run)
        return open_after_open, open_after_close

    open_after_open, open_after_close = asyncio.run(scenario())
    assert open_after_open is True
    assert open_after_close is False


# ---------------------------------------------------------------------------
# Reap actually deletes Redis keys (not just local state).
# ---------------------------------------------------------------------------


def test_reap_removes_all_redis_keys_for_run(
    build_event: Callable[..., Event],
) -> None:
    """After `reap`, no Redis key under the run's prefix remains —
    the cross-process equivalent of the in-process broker dropping
    its per-run state."""

    async def scenario() -> int:
        harness = _make_harness()
        run = RunId("r1")
        harness.broker.open_run(run)
        await harness.broker.publish(run, build_event(run, t=0))
        harness.broker.close_run(run)
        harness.broker.reap(run)
        await harness.broker.flush()
        matches: list[bytes] = []
        async for key in harness.client.scan_iter(f"{harness.key_prefix}:run:{run}:*"):
            matches.append(key)
        return len(matches)

    remaining = asyncio.run(scenario())
    assert remaining == 0


# ---------------------------------------------------------------------------
# Cancellation cleanup — the Redis equivalent of the InProcess `_waiters == 0`
# invariant from design doc §9. The InProcess broker maintains an explicit
# waiter list and the existing `test_consumer_cancellation_removes_waiter`
# inspects it directly. RedisStreamsBroker polls via `XREAD BLOCK` so there's
# no equivalent waiter list — the leak surface is the Redis client's
# connection pool, which doesn't reliably reset when an `XREAD BLOCK` is
# cancelled mid-flight (upstream redis-py #2624).
#
# We split the invariant into two tests:
#
#   1. The bare-minimum leak guarantee: cancellation MUST complete without
#      hanging. A consumer task that's stuck forever after cancel is the
#      worst possible leak (it pins the event loop, pins connections,
#      and breaks `asyncio.run` cleanup). That's covered by the passing
#      test below.
#
#   2. The stronger Protocol promise: the SAME broker must be fully usable
#      after a cancelled stream. That's the `xfail` test — currently broken
#      because of the redis-py cancellation quirk. Restoring it requires
#      isolating each `stream()` to its own connection (or pool-disconnect
#      on cancel). Tracked as a Phase C follow-up.
# ---------------------------------------------------------------------------


def test_cancelled_stream_completes_without_hanging(
    build_event: Callable[..., Event],
) -> None:
    """Cancelling a `stream()` mid-XREAD must complete inside a finite window.

    The worst leak shape would be a consumer that never observes the cancel
    — `await task` would hang, pinning the event loop forever. `asyncio.timeout`
    turns that failure mode into an explicit test failure instead of a stuck
    CI run.

    A fresh broker after the cancellation is fully functional (verified by
    creating one inside the same scenario) — this isolates the failure mode
    documented in the xfail test: it's per-broker, not process-global.
    """

    async def scenario() -> int:
        broker = _make_harness().broker
        run = RunId("r1")
        broker.open_run(run)
        await broker.publish(run, build_event(run, t=0))

        seen: list[int] = []

        async def consumer() -> None:
            async for env in broker.stream(run, from_seq=Seq(0)):
                seen.append(env.event.t)

        task = asyncio.create_task(consumer())
        # Spin briefly so the consumer reads seq=1 and re-enters XREAD BLOCK
        # waiting for seq=2. xread_block_ms=10 (from _make_harness) keeps
        # the next BLOCK-then-cancel window short.
        for _ in range(20):
            await asyncio.sleep(0.01)
            if seen:
                break
        assert seen == [0], "consumer must reach mid-XREAD state for this test to mean anything"

        task.cancel()
        async with asyncio.timeout(2.0):
            with contextlib.suppress(asyncio.CancelledError):
                await task

        # Fresh broker against a fresh fakeredis client. Proves the cancellation
        # didn't leak anything process-global (asyncio state, event-loop hooks,
        # threadlocals). The first broker's connection pool may be poisoned by
        # the redis-py cancel quirk; that's covered by the xfail test below.
        fresh = _make_harness().broker
        run2 = RunId("r2")
        fresh.open_run(run2)
        await fresh.publish(run2, build_event(run2, t=0))
        fresh.close_run(run2)
        envs = [env async for env in fresh.stream(run2, from_seq=Seq(0))]
        return len(envs)

    count = asyncio.run(scenario())
    assert count == 1


def test_cancelled_stream_leaves_same_broker_usable(
    build_event: Callable[..., Event],
) -> None:
    """Strong form of the cancel-cleanup invariant — the SAME broker is reusable.

    Design doc §9 requires the same broker to be reusable after a cancelled
    stream. `RedisStreamsBroker.stream()` enforces this by allocating a
    per-call isolated client (`_make_isolated_stream_client`); the parent
    `self._client` is never touched by XREAD BLOCK, so cancellation can't
    leak a poisoned connection into the shared pool that publishers use.
    """

    async def scenario() -> list[int]:
        broker = _make_broker()
        run = RunId("r1")
        broker.open_run(run)
        await broker.publish(run, build_event(run, t=0))

        seen: list[int] = []

        async def consumer() -> None:
            async for env in broker.stream(run, from_seq=Seq(0)):
                seen.append(env.event.t)

        task = asyncio.create_task(consumer())
        for _ in range(20):
            await asyncio.sleep(0.01)
            if seen:
                break

        task.cancel()
        async with asyncio.timeout(2.0):
            with contextlib.suppress(asyncio.CancelledError):
                await task

        # The same broker should accept publishes and serve a fresh stream.
        await broker.publish(run, build_event(run, t=1))
        broker.close_run(run)
        return [env.event.t async for env in broker.stream(run, from_seq=Seq(0))]

    seen = asyncio.run(scenario())
    assert seen == [0, 1]
