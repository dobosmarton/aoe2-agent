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
