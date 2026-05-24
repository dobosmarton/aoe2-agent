"""Cross-process integration test for `RedisStreamsBroker`.

This is the test that validates Phase C's whole reason for existing: a
publisher in OS process A and a consumer in OS process B can share an
event stream through Redis, with the Protocol contract observed
byte-for-byte on the consumer side.

Skipped automatically when no real Redis is reachable — CI and slim
local installs run the contract test suite against `fakeredis`
(in-memory) instead. To exercise this test locally:

    just arena-infra-up                                      # bring up Redis
    export REDIS_URL=redis://:$REDIS_PASSWORD@localhost:6379/0
    pytest tests/test_redis_broker_cross_process.py -v

The subprocess uses `multiprocessing` with the default `spawn` start
method on macOS — that requires the publisher target to be importable,
so `_publisher_main` is a module-level function.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from evaluation.event_broker import EventEnvelope


_REDIS_URL_ENV: str = "REDIS_URL"
_DEFAULT_REDIS_URL: str = "redis://localhost:6379/0"


def _redis_available(url: str) -> bool:
    """Quick sync ping to decide whether to skip.

    Sync `redis.Redis` (not `redis.asyncio.Redis`) so the check works
    outside an event loop, and a short socket timeout so a missing
    Redis fails fast rather than stalling test collection.
    """
    try:
        import redis
    except ImportError:
        return False
    try:
        client = redis.Redis.from_url(url, socket_timeout=0.5, socket_connect_timeout=0.5)
        client.ping()
        client.close()
    except Exception:
        return False
    return True


_REDIS_URL = os.environ.get(_REDIS_URL_ENV, _DEFAULT_REDIS_URL)

pytestmark = pytest.mark.skipif(
    not _redis_available(_REDIS_URL),
    reason=(
        f"No reachable Redis at {_REDIS_URL!r}; set {_REDIS_URL_ENV} or "
        "run `just arena-infra-up` to enable."
    ),
)


# ---------------------------------------------------------------------------
# Subprocess publisher.
# ---------------------------------------------------------------------------


def _publisher_main(redis_url: str, key_prefix: str, run_id_str: str, n_events: int) -> None:
    """Publish `n_events` synthetic `turn_start` events and close the run.

    Runs in a child process spawned by `multiprocessing.Process`. The
    parent test process connects to the same Redis with the same
    `key_prefix` and consumes from the stream.

    Why `turn_start` and not a richer payload: this test exists to
    prove cross-process delivery, not payload-shape compatibility
    (the latter is covered by `tests/test_redis_broker.py`'s codec
    round-trip test). Smallest viable Event keeps the failure modes
    isolated to networking.
    """
    import asyncio

    from evaluation.event_broker import RunId
    from evaluation.event_log import Event, TurnStartPayload
    from evaluation.redis_broker import RedisStreamsBroker
    from redis.asyncio import Redis

    async def run() -> None:
        client = Redis.from_url(redis_url)
        try:
            broker = RedisStreamsBroker(client, key_prefix=key_prefix)
            run_id = RunId(run_id_str)
            broker.open_run(run_id)
            for i in range(n_events):
                evt = Event(
                    run_id=run_id_str,
                    agent_id="publisher",
                    t=i,
                    payload=TurnStartPayload(turn_num=i),
                    ts=datetime(2026, 5, 23, 12, 0, 0, tzinfo=UTC),
                )
                await broker.publish(run_id, evt)
            broker.close_run(run_id)
            # Flush queued admin so the parent observes the close before
            # the subprocess exits and tears down its connection.
            await broker.flush()
        finally:
            await client.aclose()

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Test.
# ---------------------------------------------------------------------------


def test_cross_process_publisher_to_consumer_preserves_order() -> None:
    """Parent consumes 100 events published from a subprocess, in order.

    This is the design promise of Phase C: the SSE handler in one
    process can tail events written by a producer in another process.
    All cross-impl invariants from the in-process broker (Seq dense
    1..N, monotonic, identical to publisher's emission order) must
    hold across the process boundary.
    """
    import asyncio

    from evaluation.event_broker import RunId, Seq
    from evaluation.redis_broker import RedisStreamsBroker
    from redis.asyncio import Redis

    n_events = 100
    key_prefix = f"xprocess:{uuid.uuid4().hex[:8]}"
    run_id_str = f"r-{uuid.uuid4().hex[:8]}"

    # `spawn` is the default on macOS; explicit so this test behaves
    # identically across platforms.
    ctx = mp.get_context("spawn")
    publisher = ctx.Process(
        target=_publisher_main,
        args=(_REDIS_URL, key_prefix, run_id_str, n_events),
    )
    publisher.start()

    async def consumer() -> list[EventEnvelope]:
        client = Redis.from_url(_REDIS_URL)
        try:
            broker = RedisStreamsBroker(client, key_prefix=key_prefix)
            run_id = RunId(run_id_str)
            broker.open_run(run_id)
            collected: list[EventEnvelope] = []
            async for env in broker.stream(run_id, from_seq=Seq(0)):
                collected.append(env)
                if len(collected) >= n_events:
                    break
            broker.close_run(run_id)
            broker.reap(run_id)
            await broker.flush()
            return collected
        finally:
            await client.aclose()

    try:
        envelopes = asyncio.run(consumer())
    finally:
        publisher.join(timeout=10)
        if publisher.is_alive():
            publisher.terminate()
            publisher.join(timeout=2)

    assert publisher.exitcode == 0, "publisher subprocess failed"
    assert len(envelopes) == n_events
    assert [int(e.seq) for e in envelopes] == list(range(1, n_events + 1))
    assert [e.event.t for e in envelopes] == list(range(n_events))
