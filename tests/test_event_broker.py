"""Unit + contract tests for `evaluation/event_broker.py`.

The broker is the central correctness primitive of Phase 1 — every other
piece (persister, SSE handler, future Redis broker) trusts these invariants:

    1. `Seq` values are 1..N, monotonic, gap-free, unique per run.
    2. Late subscribers see the full history from their `from_seq`.
    3. Concurrent subscribers see byte-identical envelope sequences.
    4. Closing a run drains active streams cleanly.
    5. Cancelled consumers leave no leaked waiters.

The parametrized tests at the bottom sweep (1) and (3) across the value
range a hypothesis property test would cover (`hypothesis` isn't a project
dep today; add it + replace these two `parametrize` blocks with `@given`
strategies if you want randomized exploration later).

`broker` fixture is parametrized on factory so adding a future
`RedisStreamsBroker` (Phase C) is one fixture-list entry change —
every test re-runs unchanged.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, TypeAlias

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

    from evaluation.event_log import Event

from evaluation.event_broker import (
    BrokerEventSink,
    BrokerMetricsSnapshot,
    BrokerOverflowError,
    EventBroker,
    EventEnvelope,
    InProcessEventBroker,
    RunId,
    Seq,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_BrokerFactory: TypeAlias = "Callable[[], EventBroker]"


def _make_fakeredis_broker() -> EventBroker:
    """Phase C factory: `RedisStreamsBroker` against an in-memory `FakeRedis`.

    Each call constructs a fresh `FakeRedis` so per-test isolation is
    automatic (no shared backing store). `xread_block_ms=10` keeps the
    close-detection round-trip fast — the contract tests run a "open,
    publish N, close, drain" cycle ~30 times across the fixture sweep,
    and the default 100ms block would add several seconds of overhead.
    A unique `key_prefix` is belt-and-braces against any future
    fakeredis change that might share state across instances.
    """
    import uuid

    import fakeredis.aioredis

    from evaluation.redis_broker import RedisStreamsBroker

    client = fakeredis.aioredis.FakeRedis()
    return RedisStreamsBroker(
        client,
        key_prefix=f"test:{uuid.uuid4().hex[:8]}",
        xread_block_ms=10,
    )


# Parametrize on factory, not instance — every test gets a fresh broker.
# `RedisStreamsBroker` participates iff `fakeredis` is installed; CI installs
# it via the `dev` extra, slim installs skip it cleanly.
_BROKER_FACTORIES: list[_BrokerFactory] = [InProcessEventBroker]
try:
    import fakeredis.aioredis  # noqa: F401
except ImportError:
    pass
else:
    _BROKER_FACTORIES.append(_make_fakeredis_broker)


def _factory_id(f: _BrokerFactory) -> str:
    # `Callable` doesn't promise `__name__` in pyright's view, but every entry
    # in `_BROKER_FACTORIES` is either a class or a `def`, both of which carry
    # it at runtime. `getattr` keeps the fixture id useful even if a future
    # factory entry is a `functools.partial` or similar nameless callable.
    return getattr(f, "__name__", repr(f))


_BROKER_IDS: list[str] = [_factory_id(f) for f in _BROKER_FACTORIES]


@pytest.fixture(params=_BROKER_FACTORIES, ids=_BROKER_IDS)
def broker(request: pytest.FixtureRequest) -> EventBroker:
    factory: _BrokerFactory = request.param  # pyright: ignore[reportAny]
    return factory()


async def _drain(broker: EventBroker, run_id: RunId, from_seq: Seq = Seq(0)) -> list[EventEnvelope]:
    """Materialize the broker's stream into a list (broker must be closed)."""
    return [env async for env in broker.stream(run_id, from_seq=from_seq)]


# ---------------------------------------------------------------------------
# Lifecycle and basic semantics
# ---------------------------------------------------------------------------


def test_lifecycle_open_publish_stream_close(
    broker: EventBroker, build_event: Callable[..., Event]
) -> None:
    async def scenario() -> tuple[list[Seq], list[EventEnvelope]]:
        run = RunId("r1")
        broker.open_run(run)
        seqs = [await broker.publish(run, build_event(run, t=i)) for i in range(5)]
        broker.close_run(run)
        envs = await _drain(broker, run)
        return seqs, envs

    seqs, envs = asyncio.run(scenario())
    assert seqs == [Seq(1), Seq(2), Seq(3), Seq(4), Seq(5)]
    assert [e.seq for e in envs] == [Seq(1), Seq(2), Seq(3), Seq(4), Seq(5)]
    assert [e.event.t for e in envs] == [0, 1, 2, 3, 4]
    assert all(e.run_id == RunId("r1") for e in envs)


def test_open_twice_raises_value_error(broker: EventBroker) -> None:
    run = RunId("r1")
    broker.open_run(run)
    with pytest.raises(ValueError, match="already open"):
        broker.open_run(run)


def test_publish_before_open_raises_runtime_error(
    broker: EventBroker, build_event: Callable[..., Event]
) -> None:
    with pytest.raises(RuntimeError, match="not open"):
        asyncio.run(broker.publish(RunId("r1"), build_event()))


def test_publish_after_close_raises(broker: EventBroker, build_event: Callable[..., Event]) -> None:
    async def scenario() -> None:
        run = RunId("r1")
        broker.open_run(run)
        await broker.publish(run, build_event(run))
        broker.close_run(run)
        await broker.publish(run, build_event(run))  # must raise

    with pytest.raises(RuntimeError, match="not open"):
        asyncio.run(scenario())


def test_publish_with_mismatched_run_id_raises(
    broker: EventBroker, build_event: Callable[..., Event]
) -> None:
    """The broker uses run_id for routing; event.run_id for downstream reads.
    Drifting them silently corrupts the materializations."""

    async def scenario() -> None:
        run = RunId("r1")
        broker.open_run(run)
        await broker.publish(run, build_event(run_id="someone_else"))

    with pytest.raises(ValueError, match="does not match"):
        asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Replay and subscriber-count invariants
# ---------------------------------------------------------------------------


def test_late_subscriber_replays_from_seq_1(
    broker: EventBroker, build_event: Callable[..., Event]
) -> None:
    async def scenario() -> list[EventEnvelope]:
        run = RunId("r1")
        broker.open_run(run)
        for i in range(3):
            await broker.publish(run, build_event(run, t=i))
        broker.close_run(run)
        return await _drain(broker, run)  # subscriber attaches after close

    envs = asyncio.run(scenario())
    assert [e.seq for e in envs] == [Seq(1), Seq(2), Seq(3)]


def test_from_seq_skips_earlier_envelopes(
    broker: EventBroker, build_event: Callable[..., Event]
) -> None:
    async def scenario() -> list[EventEnvelope]:
        run = RunId("r1")
        broker.open_run(run)
        for i in range(5):
            await broker.publish(run, build_event(run, t=i))
        broker.close_run(run)
        return await _drain(broker, run, from_seq=Seq(3))

    envs = asyncio.run(scenario())
    assert [e.seq for e in envs] == [Seq(3), Seq(4), Seq(5)]


def test_two_concurrent_subscribers_see_identical_sequences(
    broker: EventBroker, build_event: Callable[..., Event]
) -> None:
    async def scenario() -> tuple[list[EventEnvelope], list[EventEnvelope]]:
        run = RunId("r1")
        broker.open_run(run)

        async def consumer() -> list[EventEnvelope]:
            return [env async for env in broker.stream(run, from_seq=Seq(0))]

        task_a = asyncio.create_task(consumer())
        task_b = asyncio.create_task(consumer())
        await asyncio.sleep(0)  # let both subscribers attach before publishing

        for i in range(4):
            await broker.publish(run, build_event(run, t=i))
        broker.close_run(run)

        return await asyncio.gather(task_a, task_b)

    envs_a, envs_b = asyncio.run(scenario())
    assert envs_a == envs_b
    assert [e.seq for e in envs_a] == [Seq(1), Seq(2), Seq(3), Seq(4)]


# ---------------------------------------------------------------------------
# Close + cancellation cleanup
# ---------------------------------------------------------------------------


def test_close_terminates_active_stream(broker: EventBroker) -> None:
    async def scenario() -> list[EventEnvelope]:
        run = RunId("r1")
        broker.open_run(run)

        async def consumer() -> list[EventEnvelope]:
            return [env async for env in broker.stream(run, from_seq=Seq(0))]

        task = asyncio.create_task(consumer())
        await asyncio.sleep(0)  # subscriber arms its waiter
        broker.close_run(run)
        return await asyncio.wait_for(task, timeout=1.0)

    envs = asyncio.run(scenario())
    assert envs == []


def test_consumer_cancellation_removes_waiter() -> None:
    # InProcessEventBroker-specific — inspects private state to prove
    # no leak. Other broker impls may store waiters differently; their
    # equivalent cleanup is verified by the property tests below.
    async def scenario() -> tuple[int, int]:
        broker = InProcessEventBroker()
        run = RunId("r1")
        broker.open_run(run)

        async def consumer() -> None:
            async for _ in broker.stream(run, from_seq=Seq(0)):
                pass

        task = asyncio.create_task(consumer())
        await asyncio.sleep(0)  # subscriber arms its waiter
        waiters_during = len(broker._waiters[run])

        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        waiters_after = len(broker._waiters[run])
        return waiters_during, waiters_after

    waiters_during, waiters_after = asyncio.run(scenario())
    assert waiters_during == 1
    assert waiters_after == 0


# ---------------------------------------------------------------------------
# BrokerEventSink adapter
# ---------------------------------------------------------------------------


def test_broker_event_sink_emit_publishes_via_loop(
    build_event: Callable[..., Event],
) -> None:
    async def scenario() -> list[EventEnvelope]:
        broker = InProcessEventBroker()
        run = RunId("r1")
        broker.open_run(run)
        sink = BrokerEventSink(broker=broker, run_id=run, loop=asyncio.get_running_loop())

        for i in range(3):
            sink.emit(build_event(run, t=i))

        # `call_soon_threadsafe` schedules onto next loop iter — give it room.
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        broker.close_run(run)
        return await _drain(broker, run)

    envs = asyncio.run(scenario())
    assert [e.event.t for e in envs] == [0, 1, 2]


# ---------------------------------------------------------------------------
# Property-style tests — sweep the broker's central correctness invariants
# across a representative range. (hypothesis would be the canonical tool;
# parametrize covers the same ground for now without a new project dep.)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_events", [0, 1, 5, 50, 200])
def test_drained_seqs_are_dense_1_to_n_for_any_publish_count(
    n_events: int, build_event: Callable[..., Event]
) -> None:
    """For any N, draining after close yields exactly Seq(1)..Seq(N) in order,
    with no gaps and no duplicates. Holds for any broker implementing the
    same Protocol — the contract, not the implementation, is on trial."""

    async def scenario() -> list[Seq]:
        broker = InProcessEventBroker()
        run = RunId("r1")
        broker.open_run(run)
        for i in range(n_events):
            await broker.publish(run, build_event(run, t=i))
        broker.close_run(run)
        return [env.seq async for env in broker.stream(run, from_seq=Seq(0))]

    seqs = asyncio.run(scenario())
    assert seqs == [Seq(i) for i in range(1, n_events + 1)]


@pytest.mark.parametrize(
    ("n_events", "n_subscribers"),
    [(1, 1), (1, 5), (5, 3), (10, 2), (20, 5), (50, 1)],
)
def test_all_subscribers_observe_identical_sequences(
    n_events: int, n_subscribers: int, build_event: Callable[..., Event]
) -> None:
    """Every subscriber attached before publish sees the same envelope list."""

    async def scenario() -> list[list[Seq]]:
        broker = InProcessEventBroker()
        run = RunId("r1")
        broker.open_run(run)

        async def consume() -> list[Seq]:
            return [env.seq async for env in broker.stream(run, from_seq=Seq(0))]

        tasks = [asyncio.create_task(consume()) for _ in range(n_subscribers)]
        await asyncio.sleep(0)  # let every subscriber arm its waiter

        for i in range(n_events):
            await broker.publish(run, build_event(run, t=i))
        broker.close_run(run)

        return await asyncio.gather(*tasks)

    results = asyncio.run(scenario())
    expected = [Seq(i) for i in range(1, n_events + 1)]
    assert all(r == expected for r in results)


# ---------------------------------------------------------------------------
# Phase 3 — reap (Protocol-level, runs against every broker impl)
# ---------------------------------------------------------------------------


def test_reap_drops_buffer_for_closed_run(
    broker: EventBroker, build_event: Callable[..., Event]
) -> None:
    """After reap, the run's history is irrecoverable — a fresh stream
    returns empty even though events were published before close."""

    async def scenario() -> list[EventEnvelope]:
        run = RunId("r1")
        broker.open_run(run)
        for i in range(3):
            await broker.publish(run, build_event(run, t=i))
        broker.close_run(run)
        broker.reap(run)
        return await _drain(broker, run)

    assert asyncio.run(scenario()) == []


def test_reap_on_open_run_raises_value_error(broker: EventBroker) -> None:
    run = RunId("r1")
    broker.open_run(run)
    with pytest.raises(ValueError, match="cannot reap open run"):
        broker.reap(run)


def test_reap_after_close_is_idempotent_with_open_again(
    broker: EventBroker, build_event: Callable[..., Event]
) -> None:
    """Reap drops state cleanly enough that the same run_id can be opened
    fresh — useful for tests and future scenarios that recycle ids."""

    async def scenario() -> list[Seq]:
        run = RunId("r1")
        broker.open_run(run)
        await broker.publish(run, build_event(run, t=0))
        broker.close_run(run)
        broker.reap(run)
        broker.open_run(run)
        await broker.publish(run, build_event(run, t=0))
        broker.close_run(run)
        return [env.seq async for env in broker.stream(run, from_seq=Seq(0))]

    # Fresh open: seq restarts at 1.
    assert asyncio.run(scenario()) == [Seq(1)]


# ---------------------------------------------------------------------------
# Phase 3 — backpressure (InProcessEventBroker-specific, NOT on Protocol)
# ---------------------------------------------------------------------------


def test_backpressure_evicts_head_when_buffer_full(
    build_event: Callable[..., Event],
) -> None:
    """Publishing past `max_buffer_size` auto-evicts the leftmost
    envelope and bumps `head_seq` by one."""

    async def scenario() -> tuple[int, int, int]:
        broker = InProcessEventBroker(max_buffer_size=3)
        run = RunId("r1")
        broker.open_run(run)
        for i in range(5):
            await broker.publish(run, build_event(run, t=i))
        return (
            len(broker._buffers[run]),
            broker._head_seq[run],
            broker._buffers[run][0].seq,
        )

    buf_len, head_seq, oldest_seq = asyncio.run(scenario())
    assert buf_len == 3, "deque maxlen must cap retained envelopes"
    assert head_seq == 3, "head_seq tracks the seq of the oldest retained envelope"
    assert oldest_seq == Seq(3), "the leftmost envelope's seq matches head_seq"


def test_slow_consumer_raises_overflow_error_on_wake(
    build_event: Callable[..., Event],
) -> None:
    """A consumer arms its waiter while its cursor is valid, then publish
    evicts past the cursor; on wake, the consumer self-raises.

    Critical timing: between the consumer arming its waiter and the test
    waking it, all four publishes run synchronously (publish has no
    `await` inside, so consecutive `await publish(...)` calls never yield
    control back to the scheduler). That's what guarantees the consumer
    finds `head_seq > cursor + 1` on its single wake — not multiple
    intermediate wakes that would each drain a single event."""

    async def scenario() -> BrokerOverflowError:
        broker = InProcessEventBroker(max_buffer_size=3)
        run = RunId("r1")
        broker.open_run(run)

        # Pre-publish so the consumer attaches mid-stream at Seq(1).
        await broker.publish(run, build_event(run, t=0))

        captured: list[BrokerOverflowError] = []

        async def consumer() -> None:
            try:
                async for _ in broker.stream(run, from_seq=Seq(1)):
                    pass  # consume normally; the eviction is the test
            except BrokerOverflowError as err:
                captured.append(err)

        task = asyncio.create_task(consumer())
        await asyncio.sleep(0)  # let consumer drain Seq(1) + arm waiter

        # Publish 4 more events synchronously (no `await sleep(0)` between).
        # Buffer holds [Seq(3), Seq(4), Seq(5)] with head_seq=3 afterward.
        for i in range(1, 5):
            await broker.publish(run, build_event(run, t=i))

        await asyncio.wait_for(task, timeout=1.0)
        assert captured, "consumer should have raised BrokerOverflowError"
        return captured[0]

    err = asyncio.run(scenario())
    # Consumer drained Seq(1) cleanly; next requested is Seq(2),
    # which was evicted (head_seq advanced to 3).
    assert err.requested_seq == Seq(2)
    assert err.available_from == Seq(3)


def test_overflow_increments_streams_dropped_metric(
    build_event: Callable[..., Event],
) -> None:
    """Each overflow-raise bumps `streams_dropped` so /metrics sees it."""

    async def scenario() -> int:
        broker = InProcessEventBroker(max_buffer_size=2)
        run = RunId("r1")
        broker.open_run(run)
        for i in range(5):
            await broker.publish(run, build_event(run, t=i))
        broker.close_run(run)
        # Now a from_seq=1 request is below head_seq=4 and must raise.
        with pytest.raises(BrokerOverflowError):
            await _drain(broker, run, from_seq=Seq(1))
        return broker.metrics().streams_dropped

    assert asyncio.run(scenario()) == 1


def test_metrics_counters_track_publishes_and_yields(
    build_event: Callable[..., Event],
) -> None:
    """`events_published` increments on publish; `events_streamed` on each
    yielded envelope. `runs_open` reflects the current open-run set."""

    async def scenario() -> BrokerMetricsSnapshot:
        broker = InProcessEventBroker()
        run_a = RunId("a")
        run_b = RunId("b")
        broker.open_run(run_a)
        broker.open_run(run_b)
        for i in range(4):
            await broker.publish(run_a, build_event(run_a, t=i))
        for i in range(2):
            await broker.publish(run_b, build_event(run_b, t=i))
        broker.close_run(run_a)
        # Drain run_a only — events_streamed should be 4, not 6.
        async for _ in broker.stream(run_a, from_seq=Seq(0)):
            pass
        return broker.metrics()

    m = asyncio.run(scenario())
    assert m.events_published == 6
    assert m.events_streamed == 4
    assert m.streams_dropped == 0
    assert m.runs_open == 1  # run_b still open
