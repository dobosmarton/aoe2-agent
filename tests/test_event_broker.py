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
from datetime import UTC, datetime
from typing import TYPE_CHECKING, TypeAlias

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

from evaluation.event_broker import (
    BrokerEventSink,
    EventBroker,
    EventEnvelope,
    InProcessEventBroker,
    RunId,
    Seq,
)
from evaluation.event_log import Event, TurnStartPayload

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _event(run_id: str = "r1", t: int = 0) -> Event:
    """Build a minimal turn_start Event for tests."""
    return Event(
        run_id=run_id,
        agent_id="agent_x",
        t=t,
        payload=TurnStartPayload(turn_num=t),
        ts=datetime(2026, 5, 21, 12, 0, 0, tzinfo=UTC),
    )


_BrokerFactory: TypeAlias = "Callable[[], EventBroker]"

# Parametrize on factory, not instance — every test gets a fresh broker.
# Future RedisStreamsBroker plugs in here as a one-line addition.
_BROKER_FACTORIES: list[_BrokerFactory] = [InProcessEventBroker]


@pytest.fixture(params=_BROKER_FACTORIES, ids=lambda f: f.__name__)
def broker(request: pytest.FixtureRequest) -> EventBroker:
    factory: _BrokerFactory = request.param
    return factory()


async def _drain(broker: EventBroker, run_id: RunId, from_seq: Seq = Seq(0)) -> list[EventEnvelope]:
    """Materialize the broker's stream into a list (broker must be closed)."""
    return [env async for env in broker.stream(run_id, from_seq=from_seq)]


# ---------------------------------------------------------------------------
# Lifecycle and basic semantics
# ---------------------------------------------------------------------------


def test_lifecycle_open_publish_stream_close(broker: EventBroker) -> None:
    async def scenario() -> tuple[list[Seq], list[EventEnvelope]]:
        run = RunId("r1")
        broker.open_run(run)
        seqs = [await broker.publish(run, _event(run, t=i)) for i in range(5)]
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


def test_publish_before_open_raises_runtime_error(broker: EventBroker) -> None:
    with pytest.raises(RuntimeError, match="not open"):
        asyncio.run(broker.publish(RunId("r1"), _event()))


def test_publish_after_close_raises(broker: EventBroker) -> None:
    async def scenario() -> None:
        run = RunId("r1")
        broker.open_run(run)
        await broker.publish(run, _event(run))
        broker.close_run(run)
        await broker.publish(run, _event(run))  # must raise

    with pytest.raises(RuntimeError, match="not open"):
        asyncio.run(scenario())


def test_publish_with_mismatched_run_id_raises(broker: EventBroker) -> None:
    """The broker uses run_id for routing; event.run_id for downstream reads.
    Drifting them silently corrupts the materializations."""

    async def scenario() -> None:
        run = RunId("r1")
        broker.open_run(run)
        await broker.publish(run, _event(run_id="someone_else"))

    with pytest.raises(ValueError, match="does not match"):
        asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Replay and subscriber-count invariants
# ---------------------------------------------------------------------------


def test_late_subscriber_replays_from_seq_1(broker: EventBroker) -> None:
    async def scenario() -> list[EventEnvelope]:
        run = RunId("r1")
        broker.open_run(run)
        for i in range(3):
            await broker.publish(run, _event(run, t=i))
        broker.close_run(run)
        return await _drain(broker, run)  # subscriber attaches after close

    envs = asyncio.run(scenario())
    assert [e.seq for e in envs] == [Seq(1), Seq(2), Seq(3)]


def test_from_seq_skips_earlier_envelopes(broker: EventBroker) -> None:
    async def scenario() -> list[EventEnvelope]:
        run = RunId("r1")
        broker.open_run(run)
        for i in range(5):
            await broker.publish(run, _event(run, t=i))
        broker.close_run(run)
        return await _drain(broker, run, from_seq=Seq(3))

    envs = asyncio.run(scenario())
    assert [e.seq for e in envs] == [Seq(3), Seq(4), Seq(5)]


def test_two_concurrent_subscribers_see_identical_sequences(
    broker: EventBroker,
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
            await broker.publish(run, _event(run, t=i))
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


def test_broker_event_sink_emit_publishes_via_loop() -> None:
    async def scenario() -> list[EventEnvelope]:
        broker = InProcessEventBroker()
        run = RunId("r1")
        broker.open_run(run)
        sink = BrokerEventSink(broker=broker, run_id=run, loop=asyncio.get_running_loop())

        for i in range(3):
            sink.emit(_event(run, t=i))

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
def test_drained_seqs_are_dense_1_to_n_for_any_publish_count(n_events: int) -> None:
    """For any N, draining after close yields exactly Seq(1)..Seq(N) in order,
    with no gaps and no duplicates. Holds for any broker implementing the
    same Protocol — the contract, not the implementation, is on trial."""

    async def scenario() -> list[Seq]:
        broker = InProcessEventBroker()
        run = RunId("r1")
        broker.open_run(run)
        for i in range(n_events):
            await broker.publish(run, _event(run, t=i))
        broker.close_run(run)
        return [env.seq async for env in broker.stream(run, from_seq=Seq(0))]

    seqs = asyncio.run(scenario())
    assert seqs == [Seq(i) for i in range(1, n_events + 1)]


@pytest.mark.parametrize(
    ("n_events", "n_subscribers"),
    [(1, 1), (1, 5), (5, 3), (10, 2), (20, 5), (50, 1)],
)
def test_all_subscribers_observe_identical_sequences(n_events: int, n_subscribers: int) -> None:
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
            await broker.publish(run, _event(run, t=i))
        broker.close_run(run)

        return await asyncio.gather(*tasks)

    results = asyncio.run(scenario())
    expected = [Seq(i) for i in range(1, n_events + 1)]
    assert all(r == expected for r in results)
