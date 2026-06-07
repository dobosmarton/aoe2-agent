"""Redis Streams broker — Phase C of the log-first SSE architecture.

Implements the same `EventBroker` Protocol as `InProcessEventBroker` but
backs each run's log with a Redis stream. Publishers and consumers can
live in different OS processes — the design promise of Phase C:

    Process A          Process B
       │                   │
       ▼                   ▼
    publish ──► Redis ──► stream
                (key: arena:run:<r>:events)

The contract is **observable equivalence** with the in-process broker:
the same parametrized test suite in `tests/test_event_broker.py` runs
against both impls and must pass byte-for-byte at the envelope level
(`Seq` values, ordering, drain-on-close, overflow semantics).

Redis primitive mapping
-----------------------
    open_run(r)   → SET arena:run:<r>:open "1" EX <grace>   (cross-process sentinel)
    close_run(r)  → DEL arena:run:<r>:open                  (consumers terminate next loop)
    reap(r)       → DEL on all three keys
    is_open(r)    → process-local view of `_open_locally`   (sync per Protocol)
    publish       → INCR arena:run:<r>:seq → XADD arena:run:<r>:events MAXLEN ~ N <seq>-0 ...
    stream        → XREAD BLOCK ... in a loop; terminate when Redis says
                    `:open` is gone AND XREAD returned no new entries

Why `INCR`+`XADD <seq>-0` instead of Redis-native time IDs: `Seq` is a
`NewType("Seq", int)` starting at 1 and totally ordered; using `INCR` and
embedding it as the left half of the stream ID keeps that identity stable
across both broker impls. Redis-native `<ms>-<n>` IDs would force a
translation table at every consumer.

Async semantics for sync Protocol methods
-----------------------------------------
The Protocol's lifecycle methods (`open_run`, `close_run`, `reap`) are
synchronous, but Redis needs network I/O for them to be observable
across processes. We resolve this by scheduling the Redis side effects
as background asyncio tasks and awaiting them at the top of any async
method (`publish` / `stream`). The contract for callers is the same as
the in-process broker's: these methods must be invoked from within an
asyncio context (`asyncio.get_event_loop()` must return a running loop).
That precondition holds for every existing call site (FastAPI handlers,
CLI commands inside `asyncio.run`, fork tasks).
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Final, cast

from evaluation.event_broker import (
    BrokerMetricsSnapshot,
    BrokerOverflowError,
    EventEnvelope,
    LiveRun,
    RunId,
    RunMeta,
    Seq,
)
from evaluation.event_log import Event, EventRow

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    from redis.asyncio import Redis
    from redis.typing import EncodableT, FieldT


_log = logging.getLogger(__name__)

# Match `InProcessEventBroker`'s buffer size so cross-impl runs with the
# same workload exhibit the same overflow behaviour. The `~` MAXLEN form
# is approximate (Redis trims in radix-tree node chunks), within a few
# dozen entries of the requested limit — fine for the recovery story.
_DEFAULT_MAX_STREAM_LEN: Final = 10_000
_DEFAULT_OPEN_TTL_SECONDS: Final = 60 * 60 * 6
_DEFAULT_XREAD_BLOCK_MS: Final = 100
_KEY_PREFIX_DEFAULT: Final = "arena"


# ---------------------------------------------------------------------------
# Key naming — centralized so a future re-namespacing (e.g. tenant prefix)
# is a one-place change.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Keys:
    events: bytes
    seq: bytes
    open: bytes


def _keys_for(prefix: str, run_id: RunId) -> _Keys:
    base = f"{prefix}:run:{run_id}"
    return _Keys(
        events=f"{base}:events".encode(),
        seq=f"{base}:seq".encode(),
        open=f"{base}:open".encode(),
    )


def _run_id_from_open_key(prefix: str, key: bytes) -> RunId:
    """Inverse of `_keys_for(...).open` — recover the run_id from an `:open`
    key. Slice the fixed `…:run:` head and `:open` tail rather than
    `split(":")`, so a future run_id containing a colon wouldn't mis-parse."""
    text = key.decode()
    head = f"{prefix}:run:"
    tail = ":open"
    return RunId(text[len(head) : -len(tail)])


def _encode_sentinel(meta: RunMeta | None) -> bytes:
    """The `:open` sentinel's *value*. `is_open_remote` only checks EXISTS, so
    the value is free real estate for run identity — JSON when we have meta,
    the legacy `b"1"` when we don't (keeps old readers and tests happy)."""
    if meta is None:
        return b"1"
    return json.dumps({"label": meta.label, "started_at": meta.started_at}).encode()


def _decode_sentinel(raw: bytes | None) -> RunMeta:
    """Parse a sentinel value into `RunMeta`. Tolerant by design: a legacy
    `b"1"`, a missing key, or any non-JSON content yields empty identity
    rather than raising mid-`live_runs` during a rolling deploy."""
    if raw is None:
        return RunMeta()
    try:
        data = cast("object", json.loads(raw))
    except (ValueError, TypeError):
        return RunMeta()
    if not isinstance(data, dict):
        return RunMeta()
    label = data.get("label")
    started_at = data.get("started_at")
    return RunMeta(
        label=label if isinstance(label, str) else None,
        started_at=started_at if isinstance(started_at, str) else None,
    )


# ---------------------------------------------------------------------------
# Stream-entry codec — reuses Event.from_row's deserialization path so the
# Redis encoding stays interchange-compatible with the DuckDB cold path.
# ---------------------------------------------------------------------------


# Short field names: stream entries are stored uncompressed per-entry, and
# the full schema lives in code, not on the wire.
_F_RUN_ID: Final = b"r"
_F_AGENT_ID: Final = b"a"
_F_T: Final = b"t"
_F_KIND: Final = b"k"
_F_PAYLOAD: Final = b"p"
_F_TS: Final = b"ts"
_F_SCHEMA_VERSION: Final = b"v"


def _event_to_fields(event: Event) -> dict[FieldT, EncodableT]:
    """Build the field-map that `XADD` writes to a Redis stream entry.

    Return type is the exact alias redis-py's `xadd` expects so basedpyright
    sees a clean assignment at the call site (no cast). `FieldT` /
    `EncodableT` widen our concrete `bytes` / `bytes | int | str` to the
    SDK's accepted scalar union; the explicit local-variable annotation
    propagates that widening to the dict literal.
    """
    fields: dict[FieldT, EncodableT] = {
        _F_RUN_ID: event.run_id,
        _F_AGENT_ID: event.agent_id,
        _F_T: event.t,
        _F_KIND: event.payload.kind,
        _F_PAYLOAD: event.payload.model_dump_json(),
        _F_TS: event.ts.isoformat(),
        _F_SCHEMA_VERSION: event.schema_version,
    }
    return fields


def _fields_to_event(fields: dict[bytes, bytes]) -> Event:
    """Inverse of `_event_to_fields`.

    Packs the field-map into the same 7-tuple shape DuckDB returns and
    delegates to `Event.from_row` — the cold-path and live-path readers
    share one deserializer, so a bug fix in one applies to both.
    """
    row: EventRow = (
        fields[_F_RUN_ID].decode(),
        fields[_F_AGENT_ID].decode(),
        int(fields[_F_T]),
        fields[_F_KIND].decode(),
        fields[_F_PAYLOAD].decode(),
        datetime.fromisoformat(fields[_F_TS].decode()),
        int(fields[_F_SCHEMA_VERSION]),
    )
    return Event.from_row(row)


def _seq_to_stream_id(seq: Seq) -> bytes:
    """`<seq>-0` is the canonical custom-ID form. Redis only requires that
    successive IDs increase, which holds because `Seq` comes from `INCR`."""
    return f"{int(seq)}-0".encode()


def _stream_id_to_seq(stream_id: bytes | str) -> Seq:
    raw = stream_id.decode() if isinstance(stream_id, bytes) else stream_id
    return Seq(int(raw.split("-", 1)[0]))


# ---------------------------------------------------------------------------
# Broker.
# ---------------------------------------------------------------------------


class RedisStreamsBroker:
    """`EventBroker` impl backed by one Redis stream per run.

    Caller owns the `redis.asyncio.Redis` client — same pattern as
    `DuckDBEventSink`'s caller-owned-connection contract. The broker
    never closes the client; whoever built it tears it down.

    Concurrency: all methods are safe to call from any coroutine on
    the same event loop. The `redis.asyncio.Redis` connection pool
    handles multiplexing. Cross-process publishing works by construction
    — two processes pointing at the same Redis URL share state through
    Redis keys.
    """

    def __init__(
        self,
        client: Redis,
        *,
        key_prefix: str = _KEY_PREFIX_DEFAULT,
        max_stream_len: int = _DEFAULT_MAX_STREAM_LEN,
        open_ttl_seconds: int = _DEFAULT_OPEN_TTL_SECONDS,
        xread_block_ms: int = _DEFAULT_XREAD_BLOCK_MS,
    ) -> None:
        self._client = client
        self._key_prefix = key_prefix
        self._max_stream_len = max_stream_len
        self._open_ttl_seconds = open_ttl_seconds
        self._xread_block_ms = xread_block_ms
        self._open_locally: set[RunId] = set()
        # Deferred Redis admin ops queued by sync lifecycle methods
        # (open_run / close_run / reap). Stored as zero-arg thunks so
        # the coroutine isn't constructed until `_drain_admin` runs —
        # avoids the "coroutine never awaited" warning when a broker
        # is built and discarded without entering an async context, and
        # sidesteps Python 3.14's deprecation of `asyncio.get_event_loop`
        # outside a running loop.
        self._pending_admin: list[Callable[[], Awaitable[object]]] = []
        self._metrics_events_published = 0
        self._metrics_events_streamed = 0
        self._metrics_streams_dropped = 0

    # ---- Internals: background admin tracking -------------------------

    def _enqueue_admin(self, build_coro: Callable[[], Awaitable[object]]) -> None:
        """Defer a Redis admin call until the next async method runs."""
        self._pending_admin.append(build_coro)

    async def flush(self) -> None:
        """Execute every queued lifecycle op (`open_run` / `close_run` /
        `reap`) against Redis, in FIFO order.

        Call this after a sequence of sync lifecycle calls when you need
        their effects to be observable cross-process *before* the next
        broker method runs — chiefly: just before a publisher process
        exits, so subscribers in other processes see the `:open`
        sentinel disappear. `publish()` and `stream()` flush
        automatically, so most callers never need this directly.

        Sequential rather than `gather`'d: admin ops are tiny SET/DEL
        commands and preserving order matters for SET/DEL of the same
        key (a rapid open+close+open sequence — rare, but a gather
        could reorder).
        """
        if not self._pending_admin:
            return
        ops = self._pending_admin
        self._pending_admin = []
        for build in ops:
            await build()

    # ---- Lifecycle ----------------------------------------------------

    def open_run(self, run_id: RunId, meta: RunMeta | None = None) -> None:
        if run_id in self._open_locally:
            raise ValueError(f"run {run_id!r} is already open")
        self._open_locally.add(run_id)
        keys = _keys_for(self._key_prefix, run_id)
        # The sentinel's *value* carries run identity (label, started_at) for
        # the cross-process `live_runs` reader; `is_open_remote` only checks
        # existence, so this is transparent to liveness.
        sentinel = _encode_sentinel(meta)
        # `ex` gives the sentinel a finite life so an abandoned producer
        # eventually self-cleans. First writer wins for the cross-process
        # case; a re-open from the same process is already blocked by the
        # local check above, and cross-process re-opens are a deployment
        # bug we won't paper over.
        self._enqueue_admin(
            lambda: self._client.set(keys.open, sentinel, ex=self._open_ttl_seconds)
        )

    def close_run(self, run_id: RunId) -> None:
        self._open_locally.discard(run_id)
        keys = _keys_for(self._key_prefix, run_id)
        self._enqueue_admin(lambda: self._client.delete(keys.open))

    def reap(self, run_id: RunId) -> None:
        if run_id in self._open_locally:
            raise ValueError(f"cannot reap open run {run_id!r}; close it first")
        keys = _keys_for(self._key_prefix, run_id)
        self._enqueue_admin(lambda: self._client.delete(keys.events, keys.seq, keys.open))

    def is_open(self, run_id: RunId) -> bool:
        """Process-local view of openness.

        Sync per Protocol; cross-process consumers see openness through
        `stream()` which queries the Redis `:open` key directly. Callers
        like `_reaper_loop` that need cross-process truth must use
        `is_open_remote()` instead.
        """
        return run_id in self._open_locally

    async def is_open_remote(self, run_id: RunId) -> bool:
        """Cross-process truth: does the `:open` sentinel exist in Redis?"""
        # Flush first so this process observes its own queued open/close before
        # answering — `open_run`/`close_run` only enqueue the SET/DEL admin op
        # (sync Protocol, async Redis). Same discipline as `publish`/`stream`.
        await self.flush()
        keys = _keys_for(self._key_prefix, run_id)
        # `exists` returns an int (count of existing keys); cast keeps
        # basedpyright's reportAny strictness happy through the SDK boundary.
        count = cast("int", await self._client.exists(keys.open))
        return count > 0

    async def live_runs(self) -> list[LiveRun]:
        """Every open run in Redis, with identity from the sentinel value and a
        live event count from `XLEN`. Cross-process: surfaces runs opened by
        any process — this is what lets the web backend list a running CLI run.

        `flush()` first so a just-opened run in *this* process (its queued SET)
        is visible before the SCAN — matters for single-process tests; a no-op
        for a pure consumer like the web backend that opens nothing itself.
        """
        await self.flush()
        pattern = f"{self._key_prefix}:run:*:open"
        runs: list[LiveRun] = []
        async for key in self._client.scan_iter(match=pattern):
            key_bytes = cast("bytes", key)
            run_id = _run_id_from_open_key(self._key_prefix, key_bytes)
            keys = _keys_for(self._key_prefix, run_id)
            raw = cast("bytes | None", await self._client.get(keys.open))
            if raw is None:
                # Sentinel expired between SCAN and GET — no longer live, skip.
                continue
            meta = _decode_sentinel(raw)
            n_events = cast("int", await self._client.xlen(keys.events))
            runs.append(
                LiveRun(
                    run_id=run_id,
                    label=meta.label,
                    started_at=meta.started_at,
                    n_events=n_events,
                )
            )
        return runs

    # ---- Publish ------------------------------------------------------

    async def publish(self, run_id: RunId, event: Event) -> Seq:
        if run_id not in self._open_locally:
            raise RuntimeError(f"run {run_id!r} is not open")
        if event.run_id != run_id:
            raise ValueError(
                f"event.run_id {event.run_id!r} does not match broker run_id {run_id!r}"
            )
        await self.flush()
        keys = _keys_for(self._key_prefix, run_id)
        seq_int = cast("int", await self._client.incr(keys.seq))
        seq = Seq(seq_int)
        await self._client.xadd(
            keys.events,
            _event_to_fields(event),
            id=_seq_to_stream_id(seq),
            maxlen=self._max_stream_len,
            approximate=True,
        )
        self._metrics_events_published += 1
        return seq

    # ---- Stream -------------------------------------------------------

    async def stream(
        self,
        run_id: RunId,
        from_seq: Seq = Seq(0),
    ) -> AsyncIterator[EventEnvelope]:
        # XREAD BLOCK cancellation can leave the connection it was using
        # in an indeterminate state — redis-py async doesn't reliably mark
        # the connection as broken when the awaiting coroutine is cancelled
        # mid-flight (upstream redis-py issue #2624). The poisoned connection
        # gets returned to the pool with a pending response still in flight
        # on the socket; a subsequent `publish()` reading that response
        # off the wire sees `None` where it expected an INCR result.
        #
        # Fix: the `except` below catches CancelledError (and the closely-
        # related GeneratorExit, which fires when an async generator is
        # closed without exhausting it — same risk shape) and force-evicts
        # every idle connection from the pool. By the time we get here,
        # redis-py's command-cleanup `finally` has returned the poisoned
        # XREAD connection to the pool's idle list, so it's caught and
        # disposed of. `inuse_connections=False` leaves connections that
        # other coroutines are actively holding (a concurrent publisher's
        # in-flight INCR) untouched. Normal exits skip the disconnect —
        # the connection used during normal XREAD-then-yield rounds is
        # clean and worth keeping in the pool for the next consumer.
        try:
            keys = _keys_for(self._key_prefix, run_id)
            # `cursor` tracks the last seq yielded. `from_seq=Seq(0)` means
            # "from the beginning" — match InProcessEventBroker.stream.
            cursor = max(0, int(from_seq) - 1)
            while True:
                # Flush at the TOP of each iteration — not just on entry — so
                # a `close_run()` called mid-stream from another coroutine
                # gets its DEL pushed to Redis before we check `is_open_remote`.
                # Without this, the consumer's first iteration flushes the
                # open-SET but subsequent iterations never see the close-DEL,
                # and the `:open` sentinel stays True forever.
                await self.flush()
                await self._check_overflow(keys, run_id, cursor)
                entries = cast(
                    "list[tuple[bytes, list[tuple[bytes, dict[bytes, bytes]]]]] | None",
                    await self._client.xread(
                        streams={keys.events: _seq_to_stream_id(Seq(cursor))},
                        block=self._xread_block_ms,
                        count=None,
                    ),
                )
                yielded_any = False
                for envelope in self._envelopes_from_xread_reply(entries, run_id):
                    cursor = int(envelope.seq)
                    self._metrics_events_streamed += 1
                    yielded_any = True
                    yield envelope
                if yielded_any:
                    continue
                # Empty XREAD: either the producer is paused or done. Ask
                # Redis (not local state) — the sentinel is the cross-process
                # truth for "is the producer still running anywhere?". Local
                # `is_open_locally` is just an in-process optimisation that's
                # already covered by Redis being the same value.
                if not await self.is_open_remote(run_id):
                    return
        except (asyncio.CancelledError, GeneratorExit):
            await self._client.connection_pool.disconnect(inuse_connections=False)
            raise

    async def _check_overflow(self, keys: _Keys, run_id: RunId, cursor: int) -> None:
        """Raise `BrokerOverflowError` if the consumer fell behind MAXLEN.

        Mirrors the in-process broker's `cursor + 1 < head_seq` check at
        `event_broker.py:292` — including the case where the consumer
        asked for `from_seq=Seq(1)` (cursor=0) but the head has already
        advanced past it. Silently yielding what's available would
        break the no-gap guarantee; overflow + reconnect is the contract.
        An empty/missing stream means no head exists yet, so no overflow
        is possible.
        """
        try:
            info = cast(
                "dict[bytes | str, object]",
                await self._client.xinfo_stream(keys.events),
            )
        except Exception as exc:
            # Stream doesn't exist yet or transient connection blip. No
            # head to fall behind; if the connection's actually dead, the
            # next XREAD raises and the caller deals with it. Broad catch
            # is justified at the SDK boundary — the project's BLE001
            # audit (pyproject.toml line 147) covers this pattern.
            _log.debug("xinfo_stream(%s) failed: %s", run_id, exc)
            return
        first_entry = info.get("first-entry") or info.get(b"first-entry")
        if first_entry is None:
            return
        first_id = cast("tuple[bytes, object]", first_entry)[0]
        head_seq = int(_stream_id_to_seq(first_id))
        if cursor + 1 < head_seq:
            self._metrics_streams_dropped += 1
            raise BrokerOverflowError(
                run_id=run_id,
                requested_seq=Seq(cursor + 1),
                available_from=Seq(head_seq),
            )

    def _envelopes_from_xread_reply(
        self,
        entries: list[tuple[bytes, list[tuple[bytes, dict[bytes, bytes]]]]] | None,
        run_id: RunId,
    ) -> list[EventEnvelope]:
        """Parse `xread` output: `[(stream_key, [(id, fields), ...])]`.

        We requested exactly one stream so we take the first row's tail.
        """
        if not entries:
            return []
        _, items = entries[0]
        return [
            EventEnvelope(
                run_id=run_id,
                seq=_stream_id_to_seq(item_id),
                event=_fields_to_event(fields),
            )
            for item_id, fields in items
        ]

    # ---- Metrics ------------------------------------------------------

    async def metrics(self) -> BrokerMetricsSnapshot:
        """Same shape as `InProcessEventBroker.metrics()`.

        Async because counting runs across a Redis cluster could require
        a SCAN; the simple case (this process's view) returns
        immediately. The async signature reserves space for a future
        SCAN-backed implementation without changing callers.
        """
        return BrokerMetricsSnapshot(
            events_published=self._metrics_events_published,
            events_streamed=self._metrics_events_streamed,
            streams_dropped=self._metrics_streams_dropped,
            runs_open=len(self._open_locally),
        )


__all__ = ["RedisStreamsBroker"]
