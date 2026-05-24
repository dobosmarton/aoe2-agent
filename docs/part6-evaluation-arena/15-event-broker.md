# Chapter 15 — Event Broker

The broker is the **single source of truth for live event consumers**. Producers (game loops, fork replays) publish into it; consumers (the SSE backend, the DuckDB persister) subscribe to it. DuckDB and the web UI are symmetric materializations — neither knows about the other.

This chapter is the kept-current reference. For the architectural rationale (why log-first, what bug it fixes, why pluggable), see the frozen design spec at [`docs/design/event-broker-architecture.md`](../design/event-broker-architecture.md), or the one-page summary in [ADR 0001](../adr/0001-broker-first-architecture.md).

## The Protocol

`evaluation/event_broker.py:130` defines the `EventBroker` Protocol. Six methods, no inheritance:

```python
class EventBroker(Protocol):
    def open_run(self, run_id: RunId) -> None
    def close_run(self, run_id: RunId) -> None
    def reap(self, run_id: RunId) -> None
    def is_open(self, run_id: RunId) -> bool
    async def publish(self, run_id: RunId, event: Event) -> Seq
    def stream(self, run_id: RunId, from_seq: Seq = Seq(0)) -> AsyncIterator[EventEnvelope]
```

Lifecycle is `open_run → publish * N → close_run → reap`. Consumers call `stream(run_id, from_seq)` once — one call covers both replay (events with `seq >= from_seq`) and live tail.

Two new types make illegal states unrepresentable at the type checker:

- `RunId = NewType("RunId", str)` — opaque per-run identity.
- `Seq = NewType("Seq", int)` — monotonic per-run sequence number assigned by the broker, starting at 1. Distinct from `Event.t` (the turn number, which multiple events per turn share). `Seq` makes ordering total across the whole run, which the SSE client needs for unambiguous dedupe on reconnect.

`EventEnvelope` (`evaluation/event_broker.py:60`) wraps `(run_id, seq, event)`. Keeping `seq` out of the `Event` dataclass means no DuckDB schema change and keeps the broker's "I assigned this" concern out of the domain model.

## Two implementations

| Impl | When to use | External deps | Cross-process? |
|---|---|---|---|
| `InProcessEventBroker` (`evaluation/event_broker.py:189`) | Default. Single-process work — CLI, tests, single-machine dev. | None | No |
| `RedisStreamsBroker` (`evaluation/redis_broker.py:178`) | Phase C and beyond. Cross-process replay — e.g. FastAPI live-tailing a CLI race. | `redis` (install via `broker-redis` extra) | Yes |

Both implement the same Protocol and pass the same parametrized contract test suite in `tests/test_event_broker.py` byte-for-byte at the envelope level. That's the design's strongest guarantee: a publisher does not know which backend is in play, and a subscriber sees identical `Seq` values, ordering, drain-on-close, and overflow semantics either way.

## Switching backends

`evaluation/broker_factory.py:60` is the single place `ARENA_BROKER_BACKEND` is read. Every callsite that needs a broker goes through `make_broker()`:

```python
broker = make_broker()   # InProcessEventBroker by default
```

Environment variables:

| Var | Default | Effect |
|---|---|---|
| `ARENA_BROKER_BACKEND` | `inprocess` | `inprocess` or `redis`. Any other value raises `ValueError` — silent fallback would hide deployment misconfigurations. |
| `REDIS_URL` | (see below) | Explicit Redis URL when backend is `redis`. Highest priority. |
| `REDIS_PASSWORD` | unset | If `REDIS_URL` is *not* set but `REDIS_PASSWORD` is, the broker auto-builds `redis://:${REDIS_PASSWORD}@localhost:6379/0`. This is the compose-stack path — see [Runbook: redis-broker-ops](../runbooks/redis-broker-ops.md). |

The `redis` Python client is **only imported inside the `"redis"` branch** (`broker_factory.py:76`), so a slim install (`pip install -e .` without extras) keeps working without it.

## Backpressure semantics

In-process broker (`event_broker.py:189`):
- Per-run buffer is a `deque` with `maxlen=10_000`.
- When full, the next `publish` evicts the oldest envelope (O(1)) and bumps `head_seq` by one.
- A consumer whose cursor falls below `head_seq` self-raises `BrokerOverflowError(run_id, requested_seq, available_from)` from inside `stream()` on the next wake-up. The slow consumer self-evicts; publishers never block or raise on buffer pressure.
- The SSE handler (`arena/web/server.py:296`, `_stream_from_broker`) catches the error, emits a final `event: overflow` SSE line carrying `available_from`, and returns. The frontend reconnects with `?from_seq=<available_from>` and accepts the gap — surfacing the loss is the contract; silent partial reads would be worse.

Redis broker (`redis_broker.py:178`):
- `XADD MAXLEN ~ 10_000` matches the in-process buffer size (the `~` form is approximate — Redis trims in radix-tree node chunks, within a few dozen entries of the limit).
- Overflow is detected by comparing the consumer's cursor against `XINFO STREAM first-entry` (`redis_broker.py:381`, `_check_overflow`) and raises the same `BrokerOverflowError`.

## Cross-process correctness (Redis only)

The Redis impl resolves a tricky impedance mismatch: the Protocol's `open_run` / `close_run` / `reap` are synchronous, but Redis admin needs network I/O for the side effect to be visible to other processes. The fix is a `_pending_admin` queue of zero-arg thunks (`redis_broker.py:214`) drained at the top of every `publish` / `stream` call via `flush()`. Callers who need the open/close sentinel observable cross-process *before* the next broker method runs can call `await broker.flush()` explicitly — chiefly: just before a publisher process exits.

The `:open` sentinel uses `SET ... EX <grace>` (`redis_broker.py:260`) so an abandoned producer eventually self-cleans. `is_open()` is process-local per Protocol; cross-process truth is `is_open_remote()` which reads the Redis sentinel directly (`redis_broker.py:283`).

There is one known SDK gotcha hardened against in code: `XREAD BLOCK` cancellation can leave the redis-py connection in an indeterminate state (upstream issue #2624). The `stream()` method catches `CancelledError` / `GeneratorExit` and force-evicts idle connections from the pool (`redis_broker.py:340–379`). Without this, a subsequent `publish` reads the stale XREAD response off the wire and sees `None` where it expected an INCR result.

## Metrics

Both impls expose `metrics()` returning `BrokerMetricsSnapshot` (`event_broker.py:100`):

```json
{
  "events_published": 1234,
  "events_streamed": 1200,
  "streams_dropped": 0,
  "runs_open": 1
}
```

The FastAPI `/metrics` endpoint (`arena/web/server.py:362`) dispatches on the concrete type — `metrics()` is intentionally **not** on the `EventBroker` Protocol, because each impl has a different counter surface (in-process: pure dataclass; Redis: an `await` on Redis state). At N=2 impls, an `isinstance` branch is less ceremony than a `BrokerMetrics` Protocol. The Redis import in the metrics handler is lazy so the slim install never has to import `redis` just to start the web server in in-process mode.

## Producer adapter

`BrokerEventSink` (`event_broker.py:343`) is the bridge between the existing sync `evaluation.event_log.EventSink.emit(event) -> None` API and the broker's async `publish`. It does:

```python
self.loop.call_soon_threadsafe(
    asyncio.create_task,
    self.broker.publish(self.run_id, event),
)
```

This keeps `emit()` non-blocking and preserves FIFO order per loop thread. The fork replay path uses it directly (`arena/web/forks.py:228`). The CLI path uses the higher-level `MultiRunBrokerSink` (`evaluation/duckdb_persister.py:99`), which auto-opens runs on first emit and routes each `run_id` to its own drainer — see [Chapter 16](./16-duckdb-persister-and-replay.md).

## The invariant you should not break

The in-process broker's overflow correctness rests on **single asyncio event loop semantics**: `publish` runs to completion between consumer `await`s, so a suspended consumer's cursor cannot be evicted mid-flight. Only on wake does it re-read `head_seq` and self-raise. Moving the broker itself to a thread pool would invalidate this. A future third backend (NATS? Kafka?) must either preserve the invariant or replace the eviction strategy. The contract tests in `tests/test_event_broker.py` will catch the visible drift, but a code reviewer needs to be aware of the invisible one.

## Related reading

- [Chapter 16](./16-duckdb-persister-and-replay.md) — DuckDB persister + cold-path reader (`stream_cold`).
- [Chapter 19](../part7-arena-web/19-web-architecture.md) — how the SSE backend consumes the broker.
- [Runbook: switching-broker-backend](../runbooks/switching-broker-backend.md) — operational checklist.
- [Runbook: debug-stuck-fork](../runbooks/debug-stuck-fork.md) — using `/metrics` to find a leak.
