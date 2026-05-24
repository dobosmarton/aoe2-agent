# Event Broker Architecture — Log-First Live Streaming

**Status:** Proposed
**Author:** Marton (planning session 2026-05-21)
**Supersedes:** ad-hoc `LiveRunRegistry` + `BroadcastingSink` (Phase 9 design)

---

## 1. Context

### 1.1 What prompted this

Phase 9 introduced operator-driven forks: `POST /forks` snapshots a parent run, optionally mutates the world, then spawns an async N-turn replay. The replay holds a read-write DuckDB connection to the child run's `.duckdb` file for the entire game loop. Meanwhile, the frontend immediately opens `GET /events?run_id=<child>` to live-tail the replay.

The first SSE call crashes:

```
_duckdb.ConnectionException: Connection Error: Can't open a connection to same
database file with a different configuration than existing connections
```

DuckDB's in-process invariant: every connection to a file in one process must share access mode. The replay holds it read-write; the SSE handler's `_resolve_run_optional` scans `logs/arena/*/*.duckdb` opening each `read_only=True` to find the run; collision on the active child file.

### 1.2 Why a small patch is the wrong fix

Two tactical options were considered (drop `read_only=True`, or route the live path past the scan by tracking `db_path` in the registry). Both work, both ship in half a day, both encode the same load-bearing assumption: **"readers open the writer's DuckDB file."** That assumption is fine in a single-process FastAPI app and instantly false the day any of these happen:

- Replay moves to a worker process (DuckDB's cross-process rule: one writer system-wide, period — no concurrent readers, even RO).
- Frontend dev box talks to a backend on a VM (already happening per project memory).
- Multiple materializers want the event stream (DuckDB + analytical OLAP + Langfuse traces — Langfuse containers are already deployed).
- Parallel forks fan out across machines.

Each of those is a forced rewrite of the SSE layer under deadline pressure if we bake in the file-coupling assumption now.

### 1.3 Intended outcome

A log-first architecture where the event log is the single source of truth for live consumers and DuckDB is one of N materializations. The bug disappears as a side effect; the race-window dedupe code disappears as a side effect; cold reads stay untouched; and the day we go distributed, the change is swapping one broker implementation for another, not rewriting the SSE handler.

---

## 2. North Star

> **The event log is the source of truth. DuckDB and SSE clients are two materializations that consume it. Neither knows about the other.**

Every design question below resolves against that sentence: does the answer keep producers and consumers coupled only through the log?

---

## 3. Conceptual Architecture

```
                            ┌──────────────────────┐
                            │       Producer       │
                            │ (synth_game_loop,    │
                            │  fork snapshot,      │
                            │  mutation events)    │
                            └──────────┬───────────┘
                                       │ publish(event)
                                       ▼
                            ┌──────────────────────┐
                            │     EventBroker      │
                            │  (Protocol; in-proc  │
                            │   impl today; Redis/ │
                            │   NATS swappable)    │
                            └─┬─────────────────┬──┘
                              │                 │
                              │ stream(         │ stream(
                              │  run_id,        │  run_id,
                              │  from_seq=0)    │  from_seq=0)
                              ▼                 ▼
                  ┌──────────────────┐  ┌──────────────────┐
                  │ DuckDBPersister  │  │  SSE Handler     │
                  │ (consumer)       │  │  (consumer)      │
                  └──────────────────┘  └──────────────────┘
```

Two consumers, symmetric. Neither opens the other's files. The producer never knows who is listening or whether persistence has caught up.

---

## 4. Core Abstractions

All new code lives in `evaluation/event_broker.py` (broker is a domain concern, not a web concern — the web layer depends on `evaluation`, never the reverse).

### 4.1 `EventEnvelope` — the unit of streaming

```python
# evaluation/event_broker.py
from dataclasses import dataclass
from typing import NewType
from evaluation.event_log import Event   # frozen dataclass; do not modify

RunId = NewType("RunId", str)
Seq   = NewType("Seq", int)   # monotonic & UNIQUE per run, assigned by broker

@dataclass(frozen=True, slots=True)
class EventEnvelope:
    run_id: RunId
    seq:    Seq
    event:  Event
```

**Why a wrapper, not a field on `Event`:** `Event` is a frozen domain dataclass owned by `evaluation/event_log.py` and used by all producers (race, ranking, synth_game_loop, fork). Wrapping isolates the broker's "I assigned this sequence number" concern from the domain. It also means no DuckDB schema change — `seq` is a runtime concept that lives only in the broker and the SSE wire format if needed.

**Why a separate `Seq` distinct from `t`:** confirmed by exploration — multiple events share the same `t` within a turn (~8 events per turn: turn_start, observation, llm_prompt, llm_response, action, action_result, world_mutation, metric). Today's `_live_event_stream` filters live-tailed events with `event.t <= max_t_seen`, which **silently drops legitimate same-turn events**. `Seq` makes ordering total and dedupe unambiguous.

### 4.2 `EventBroker` — the interface

```python
from typing import Protocol, AsyncIterator

class EventBroker(Protocol):
    """Pub/sub with replay-from-offset semantics.

    Producers call open_run → publish*N → close_run.
    Consumers call stream(run_id, from_seq) — one call, replay+tail.
    """

    def open_run(self,  run_id: RunId) -> None: ...
    def close_run(self, run_id: RunId) -> None: ...
    def is_open(self,   run_id: RunId) -> bool: ...

    async def publish(self, run_id: RunId, event: Event) -> Seq: ...

    def stream(
        self,
        run_id: RunId,
        from_seq: Seq = Seq(0),
    ) -> AsyncIterator[EventEnvelope]: ...
```

**Design choices and the principles behind them:**

| Choice | Principle |
|---|---|
| `Protocol`, not ABC | Structural typing; test doubles are plain classes — no inheritance, no `super()`, no `@abstractmethod` boilerplate |
| Explicit `open_run`/`close_run` | Lifecycle is contract, not implicit-on-first-publish; ownership testable |
| `stream` does replay + tail in one call | Kills the subscribe-then-read-then-dedupe-by-t dance; one concept not three |
| `publish` returns `Seq` | Producers get a receipt — useful for at-least-once durability later, immediately useful for tests |
| `RunId`/`Seq` NewTypes | Wrong-type-as-arg becomes a mypy error, not a runtime bug |
| No subscribe/unsubscribe API | Async generator's `finally` is the natural unregister; one concept not two |

### 4.3 `InProcessEventBroker` — Phase 1 implementation

```python
import asyncio
from collections import defaultdict
from itertools import count

class InProcessEventBroker(EventBroker):
    """Single-process, asyncio-native broker.

    Retains every event for the lifetime of the run (open_run → close_run).
    Late subscribers replay from the retained buffer, then tail live.

    Memory: O(events_per_run × concurrent_runs). For arena replays
    (hundreds of events per run), negligible.
    """

    def __init__(self) -> None:
        self._buffers: dict[RunId, list[EventEnvelope]] = {}
        self._seq:     dict[RunId, count]               = {}
        self._open:    set[RunId]                       = set()
        self._waiters: dict[RunId, list[asyncio.Event]] = defaultdict(list)

    def open_run(self, run_id: RunId) -> None:
        if run_id in self._open:
            raise ValueError(f"run {run_id!r} already open")
        self._buffers[run_id] = []
        self._seq[run_id]     = count(1)
        self._open.add(run_id)

    def close_run(self, run_id: RunId) -> None:
        self._open.discard(run_id)
        for w in self._waiters.pop(run_id, ()):
            w.set()
        # Buffer kept; reclaimed via explicit reap() if added later.

    def is_open(self, run_id: RunId) -> bool:
        return run_id in self._open

    async def publish(self, run_id: RunId, event: Event) -> Seq:
        if run_id not in self._open:
            raise RuntimeError(f"run {run_id!r} is not open")
        seq = Seq(next(self._seq[run_id]))
        self._buffers[run_id].append(EventEnvelope(run_id, seq, event))
        for w in self._waiters[run_id]:
            w.set()
        return seq

    async def stream(
        self,
        run_id: RunId,
        from_seq: Seq = Seq(0),
    ) -> AsyncIterator[EventEnvelope]:
        cursor = max(0, int(from_seq) - 1)
        while True:
            buf = self._buffers.get(run_id, [])
            while cursor < len(buf):
                yield buf[cursor]
                cursor += 1
            if not self.is_open(run_id):
                return
            waiter = asyncio.Event()
            self._waiters[run_id].append(waiter)
            try:
                await waiter.wait()
            finally:
                self._waiters[run_id] = [
                    w for w in self._waiters[run_id] if w is not waiter
                ]
```

**Clean-code properties:**

- Single responsibility — broker only brokers; no DuckDB/SSE/FastAPI imports.
- One shared append-only buffer + wake-up events, not N per-subscriber queues. Memory is O(events), not O(events × consumers).
- All deliveries are immutable frozen envelopes; the only mutable state is the buffer (append-only) and the lifecycle set.
- No `call_soon_threadsafe` here — thread-bridge concerns belong at the producer adapter, not inside the broker.

### 4.4 `BrokerEventSink` — adapter at the producer boundary

```python
@dataclass(frozen=True, slots=True)
class BrokerEventSink:
    """EventSink adapter: sync emit() bridges to async publish().

    The producer (synth_game_loop, fork) uses the existing sync EventSink
    Protocol. This adapter schedules publish onto the broker's loop.
    """
    broker: EventBroker
    run_id: RunId
    loop:   asyncio.AbstractEventLoop

    def emit(self, event: Event) -> None:
        # FIFO per loop preserves order across same-thread emits.
        self.loop.call_soon_threadsafe(
            asyncio.create_task,
            self.broker.publish(self.run_id, event),
        )
```

### 4.5 `DuckDBPersister` — DuckDB becomes a consumer

```python
# evaluation/duckdb_persister.py
async def persist_to_duckdb(
    broker: EventBroker,
    run_id: RunId,
    db_path: Path,
) -> None:
    """Subscribe from seq=0, write every event to DuckDB until close.

    Spawned as a background task alongside the producer. The persister
    is the SOLE owner of the DuckDB file — no other process or task
    opens it.
    """
    with duckdb.connect(str(db_path)) as conn:
        sink = DuckDBEventSink(conn)   # existing class; takes a conn
        async for env in broker.stream(run_id, from_seq=Seq(0)):
            sink.emit(env.event)
```

**This is the structural inversion that fixes everything:** nobody but the persister opens the writer's DuckDB file. The SSE handler doesn't touch DuckDB for live runs at all. The DuckDB collision bug is gone by construction.

### 4.6 Cold-run reader (post-finalize)

After `close_run`, the broker no longer serves the run. Cold-run readers go to DuckDB with `read_only=True` (safe — no in-process writer remains). To preserve the `EventEnvelope` contract end to end:

```python
# evaluation/event_log.py (or new evaluation/cold_stream.py)
def stream_cold(db_path: Path, run_id: RunId) -> Iterator[EventEnvelope]:
    """Read finalized events from DuckDB, assigning seq from row order."""
    with duckdb.connect(str(db_path), read_only=True) as conn:
        rows = conn.execute(
            "SELECT t, payload_json FROM events "
            "WHERE run_id=? ORDER BY t, rowid",   # rowid: same-t tiebreak
            [run_id],
        ).fetchall()
    for i, (_t, payload_json) in enumerate(rows, start=1):
        yield EventEnvelope(
            run_id=run_id,
            seq=Seq(i),
            event=_event_from_payload(payload_json),  # see verification §10
        )
```

`ORDER BY t, rowid` is doing real work — it fixes the same-turn ordering ambiguity the current code papers over. Cold and live consumers now share an identical contract.

---

## 5. The new SSE handler

```python
# arena/web/server.py
@app.get("/events")
async def events(
    run_id: str           = Query(..., min_length=1),
    broker: EventBroker   = Depends(get_broker),
) -> StreamingResponse:
    return StreamingResponse(
        _sse(broker, RunId(run_id)),
        media_type="text/event-stream",
    )

async def _sse(broker: EventBroker, run_id: RunId) -> AsyncIterator[str]:
    if broker.is_open(run_id):
        async for env in broker.stream(run_id, from_seq=Seq(0)):
            yield f"data: {env.event.payload.model_dump_json()}\n\n"
    else:
        db_path = await asyncio.to_thread(_resolve_cold_db, run_id, _logs_root())
        for env in stream_cold(db_path, run_id):
            yield f"data: {env.event.payload.model_dump_json()}\n\n"
```

Before vs after:

| Concern | Before | After |
|---|---|---|
| LOC in live-stream handler | ~30 | 2 |
| DuckDB opens per live request | 1 + N scan opens | 0 |
| Race-window dedupe | manual `t <=` filter (latently buggy) | none — `Seq` is unique + monotonic |
| Concept count | Subscription + queue + register + subscribe + unsubscribe | one: `stream` |
| Process-local assumption | hard-coded in handler | isolated to broker impl |

---

## 6. File / module layout

**New files:**
- `evaluation/event_broker.py` — `EventBroker`, `EventEnvelope`, `RunId`, `Seq`, `InProcessEventBroker`, `BrokerEventSink`
- `evaluation/duckdb_persister.py` — `persist_to_duckdb()` coroutine
- `tests/evaluation/test_event_broker.py` — broker contract tests + property tests
- `tests/arena/web/test_live_sse_during_replay.py` — regression test for the original bug

**Modified files:**
- `arena/web/server.py` — `_sse` handler in §5; `Depends(get_broker)`; `_resolve_run_optional`, `_stream_existing_rows`, `_live_event_stream` deleted
- `arena/web/forks.py` — `create_fork` uses `broker.open_run` + `await broker.publish(...)` for snapshot/mutation events; spawns `persist_to_duckdb` task; replay uses `BrokerEventSink`
- `evaluation/event_log.py` — possibly add `stream_cold()` helper (or new `cold_stream.py`)

**Deleted:**
- `arena/web/live.py` — `LiveRunRegistry`, `_Subscription`, `BroadcastingSink` are all superseded

---

## 7. Reused existing functions / utilities

| Existing | Where it lives | How the plan reuses it |
|---|---|---|
| `Event` dataclass | `evaluation/event_log.py:194` | Wrapped, never modified |
| `DuckDBEventSink` | `evaluation/event_log.py:241` | Used by the persister; lifecycle (caller-owned connection) preserved |
| `SCHEMA_VERSION`, `_CREATE_TABLE_SQL` | `evaluation/event_log.py` | Untouched — no schema migration |
| `fork()` (snapshot logic) | `evaluation/fork.py:73` | Unchanged; its emitted Event flows through the broker |
| `synth_game_loop` + `build_synth_invoke` | `arena/invoke.py`, etc. | Unchanged; the replay's sink swaps from `BroadcastingSink` to `BrokerEventSink` |
| FastAPI `Depends` injection | already used for `registry`/`fork_tasks` in `server.py` | Same pattern for `get_broker` — zero new infra |
| Pydantic v2 `model_dump_json` on `Payload` | `evaluation/event_log.py` | SSE serialization stays identical to today's `live.py:98` shape |

---

## 8. Migration phases

Each phase ships independently, stays reversible, and leaves the system green.

### Phase 0 — Prep (~0.5 day)

- Read `evaluation/event_log.py` end to end. Confirm `Event` reconstruction from `payload_json` works (today's `_stream_events_sync` does this — extract or call the same helper).
- Write the regression test in `tests/arena/web/test_live_sse_during_replay.py`. It should:
  1. Open a RW DuckDB connection to a fake child file, hold it.
  2. Hit `_live_event_stream` for that run.
  3. Today: fails with `ConnectionException`. Post-migration: passes.

### Phase 1 — Add the broker behind a feature flag (~2 days)

- Implement `evaluation/event_broker.py` and `evaluation/duckdb_persister.py`.
- Add `ARENA_BROKER_ENABLED` env var (default `false`). When true, `/events` uses the new path; when false, current path. This is the *only* shim in the entire plan and dies in Phase 2.
- Write tests:
  - Unit: lifecycle (open → publish → stream → close → re-stream is empty).
  - Unit: late subscriber replays from seq=1.
  - Unit: two concurrent subscribers see byte-identical sequences.
  - Property test (hypothesis): for any interleaving of `publish(N)` and `stream(K)`, every stream yields events 1..N in order, no gaps, no duplicates.
  - Integration: regression test from Phase 0 turns green when flag is on.

### Phase 2 — Cutover & delete (~1 day)

- Update `create_fork` (`arena/web/forks.py`):
  - Replace synchronous RW DuckDB write of snapshot+mutation events with `broker.open_run(child_run_id)` + `await broker.publish(...)`.
  - Spawn `asyncio.create_task(persist_to_duckdb(broker, child_run_id, child_db))`.
  - Replace `BroadcastingSink` with `BrokerEventSink` for the replay task.
  - On replay completion: `broker.close_run(child_run_id)`.
- Update `/events` handler to the simplified form in §5.
- Delete `arena/web/live.py` (and `from arena.web.live import …` references).
- Delete `_resolve_run_optional`, `_stream_existing_rows`, `_live_event_stream` from `server.py`.
- Delete `ARENA_BROKER_ENABLED`.
- All tests still pass. The regression test is now load-bearing — keep forever.

### Phase 2.5 — Roll non-fork producers onto the broker (~1 day, optional)

Today `arena/__main__.py`, `arena/race.py`, `arena/ranking.py` all write directly to DuckDB via `DuckDBEventSink`. They're not live-streamed today (no SSE consumer). Rolling them onto the broker:
- Lets the same SSE handler tail any in-flight arena run, not just forks.
- Unifies the producer story (everyone publishes; persister writes).
- Risk: low — just an EventSink swap at the call sites.

Defer until someone wants live UI for non-fork runs.

### Phase 3 — Hardening (~1 day, opt-in)

- **Backpressure:** bound the broker's per-run buffer to e.g. 10k envelopes; on overflow, drop the slow consumer (its `stream` generator raises). The client reconnects with `from_seq=last_seen` and the broker replays from the buffer — recovery is well-defined because of the seq design.
- **Memory reclamation:** `broker.reap(run_id)` after grace period post-close.
- **Metrics:** counters for `events_published`, `events_streamed`, `streams_dropped`, `runs_open`. Wire to `/metrics`.

### Phase C — Distributed swap (shipped, opt-in)

`RedisStreamsBroker(EventBroker)` implements the full Protocol against a Redis stream per run. Selected at process startup via `ARENA_BROKER_BACKEND=redis` (default remains `inprocess`); the only callsite change is `evaluation/broker_factory.make_broker()` reading the env. Cold-path readers continue using DuckDB; live-path readers transparently get cross-process semantics — a producer in process A and an SSE handler in process B share state through Redis keys.

Key design decisions:
- **Seq mapping**: `INCR :seq` + `XADD MAXLEN ~ N {seq}-0` keeps `Seq = NewType(int)` stable across both impls. Native Redis IDs (`<ms>-<n>`) would have forced a translation table at every consumer.
- **Liveness sentinel**: `SET :open 1 EX <grace>` written by `open_run`, `DEL` by `close_run`; `stream()` watches Redis (not the producer's `_open_locally`) so cross-process consumers terminate cleanly.
- **Sync lifecycle methods queue thunks** (`_pending_admin`) drained at the top of each `stream()` iteration and `publish()` — keeps the Protocol surface unchanged while honoring Redis's async-only client.
- **Codec reuse**: `_fields_to_event` packs Redis stream fields into the same `EventRow` tuple shape DuckDB returns, then delegates to `Event.from_row`. One deserializer for cold and live paths.
- **Metrics dispatch**: `metrics()` stays OFF the Protocol (impl-specific surface). The `/metrics` endpoint dispatches via `isinstance` — at N=2 impls, simpler than a new `BrokerMetrics` Protocol.

NATS JetStream was the alternative considered and explicitly deferred: Redis was already in `docker-compose.yml` for Langfuse, so Phase C added zero infra. Backend choice can be revisited as a Phase D follow-up.

---

## 9. Testing strategy

Three layers, three failure modes:

| Layer | Proves | Doesn't prove |
|---|---|---|
| Unit (broker) | Pub/sub semantics, ordering, lifecycle, replay correctness | Wiring |
| Contract (parametrized) | Any `EventBroker` impl satisfies the same property tests | Performance |
| Integration | `create_fork` + persister + SSE work together; no DB collision | Cross-process behavior |

**Contract test pattern** — the key investment for Phase C swap-ability:

```python
@pytest.mark.parametrize("broker_factory", [
    InProcessEventBroker,
    # RedisStreamsBroker,    # when implemented
])
@pytest.mark.asyncio
async def test_broker_contract(broker_factory):
    broker = broker_factory()
    broker.open_run(RunId("r1"))
    for i in range(5):
        await broker.publish(RunId("r1"), _make_event(t=i))
    broker.close_run(RunId("r1"))

    seqs = [env.seq async for env in broker.stream(RunId("r1"), Seq(0))]
    assert seqs == [Seq(1), Seq(2), Seq(3), Seq(4), Seq(5)]
```

Property test (hypothesis): given any sequence of `publish`/`stream` operations interleaved, every stream consumer must observe events 1..N in order, exactly once, with no gaps. This is the invariant that defines correctness — if it holds, the implementation details below it don't matter.

---

## 10. Verification — how to confirm the plan worked

**Local end-to-end smoke (after Phase 2):**

1. `just arena-infra-up` — start backing services.
2. Start backend with `ANTHROPIC_API_KEY` set; start frontend (`npm run dev` in `arena/web/ui`).
3. In the UI: pick a finalized parent run, scrub timeline to a turn boundary, hit fork.
4. Observe in DevTools Network tab:
   - `POST /forks` → 200 with `child_run_id`.
   - `GET /events?run_id=<child>` → 200 stays open, SSE events arrive within ~1 sec.
5. Backend logs: **no `ConnectionException`**, no stack traces. Persister log line per turn.
6. After replay completes (~30 sec), open a fresh tab and request `/events?run_id=<child>` — should fall through to the cold path and replay the full event tape from DuckDB.

**Automated:**

- `pytest tests/evaluation/test_event_broker.py -v` — broker contract green.
- `pytest tests/arena/web/test_live_sse_during_replay.py -v` — regression test green (was the original bug).
- `pytest` (full suite) — no other regressions; `test_web_forks.py` updated to test against the broker, not `LiveRunRegistry`.

**Distributed verification (Phase C — shipped):**

- `pytest tests/test_event_broker.py -v` runs the 20-test contract suite against **both** impls (~41 cases total) via the parametrized `_BROKER_FACTORIES` fixture. The Redis row uses `fakeredis.aioredis.FakeRedis` so CI doesn't need a real Redis.
- `pytest tests/test_redis_broker.py -v` covers Redis-only concerns: codec round-trip, MAXLEN trimming, XINFO-based overflow detection, metrics counters, `is_open_remote()` sentinel lifecycle.
- `pytest tests/test_redis_broker_cross_process.py -v` is the headline verification: a `multiprocessing.Process` publisher emits 100 events; the parent test process consumes them via a separate `RedisStreamsBroker` instance against the same real Redis, asserts byte-identical Seq sequence. Gated on `redis://localhost:6379/0` reachability (skips cleanly when no Redis is running).

---

## 11. Open verification items (resolve at Phase 0 kickoff)

Things confirmed by exploration on 2026-05-21:
- ✅ `Event` is a frozen dataclass; `Event.payload` is Pydantic v2.
- ✅ `DuckDBEventSink.__init__` takes a caller-owned connection; no internal close.
- ✅ Schema in `evaluation/event_log.py:223-232`; no migration needed.
- ✅ Multiple producers exist; Phase 1 scopes to the fork path.
- ✅ Frontend uses `turn_num`, not `t`; backend is free to evolve `t`/`seq` semantics.
- ✅ Only `tests/test_web_forks.py` references `LiveRunRegistry`; one test file to update.

Things to verify on first read in Phase 0:
- Exact helper used today to reconstruct `Event` from `payload_json` (likely in `_stream_events_sync` in `server.py`) — extract for reuse in `stream_cold()`.
- Whether `_event_from_payload` needs the `kind` column too (the schema has it; today's helper may or may not).

Resolved during Phase C (2026-05-23):
- ✅ `RedisStreamsBroker` satisfies the contract test suite under `fakeredis`; the parametrized fixture at `tests/test_event_broker.py:55` now sweeps both impls.
- ✅ Cross-process publish/consume verified end-to-end in `tests/test_redis_broker_cross_process.py` (real Redis, `multiprocessing.spawn` subprocess).
- ✅ Existing `Event.from_row` + `payload.model_dump_json()` were enough — no `event_codec.py` extraction was needed (the conditional extraction in the Phase C plan was correctly skipped).

---

## 12. Effort estimate

| Phase | Effort | Status |
|---|---|---|
| 0 — Prep & regression test | 0.5 day | ✅ shipped |
| 1 — Broker + persister + tests | 1.5–2 days | ✅ shipped (`d408401`) |
| 2 — Cutover & deletions | 1 day | ✅ shipped (`74eb7c6`) |
| 2.5 — Non-fork producers (optional) | 1 day | ✅ shipped (`60da488`) |
| 3 — Hardening (optional) | 1 day | ✅ shipped (`6b53a0c`) |
| C — Distributed swap (Redis Streams) | 1 day | ✅ shipped |
| **Total to fix the bug correctly** | **~5 working days** | |

For reference: the tactical patch ("option B") was ~0.5 day, at the cost of locking in the file-coupling assumption.

---

## 13. Cross-walk: Python & clean-code principles

| Principle | Where it shows up |
|---|---|
| Depend on abstractions (DIP) | `EventBroker` Protocol; `Depends(get_broker)` |
| Single Responsibility | Broker, persister, SSE handler each do one thing |
| Open/Closed | New broker impls require zero consumer changes |
| Composition over inheritance | `BrokerEventSink` has-a broker; doesn't subclass it |
| Immutability where possible | `EventEnvelope` frozen+slotted; events append-only |
| Explicit lifecycle | `open_run` / `close_run` instead of implicit-on-first-publish |
| One concept, not two | `stream` replaces subscribe + queue + iterate |
| Make illegal states unrepresentable | `RunId`, `Seq` NewTypes; broker rejects publish-before-open |
| Type the boundaries | `Protocol`, `NewType`, `AsyncIterator[EventEnvelope]` |
| Tests are documentation | Contract test is the broker spec |
| Delete code, don't accumulate it | Phase 2 step explicitly enumerates deletions |
