# ADR 0001 — Broker-first event architecture

**Status:** Accepted (2026-05). Shipped through Phases 0–C.
**Context:** [`docs/design/event-broker-architecture.md`](../design/event-broker-architecture.md) (frozen full spec).

## Decision

Make the event broker the single source of truth for live event consumers. DuckDB and the web UI are symmetric materializations of the broker stream — neither knows about the other. Delete the old `live.py` shim and `LiveRunRegistry` / `BroadcastingSink`.

## What we considered

1. **Status-quo patch** — drop `read_only=True` from the SSE handler's DuckDB scan, or route the live path past the scan by tracking `db_path` in a registry.
2. **Broker-first redesign** — abstract producers and consumers behind an `EventBroker` Protocol, push DuckDB to the side as one of N consumers.

## Why broker-first

The status-quo patches both work and both ship in half a day. They also both encode the load-bearing assumption that **"readers open the writer's DuckDB file."** That assumption is fine in a single-process FastAPI app and instantly false the day any of these happen:

- Replay moves to a worker process. DuckDB's cross-process rule: one writer system-wide, no concurrent readers — not even RO.
- Frontend dev box talks to a backend on a VM (already happening, per past project context).
- Multiple materializers want the event stream (DuckDB + analytical OLAP + Langfuse).
- Parallel forks fan out across machines.

Each of those is a forced rewrite of the SSE layer under deadline pressure if we bake in the file-coupling assumption now. The broker-first design moves that assumption out of the SSE layer entirely.

## Consequences

**Positive**
- The Phase 9 DuckDB collision bug disappears as a side effect — only the persister opens the writer's file.
- Race-window dedupe code disappears as a side effect — `Seq` ordering is total per-run.
- The day we go distributed, swapping in `RedisStreamsBroker` is a one-line factory change (it shipped as Phase C; see [ADR 0002](./0002-redis-streams-for-cross-process.md)).
- Per-run cold reads stay untouched (`stream_cold` over read-only DuckDB).

**Negative**
- One more abstraction layer; producers now go through `BrokerEventSink` instead of writing to DuckDB directly.
- The single-loop invariant on `InProcessEventBroker` (publishers can't run between consumer awaits) is invisible until you try to add a thread-pooled backend; the test suite catches the visible part but reviewers must be aware of the invisible part.

## Sequencing

- **Phase 0** — `Event.from_row` + DuckDB collision regression test.
- **Phase 1** — `EventBroker` Protocol + `InProcessEventBroker` + DuckDB persister behind `ARENA_BROKER_ENABLED`.
- **Phase 2** — Cutover. Delete `live.py`. Remove the gate; broker is unconditional.
- **Phase 2.5** — Roll `race` / `smoke` / `rank` CLIs onto the broker.
- **Phase 3** — Bounded buffers, explicit `reap()`, `/metrics` endpoint, overflow backpressure semantics.
- **Phase C** — Redis Streams broker + fakeredis contract tests for byte-for-byte equivalence.

## Related

- [Chapter 15 — Event Broker](../part6-evaluation-arena/15-event-broker.md) — current-state reference.
- [ADR 0002 — Redis Streams](./0002-redis-streams-for-cross-process.md) — the Phase C backend choice.
