# `core/` — Shared Types

Pure-Python leaf package with zero I/O. The only dependency is `pydantic`.
Every other package in the workspace consumes types from here; nothing here
imports from another workspace package. Breaking that rule re-introduces the
back-edges the broker rewrite was designed to eliminate.

## What's here

```
packages/core/src/
├── event_log.py     # Event, Payload union (9 kinds), EventSink Protocol, WorldStateSnapshot
├── world_state.py   # WorldState dataclass + AGE_SEQUENCE
└── entity.py        # DetectedEntity dataclass
```

These are the types that cross package boundaries:

- **`Event`** / **`Payload`** — the wire format every consumer reads and every producer writes. DuckDB and Redis materializations bind to this shape.
- **`EventSink`** Protocol — producer-side contract; `evaluation.DuckDBEventSink`, `evaluation.BrokerEventSink`, and the in-memory test sinks all implement it.
- **`WorldState`** — the canonical mid-game state of the synth arena. The simulator (economy, perception projection) lives in `evaluation.world_sim` and operates on this type.
- **`DetectedEntity`** — the schema YOLO inference emits AND synthetic perception projects into. Putting it here breaks the would-be cycle between `detection` and `evaluation`.

## Where to read more

- [Chapter 15 — Event Broker](../../docs/part6-evaluation-arena/15-event-broker.md) — how the `Event` / `EventSink` Protocol is consumed.
- [Chapter 16 — DuckDB Persister and Replay](../../docs/part6-evaluation-arena/16-duckdb-persister-and-replay.md) — the nine `Payload` kinds and their DuckDB schema.
- [Chapter 18 — Synthetic World Sim](../../docs/part6-evaluation-arena/18-synthetic-world-sim.md) — what consumers do with `WorldState`.
