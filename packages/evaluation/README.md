# `evaluation/` — Event Broker, Persister, World Sim

Pure infrastructure: the live event substrate, the DuckDB cold-path
materialization, snapshot-fork, and the synthetic AoE2-lite economy
the arena races against. **Zero coupling to `gameplay_agent`** —
scenario-runner helpers that need the real agent live in `gameplay-agent`
now.

## What's here

```
packages/evaluation/src/
├── event_broker.py       # Protocol + InProcessEventBroker (Phase 1/3)
├── redis_broker.py       # RedisStreamsBroker (Phase C cross-process)
├── broker_factory.py     # make_broker() — single place ARENA_BROKER_BACKEND is read
├── event_log.py          # DuckDBEventSink + stream_cold (types live in `core`)
├── duckdb_persister.py   # persist_to_duckdb, MultiRunBrokerSink (CLI case)
├── fork.py               # Snapshot-fork from a parent turn_start event
└── world_sim.py          # AoE2-lite economy + render() for synthetic perception
```

Types (`Event`, `Payload`, `EventSink` Protocol, `WorldState`,
`DetectedEntity`) live in `core`. `event_log.py` re-exports them for
backwards-compat within `evaluation`; new consumers should
`from core import ...` directly.

## Common entry points

```bash
# Run a single scenario fixture (the runner lives in gameplay-agent now)
uv run --package gameplay-agent python -m gameplay_agent.scenario_runner \
    packages/gameplay-agent/src/scenarios/age_up_gate_fires.yaml

# Run every fixture
just eval-all
```

## Reading order

- Broker mechanics: [event broker](../../docs/arena/event-broker.md).
- DuckDB persister + the cold path + fork primitive: [event log and replay](../../docs/arena/event-log-and-replay.md).
- World sim economy + `render()` perception projection: [world sim](../../docs/arena/world-sim.md).

## Invariants worth knowing

- **Only one writer per DuckDB file.** The persister owns the connection RW; readers (cold path, web UI) open read-only. Breaking this triggers DuckDB's cross-connection invariant — the bug that motivated [ADR 0001](../../docs/adr/0001-broker-first-architecture.md).
- **`Seq` is per-run and totally ordered.** Multiple events can share `Event.t` (turn number); `Seq` resolves their ordering.
- **No back-edges to `gameplay-agent`.** This was the layering fix that unblocked the workspace split. Anything that needs `ExecutorProvider`, `AgentMemory`, etc. lives in `gameplay-agent` (or `arena` / `arena-web` for things that depend on both).
- **`render()` never touches global `random` state.** Local `random.Random(seed)`. Determinism is the point.
