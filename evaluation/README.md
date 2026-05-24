# `evaluation/` — Event Broker, World Sim, Scenario Runner

Three responsibilities live in this package:

- **Event broker** — the live event substrate that every arena run flows through.
- **DuckDB persistence + fork primitive** — the cold-path materialization, and the snapshot-fork that powers the web UI's branching.
- **Scenario harness** — the multi-turn test runner (`runner.py`) for `evaluation/scenarios/*.yaml`.

## What's here

```
evaluation/
├── event_broker.py       # Protocol + InProcessEventBroker (Phase 1/3)
├── redis_broker.py       # RedisStreamsBroker (Phase C cross-process)
├── broker_factory.py     # make_broker() — single place ARENA_BROKER_BACKEND is read
├── event_log.py          # Event/Payload Pydantic types, DuckDBEventSink, stream_cold
├── duckdb_persister.py   # persist_to_duckdb, MultiRunBrokerSink (CLI case)
├── fork.py               # Snapshot-fork from a parent turn_start event
├── world_sim.py          # AoE2-lite economy + render() for synthetic perception
├── runner.py             # python -m evaluation.runner — multi-turn scenario harness
├── assertions.py         # Assertion DSL evaluated by runner
├── context_builder.py    # Builds executor context from a scenario fixture
├── fixture_builder.py    # Builds fixtures from real-game logs
├── log_to_scenario.py    # Converts a structlog game.txt into a scenario YAML
├── strategist_eval.py    # Standalone strategist evaluation harness
├── test_isolation.py     # _isolate_memories_dir, _mock_executor, _seed_detected_entities
└── scenarios/            # *.yaml — one fixture per file
```

## Common entry points

```bash
# Run a single scenario fixture
python -m evaluation.runner evaluation/scenarios/age_up_gate_fires.yaml

# Run every fixture (~$0.50 with default Haiku)
python -m evaluation.runner --all
# also: just eval-all
```

The arena CLIs (`arena/__main__.py`) consume the broker via `evaluation.broker_factory.make_broker()` and the persister via `evaluation.duckdb_persister.MultiRunBrokerSink`.

## Reading order

- Broker mechanics: [Chapter 15](../docs/part6-evaluation-arena/15-event-broker.md).
- DuckDB persister + the cold path + fork primitive: [Chapter 16](../docs/part6-evaluation-arena/16-duckdb-persister-and-replay.md).
- World sim economy + `render()` perception projection: [Chapter 18](../docs/part6-evaluation-arena/18-synthetic-world-sim.md).
- Scenario runner DSL: existing scenario YAMLs under `evaluation/scenarios/` plus `evaluation/assertions.py`.

## Invariants worth knowing

- **Only one writer per DuckDB file.** The persister owns the connection RW; readers (cold path, web UI) open read-only. Breaking this triggers DuckDB's cross-connection invariant — the bug that motivated [ADR 0001](../docs/adr/0001-broker-first-architecture.md).
- **`Seq` is per-run and totally ordered.** Multiple events can share `Event.t` (turn number); `Seq` resolves their ordering.
- **`event_log.py` and `world_sim.py` are leaf modules.** They don't import from `arena/` or `gameplay_agent/`. Keep it that way.
- **`render()` never touches global `random` state.** Local `random.Random(seed)`. Determinism is the point.
