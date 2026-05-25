# Chapter 16 — DuckDB Persister and Replay

DuckDB is **one of N broker consumers**, not the primary store. The live story is the broker (Chapter 15); DuckDB just happens to be the consumer we run all the time, and it's the substrate for post-mortem queries, fork replays, and cold-path SSE for finalized runs.

The structural inversion this chapter documents — *nobody but the persister opens the writer's DuckDB file* — is what makes the rest of the architecture work. The original Phase 9 bug (concurrent RW/RO opens of the same DuckDB file crashing on the in-process invariant) disappears as a side effect of this rule.

## The events table

Schema lives in `packages/evaluation/src/event_log.py:255` and is unchanged since Phase 4:

```sql
CREATE TABLE IF NOT EXISTS events (
    run_id           VARCHAR,
    agent_id         VARCHAR,
    t                INTEGER,        -- turn number, 0-indexed
    kind             VARCHAR,        -- payload.kind discriminator
    payload_json     VARCHAR,        -- Pydantic .model_dump_json()
    ts               TIMESTAMP,
    schema_version   INTEGER         -- payload schema version, currently 1
)
```

Forward compatibility lives in `schema_version`. Multiple `run_id`s share one table (CLI commands write many run_ids per file), so every query filters by `run_id`.

### The 9 payload kinds

Each `kind` maps to a frozen Pydantic model defined in `packages/evaluation/src/event_log.py:115–175`:

| Kind | Purpose | Carries |
|---|---|---|
| `turn_start` | Begin a turn (and snapshot the world for fork) | `turn_num`, optional `WorldStateSnapshot` |
| `observation` | What the agent perceived | `entity_count`, sorted unique `classes` |
| `llm_prompt` | What the LLM saw | human-readable `state_summary` |
| `llm_response` | What the LLM said | `actions`, `reasoning`, `cost_usd` |
| `action` | One action being executed | `index_in_turn`, raw `action` dict |
| `action_result` | What changed | `index_in_turn`, `action_type`, `state_changed` |
| `world_mutation` | Operator-driven world edit (fork mutate) | `before_summary`, `after_summary`, `reason` |
| `fork` | This run was branched from another | `parent_run_id`, `parent_t`, `mutation_summary` |
| `metric` | Free-form named scalar (ranking emits these) | `name`, `value` |

The discriminated `Payload` union (`event_log.py:178`) is validated by a single cached `TypeAdapter` — same adapter used by both DuckDB row-loading and Redis stream-entry decoding (see Chapter 15's "the wire format" note).

## The two persister flavours

`packages/evaluation/src/duckdb_persister.py` exports two related primitives:

### Fork case — one run, one file

```python
broker.open_run(run_id)
asyncio.create_task(persist_to_duckdb(broker, run_id, db_path))
# producer publishes ...
broker.close_run(run_id)   # persister drains and returns
```

`persist_to_duckdb` (`packages/evaluation/src/duckdb_persister.py:59`) opens the DuckDB file exclusively, subscribes from `Seq(0)` so no events are missed, and exits when the broker closes the run. Used by the fork endpoint (`apps/api/src/forks.py:324`).

### CLI case — many runs, one file

```python
broker = make_broker()
with duckdb.connect(db_path) as conn:
    db_sink = DuckDBEventSink(conn)
    sink = MultiRunBrokerSink(broker, db_sink, asyncio.get_running_loop())
    try:
        await producer(sink)
    finally:
        await sink.close_all()
```

`MultiRunBrokerSink` (`duckdb_persister.py:99`) absorbs the mismatch between CLI producers (which surface `run_id`s only at emit time, inside `synth_game_loop`) and the broker's "open before publish" lifecycle. On first sight of a new `event.run_id` it:

1. `broker.open_run(rid)` — must come first; publish raises on closed runs.
2. Spawns `persist_to_duckdb_via_sink(broker, rid, db_sink)` to drain into the *shared* DuckDB connection.
3. Schedules the publish via `loop.call_soon_threadsafe` so the open-or-publish dance is atomic on the loop thread.

A single shared `DuckDBEventSink` keeps the "one file per CLI command" invariant — multiple `run_id`s naturally share the `events` table via the `run_id` column. Per-run persister coroutines drain in parallel but `emit` serializes on the loop thread anyway.

### `close_all` is load-bearing

`MultiRunBrokerSink.close_all()` (`duckdb_persister.py:155`) is what makes CLI flows leak-free. It:

1. Does a **two-tick `asyncio.sleep(0)` drain** so the queued `call_soon_threadsafe` callbacks fire and the publish tasks they schedule get to run. (Same pattern as `apps/api/src/forks.py:_replay`.)
2. `await gather`s any still-pending publishes — otherwise a publish-after-close would raise.
3. Closes every opened run, awaits every persister, and finally `reap`s — a single process can run thousands of `synth_game_loop` invocations through this sink and never accumulate per-run state past `close_all`.

The server uses a grace-period reaper instead (see Chapter 19); the CLI process is exiting so no grace is needed.

## The cold-path reader

For runs that have been closed and reaped from the broker buffer, consumers fall back to `stream_cold` (`packages/evaluation/src/event_log.py:305`):

```python
def stream_cold(db_path: Path, run_id: str) -> Iterator[EventEnvelope]:
    with duckdb.connect(str(db_path), read_only=True) as conn:
        rows = conn.execute(
            "SELECT * FROM events WHERE run_id=? ORDER BY t, rowid", [run_id]
        ).fetchall()
    for i, row in enumerate(rows, start=1):
        yield EventEnvelope(run_id=RunId(run_id), seq=Seq(i), event=Event.from_row(row))
```

Three things to know:

- `ORDER BY t, rowid` is load-bearing. Same-turn events share `t`; `rowid` is DuckDB's stable insert-order tiebreak. Without it, replays silently reorder events that share a turn number.
- `Seq` is reassigned from row index (1-indexed). Cold readers see the same envelope shape the broker delivers live — that uniformity is what lets the SSE endpoint switch transparently between the broker (live runs) and `stream_cold` (finalized runs); see `apps/api/src/server.py:330` (`events()`).
- The caller must guarantee no in-process writer holds `db_path` RW. The persister's "exclusive RW for the run's lifetime" rule is the other half of this contract.

`Event.from_row` (`event_log.py:217`) is the inverse of `DuckDBEventSink.emit` — it validates `payload_json` through the cached `TypeAdapter` and ignores the redundant `kind` column (the discriminator inside the JSON is the source of truth).

## Forking off a turn

`packages/evaluation/src/fork.py:73` (`fork`) is the primitive every fork flow goes through:

```python
new_run_id, forked_state = fork(
    conn=parent_conn,
    parent_run_id=parent_run_id,
    parent_t=parent_t,
    sink=event_sink,
    mutation_fn=None,        # or a Callable[[WorldState], WorldState]
)
```

It reads the `WorldStateSnapshot` embedded in the parent's `turn_start` event (every `turn_start` after Phase 5 carries one — `event_log.py:115`), optionally applies a mutation, emits a `ForkPayload` tagged with the new `run_id`, and returns `(new_run_id, forked_state)` ready to feed into `synth_game_loop`.

`ForkError` is raised when the parent has no `turn_start` at the requested `t`, or when the `turn_start` is a Phase 4 legacy event with no snapshot. Schema versioning is what lets us catch this cleanly instead of silently restoring an empty world.

The HTTP wrapper around `fork()` plus the `n_turn` async replay is in `apps/api/src/forks.py` — see [Chapter 19](../part7-arena-web/19-web-architecture.md) and [Chapter 20](../part7-arena-web/20-fork-and-diff-ui.md).

## Why the persister is "just a consumer"

Reframing DuckDB as a broker consumer (rather than the source of truth) buys three properties that the original Phase 9 design didn't have:

1. **No file-mode collisions.** Only the persister opens the writer's file. SSE reads from the broker for live runs; the cold path opens read-only only for finalized runs.
2. **N consumers, same wire.** Adding a Langfuse mirror or an OLAP store is a new `async for envelope in broker.stream(...)` coroutine — no producer changes, no DuckDB churn.
3. **Cold and live look the same to readers.** Same `EventEnvelope` shape, same `Seq` semantics, swap is transparent in `apps/api/src/server.py:events()`.

## Related reading

- [Chapter 15](./15-event-broker.md) — the broker Protocol the persister consumes.
- [Chapter 20 — Fork and Diff UI](../part7-arena-web/20-fork-and-diff-ui.md) — the operator-facing surface that drives `fork()`.
- [`docs/design/event-broker-architecture.md`](../design/event-broker-architecture.md) §C — the original payload-kind table (frozen historical spec).
