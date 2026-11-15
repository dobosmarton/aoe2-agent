# ADR 0001: broker-first event architecture

Status: accepted.

## Context

An arena run has several readers. The DuckDB log keeps every event for post-mortem queries, ranking and forks. The web UI shows a run while it plays and must resume after a dropped connection without missing or repeating events. Fork replays started from the web API produce runs of their own that the same UI follows.

DuckDB allows one read-write process per file, and a read-only open from another process fails while that writer holds the file. Any design in which live readers open the file being written breaks as soon as the producer and the reader are different processes, as the arena CLI and the web server are.

## Decision

Producers publish events to an `EventBroker` (`packages/evaluation/src/event_broker.py`), and every reader is a broker consumer that calls `stream(run_id, from_seq)`.

- The broker assigns each event a per-run `Seq` starting at 1 and delivers `EventEnvelope(run_id, seq, event)`. A consumer that reconnects passes the next `Seq` it needs.
- The DuckDB persister is one consumer. It is the only code that opens a run's file for writing, and the file is released once the run is closed and its stream drained.
- Live readers stream from the broker. Readers of finished runs use `stream_cold`, which opens the file read-only and numbers rows the same way, so both paths deliver the same envelope shape.
- `make_broker()` chooses the implementation, so producers and consumers do not depend on a backend.

## Consequences

- No reader opens a file that a writer holds. The web server checks `is_open_remote` and streams live runs from the broker and finished runs from DuckDB.
- Adding a consumer means one more `stream` subscription; producers do not change.
- Each producer needs an adapter (`BrokerEventSink`, `MultiRunBrokerSink`) and has to keep the lifecycle order: open the run before publishing, let queued publishes finish before `close_run`, wait for the persister before treating the file as complete, then `reap`.
- The broker keeps at most 10,000 events per run. A consumer that falls further behind gets `BrokerOverflowError` and has to accept a gap; the DuckDB file still has the full run.
- The in-process broker is correct only while all callers share one asyncio event loop.
- With the in-process broker, the web server cannot watch a run produced by a separate CLI process. That needs a cross-process backend ([ADR 0002](0002-redis-streams-for-cross-process.md)).

## Alternatives considered

| Option | Why it was rejected |
|---|---|
| The SSE handler reads the run's DuckDB file directly, polling for new rows | Fails while the writer holds the file in another process, and couples every reader to the file layout |
| A process-local registry that maps live runs to their files or forwards events to open SSE connections | Works only when producer and web server share a process, and leaves DuckDB as the point every reader depends on |

## Related

- [Event broker](../arena/event-broker.md)
- [Event log and replay](../arena/event-log-and-replay.md)
