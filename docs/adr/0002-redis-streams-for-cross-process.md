# ADR 0002: Redis Streams as the cross-process broker

Status: accepted.

## Context

The in-process broker from [ADR 0001](0001-broker-first-architecture.md) lives inside one Python process. The arena CLI and the web server are separate processes, so showing a CLI run live in the dashboard needs a backend that both can reach and that keeps the `EventBroker` contract: per-run `Seq` from 1, replay from any `Seq`, a bounded buffer with a detectable overflow, and an open or closed status that every process can read. The compose stack in `docker-compose.yml` already runs Redis for Langfuse.

## Decision

`RedisStreamsBroker` (`packages/evaluation/src/redis_broker.py`) implements `EventBroker` with one Redis stream per run. `ARENA_BROKER_BACKEND=redis` selects it.

- `publish` takes the next `Seq` from `INCR arena:run:<id>:seq` and writes the event with `XADD arena:run:<id>:events MAXLEN ~ 10000` and the explicit ID `<seq>-0`, so stream IDs and `Seq` are the same number.
- `stream` reads with `XREAD BLOCK 100` from the last delivered ID and detects overflow from the stream's first entry.
- `open_run` writes `arena:run:<id>:open` with a 6-hour TTL, holding the run's label and start time as JSON. `is_open_remote` and `live_runs` read these keys.
- The `redis` package is imported only when this backend is selected.

## Consequences

- A CLI run is visible live in the web server, and any number of processes can consume a run.
- The contract test suite (`tests/test_event_broker.py`) runs against both implementations, the Redis one through `fakeredis`.
- The stack gains no extra service.
- The protocol's lifecycle methods are synchronous, so the Redis broker queues their `SET` and `DEL` until its next async call. A process that makes no further call after `reap` leaves the run's keys in Redis.
- Cancelling a blocked `XREAD` can leave a redis-py connection with an unread reply (redis-py issue #2624). The broker drops idle pool connections whenever a stream is cancelled.
- `INCR` and `XADD` are two round trips, so concurrent publishes to one run can reach `XADD` out of order, and Redis rejects the lower ID.
- `MAXLEN ~` trims approximately, and the metrics counters cover only the process that reports them.

## Alternatives considered

| Option | Why it was rejected |
|---|---|
| NATS JetStream | Fits the model, but adds a service and its credentials to a stack that already runs Redis |
| Kafka or Redpanda | Operational weight far beyond a handful of runs on one machine |
| Postgres `LISTEN`/`NOTIFY` | No replay: a notification reaches only listeners connected at that moment, and payloads are size-limited |
| Redis-generated stream IDs (`<ms>-<n>`) | Every consumer would need to translate them to `Seq`, and the two brokers would number events differently |

## Related

- [Event broker](../arena/event-broker.md)
- [Redis broker operations](../runbooks/redis-broker-ops.md)
