# ADR 0002 — Redis Streams as the cross-process broker backend

**Status:** Accepted (2026-05). Shipped as Phase C.
**Context:** [ADR 0001](./0001-broker-first-architecture.md) created the broker abstraction. This ADR picks the cross-process backend.

## Decision

Use Redis Streams (`XADD` / `XREAD BLOCK` per run) as the second `EventBroker` implementation, behind `ARENA_BROKER_BACKEND=redis`. Use `INCR` for the `Seq` source and embed it as the left half of the stream ID.

## What we considered

| Option | Pros | Cons |
|---|---|---|
| **Redis Streams** | Already in the compose stack (Langfuse needs it). Native `XADD MAXLEN` matches the in-process deque. `BLOCK` makes consumers cheap. Single-binary ops. | Redis isn't a "real" message broker — replication semantics aren't NATS/Kafka grade. |
| **NATS JetStream** | Purpose-built for this; better delivery guarantees. | New ops surface; another container, another set of credentials. |
| **Kafka / Redpanda** | Industry standard. | Ridiculous overkill for our load; ops complexity dwarfs the architecture. |
| **Postgres `LISTEN/NOTIFY`** | Postgres already in stack. | NOTIFY drops messages on disconnect; no replay; payload size limit. |

## Why Redis Streams

- **Already deployed.** The compose stack runs Redis with `--requirepass ${REDIS_PASSWORD}` for Langfuse. Reusing the container is one shared dependency vs two.
- **The primitives map cleanly.** `XADD` is publish, `XREAD BLOCK` is consumer wait, `MAXLEN ~ N` is the bounded-buffer policy, custom `<seq>-0` IDs preserve our `Seq` identity. The whole impl in `packages/evaluation/src/redis_broker.py` is 458 lines including the cross-process `:open` sentinel and the SDK-bug workarounds.
- **Observable equivalence with in-process.** Same parametrized contract test (`tests/test_event_broker.py`) runs against both impls and passes byte-for-byte at the envelope level. That's the design's strongest guarantee.
- **Local dev story is fine.** `fakeredis` covers CI; real-Redis local work needs the `broker-redis` extra.

## Why we kept `Seq` from INCR (not native Redis IDs)

`Seq` is a `NewType("Seq", int)` starting at 1 and totally ordered. We could have used Redis-native `<ms>-<n>` IDs and translated at the consumer boundary, but every consumer would then need its own translation table. `INCR` is one round-trip per publish, embedded into the stream ID as `<seq>-0`, and the identity stays stable across both impls. Cleaner everywhere.

## Subtle correctness items we hit

These are documented in the source where they apply, but worth surfacing for ADR future-readers:

- **Sync lifecycle on async backend.** `open_run` / `close_run` / `reap` are Protocol-sync; Redis needs network I/O to be observable cross-process. Resolved with a `_pending_admin` thunk queue drained at the top of every async method via `flush()` (`packages/evaluation/src/redis_broker.py:225`). Sync methods stage the side effect; the next async call lands it.
- **`XREAD BLOCK` cancellation poisons the connection.** Upstream redis-py issue #2624. A cancelled consumer leaves a pending response on the socket; the next `publish` reads it and sees `None` where it expected an INCR result. The fix is `connection_pool.disconnect(inuse_connections=False)` on the `CancelledError` / `GeneratorExit` path (`redis_broker.py:340–379`). Only idle connections are evicted, so concurrent publishers aren't disrupted.
- **Cross-process truth via the `:open` sentinel.** `is_open()` is process-local per Protocol (sync). `is_open_remote()` does the Redis round-trip when callers (like the server's reaper loop) need the cross-process answer.

## Consequences

**Positive**
- Cross-process producer / consumer separation, which unlocks the FastAPI-on-VM + UI-on-laptop topology.
- Existing compose stack covers the production dependency.
- One `ARENA_BROKER_BACKEND` env var is the operator surface — see [Runbook: switching-broker-backend](../runbooks/switching-broker-backend.md).
- `REDIS_PASSWORD` auto-builds the URL when `REDIS_URL` isn't set (compose-stack default path).

**Negative**
- The `redis` Python client is an optional dep (good — slim install keeps working) but means a `broker-redis` extra to install.
- The two SDK gotchas above are real complexity we paid for in source comments and tests.
- A future third backend (NATS, Kafka) needs to pass the same contract tests *and* not break the single-loop-eviction invariant of the in-process impl.

## Related

- [Chapter 15 — Event Broker](../part6-evaluation-arena/15-event-broker.md) — current-state reference.
- [Runbook: redis-broker-ops](../runbooks/redis-broker-ops.md) — operational checklist.
