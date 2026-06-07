# Appendix B — Event brokers and Redis Streams

This appendix is the reference behind [Chapter 15 (Event Broker)](../part6-evaluation-arena/15-event-broker.md), [Chapter 16 (DuckDB Persister and Replay)](../part6-evaluation-arena/16-duckdb-persister-and-replay.md), and [ADR 0001 (Broker-First Architecture)](../adr/0001-broker-first-architecture.md). The chapters describe *what we built*; this appendix explains *the family of solutions our broker is one member of, why Redis Streams beat the alternatives at our scale, and the parts of distributed-systems theory you need to be comfortable with the design.*

If you're new to event-driven architectures, read this first. If you've shipped one before, skim to §B.5 for the Redis Streams operational details.

---

## B.1 The problem an event broker solves

You have a *producer* of events (in our case: a running game agent emitting `turn_start`, `observation`, `llm_response`, `action`, … events as it plays) and one or more *consumers* (in our case: the DuckDB persister writing to disk, the SSE endpoint pushing to browsers, the ranking pipeline computing scores, plus future consumers we haven't built yet).

The naive design connects producer to each consumer directly:

```
producer → DuckDB writer
producer → SSE clients
producer → ranking pipeline
```

This couples the producer to every consumer. Adding a consumer means changing the producer. A slow consumer slows the producer. A crashed consumer might lose events. Etc.

An **event broker** is a middle tier that decouples them:

```
producer → broker → DuckDB writer
                  → SSE clients
                  → ranking pipeline
```

The producer talks to *one* thing (the broker). Each consumer subscribes independently. Adding a new consumer is a new subscription — no producer changes. The broker absorbs the timing mismatch (a fast producer + slow consumer is fine as long as the broker can buffer).

---

## B.2 The three flavours: queue vs pub/sub vs log

Brokers come in three flavours that are easy to confuse because the words are used loosely.

### Queue (work distribution)

Each message is delivered to **exactly one consumer** out of a pool. RabbitMQ work-queues, AWS SQS, Celery. Use when you have N workers and want load-balanced work distribution.

| Strength | Weakness |
|---|---|
| Built-in load balancing across workers. | A consumed message is *gone* — you can't replay it for a new consumer later. |
| Backpressure is natural: workers ack at their own pace. | Inherently single-purpose: one queue per job type. |

### Pub/sub (broadcast, fire-and-forget)

Each message is delivered to **every subscriber that's currently connected**. Redis `PUBLISH`/`SUBSCRIBE`, MQTT, classic JMS topics. Late subscribers miss earlier messages — the broker doesn't persist anything.

| Strength | Weakness |
|---|---|
| Trivial fan-out. | No history. Connect late → see nothing. |
| Lowest possible latency. | If a subscriber crashes, the messages it was about to process are lost. |

### Log (append-only, replayable)

Each message is **persisted in order**; consumers track their own *offset* into the log and can rewind, replay, or join late. Kafka, AWS Kinesis, **Redis Streams**, NATS JetStream.

| Strength | Weakness |
|---|---|
| Replayable from any offset → late-joining consumers see the full history. | More storage (you're keeping events around). |
| Multiple independent consumers reading the same stream at different paces. | Operational complexity is higher than pub/sub. |
| Audit trail comes for free — every event ever produced is on disk. | Requires deciding on retention (how long to keep events). |

The log model is the only one that works for our case because the SSE endpoint needs to deliver every event a UI client missed during a reconnect, and the DuckDB persister, ranking pipeline, and future consumers all read at different paces. We need offset-based replay, not fire-and-forget broadcast.

---

## B.3 Why Redis Streams and not Kafka / NATS / Postgres LISTEN

We considered four log-shaped backends seriously.

### Kafka

The reference design. Excellent throughput (millions of events/sec), partition-based parallelism, ironclad durability, mature ecosystem. **Massive operational overhead** for our scale: a Kafka cluster needs Zookeeper (or KRaft), multiple broker nodes for HA, partition planning, schema-registry decisions. At our peak of maybe ~100 events/sec across all running games, Kafka is several orders of magnitude over-provisioned. A single developer running the system on a laptop should not need to operate a 3-node distributed cluster.

### NATS JetStream

Closer in scale to what we need. Lightweight, single binary, supports streams with replay. Genuinely a good choice. The reason we didn't pick it: we already had Redis as a dependency (for caching, rate limiting in the arena web), and adding a second piece of stateful infra felt like premature complexity. "Pick the boring backend you already run" was the decisive factor.

### Postgres LISTEN/NOTIFY (or row triggers as an event log)

Tempting because we have Postgres elsewhere. Two problems: `LISTEN/NOTIFY` is fire-and-forget (no replay), and using an `events` table as a stream means polling — slow, and prone to lock contention on the write side.

### Redis Streams (XADD / XREAD)

What we ship. Redis has had streams as a first-class data structure since 5.0 (2018). Combines pub/sub's simplicity with log semantics:

- `XADD stream * key value …` appends an event with a server-assigned monotonic ID (`<timestamp>-<seq>`).
- `XREAD BLOCK <ms> STREAMS stream <id>` reads events with ID > given (the consumer's offset). The `BLOCK` makes the call wait for new events instead of polling.
- `XREADGROUP` adds consumer-group semantics if you want load-balanced consumption within a group (we don't use this — every consumer wants every event).
- `XLEN`, `XTRIM`, `MAXLEN` for retention policy.

It's *zero operational overhead beyond the Redis we already run*, has the right semantics, and is fast enough that we won't outgrow it for several orders of magnitude.

See [ADR 0002](../adr/0002-redis-streams-for-cross-process.md) for the formal decision write-up.

---

## B.4 Backpressure: the thing every broker eventually forces you to think about

If your producer is consistently faster than the slowest consumer, *something* has to give. There are three policies, none of which is wrong, but you have to pick one consciously.

### Block the producer (lossless, slow)

The producer's `publish()` call waits until the broker has buffer space. Memory-safe, never loses events, but a slow consumer slows the whole system. Default in most queue-based brokers.

### Drop oldest (lossy, fast)

The broker enforces a max buffer size; when full, it discards the oldest events to make room. The producer never blocks. Events are lost, but only the events that were already stale.

### Drop newest (lossy, fast, different failure mode)

When full, discard the *new* event instead of the old. Useful when older events represent more important state (e.g., the first error matters more than later ones).

**Our policy:** drop-oldest with high-water-mark logging. Redis Streams' `XADD … MAXLEN ~ N` does this natively — `~` is "approximate trim" (faster). The broker emits an `overflow` event when trimming happens so we can detect persistent backpressure problems. See `packages/evaluation/src/event_broker.py` for the implementation.

The reason drop-oldest is right for our system: every event eventually lands in DuckDB *anyway* (the persister is the slow consumer; the broker is the hot buffer). The broker only needs to hold the last few seconds of events for live SSE subscribers to catch up after a reconnect. Older events are queryable from DuckDB.

---

## B.5 Operational gotchas with Redis Streams

A practical checklist of things that bit us, distilled:

- **`XREAD BLOCK 0` vs `BLOCK <ms>`** — `0` is "block forever." Use a finite timeout (we use 5s) so your consumer can periodically check its own shutdown signal and gracefully exit. Otherwise `Ctrl-C` won't deliver until the next event arrives.
- **The starting ID matters.** `$` means "from now on, only new events." `0` means "from the start of time." A consumer that crashes and restarts with `$` will miss everything that happened during the outage. Always persist the last-seen ID somewhere if you care about not missing events.
- **`MAXLEN ~ N` is approximate.** The `~` lets Redis trim in efficient chunks; you'll occasionally find slightly more than N entries. Don't rely on exact bounds. For exact: drop the `~`, but pay a performance cost.
- **Connection pooling.** The `redis-py` async client has a known pitfall (SDK issue #2624) where a single connection blocked in `XREAD` can starve other operations. We use a separate connection per long-lived subscription.
- **Stream IDs are monotonic per-stream, not globally.** Don't try to merge IDs across streams expecting global ordering.

---

## B.6 The single-loop invariant (and why it matters)

A subtle but important property of our broker: **all `publish()` and subscriber dispatch happens on a single asyncio event loop**. This is what makes the in-process broker correct without locks.

When the persister and SSE endpoints subscribe, they get an `async for envelope in broker.stream(run_id)` coroutine. The broker delivers events by `await queue.put(envelope)` — a coroutine yield, not a synchronous call. As long as everyone is on the same loop, "publish then read" is sequentially consistent without explicit synchronization.

The CLI breaks this invariant when `synth_game_loop` runs in a worker thread. That's why `MultiRunBrokerSink` uses `loop.call_soon_threadsafe(...)` to marshal cross-thread publishes back onto the main loop. See [Chapter 16 §"close_all is load-bearing"](../part6-evaluation-arena/16-duckdb-persister-and-replay.md) for the details.

The Redis backend is naturally cross-process — `XADD` from any process is safe — so the single-loop concern doesn't apply once you switch to Redis. But the in-process broker is faster (no serialization, no network), and you need to understand the invariant to know when each is safe.

---

## B.7 When does this design break?

A few growth scenarios that would force a redesign:

- **>10k events/sec sustained.** Redis Streams can do it on beefy hardware, but you'd start wanting Kafka's partition-based parallelism. Not a concern today.
- **Cross-region consumers.** Redis replication is async and not the same shape as Kafka mirror-maker. If you need a consumer in another datacenter, Kafka or a managed log service (Confluent Cloud, AWS MSK, Warpstream) becomes attractive.
- **Strict at-least-once semantics under broker crashes.** Redis Streams persists if AOF is configured (which we do), but the durability story is weaker than Kafka's quorum-replicated log. For payment-grade workloads, this would matter.

For an evaluation arena running on dev laptops and one small VPS, none of these apply.

---

## Further reading

- Martin Kleppmann, *Designing Data-Intensive Applications*, Chapter 11 ("Stream Processing"). The reference textbook.
- Redis docs: [*Introduction to Redis Streams*](https://redis.io/docs/data-types/streams/) — short, well-written.
- Jay Kreps, [*The Log: What every software engineer should know about real-time data's unifying abstraction*](https://engineering.linkedin.com/distributed-systems/log-what-every-software-engineer-should-know-about-real-time-datas-unifying) (LinkedIn 2013) — the essay that popularized the log-shaped broker model.

---

**Cross-references:**

- [Chapter 15 — Event Broker](../part6-evaluation-arena/15-event-broker.md) — our `BrokerProtocol`, in-process vs Redis backends, `/metrics` surface.
- [Chapter 16 — DuckDB Persister and Replay](../part6-evaluation-arena/16-duckdb-persister-and-replay.md) — the cold-path consumer that reads from the broker.
- [ADR 0001](../adr/0001-broker-first-architecture.md) and [ADR 0002](../adr/0002-redis-streams-for-cross-process.md) — the formal decisions.
- [Runbook — Redis broker operations](../runbooks/redis-broker-ops.md) — when something is wrong with Redis in production.
