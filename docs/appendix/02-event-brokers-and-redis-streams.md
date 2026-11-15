# Event brokers and Redis Streams

An event broker sits between the code that produces events and the code that reads them, so neither side has to know about the other. This primer covers the broker models, offsets, backpressure and the Redis Streams commands that the arena's [event broker](../arena/event-broker.md) is built on.

## Why a broker

Without a broker, a producer calls each consumer directly: it writes to the database, pushes to every open browser connection, and so on. Adding a consumer means changing the producer, and a slow consumer slows the producer down. With a broker, the producer publishes once and each consumer subscribes on its own. The broker holds events until the slower consumers catch up.

## Queue, pub/sub and log

| Model | Who receives a message | Late subscriber | Examples |
|---|---|---|---|
| Queue | Exactly one consumer from a pool | Gets only what is still queued; consumed messages are gone | RabbitMQ work queues, SQS, Celery |
| Pub/sub | Every subscriber connected at that moment | Sees nothing published before it connected | Redis `PUBLISH`/`SUBSCRIBE`, MQTT |
| Log | Every consumer, each at its own pace | Reads from any retained position | Kafka, Kinesis, Redis Streams, NATS JetStream |

A queue spreads work across workers. Pub/sub fans out with minimal latency but keeps no history. A log keeps events in order and lets each consumer track its own position, which is what a reconnecting viewer needs: it asks for everything after the last event it saw. The cost is retention: the broker has to store events and decide when to drop them.

The arena broker is a log with one stream per run. The DuckDB persister and each browser connection are independent consumers of the same run.

## Offsets

In a log, an offset is a consumer's position. The broker numbers entries in order, and a consumer asks for everything at or after a number. Kafka calls it an offset, Redis Streams uses entry IDs, and the arena calls it `Seq`. Because the consumer owns its offset, the broker does not track who has read what, and a consumer can reread from the start.

Offsets order events within one log only. Two runs are separate streams, and their numbers cannot be compared.

## Backpressure

When a producer is faster than a consumer, the backlog grows until something gives. There are three standard policies:

| Policy | Producer | Events | Typical use |
|---|---|---|---|
| Block the producer | Waits for space | None lost | Work queues where every message must be processed |
| Drop oldest | Never waits | Oldest discarded | Live views, where recent events matter most |
| Drop newest | Never waits | Incoming events discarded | Cases where the first events matter most |

The arena broker drops oldest, with a 10,000-event limit per run. A consumer whose next event has already been dropped receives an error naming the oldest event still available, so the gap is visible to it. The full history is in DuckDB.

## Redis Streams

A Redis stream is an append-only log stored under one key. Each entry has an ID and a small map of fields.

| Command | Does |
|---|---|
| `XADD key [MAXLEN ~ n] id field value ...` | Appends an entry. `*` lets Redis generate the ID; an explicit ID must be greater than the last one |
| `XREAD [BLOCK ms] STREAMS key id` | Returns entries with an ID greater than `id`. `BLOCK` waits up to `ms` for entries to arrive; `BLOCK 0` waits forever |
| `XRANGE key start end [COUNT n]`, `XREVRANGE` | Reads a range of entries, oldest or newest first |
| `XLEN key` | Number of entries |
| `XINFO STREAM key` | Length, first and last entry, and other metadata |
| `XTRIM key MAXLEN ~ n` | Trims the stream to about `n` entries |
| `XREADGROUP` | Reads as part of a consumer group, where each entry goes to one member of the group |

Points that matter in practice:

- Generated IDs have the form `<milliseconds>-<sequence>`. An application can supply its own IDs instead, as long as each is larger than the previous one. The arena writes `<seq>-0` so that the stream ID is the `Seq`.
- `MAXLEN ~ n` trims in whole internal nodes, so the stream can hold somewhat more than `n` entries. Exact trimming (no `~`) costs more.
- `XREAD` with `$` returns only entries added after the call; with `0` it starts at the beginning. A consumer that restarts with `$` misses everything published while it was away, so consumers that must not miss events keep their last ID.
- A finite `BLOCK` lets a consumer return periodically to check other conditions, such as whether the producer has finished.
- Consumer groups distribute entries across workers like a queue. A log where every consumer wants every event, like the arena's, uses plain `XREAD`.

Streams sit beside ordinary keys. A key written with `SET key value EX seconds` expires on its own, which makes it a liveness marker: if the producer dies without deleting it, the key disappears after the TTL. `SCAN` with `MATCH` walks the keyspace in batches without blocking the server, where `KEYS` does not.

## Durability

Redis keeps data in memory. With `appendonly yes`, as in the compose file, it also logs every write to an append-only file and replays it on restart, so streams survive a container restart. With `maxmemory-policy noeviction`, Redis refuses writes once a memory limit is reached instead of evicting keys; the compose file sets no limit, so streams can grow until host memory runs out. Untrimmed streams and forgotten keys therefore stay until something deletes them.

## Further reading

- Redis documentation, [Redis Streams](https://redis.io/docs/latest/develop/data-types/streams/).
- Martin Kleppmann, *Designing Data-Intensive Applications*, chapter 11, "Stream Processing".
- Jay Kreps, [The Log: What every software engineer should know about real-time data's unifying abstraction](https://engineering.linkedin.com/distributed-systems/log-what-every-software-engineer-should-know-about-real-time-datas-unifying).
