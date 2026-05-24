# Runbook — Redis Broker Operations

Operational checklist for the Redis backend of the event broker. For the architecture rationale see [Chapter 15](../part6-evaluation-arena/15-event-broker.md) and [ADR 0002](../adr/0002-redis-streams-for-cross-process.md).

## Bring it up (compose stack)

The compose Redis is configured for AUTH with `REDIS_PASSWORD` (see `docker-compose.yml`):

```bash
# Generate a password (and the other compose secrets) if .env is fresh
{
  echo "REDIS_PASSWORD=$(openssl rand -base64 24 | tr -d '=+/')"
} >> .env   # plus the other vars from README.md "Synthetic Arena infrastructure"

# Start the stack — Redis healthcheck passes when redis-cli -a $REDIS_PASSWORD ping returns PONG
just arena-infra-up

# Verify
just arena-infra-status | grep -i redis
# Expected: arena-redis  Up X seconds (healthy)
```

The healthcheck command is `redis-cli -a "${REDIS_PASSWORD}" ping` (every 5s, 10 retries). If it never goes healthy, `docker compose logs redis` is the first stop — most often the `--requirepass` value mismatches what the healthcheck sends, which means `REDIS_PASSWORD` isn't being substituted (`.env` not loaded by your compose invocation).

## Point the agent at it

Two paths, depending on what's already in your shell:

```bash
# Path 1 — your shell already has REDIS_PASSWORD (e.g. you sourced .env)
export ARENA_BROKER_BACKEND=redis
# make_broker() auto-builds redis://:${REDIS_PASSWORD}@localhost:6379/0
just arena-smoke   # or any arena CLI

# Path 2 — explicit URL
export ARENA_BROKER_BACKEND=redis
export REDIS_URL="redis://:$(grep ^REDIS_PASSWORD .env | cut -d= -f2)@localhost:6379/0"
just arena-smoke
```

`just arena-broker-redis-env` prints both export lines if you prefer `eval`-ing the recipe.

## Verify connectivity

```bash
# Round-trip
redis-cli -a "$REDIS_PASSWORD" ping
# PONG

# Inspect a live stream during an in-flight run
redis-cli -a "$REDIS_PASSWORD" --scan --pattern 'arena:run:*:events'
# arena:run:<uuid>:events

# Tail the head of a stream
redis-cli -a "$REDIS_PASSWORD" XINFO STREAM arena:run:<uuid>:events
# Shows length, first-entry, last-entry — first-entry is the head_seq the
# backend uses for overflow detection (see redis_broker.py:381)

# See which runs the broker thinks are open
redis-cli -a "$REDIS_PASSWORD" --scan --pattern 'arena:run:*:open'
```

The three key namespaces per run (`packages/evaluation/src/redis_broker.py:96`):

| Key | Type | Purpose |
|---|---|---|
| `arena:run:<uuid>:open` | string `"1"` with EX (6h default) | Cross-process "producer still alive" sentinel. |
| `arena:run:<uuid>:seq` | integer | `INCR` source for `Seq` values. |
| `arena:run:<uuid>:events` | stream (MAXLEN ~ 10000) | The actual event log. Stream ID is `<seq>-0`. |

## Rotate `REDIS_PASSWORD`

The compose Redis re-reads the password from env when the container restarts. The process is:

1. Generate a new password and update `.env` (`REDIS_PASSWORD=<new>`).
2. `docker compose up -d --force-recreate redis` (recreates only Redis, leaves Langfuse / Postgres alone).
3. Verify: `just arena-infra-status | grep redis` is healthy.
4. Restart anything that holds an open Redis connection (the arena web server, any in-flight CLI). The `make_broker()` factory reads `REDIS_PASSWORD` at call time, not import time, so a fresh process picks up the new value.

⚠️ The old password is invalidated immediately. Any process still holding an open connection authenticated with the old password will start getting auth errors on the next command — restart them. Langfuse will also need restarting if you rotate while it's running.

## Disk and memory limits

The compose Redis runs with `appendonly yes` and writes to the `redis-data` named volume. We don't set a `maxmemory` policy — the assumption is that arena runs are bounded in size (10k entries per stream × O(active runs)) and the host has enough RAM. If you start using this for higher-throughput workloads, set `--maxmemory` and pick a policy explicitly.

Persistence model: AOF (append-only file). Good for crash recovery, bad if you don't want history. To wipe everything: `just arena-infra-nuke` deletes the volume; on next `up` Redis starts empty.

## When something is wrong

| Symptom | Likely cause | Fix |
|---|---|---|
| `redis.exceptions.AuthenticationError` on first publish | `REDIS_PASSWORD` env var not set; `make_broker()` falls back to bare `redis://localhost:6379/0` which has no auth | Source `.env` or set `REDIS_URL` explicitly. |
| `redis.exceptions.ConnectionError: Error 61 connecting to localhost:6379` | Redis container not running, or not bound to `localhost` | `docker compose ps redis` — should be Up. Check `docker compose logs redis`. |
| `BrokerOverflowError` raised in consumers right after starting | Consumer asked for `from_seq` below stream's `first-entry` | Reconnect with `from_seq=available_from` from the overflow event (frontend does this automatically). |
| `XREAD` consumer hangs forever, never sees a published event | Publisher and consumer on different brokers (one on Redis, one on InProcess) | Both processes need `ARENA_BROKER_BACKEND=redis` and the same `REDIS_URL`. |
| Stream `XINFO` shows growing length but consumers see nothing | Connection-pool poisoning from a cancelled `XREAD BLOCK` (upstream redis-py #2624) | The broker has a workaround (`packages/evaluation/src/redis_broker.py:340–379`) that should handle this; if you still see it, file an issue with reproducer. |
| `flush()` never lands the `:open` DELETE | Pub or stream methods aren't being called after the sync `close_run()` | Call `await broker.flush()` explicitly before the producer process exits, OR let the FastAPI lifespan handle teardown. |

## Where things live

- Broker impl: `packages/evaluation/src/redis_broker.py`.
- Factory + env-var read: `packages/evaluation/src/broker_factory.py`.
- Healthcheck + compose config: `docker-compose.yml` (Redis service starts at line 155).
- Compose env-var template: `env.example` (look for `REDIS_PASSWORD`).
