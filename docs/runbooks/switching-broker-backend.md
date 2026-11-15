# Switching the broker backend

`ARENA_BROKER_BACKEND` selects the [event broker](../arena/event-broker.md) implementation in every process that calls `make_broker()`: the arena CLI and the web server. Run commands from the root of the agent repository.

| Backend | Value | Use when |
|---|---|---|
| `InProcessEventBroker` | unset or `inprocess` | Default. CLI runs, tests, and viewing finished runs or forks in the web UI |
| `RedisStreamsBroker` | `redis` | The web UI has to show a CLI run while it is still playing |

With the in-process backend the web server sees a CLI run only after the CLI exits and releases the DuckDB file. On Redis, events can go missing under concurrent publishes ([event broker](../arena/event-broker.md#redis)), so prefer the in-process backend when the stored log has to be complete.

## Before switching to Redis

1. Install the `redis` package: `just install` (it is in the `dev` dependency group) or `uv sync --all-extras` (the `broker-redis` extra of `evaluation`).
2. Make Redis reachable at `localhost:6379`, or set `REDIS_URL`. The compose Redis does not publish its port by default; see [Redis broker operations](redis-broker-ops.md#reach-redis-from-the-host).

## Switch

The CLI and the web server must both use Redis, and the same Redis.

| Process | In-process | Redis |
|---|---|---|
| Web server | `just arena-web-dev` | `just arena-web-dev-redis` |
| Race | `just arena-race` | `just arena-race-redis` |
| Rank | `just arena-rank` | `just arena-rank-redis` |
| Smoke | `just arena-smoke` | `ARENA_BROKER_BACKEND=redis just arena-smoke` |

`just` loads `.env`, so `REDIS_PASSWORD` reaches every recipe. A process started without `just` needs `ARENA_BROKER_BACKEND` and either `REDIS_PASSWORD` or `REDIS_URL` in its own environment.

## Check which backend a shell selects

```bash
uv run --package evaluation python -c "from evaluation.broker_factory import make_broker; print(type(make_broker()).__name__)"
```

It prints `InProcessEventBroker` or `RedisStreamsBroker`. Building the Redis client does not connect, so this does not prove Redis is reachable.

## Verify end to end

```bash
# terminal 1
just arena-web-dev-redis

# terminal 2
just arena-rank-redis

# terminal 3, while the ranking runs
curl -s localhost:8000/runs | jq '[.[] | select(.status == "running") | {run_id, label, n_events}]'
curl -sN "localhost:8000/events?run_id=<run_id>" | head -3
```

`/runs` lists each open run with `"status": "running"` and `label` `rank`, and `/events` prints `data: {...}` lines while the run plays. After the CLI exits, the same runs appear as `complete` with a `db_path`. `smoke` finishes too quickly to catch live.

`GET /metrics` reports only the web server's own counters. On Redis, `events_streamed` rises as SSE clients read, while `events_published` and `runs_open` count only forks started from the UI.

## Switch back

Run the recipes without `-redis`, or `unset ARENA_BROKER_BACKEND`. Redis is not touched; keys left by earlier CLI runs stay until you [clean them up](redis-broker-ops.md#clean-up-leftover-run-keys).

## Symptoms

| Symptom | Cause | Fix |
|---|---|---|
| `ValueError: unknown ARENA_BROKER_BACKEND=''` | The variable is exported but empty. Case and surrounding spaces are ignored, but empty is not | `unset ARENA_BROKER_BACKEND` or set a value |
| `ModuleNotFoundError: No module named 'redis'` | Installed with `just install-prod` (`uv sync --no-dev`) | `just install` or `uv sync --all-extras` |
| `/runs` does not list the CLI run until it ends, and `/events` returns 503 while it plays | One of the two processes is on the in-process backend | Start both with the `-redis` recipes |
| Both on Redis, but the run never appears | They point at different Redis servers (`REDIS_URL` in one shell only) | Use the same `REDIS_URL`, or unset it in both |
| `ConnectionError` or `AuthenticationError` | Redis unreachable, or no password in the environment | [Redis broker operations](redis-broker-ops.md#symptoms) |

## Contract tests

```bash
just test tests/test_event_broker.py tests/test_broker_factory.py tests/test_redis_broker.py
```

These run both implementations, Redis through `fakeredis`. The cross-process test needs a real Redis and skips when it cannot connect:

```bash
set -a; . ./.env; set +a
REDIS_URL="redis://:$REDIS_PASSWORD@localhost:6379/0" just test tests/test_redis_broker_cross_process.py
```
