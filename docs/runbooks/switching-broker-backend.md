# Runbook — Switching the Broker Backend

When to use each backend, how to flip between them, and how to verify the switch took.

## The two backends

| Backend | `ARENA_BROKER_BACKEND` | When to use |
|---|---|---|
| In-process | unset, `inprocess` (default) | Single-process work: CLI race/rank/smoke, tests, single-machine dev where the SSE backend and the CLI live in the same Python process. |
| Redis Streams | `redis` | Cross-process: a CLI writes events from one process while the FastAPI server lives-tails them from another, or producers / consumers on different hosts. |

If you're not sure which you need, you need in-process. Reach for Redis when you have a specific cross-process need.

## Switching to Redis

Prereqs:

1. Install the Redis client: `pip install -e ".[broker-redis]"` (or `uv sync --extra broker-redis`).
2. Have Redis running. Easiest: `just arena-infra-up` (compose stack with AUTH). See [Runbook: redis-broker-ops](./redis-broker-ops.md).
3. `REDIS_PASSWORD` in env (if using compose) or `REDIS_URL` set explicitly.

Then:

```bash
export ARENA_BROKER_BACKEND=redis
# verify the factory will pick the right backend
python -c "from evaluation.broker_factory import make_broker; print(type(make_broker()).__name__)"
# RedisStreamsBroker
```

Both the producer process *and* the consumer process need the env var set. If the CLI has `redis` but the web server has the default, the CLI writes to Redis while the server live-tails the empty in-process broker — symptom: `/events` shows `Streaming` but no events arrive.

## Switching back to in-process

```bash
unset ARENA_BROKER_BACKEND   # or set to "inprocess"
# verify
python -c "from evaluation.broker_factory import make_broker; print(type(make_broker()).__name__)"
# InProcessEventBroker
```

No teardown of Redis needed; the in-process broker simply ignores it.

## Verifying end-to-end

The strongest verification is a live round-trip: start the web server in one shell, run a CLI race in another, watch the run appear in `/runs` and the events stream live.

```bash
# Terminal 1
export ARENA_BROKER_BACKEND=redis
just arena-web-dev          # or `just arena-web-dev-redis` (sets the env var for you)

# Terminal 2 (same export)
export ARENA_BROKER_BACKEND=redis
just arena-smoke            # or `just arena-rank-redis` / `just arena-race-redis`

# Terminal 3
curl -s http://localhost:8000/runs | jq '.[0]'
# Should show {"status": "running", ...} while the run is in flight (surfaced
# from the broker via live_runs), flipping to "complete" once it finalizes.

curl -sN "http://localhost:8000/events?run_id=$(curl -s http://localhost:8000/runs | jq -r '.[0].run_id')" | head -5
# Should stream events from t=0 onwards
```

If the CLI run completes and the web server's `/runs` shows it but `/events` is empty when streamed live, the broker switch worked but something else is wrong. Check the broker's `/metrics`:

```bash
curl -s http://localhost:8000/metrics | jq
# {"events_published": N, "events_streamed": M, "streams_dropped": 0, "runs_open": 0}
```

`events_published` should be ≥ the CLI's published count. If it's zero, the producer is not actually writing to this broker — re-check the env var in both processes.

## Common gotchas

- **`ARENA_BROKER_BACKEND` set in a tmux pane that's no longer your producer.** Both producer and consumer processes need it set in *their* env. `printenv ARENA_BROKER_BACKEND` in each shell is the quickest sanity check.
- **`make_broker()` reads the env var at call time, not import time.** So `os.environ["ARENA_BROKER_BACKEND"] = "redis"; from evaluation.broker_factory import make_broker; make_broker()` works. The factory is intentionally not module-level cached for this reason.
- **The compose Redis uses AUTH.** A raw `REDIS_URL=redis://localhost:6379/0` without the password gets `AuthenticationError`. Use `REDIS_PASSWORD` (and let `make_broker()` build the URL) unless you have a specific reason to set `REDIS_URL` directly — see [Runbook: redis-broker-ops](./redis-broker-ops.md).
- **Unknown backend values raise `ValueError` immediately.** `ARENA_BROKER_BACKEND=Redis` (capital R) is rejected — the factory lowercases but doesn't fuzzy-match. This is deliberate (`packages/evaluation/src/broker_factory.py:81`): silent fallback to in-process would hide deployment misconfigurations.

## Contract test

Whether you've made changes to either broker impl or you're just trying to convince yourself the equivalence holds, run:

```bash
pytest tests/test_event_broker.py -v
```

The suite is parametrized over both impls (using `fakeredis` for the Redis one). It verifies envelope equivalence, ordering, drain-on-close, and overflow semantics. If a behaviour difference between the impls slips through, this is what catches it before deployment.
