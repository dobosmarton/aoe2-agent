# Redis broker operations

Procedures for the compose Redis that backs `RedisStreamsBroker`. Run every command from the root of the agent repository, where `docker-compose.yml`, `.env` and the `justfile` are. How the broker uses Redis is in [event broker](../arena/event-broker.md).

## Start Redis

1. Set `REDIS_PASSWORD` in `.env` (`.env.example` has it under "Synthetic Arena: Redis"). Use a hex password, because `make_broker()` puts the password into a URL without escaping:

   ```bash
   openssl rand -hex 24
   ```

2. Start Redis alone, or the whole stack:

   ```bash
   docker compose up -d redis
   just arena-infra-up          # docker compose up -d --wait, every service
   ```

3. Check it is healthy. The healthcheck runs `redis-cli -a $REDIS_PASSWORD ping` every 5 s with 10 retries.

   ```bash
   docker compose ps redis
   ```

Redis runs with `--appendonly yes` and `--maxmemory-policy noeviction` on the `redis-data` volume, with no memory limit. `just arena-infra-nuke` (`docker compose down -v`) deletes the volume along with every other volume in the stack.

## Reach Redis from the host

`docker-compose.yml` publishes no Redis port, and `make_broker()` connects to `localhost:6379` unless `REDIS_URL` says otherwise. Arena processes run on the host, so add a `docker-compose.override.yml` next to `docker-compose.yml`. `docker compose` merges it automatically, including inside `just arena-infra-up`:

```yaml
services:
  redis:
    ports:
      - "127.0.0.1:6379:6379"
```

Apply it with `docker compose up -d redis`. `docker-compose.ci.yml` has the same mapping, but it also puts every volume on tmpfs. To use a Redis outside the stack, set `REDIS_URL` instead.

## Point the arena at it

`just` loads `.env` (`set dotenv-load`), so every recipe has `REDIS_PASSWORD`, and `make_broker()` builds `redis://:$REDIS_PASSWORD@localhost:6379/0` from it. The `-redis` recipes also set the backend:

```bash
just arena-web-dev-redis     # web server on :8000
just arena-rank-redis        # or arena-race-redis
```

Commands run outside `just` need the variables in their own environment. Selecting and verifying the backend is in [switching the broker backend](switching-broker-backend.md).

## Inspect runs

Load `.env` and define a helper that runs `redis-cli` inside the container:

```bash
set -a; . ./.env; set +a
rcli() { docker compose exec -T redis redis-cli -a "$REDIS_PASSWORD" --no-auth-warning "$@"; }
```

| Question | Command |
|---|---|
| Is Redis up? | `rcli ping` |
| Which runs are open? | `rcli --scan --pattern 'arena:run:*:open'` |
| A run's label and start time | `rcli GET arena:run:<id>:open` |
| Time before an abandoned run stops showing as open | `rcli TTL arena:run:<id>:open` |
| Events stored for a run | `rcli XLEN arena:run:<id>:events` |
| Oldest retained `Seq` (the overflow boundary) | `rcli XINFO STREAM arena:run:<id>:events`, field `first-entry` |
| The last three events | `rcli XREVRANGE arena:run:<id>:events + - COUNT 3` |

| Key | Type | Lifetime |
|---|---|---|
| `arena:run:<id>:open` | string: JSON label and start time, or `1` | Set by `open_run` with a 6 h TTL, deleted by `close_run` |
| `arena:run:<id>:seq` | integer | Until `reap` reaches Redis |
| `arena:run:<id>:events` | stream, IDs `<seq>-0`, about 10,000 entries | Until `reap` reaches Redis |

## Clean up leftover run keys

The CLI's `close_all()` calls `reap`, but the Redis broker only queues that `DEL` and sends it on its next async call, which never comes because the CLI exits. Every CLI run on the Redis backend therefore leaves its `:seq` and `:events` keys behind, with no TTL. Fork runs are reaped by the web server 30 minutes after they close, and the `DEL` goes out with its next broker call.

Delete the keys of every run that has no `:open` key:

```bash
docker compose exec -T redis sh -c '
  for k in $(redis-cli -a "$1" --no-auth-warning --scan --pattern "arena:run:*:seq"); do
    run=${k%:seq}
    if [ "$(redis-cli -a "$1" --no-auth-warning EXISTS "$run:open")" = 0 ]; then
      redis-cli -a "$1" --no-auth-warning DEL "$run:seq" "$run:events"
    fi
  done' _ "$REDIS_PASSWORD"
```

A run's events are already in its DuckDB file once the CLI has exited, so nothing is lost.

## Rotate the password

1. Write the replacement password to `.env` as `REDIS_PASSWORD`.
2. Recreate Redis and the two services that authenticate to it:

   ```bash
   docker compose up -d --force-recreate redis langfuse-web langfuse-worker
   ```

3. Restart every arena process that holds a Redis connection (the web server, any running CLI). `make_broker()` reads `REDIS_PASSWORD` when it is called, so a restarted process reads the current value.

## Symptoms

| Symptom | Cause | Fix |
|---|---|---|
| `ConnectionError` connecting to `localhost:6379` | Redis is not running, or its port is not published | `docker compose ps redis`; add the override from [Reach Redis from the host](#reach-redis-from-the-host) |
| `AuthenticationError` on the first publish | `REDIS_PASSWORD` is not in the process environment, so the URL has no password | Run through `just`, or `set -a; . ./.env; set +a` first, or set `REDIS_URL` |
| `ValueError: Port could not be cast to integer value` when the broker is built | The password contains `/`, `#` or `?`, which end the host part of the auto-built URL (a `%` instead corrupts the password and gives `AuthenticationError`) | Use a hex password, or set `REDIS_URL` with the password percent-encoded |
| Log line "Task exception was never retrieved" with `ResponseError: The ID specified in XADD is equal or smaller than the target stream top item`; the DuckDB log has fewer rows than expected | Concurrent publishes to one run reached `XADD` out of order (see [event broker](../arena/event-broker.md#redis)) | Use the in-process backend when the full log matters |
| A run stays in `/runs` as running after its producer died | The `:open` key outlives the process until its TTL | Wait for the TTL, or `rcli DEL arena:run:<id>:open` |
| `BrokerOverflowError` in a consumer | It asked for, or fell behind to, a `Seq` older than `first-entry` | The dashboard stops at the gap and does not resume; reopen the run once it finishes to read it in full from DuckDB |
| Memory grows across many runs | Leftover `:seq` and `:events` keys | [Clean up leftover run keys](#clean-up-leftover-run-keys) |
