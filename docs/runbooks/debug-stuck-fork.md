# Debugging a stuck fork

Use this when `POST /forks` fails, or when a fork's child run never shows the turns it should. The fork code is `apps/api/src/forks.py` (`create_fork`, `_replay`); the request lifecycle is in [arena web architecture](../web/architecture.md#forks). Commands assume the backend on `localhost:8000` and run from the agent repo root.

## Find the symptom

| Symptom | Go to |
|---|---|
| Operator panel shows an error badge, no child opens | [The request fails](#the-request-fails) |
| Child page says Complete with fewer turns than requested | [The replay died](#the-replay-died) |
| Child page says Streaming, event count stops growing | [The replay is parked](#the-replay-is-parked) |
| Child stays `running` in the sidebar long after it should have ended (Redis only) | [Orphaned open marker](#orphaned-open-marker) |
| Child page says Error, or `/events` returns 404 or 503 for it | [The child log is unreadable](#the-child-log-is-unreadable) |

First checks, for every case:

```bash
curl -s localhost:8000/health
curl -s localhost:8000/metrics | jq
curl -s localhost:8000/runs | jq '.[] | select(.run_id=="<child_id>")'
```

`/metrics` counters belong to the backend process. `runs_open` counts runs this process has open, which for the backend means forks in progress.

## The request fails

The badge shows the server's `detail` text.

| Status and detail | Cause | Fix |
|---|---|---|
| 404 `parent run_id '...' not found in ...` | No log under `ARENA_LOGS_ROOT` holds the parent | Confirm the parent is in `/runs` with `status: "complete"` and that the backend runs from the directory or `ARENA_LOGS_ROOT` that holds the logs |
| 422 `no turn_start event at run_id=... t=...` | `parent_t` is outside the parent's turns | Pick a turn the parent's timeline shows |
| 422 `... has no state snapshot` | That `turn_start` carries no state | Fork from another turn |
| 422 validation error | `n_turns` above 200, a negative population or pop cap, or an unknown mutation field | Fix the form values |
| 500, DuckDB lock error in the backend log | Another process (an `arena race` or `rank`) has a log under `ARENA_LOGS_ROOT` open for writing. `_resolve_parent_db` opens every file newest first and does not skip locked ones | Wait for the CLI run to finish, then retry |
| 500, Redis connection error in the log | Redis backend unreachable | `redis-cli -a "$REDIS_PASSWORD" ping`, then see [Redis broker operations](redis-broker-ops.md) |

## The replay died

`_replay` logs any exception and still closes the run and finishes the persister, so the child ends up as a short, finalized run.

```bash
# in the backend's terminal output
grep -E "fork replay failed for run_id=<child_id>|persister failed for run_id=<child_id>" <backend log>
```

| Log line | Cause | Fix |
|---|---|---|
| `fork replay failed`, an authentication error from the model API | `AOE2_LLM_API_KEY` missing or invalid in the backend's environment | Export it (or put it in `.env` and start through `just`), restart the backend, fork again |
| `fork replay failed`, rate limit or API error | The fork profile (`DEFAULT_FORK_PROFILE`: wire `openai`, model `gpt-5.6-luna`) hit a provider error after the SDK's retries | Retry later |
| `persister failed` | The child log could not be written | Check disk space and permissions under `ARENA_LOGS_ROOT`; the child's events stay only in the broker until the reaper drops them |

## The replay is parked

The run is open but no events arrive. Each replay turn is one model call, so a slow or hanging call stalls the stream.

1. Run `curl -s localhost:8000/metrics | jq` twice, a minute apart. `runs_open` of 1 or more with `events_published` unchanged means a replay is waiting.
2. With Redis, look at the child's keys:

   ```bash
   redis-cli -a "$REDIS_PASSWORD" EXISTS arena:run:<child_id>:open    # 1 = open
   redis-cli -a "$REDIS_PASSWORD" XLEN arena:run:<child_id>:events
   ```

   Open with a fixed `XLEN` means the producer is stuck, not the reader.
3. Wait for the SDK timeout, which ends the replay through the path in [The replay died](#the-replay-died).
4. If it does not end, restart the backend. Shutdown cancels every task in `app.state.fork_tasks`; the child log keeps the events written before the cancel. Fork again from the parent.

With the Redis backend, a publish that fails inside the replay runs in a task nobody awaits (`BrokerEventSink.emit` schedules it with `call_soon_threadsafe`). It appears in the log only as `Task exception was never retrieved`, and the turn's events are missing from the stream and the log.

## Orphaned open marker

With Redis, `open_run` sets `arena:run:<id>:open` with a six-hour TTL. If the backend dies mid-fork, nothing deletes it: `/runs` keeps listing the child as `running`, and `/events` stays in live mode, waiting on an empty stream until the key expires.

```bash
redis-cli -a "$REDIS_PASSWORD" TTL arena:run:<child_id>:open
redis-cli -a "$REDIS_PASSWORD" DEL arena:run:<child_id>:open
```

After the `DEL`, `/events` serves the child from its log file. See [Redis broker operations](redis-broker-ops.md) for cleaning up the stream and sequence keys.

## The child log is unreadable

Once the run closes, `/events` reads the child from `<ARENA_LOGS_ROOT>/<UTC date>/fork-*.duckdb`.

| Response | Cause | Fix |
|---|---|---|
| 503 `... locked by an active writer` | The child is not in any readable file and some other file is locked by a writer | Retry after the CLI run that holds the lock finishes |
| 404 `run_id ... not found` | The persister failed, or the backend's logs root differs from where the child was written | Check the `persister failed` log line and `ls logs/arena/*/fork-*.duckdb` |

In the dashboard a 404 or 503 shows as an Error badge, because the stream fails before it opens.

## Related

- [Run viewer](../web/run-viewer.md)
- [Event broker](../arena/event-broker.md)
- [Event log and replay](../arena/event-log-and-replay.md)
