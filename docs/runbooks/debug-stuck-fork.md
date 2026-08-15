# Runbook — Debugging a Stuck Fork or Replay

When `POST /forks` returns a `child_run_id` but the run never finishes — no `Complete` status, no events past the first few, eventually hits the broker buffer and starts dropping. This runbook walks through the diagnosis order.

## First, decide what "stuck" means

| Frontend says | What it usually means |
|---|---|
| `Streaming` + low event count + no growth | The replay is blocked or crashed; check the server logs. |
| `Streaming` + steady growth + then `Error` | Broker overflow without auto-reconnect, or backend crash. |
| `Closed` immediately | Replay exited early; check for an exception in server logs. |
| Run never appears in `/runs` | Replay crashed before any event was persisted; check server logs. |

## Step 1 — read the server log

The replay task uses `logger.exception(...)` on any exception (`apps/api/src/forks.py:248`). The trace shows up wherever `uvicorn` is writing logs. Most "stuck" cases turn out to be:

- `openai.APIError` or `anthropic.APIError`, whichever adapter `AOE2_LLM_WIRE` selected — bad/expired `AOE2_LLM_API_KEY`, rate limit, network blip. The replay aborts cleanly; check `/metrics → runs_open` to confirm the run was closed.
- `ValueError: unknown AOE2_LLM_WIRE=...` at import — a typo in the adapter name. Since `2e2717e` this raises rather than silently running the default, so the process never starts.
- `pydantic.ValidationError` — LLM returned malformed JSON that the action parser couldn't fix. Rare; usually intermittent.
- `RuntimeError: run X is not open` — lifecycle ordering bug. Should not happen in shipped code; if you see it, take a stack trace.

## Step 2 — check `/metrics`

```bash
curl -s http://localhost:8000/metrics | jq
```

| Counter | Meaning | What to look for |
|---|---|---|
| `events_published` | Total publishes since server start | Should grow during an active replay. |
| `events_streamed` | Total envelopes yielded to consumers | Should grow with each SSE consumer. |
| `streams_dropped` | Consumers that hit `BrokerOverflowError` | Non-zero ⇒ a consumer fell behind; expected to be small / zero. |
| `runs_open` | Currently-publishing runs | If non-zero long after the producer should be done, the producer didn't close the run. |

A stuck replay typically shows `runs_open > 0` and `events_published` constant for an unusually long time. The reaper grace period is 30 min by default — if a run is stuck for >30 min without progress, the reaper will eventually drop the buffer, and any reader will start getting `BrokerOverflowError` (or 404 on cold-path fallback if the run wasn't persisted yet).

## Step 3 — inspect the fork tasks set

If you have access to the running process, this is the gold standard:

```python
# In a REPL attached to the server (or via a debug endpoint you add)
import asyncio
for task in app.state.fork_tasks:
    print(task.get_name(), task.done(), task.get_coro())
```

A task that's `done()=False` but `get_coro()` shows it parked on a specific `await` tells you exactly where the replay is stuck. Common stuck spots:

- `await asyncio.sleep(0)` inside the two-tick drain — would mean the loop is wedged; almost never the actual cause.
- `await self.broker.publish(...)` for Redis — Redis is unreachable or the connection pool is exhausted. Check `redis-cli ping`.
- `await synth_game_loop(...)` deep inside the wire's API call (`chat.completions.create` on the `openai`/`zen` adapters, `messages.create` on `anthropic`) — the model call is hanging; usually the SDK times out eventually but can sit for a while.

## Step 4 — Redis-backend specific checks

```bash
# Is the producer's :open sentinel still there?
redis-cli -a "$REDIS_PASSWORD" EXISTS arena:run:<child_run_id>:open
# 1 = open, 0 = closed/never-opened

# How many events did the producer write?
redis-cli -a "$REDIS_PASSWORD" XLEN arena:run:<child_run_id>:events

# Has the consumer fallen behind the head?
redis-cli -a "$REDIS_PASSWORD" XINFO STREAM arena:run:<child_run_id>:events
# Look at first-entry vs last-entry IDs
```

If `EXISTS` returns 0 but the stream has entries, the producer closed normally but the consumer hasn't terminated — likely the consumer's `await self.flush()` queue isn't draining. See [Runbook: redis-broker-ops](./redis-broker-ops.md).

If `EXISTS` returns 1 and the stream length is stable, the producer is stuck mid-replay (back to step 1 — server logs).

## Step 5 — recover

Options, in order of preference:

1. **Wait it out.** If the producer is genuinely doing an LLM call, give it 60s.
2. **Restart the server.** The lifespan teardown cancels all `fork_tasks` and `gather`s them with `return_exceptions=True` (`apps/api/src/server.py:241`), so a clean restart drains. Existing finalized runs in DuckDB are unaffected.
3. **Manual reap (Redis).** If the run is wedged with no persister catching up, you can delete the Redis keys directly:
   ```bash
   redis-cli -a "$REDIS_PASSWORD" DEL arena:run:<rid>:events arena:run:<rid>:seq arena:run:<rid>:open
   ```
   This is destructive — any consumer mid-stream will get `BrokerOverflowError` or empty results. Only do this if you're already going to restart everything.
4. **Manual reap (in-process).** Restart the server. There's no "reap one run" endpoint by design; the reaper does it on a schedule.

## Step 6 — file the bug

If the stuck case is reproducible, capture:

- The server log around the time of the fork (everything between the POST and the symptom).
- The `/metrics` output before and after the stuck state began.
- The fork request body (`ForkRequest` JSON).
- For Redis: `XLEN` and `XINFO STREAM` output.

The replay path (`apps/api/src/forks.py:_replay`) has annotated lifecycle ordering — any new "stuck" failure mode is usually a violation of that ordering or a new SDK gotcha worth recording in the source as a comment.

## Related

- [Chapter 19](../part7-arena-web/19-web-architecture.md) — replay lifecycle.
- [Chapter 15](../part6-evaluation-arena/15-event-broker.md) — broker semantics including overflow.
- [Runbook: redis-broker-ops](./redis-broker-ops.md) — Redis health checks.
