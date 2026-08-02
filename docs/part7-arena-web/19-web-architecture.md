# Chapter 19 — Arena Web Architecture

`apps/api/src/` is the operator-facing surface for inspecting and steering arena runs. It has two halves:

- **Backend** (`apps/api/src/server.py`) — FastAPI + SSE. Reads from the event broker for live runs, falls back to a read-only DuckDB scan for finalized ones. Hosts the `/forks` endpoint that branches a parent run into a child replay.
- **Frontend** (`apps/dashboard/`) — Vite + React 19 + Tailwind v4 + TanStack Router/Query + React Aria Components. Connects to the backend over SSE, renders a Timeline scrubber and a World/Trace/Diff/Operator tab layout for a single run, an experiment overview for comparing the parallel runs of one `rank`/`race` operation, and posts mutation patches to `/forks`.

Both are optional. They sit on top of the broker and the DuckDB log — the agent and arena CLIs work without them.

The dashboard also hosts a **second, unrelated backend**: the detection training tracker (`apps/training-api`, port 8100), mounted under `/training/*` routes. It shares nothing with the arena API but the SPA shell — see [Chapter 24](../part3-entity-detection/24-training-tracker.md).

<aside class="concept" data-title="SSE vs WebSocket vs long-polling (why SSE was the right call)">

Three ways for a server to push updates to a browser without the browser asking:

- **Long-polling.** Browser opens a normal HTTP request; server holds it open until it has data; closes; browser immediately reopens. Simple, but every push pays the overhead of an HTTP request, and reconnect timing is brittle.
- **WebSocket.** Full-duplex, persistent TCP-over-HTTP-upgrade connection. The fastest and most flexible — but it's binary-framed, requires its own auth/proxy story (no automatic cookies on upgrade in some browsers), and gives you no built-in reconnect or replay.
- **SSE (Server-Sent Events).** One-way HTTP stream of `text/event-stream` chunks. Native browser API (`EventSource`), automatic reconnect with last-seen event ID, plays nicely with HTTP/2 multiplexing, and "just works" through any HTTP proxy that doesn't aggressively buffer.

Our needs are dead-simple: *server → browser, fan-out telemetry, replay from a known offset.* That maps exactly onto SSE. We don't need browser → server messages (the operator mutations go through a normal POST `/forks`), so the half-duplex restriction costs us nothing, and the built-in reconnect-with-`Last-Event-ID` semantics map perfectly onto the broker's `Seq` numbers — a reconnecting client resumes where it left off without us writing any code. Picking WebSocket would have meant rebuilding all of that machinery to gain duplex we don't use.

</aside>

<aside class="prereqs">

FastAPI basics (decorators, dependency injection, lifespan). [Server-Sent Events (SSE)](../glossary.md#s) — there's a callout further down this chapter that compares SSE to WebSocket and long-polling.

</aside>

## URL contract (backend)

The HTTP contract is frozen so the frontend can evolve independently:

| Method | Path | Returns | Purpose |
|---|---|---|---|
| `GET` | `/health` | `{"status": "ok"}` | Liveness ping. |
| `GET` | `/runs` | `list[RunSummary]`, newest first | Live runs from the broker (`status: "running"`) merged over finalized runs read from every DuckDB file under `ARENA_LOGS_ROOT` (`status: "complete"`). |
| `GET` | `/runs/summaries` | `list[RunMetrics]` | Per-run end-of-run metrics (`profile_name`, final age/population/economy, cost, turns) for the experiment overview. Finalized runs only — a live operation's file is writer-locked until it finalizes. |
| `GET` | `/runs/series?db_path=X` | `list[RunSeries]` | Per-turn resource trajectories for every run in one operation's DuckDB file (the overview's per-resource charts). `db_path` is validated to resolve under `ARENA_LOGS_ROOT` (path-traversal guard). |
| `GET` | `/events?run_id=X&from_seq=N` | `text/event-stream` | Replay + live-tail. Switches to live broker mode when `broker.is_open_remote(run_id)`, falls back to cold DuckDB scan otherwise. |
| `POST` | `/forks` | `ForkResponse` | Snapshot the parent at `parent_t`, optionally mutate, schedule an N-turn async replay. |
| `GET` | `/metrics` | `BrokerMetricsSnapshot` JSON | Operational counters (see Chapter 15). Backend-agnostic via `isinstance` dispatch. |

SSE line shape: `data: <payload_json>\n\n` where `<payload_json>` is the raw `Payload.model_dump_json()` from `packages/evaluation/src/event_log.py`. The frontend parses it and matches on the embedded `kind` discriminator. On overflow, the backend emits a final `event: overflow\ndata: {"available_from": N}\n\n` line; the frontend reconnects with `?from_seq=N` and accepts the gap (see Chapter 15's backpressure section).

## Lifespan and shared state

`apps/api/src/server.py:224` (`lifespan`) is the FastAPI lifespan context. On startup it:

1. Calls `make_broker()` — picks the backend per `ARENA_BROKER_BACKEND`.
2. Constructs a `_ReaperRegistry` (`server.py:90`) for wall-clock-based buffer reap.
3. Initializes an `app.state.fork_tasks: set[asyncio.Task]` for tracking in-flight fork replays.
4. Starts `_reaper_loop` as a background task — scans every `grace_period / 2` (default `15min`) and reaps runs whose close-time is older than the grace.

On shutdown: cancel the reaper *before* the fork tasks (otherwise the reaper could race shutdown and reap a run mid-replay), then cancel any in-flight forks. The ordering is load-bearing — `server.py:236–246`.

`app.state.broker`, `app.state.reaper`, `app.state.fork_tasks` are exposed via three FastAPI dependencies (`get_broker`, `get_reaper`, `get_fork_tasks`). The dependency boundary uses `cast` rather than `isinstance` because the lifespan is the single writer of these slots — a runtime `isinstance` check would be hostile to the multi-backend broker design.

## `/runs` — live + cold

`server.py:372` (`runs`) is symmetric with `/events`: the broker is the source of truth for in-progress runs, the cold DuckDB scan for finalized ones. It calls `broker.live_runs()` (mapped to `RunSummary(status="running", db_path="", …)` by `_live_summaries`, `server.py:344`), `_list_runs` for the cold DuckDB rows (`status="complete"`), and `_merge_runs` (`server.py:363`) concatenates them — **live wins** on a run_id collision, which only happens during the brief window after a run closes but before its writer process releases the DuckDB lock. A live run's `db_path` is empty; the frontend keys and selects by `run_id`, never the path (Chapter 15's [live-run discovery](../part6-evaluation-arena/15-event-broker.md#live-run-discovery)).

## `/events` — live vs cold

`server.py:415` (`events`) is the load-bearing dispatch:

```python
typed_run = RunId(run_id)
if await broker.is_open_remote(typed_run):
    return StreamingResponse(_stream_from_broker(broker, typed_run, Seq(from_seq)), ...)
db_path = await asyncio.to_thread(_resolve_run, run_id, _logs_root())
return StreamingResponse(_stream_from_cold(db_path, typed_run), ...)
```

It dispatches on `is_open_remote`, **not** `is_open` — the web process never opened the run (a separate CLI process did), so the process-local `is_open` would be `False` and we'd wrongly fall through to the writer-locked DuckDB. `is_open_remote` is the cross-process liveness signal; for the in-process broker the two coincide, so single-process forks and the test suite are unaffected.

The frontend doesn't need to know which path it's getting. The byte-equivalence guarantee (broker path emits `payload.model_dump_json()`; cold path emits the same via `stream_cold`, guarded by `test_payload_roundtrip_is_byte_stable`) is what makes this transparent.

`_resolve_run` (`server.py:242`) is a newest-first scan over `logs/arena/*/*.duckdb`. It opens each file read-only via `_connect_read_only` (`server.py:183`), which **skips a file a writer holds locked** rather than erroring — a separate-process live run holds its own DuckDB RW, and DuckDB is single-writer. Such runs are served from the broker (above), not cold; if the requested run is in none of the *readable* files but a locked one might hold it, the handler returns 503 (transient) instead of 404 (permanent). Throws 404 if no file contains the run and none are locked.

`_stream_from_broker` (`server.py:296`) catches `BrokerOverflowError` and emits the overflow SSE line. Cold path (`_stream_from_cold` at `server.py:317`) is synchronous because DuckDB iteration is blocking — Starlette drives it on its thread pool, which is honest about the cost instead of hiding it behind `to_thread`.

## `/forks` — branching a run

`server.py:385` is a thin handler; the work happens in `apps/api/src/forks.py:create_fork`. The flow:

1. Locate the parent's DuckDB file (`_resolve_parent_db`, `forks.py:119`) — newest-first scan, raises `FileNotFoundError` → 404.
2. Open the parent read-only, call `evaluation.fork.fork()` to snapshot the parent's `turn_start` state. Capture the fork event into an in-memory `_CapturingSink` (`forks.py:183`) — the fork primitive is sync but we need to publish via async broker.
3. `broker.open_run(typed_run)`. Publish the fork event(s) and (optionally) a `WorldMutationPayload` describing the before/after if a mutation patch was applied.
4. Spawn `persist_to_duckdb(broker, typed_run, child_db)` — drains the broker into a new per-run DuckDB file under `logs/arena/<date>/fork-<HHMMSSμs>.duckdb`.
5. Spawn `_replay(...)` (`forks.py:203`) — runs `synth_game_loop` for `n_turns`, publishing through a `BrokerEventSink`. On exit:
   - Two-tick `asyncio.sleep(0)` drain so queued `call_soon_threadsafe` publishes fire.
   - `broker.close_run(typed_run)`.
   - `await persist_task` — guarantees DuckDB is written before any cold-path reader sees the run finalized.
   - `on_close(typed_run)` — tells the reaper registry to start the grace timer.

The lifecycle ordering in both `create_fork` (head) and `_replay` (tail) is annotated load-bearing in the source. Reordering either set of steps will cause publish-after-close races or premature reaps — there's a banner comment in the code, do not move them without a test.

Fork tasks are tracked in `app.state.fork_tasks` (a strong-reference set). Without the strong reference, asyncio may GC mid-execution; `add_done_callback(fork_tasks.discard)` keeps the set bounded. Same pattern is used by `MultiRunBrokerSink._pending_publishes` (Chapter 16).

### MutationPatch

`forks.py:57` — frozen Pydantic model with `extra="forbid"`. Only seven WorldState fields are mutable from outside: `food, wood, gold, stone, population, pop_cap, age`. The `age` field is a typed Literal `Dark Age | Feudal Age | Castle Age | Imperial Age` so the API rejects typos at request validation. `is_empty()` short-circuits the no-op patch case — no `world_mutation` event is emitted when the patch has no effect.

## Frontend topology

The entry point is `apps/dashboard/src/main.tsx`, which mounts a TanStack Router `RouterProvider` wrapped in a TanStack Query `QueryClientProvider`. There is no `App.tsx` — the layout is assembled by nested routes. The arena shape is a 2-column grid:

```
┌─ aside (300px) ──┬─ main ───────────────────────────────────────┐
│  AoE2 Arena      │ <run-id>     [Streaming · 142 events]        │
│  Event log replay├───────────────────────────────────────────────┤
│                  │ [ World ] [ Trace ] [ Diff ] [ Operator ]    │
│  ┌────────────┐  │                                               │
│  │ run-list   │  │   <Tab content>                               │
│  │            │  │                                               │
│  │            │  │                                               │
│  └────────────┘  ├───────────────────────────────────────────────┤
│                  │   Timeline scrubber  ────●────────            │
└──────────────────┴───────────────────────────────────────────────┘
```

### Route table

Routes are file-based under `src/routes/`. The `@tanstack/router-plugin` Vite plugin generates `src/routeTree.gen.ts` from that directory; the generated file **is committed**, because `bun run build` runs `tsc -b` before Vite, so a missing tree would fail a fresh clone's build before the plugin ever ran.

| Route file | URL | What it renders |
|---|---|---|
| `index.tsx` | `/` | `redirect` to `/runs`. |
| `_arena.tsx` | — (pathless) | `ArenaLayout` — the shared run sidebar. Contributes nothing to the URL. |
| `_arena.runs.index.tsx` | `/runs` | "No run selected" empty state. |
| `_arena.runs.$runId.tsx` | `/runs/<id>` | Run detail: header, sibling strip, four tabs, Timeline. |
| `_arena.experiments.$key.tsx` | `/experiments/<key>` | Experiment overview for one run group. |
| `training.tsx` | — | `TrainingLayout` — the tracker's own nav shell. |
| `training.index.tsx` | `/training` | `redirect` to `/training/coverage`. |
| `training.coverage.tsx` | `/training/coverage` | Per-class coverage matrix ([Chapter 24](../part3-entity-detection/24-training-tracker.md)). |
| `training.images.tsx` | `/training/images` | Paginated image table + annotation lightbox. |

The `_arena` prefix is TanStack Router's **pathless layout** convention: the segment groups children under a shared component without appearing in the URL. That's what lets `/runs/<id>` and `/experiments/<key>` share one sidebar instance — switching between them doesn't remount it.

### State: URL, server cache, stream

State is split three ways, and the split is the main thing to understand about this frontend:

- **URL search params** own view state. `/runs/$runId` validates `?turn=<n>&tab=<world|trace|diff|operator>` in `validateSearch`. Both keys are optional, and defaults are applied at *read* time rather than written into the URL, so a bare `/runs/<id>` stays clean. An absent `turn` means "pinned to the newest turn" — `selectedTurn = turn ?? maxTurn` — which is why the scrubber auto-advances with the stream and stops advancing the moment the user drags it. That single expression replaced the two effects the old `App.tsx` needed (pin-to-latest, reset-on-run-change), and switching runs resets the turn for free because the param lives in a URL that changed. Scrubber writes use `replace: true` so dragging doesn't bury the Back button under one history entry per turn.
- **TanStack Query** owns server reads. Every fetch is a query-option factory in `lib/queries.ts` (`runsQueryOptions`, `runSummariesQueryOptions`, `runSeriesQueryOptions`, plus the `tracker*` family). Keys are hierarchical — invalidating `["runs"]` also sweeps `["runs","summaries"]` and `["runs","series"]`. Route `loader`s call `context.queryClient.ensureQueryData(...)` so data is in flight before the component renders; the component then reads the same key with `useQuery` and gets the cached result. `runSeriesQueryOptions("")` is `enabled: false` rather than firing a request that would 404 — a live operation has no finalized DuckDB file yet.
- **`useEvents(runId)`** (`hooks/use-events.ts`) stays a hand-written hook, not a query. It opens an `EventSource` against `/events?run_id=...`, accumulates events, and exposes the SSE status union (`idle | connecting | open | closed | error`). A stream that's appended to over minutes and reconnects on a custom overflow signal is not what a request-cache models, so it was deliberately left outside Query.

`main.tsx` sets `defaultPreloadStaleTime: 0` on the router: Query owns caching, and without this the router would layer its own preload staleness on top and the two would disagree about when data is fresh.

The sidebar (`components/run-list.tsx`, rendered by `layouts/arena-layout.tsx`) groups runs into operations via `lib/run-grouping.ts` — all runs of a `rank`/`race` share one DuckDB file, hence one `db_path`. Two destinations:

- **Run detail** (click a run) — the four World/Trace/Diff/Operator panels under `src/panels/` (see [Chapter 20](./20-fork-and-diff-ui.md)), plus a sibling strip to jump between the operation's parallel runs. The Timeline (`src/components/timeline.tsx`) sits outside the `Tabs` so the scrubber survives tab switches.
- **Experiment overview** (click a group header) — `panels/experiment-overview.tsx`: a leaderboard sorted by the same lexicographic composite as `arena.ranking.composite_score`, per-run comparison bars (final population, total cost), and per-resource trajectory charts averaged per profile. Lets you pick the best/worst run, then drill into any row.

<aside class="concept" data-title="Why put the scrubber position in the URL?">

The obvious home for "which turn is selected" is `useState`. Moving it into the query string buys four things that state in a component cannot:

1. **Shareable.** `/runs/abc?turn=17&tab=trace` is a link to *exactly* what you were looking at. Debugging conversations become URLs instead of "scrub to about turn 17 and open Trace".
2. **Back/forward works.** The browser's history stack is the undo stack for navigation, for free.
3. **Reload-survivable.** State that lives in React dies on refresh; state in the URL doesn't.
4. **Reset-on-navigation is automatic.** Switching runs changes the URL, so the turn resets without an effect watching `runId`.

The cost is that every write is a navigation, which is why the scrubber passes `replace: true` — a 200-turn drag would otherwise push 200 history entries and make the Back button useless. The rule of thumb: **view state that a user would want to link to belongs in the URL; everything else belongs in a component.** Ephemeral things here (form drafts in the Operator panel, lightbox zoom/pan) stay in `useState` precisely because nobody wants to bookmark them.

</aside>

## Backend / frontend wiring

Three wiring modes are supported:

| Mode | When | What you set |
|---|---|---|
| Vite dev proxy | Local dev — UI on :5173, FastAPI on :8000 | Nothing. `vite.config.ts` proxies `/runs`, `/events`, `/forks`, `/health` to `http://localhost:8000`, and the tracker's `/classes`, `/images`, `/annotations`, `/datasets`, `/stats`, `/thumbs`, `/raw` to `http://localhost:8100`. |
| Cross-origin dev | UI local, backend on a VM | `VITE_API_BASE_URL=http://vm:8000` in `apps/dashboard/.env.local`, plus `ARENA_WEB_CORS_ORIGINS` on the backend to allow the SPA origin. |
| Prod build | SPA served from the API origin | Build with `bun run build`, mount `dist/` behind FastAPI (not wired by default — the contract above is enough to do it). |

One collision is worth knowing about: since the routing migration, **`/runs` is both an API path and a client route** (`/runs`, `/runs/$runId`). Vite matches the proxy before the SPA fallback, so a browser navigation to `/runs/<id>` would render the raw JSON run list. The proxy entry carries a `bypass` that returns `/index.html` when the request's `Accept` header includes `text/html`: document requests ask for HTML, `fetch`/`XHR` never do, so that one header cleanly separates "show me the app" from "give me the data" (`vite.config.ts:41`).

See [Chapter 21 — Running the UI Locally](./21-running-the-ui-locally.md) for the actual recipes.

## What's intentionally *not* in the web stack

- **No auth.** Local-dev tool. If you expose it to the internet you need a proxy in front.
- **No persistent UI state.** Run selection / scrubber position live in React state only; reload starts fresh.
- **No write API on `/events`.** Events flow in one direction only — broker → UI. `POST /forks` is the only state-mutating endpoint.
- **No Langfuse sink yet.** Phase 10+. The architecture (Chapter 16) supports adding it as another broker consumer without touching the producers.

## Related reading

- [Chapter 15 — Event Broker](../part6-evaluation-arena/15-event-broker.md) — the source of `/events`.
- [Chapter 16 — DuckDB Persister and Replay](../part6-evaluation-arena/16-duckdb-persister-and-replay.md) — the source of the cold-path fallback and the fork primitive.
- [Chapter 20 — Fork and Diff UI](./20-fork-and-diff-ui.md) — the four tabs in detail.
- [Chapter 21 — Running the UI Locally](./21-running-the-ui-locally.md) — dev workflow.
- [Chapter 24 — Detection Training Tracker](../part3-entity-detection/24-training-tracker.md) — the other backend behind this SPA.
- [ADR 0005 — Vite / React / Tailwind for arena UI](../adr/0005-vite-react-tailwind-for-arena-ui.md) — the original scaffold decision.
- [ADR 0006 — TanStack Router + Query and React Aria](../adr/0006-tanstack-router-query-react-aria.md) — what replaced the hand-rolled state and Radix.
