# Chapter 21 — Running the UI Locally

This is the recipe for getting from a fresh clone to "click around past runs in a browser" in three commands.

<aside class="prereqs">

[Chapter 19 — Web Architecture](./19-web-architecture.md) and [Chapter 20 — Fork and Diff UI](./20-fork-and-diff-ui.md) describe what you're running. This chapter is the recipe. The **Prerequisites** section below covers installation prereqs (bun, Python) — distinct from the conceptual ones here.

</aside>

## Prerequisites

- Python env set up per the root README (`uv sync` or `pip install -e ".[dev,server]"`). The web backend is part of the base install; no extra extra is needed.
- [Bun](https://bun.sh) for the frontend (`brew install oven-sh/bun/bun` on macOS). The justfile recipes assume `bun`. If you prefer `pnpm`/`npm`, the project is a standard Vite app and works fine — adjust the commands.
- At least one DuckDB log under `logs/arena/`. If the directory is empty, run `just arena-smoke` once to generate `logs/arena/<today>/smoke-*.duckdb`.

## The fastest path — two terminals

```bash
# Terminal 1: backend on :8000
just arena-web-dev

# Terminal 2: frontend on :5173 (proxies /runs, /events, /forks, /health to :8000)
just arena-ui-dev
```

Browse to <http://localhost:5173>. `/` redirects to `/runs`; pick a run from the left sidebar.

`just arena-web-dev` resolves to `python -m arena_web --port 8000` (`apps/api/src/__main__.py`). `just arena-ui-dev` runs `bun install` (idempotent) then `bun run dev`.

## The training tracker (third terminal, optional)

The same SPA hosts the detection training tracker under `/training/*`, backed by a **separate** service on :8100 ([Chapter 24](../part3-entity-detection/24-training-tracker.md)). It's independent of the arena backend — you can run either, both, or neither:

```bash
just training-seed        # once: populate logs/training/tracker.db from disk
just training-api-dev     # Terminal 3: tracker API on :8100
```

Then browse to <http://localhost:5173/training>. Without :8100 running, the arena routes work normally and only `/training/*` shows load errors.

## Watching a run live (cross-process)

The recipe above replays *finalized* runs. To watch a run **as it happens**, the producer (the arena CLI) and the consumer (the web backend) must share a Redis broker — an in-process broker is private to each process, so the only thing they'd otherwise share is the writer-locked DuckDB file. The `*-redis` recipes set `ARENA_BROKER_BACKEND=redis` for you (`set dotenv-load` puts `REDIS_PASSWORD` in scope, and `make_broker()` builds the AUTH'd URL):

```bash
just arena-infra-up          # 1. Redis (+ compose stack) up
just arena-web-dev-redis     # 2. backend on :8000, redis mode
just arena-ui-dev            # 3. dashboard on :5173
just arena-rank-redis        # 4. a run, in another shell (or arena-race-redis)
```

No API key? `ARENA_BROKER_BACKEND=redis just arena-smoke` does a mock-LLM run with the same wiring (it's fast, so the live window is short). The run shows in the sidebar with a pulsing **live** badge while in progress (`status: "running"` from `/runs`) and the World panel fills in as turns arrive. The list is fetched once per page load, so reload to pick up newly-started runs; when the run finishes it becomes a normal finalized run (`status: "complete"`), readable from its DuckDB file even after Redis reaps the stream. See [Runbook: switching-broker-backend](../runbooks/switching-broker-backend.md).

## Configuration knobs

### Backend

| Env Var | Default | Purpose |
|---|---|---|
| `ARENA_LOGS_ROOT` | `logs/arena` | Where to look for DuckDB files. Useful for test isolation or pointing at a remote-mounted log dir. |
| `ARENA_WEB_CORS_ORIGINS` | `http://localhost:5173, http://localhost:8000` | Comma-separated allow-list. Add your SPA origin when running cross-host. |
| `ARENA_BROKER_BACKEND` | `inprocess` | Set to `redis` to live-tail a CLI run started in another process. See [Runbook: switching-broker-backend](../runbooks/switching-broker-backend.md). |
| `ANTHROPIC_API_KEY` | — | Required only if you use the Operator panel (it calls `/forks`, which runs an LLM-driven replay). Listing and replaying runs works without it. |

### Frontend

| Env Var | Default | Purpose |
|---|---|---|
| `VITE_API_BASE_URL` | unset (use Vite dev proxy) | Set to a full `http(s)://host:port` URL to bypass the proxy. Necessary when the backend is on a different machine. Live in `apps/dashboard/.env.local`; the example is `apps/dashboard/.env.example`. |

### Training tracker (`apps/training-api`, :8100)

| Env Var | Default | Purpose |
|---|---|---|
| `TRAINING_API_DB` | `logs/training/tracker.db` | SQLite file. Delete it and re-run `just training-seed` to rebuild from disk. |
| `TRAINING_API_RAW_IMAGES` | `packages/detection/src/real_screenshots/raw` | Where `ingest.py` looks for screenshots. |
| `TRAINING_API_DATASET_ROOT` | `packages/detection/src` | Parent of the `training_data_*` dirs. |
| `TRAINING_API_CLASSES_YAML` | `.../training/config/classes.yaml` | The 60-class schema served at `/classes`. |
| `TRAINING_API_THUMB_CACHE` | `logs/training/thumbs` | Generated thumbnails; safe to delete. |
| `TRAINING_API_CORS_ORIGINS` | `http://localhost:5173,http://localhost:8100` | Comma-separated allow-list. |

All paths resolve relative to the repo root, not the process CWD — `uv run` resolves from odd places, so `config.py` anchors them off its own file location.

The proxy paths are listed in `apps/dashboard/vite.config.ts`: `/runs`, `/events`, `/forks`, `/health` → :8000, and `/classes`, `/images`, `/annotations`, `/datasets`, `/stats`, `/thumbs`, `/raw` → :8100. When `VITE_API_BASE_URL` is set, the frontend prepends it to every API call instead of using relative URLs.

<aside class="concept" data-title="When an API path collides with a client route">

`/runs` is now both a backend endpoint *and* a page in the SPA (`/runs`, `/runs/<id>`). Vite matches proxy rules **before** the SPA history fallback, so typing `localhost:5173/runs/abc` into the address bar would hand you the raw JSON run list instead of the app.

The fix is a `bypass` on that one proxy entry:

```ts
"/runs": {
  target: "http://localhost:8000",
  bypass: (req) =>
    req.headers.accept?.includes("text/html") === true ? "/index.html" : undefined,
},
```

Returning a path from `bypass` means "serve this instead of proxying"; returning `undefined` means "proxy normally". The discriminator is the `Accept` header: browsers requesting a *document* send `text/html`, while `fetch()` and `XHR` never do. One header cleanly separates "show me the app" from "give me the data".

The general lesson is worth carrying to any SPA-over-API setup: **once your client routes and your API paths share a namespace, something has to disambiguate them.** The alternatives are prefixing the API (`/api/runs`) or prefixing the app (`/app/runs`) — both cleaner, both a breaking change to a frozen URL contract. The `Accept` bypass buys the same result without touching the contract, at the cost of one non-obvious line that this callout exists to explain.

</aside>

<aside class="concept" data-title="The Vite dev proxy (what it does and why it disappears in production)">

In dev, your browser talks to `localhost:5173` (Vite) and your backend listens on `localhost:8000` (FastAPI). Cross-origin requests (`5173 → 8000`) would normally trigger CORS preflights and cookie quirks. The Vite dev proxy makes that go away by forwarding any request matching configured paths from the dev server to the backend, server-side:

```ts
// vite.config.ts (sketch)
server: {
  proxy: {
    '/events': { target: 'http://localhost:8000', changeOrigin: true },
    '/runs':   'http://localhost:8000',
    '/forks':  'http://localhost:8000',
    '/health': 'http://localhost:8000',
  },
}
```

To the browser, it looks like `/events` is served by `localhost:5173` — same origin, no CORS preflight, cookies attach by default. To FastAPI, the request arrives just like any direct call. The proxy is **dev-only** — `vite build` produces a static `dist/` bundle that's served from somewhere else entirely (often the FastAPI process itself in production), and the `/events` requests then go to the same origin. That's why `VITE_API_BASE_URL` exists: it lets you bypass the proxy in dev to point at a backend on another machine, *without* changing any of the frontend's `fetch` calls — they remain relative paths and the env var is prepended at build time.

The pattern generalizes to webpack-dev-server, Next.js's `rewrites`, and esbuild's serve mode. They all solve the same "cross-origin in dev, same-origin in prod" problem.

</aside>

## Three deployment modes

### 1. Local dev (default)

Backend and frontend both on localhost, Vite proxy in the middle. No CORS work, no env vars beyond defaults. This is what `just arena-web-dev` + `just arena-ui-dev` give you.

### 2. Backend on a VM, frontend on your laptop

```bash
# On the VM
export ARENA_WEB_CORS_ORIGINS="http://localhost:5173,http://localhost:5174"
just arena-web-dev   # binds to 127.0.0.1 by default; pass --host 0.0.0.0 if needed
```

```bash
# On the laptop
echo 'VITE_API_BASE_URL=http://vm.local:8000' > apps/dashboard/.env.local
just arena-ui-dev
```

The CORS list on the backend must include the laptop's SPA origin. If you change it after the server is up, restart `arena-web-dev` — the lifespan reads the env var once at startup.

### 3. Production build

```bash
just arena-web-build     # writes apps/dashboard/dist/
```

The `dist/` bundle is a standard Vite static site — drop it behind any HTTP server, or mount it from FastAPI with a `StaticFiles` mount. Wiring that into `server.py` is intentionally not done; the URL contract in [Chapter 19](./19-web-architecture.md) is the stable surface, and how you serve the bundle is up to your deployment.

## Smoke test

After bringing both halves up, hit the API directly to confirm the backend is fine before debugging the UI:

```bash
curl -s http://localhost:8000/health
# {"status": "ok"}

curl -s http://localhost:8000/runs | jq '. | length'
# 1   (or more, depending on how many runs you have)

curl -sN "http://localhost:8000/events?run_id=$(curl -s http://localhost:8000/runs | jq -r '.[0].run_id')" | head -3
# data: {"kind":"turn_start","turn_num":0,"state":{...}}
# data: {"kind":"observation","entity_count":...,...}
# data: {"kind":"llm_prompt","state_summary":"..."}
```

If `/events` hangs with no output, the run id is for a live run on a broker the web server can't see (e.g. you started a CLI race in another process without `ARENA_BROKER_BACKEND=redis` on both ends). See [Runbook: switching-broker-backend](../runbooks/switching-broker-backend.md).

## Fork-and-replay flow

The end-to-end Operator workflow:

1. Open a finalized run from the sidebar.
2. Scrub the timeline to an interesting turn (say t=15).
3. Switch to the **Operator** tab. `parent_t` is pre-filled with 15.
4. Tick "mutation" and set `food: 1500, age: Feudal Age` (or whatever counterfactual you're testing).
5. Set `n_turns: 20`, give a `reason`, submit.
6. The new run appears in the sidebar; the timeline starts streaming live as `_replay` publishes events.
7. Switch to the **Diff** tab to see parent vs child side-by-side.

For this to work you need `ANTHROPIC_API_KEY` exported in the shell that started `arena-web-dev` — the fork replay drives a real LLM. The default fork profile is Haiku at temperature 0.5 (`apps/api/src/forks.py:95`, `DEFAULT_FORK_PROFILE`). Cost per 10-turn fork is roughly the same per-profile cost as one race-instance variant from `arena race` — see Chapter 17 for the breakdown.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Sidebar empty | No DuckDB files under `logs/arena/` | `just arena-smoke` to generate one, or set `ARENA_LOGS_ROOT` to a populated directory. |
| `404 /runs` from the UI | Vite proxy not running, or `VITE_API_BASE_URL` is set but the backend isn't reachable at that URL | `curl` the backend directly; if it works, the issue is in the frontend wiring. |
| SSE shows `Error` status | CORS rejection (you'll see a console error in DevTools) or the backend died | Check the backend logs and `ARENA_WEB_CORS_ORIGINS`. |
| Operator tab returns 500 | `ANTHROPIC_API_KEY` not set on the backend | Export it in the shell that runs `arena-web-dev` and restart. |
| `/events` returns 404 | Run isn't in any DuckDB under `ARENA_LOGS_ROOT` | Confirm with `curl /runs`; the run may have been written to a different log root. |
| Run shows as `Streaming` but no events | Live broker mode but the producer's broker doesn't match the server's | Both producer and server need the same `ARENA_BROKER_BACKEND`. See [Runbook: switching-broker-backend](../runbooks/switching-broker-backend.md). |
| Navigating to `/runs/<id>` shows raw JSON | Dev proxy matched `/runs` before the SPA fallback | The `bypass` on the `/runs` proxy entry handles this — check it wasn't removed from `vite.config.ts`. |
| `/training/*` pages error, arena pages fine | Tracker API on :8100 isn't running | `just training-api-dev`. The two backends are independent. |
| `/training/coverage` loads but everything is zero | Tracker DB is empty | `just training-seed`. |
| Tracker images 404 on `/raw` or `/thumbs` | DB rows point at paths that moved | Re-run `just training-seed`; it upserts by path. |

## Related reading

- [Chapter 19](./19-web-architecture.md) — what the backend does.
- [Chapter 20](./20-fork-and-diff-ui.md) — what each tab does.
- [Chapter 24](../part3-entity-detection/24-training-tracker.md) — the tracker behind `/training/*`.
- [Runbook: redis-broker-ops](../runbooks/redis-broker-ops.md) — bringing up the compose stack when you need cross-process replay.
