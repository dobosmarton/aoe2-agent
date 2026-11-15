# Running the arena web stack locally

The backend (`apps/api`) runs with uv and the dashboard (`apps/dashboard`) with Bun. Both start from `justfile` recipes in the agent repo root. The backend reads DuckDB logs from `logs/arena/`; `just arena-smoke` writes one with a mock model and no API key.

## Replaying finished runs

```bash
just arena-web-dev        # backend on 127.0.0.1:8000
just arena-ui-dev         # dashboard on localhost:5173
```

Open `http://localhost:5173`; `/` redirects to `/runs`. The backend uses the in-process broker, which is enough to replay any finalized run and to watch forks created from the dashboard, because the fork replay runs inside the backend process.

| Recipe | Runs |
|---|---|
| `just arena-web-dev [port]` | `uv run --package arena-web aoe2-arena-web --port 8000` |
| `just arena-web-dev-redis [port]` | Same, with `ARENA_BROKER_BACKEND=redis` |
| `just arena-ui-install` | `bun install` at the repo root, which fills the hoisted `node_modules/` for the dashboard and the landing site |
| `just arena-ui-dev` | `arena-ui-install`, then `bun run dev` in `apps/dashboard` |
| `just arena-web-build` | `arena-ui-install`, then `bun run build` (`tsc -b && vite build`) into `apps/dashboard/dist/` |
| `just landing-dev` | `arena-ui-install`, then `astro dev` in `apps/landing` |
| `just landing-build` | `astro build` plus `pagefind --site dist` |
| `just capture-screenshots` | Playwright script that captures dashboard screenshots for the landing page, starting the backend and dashboard if they are not up |

The dashboard also has `bun run typecheck` and `bun run lint`. CI runs the dashboard typecheck and build with `bun --filter arena-web-ui`.

To check the backend without the UI:

```bash
curl -s localhost:8000/health
curl -s localhost:8000/runs | jq length
curl -sN "localhost:8000/events?run_id=<run_id>" | head -3
```

## Watching a CLI run live

An arena CLI run (`race`, `rank`, `smoke`) is a separate process. With the in-process broker its events stay in that process, and its log file is locked by the writer until the run finishes, so the dashboard sees it only afterwards. To tail it live, the CLI and the backend must share a Redis broker:

```bash
just arena-infra-up          # docker compose stack, including Redis
just arena-web-dev-redis     # backend in redis mode
just arena-ui-dev
just arena-rank-redis        # or arena-race-redis
```

`set dotenv-load` in the justfile loads `.env`, so `REDIS_PASSWORD` is in scope and `make_broker()` builds `redis://:<password>@localhost:6379/0` from it. The Redis service in `docker-compose.yml` publishes no host port (only `docker-compose.ci.yml` maps `127.0.0.1:6379`), so a host process reaches it only if a port mapping or `REDIS_URL` points at it. Broker setup and Redis operations are in [switching the broker backend](../runbooks/switching-broker-backend.md) and [Redis broker operations](../runbooks/redis-broker-ops.md).

The sidebar does not poll. A run started after the page loaded appears on reload or when the browser window regains focus.

## Environment variables

| Variable | Read by | Default | Effect |
|---|---|---|---|
| `ARENA_LOGS_ROOT` | backend | `logs/arena` (relative to the working directory) | Directory holding `<date>/*.duckdb` logs; fork logs are written here too |
| `ARENA_WEB_CORS_ORIGINS` | backend, at import | `http://localhost:5173,http://localhost:8000` | Allowed browser origins |
| `ARENA_BROKER_BACKEND` | backend, CLIs | `inprocess` | `inprocess` or `redis`; anything else fails at startup |
| `REDIS_URL` | `make_broker` | unset | Full Redis URL; wins over `REDIS_PASSWORD` |
| `REDIS_PASSWORD` | `make_broker` | unset | Builds an authenticated localhost URL when `REDIS_URL` is unset |
| `AOE2_LLM_API_KEY` | fork replay | empty | Credential for the fork profile's model calls; only `POST /forks` needs it |
| `VITE_API_BASE_URL` | dashboard, at build or dev-server start | empty (same origin) | Backend origin prefixed to every API call |
| `SITE_URL` | landing | `https://aoe2-llm-arena.example.com` | Canonical site URL for the Astro build |

`VITE_API_BASE_URL` goes in `apps/dashboard/.env.local`; `apps/dashboard/.env.example` is the template. Vite inlines it, so a built bundle keeps the value it was built with.

## Dev proxy

`apps/dashboard/vite.config.ts` serves on port 5173 and proxies `/runs`, `/events`, `/forks` and `/health` to `http://localhost:8000`, so the dashboard calls the API with relative URLs and needs no CORS. `/metrics` is not proxied; query it on port 8000.

`/runs` is both an API path and a client route (`/runs`, `/runs/<id>`), and Vite matches proxy rules before its SPA fallback. The `/runs` entry therefore has a `bypass` that serves `/index.html` when the request's `Accept` header contains `text/html`. A browser loading the page sends that header; `fetch` does not.

```ts
"/runs": {
  target: "http://localhost:8000",
  bypass: (req) =>
    req.headers.accept?.includes("text/html") === true ? "/index.html" : undefined,
},
```

## Deployment modes

| Mode | Backend | Dashboard |
|---|---|---|
| Single machine | `just arena-web-dev` | `just arena-ui-dev`; the proxy forwards API calls |
| Backend on another host | `uv run --package arena-web aoe2-arena-web --host 0.0.0.0`, with the dashboard's origin in `ARENA_WEB_CORS_ORIGINS` | `VITE_API_BASE_URL=http://<host>:8000`, then `just arena-ui-dev` |
| Static build | Any of the above | `just arena-web-build`, then serve `apps/dashboard/dist/` from a static server |

The backend does not serve the built bundle. A static deployment either builds with `VITE_API_BASE_URL` set and allows that origin through CORS, or puts the bundle and the API behind one reverse proxy. That proxy needs the same `Accept` split for `/runs` and a fallback to `index.html` for client routes. The backend has no authentication, so it should not face an untrusted network.

## Landing site

`just landing-dev` starts the Astro dev server for `apps/landing`, a static documentation site built from the Markdown under the agent repo's `docs/` directory. It is independent of the backend; only `just capture-screenshots` needs the backend and dashboard running. See [arena web architecture](architecture.md#landing-site).
