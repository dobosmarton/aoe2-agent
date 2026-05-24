# `arena/web/` — Arena Replay Backend + UI

FastAPI + SSE backend, plus a Vite/React/Tailwind SPA, for inspecting arena event logs and creating operator-driven forks.

## What's here

```
arena/web/
├── __main__.py     # CLI: python -m arena.web --port 8000
├── server.py       # FastAPI app: /runs, /events, /forks, /metrics, /health
├── forks.py        # /forks handler + create_fork + background replay
└── ui/             # Vite + React 19 + Tailwind v4 + Radix UI primitives
    ├── package.json
    ├── vite.config.ts       # dev proxy for /runs, /events, /forks, /health
    ├── .env.example         # VITE_API_BASE_URL for cross-host dev
    └── src/
        ├── App.tsx          # 2-column layout: sidebar + Tabs (World/Trace/Diff/Operator) + Timeline
        ├── components/      # ui/, run-list, timeline, state-summary, empty-state
        ├── hooks/           # use-runs, use-events
        ├── lib/             # api, event-utils, events, utils
        └── panels/          # world, trace, diff, operator
```

## Two-terminal dev

```bash
# Backend on :8000
just arena-web-dev

# Frontend on :5173 (proxies API calls to :8000)
just arena-ui-dev
```

Browse <http://localhost:5173>, pick a run.

## URL contract

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness. |
| `GET` | `/runs` | List runs from every DuckDB under `ARENA_LOGS_ROOT`. |
| `GET` | `/events?run_id=X&from_seq=N` | SSE — live broker for active runs, cold DuckDB scan for finalized. |
| `POST` | `/forks` | Snapshot + optionally mutate + schedule N-turn async replay. |
| `GET` | `/metrics` | Broker counters (`events_published`, `events_streamed`, `streams_dropped`, `runs_open`). |

## Reading order

- [Chapter 19 — Web Architecture](../../docs/part7-arena-web/19-web-architecture.md) — backend lifespan, dependencies, `/events` dispatch.
- [Chapter 20 — Fork and Diff UI](../../docs/part7-arena-web/20-fork-and-diff-ui.md) — what each tab does.
- [Chapter 21 — Running the UI Locally](../../docs/part7-arena-web/21-running-the-ui-locally.md) — dev workflow, env vars, troubleshooting.
- [ADR 0005](../../docs/adr/0005-vite-react-tailwind-for-arena-ui.md) — frontend stack choice.

## Notes for contributors

- The lifecycle ordering inside `forks.py:_replay` and `create_fork` is load-bearing. Don't reorder without a test — see the banner comments in source.
- The SSE handler's broker/cold dispatch in `server.py:events()` relies on byte-equivalent serialization between the broker and `stream_cold`. The `test_payload_roundtrip_is_byte_stable` test guards this.
- The UI is intentionally library-light: no TanStack Query, no Redux. State is in three hooks + local component state.
