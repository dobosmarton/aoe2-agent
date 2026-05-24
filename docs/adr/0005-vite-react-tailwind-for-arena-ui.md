# ADR 0005 — Vite + React 19 + Tailwind v4 + Radix UI for the arena UI

**Status:** Accepted (2026-05). Shipped as the Phase-2 frontend scaffold.
**Context:** [Chapter 19 — Arena Web Architecture](../part7-arena-web/19-web-architecture.md).

## Decision

Build the arena replay UI as a Vite-bundled React 19 SPA, styled with Tailwind v4 via `@tailwindcss/vite`, with Radix UI primitives wrapped in shadcn-style local components. Use Bun for package management and the dev server. Use plain `EventSource` + custom hooks for SSE; no data-fetching library.

## What we considered

| Option | Pros | Cons |
|---|---|---|
| **Vite + React** | Mature SPA story; great dev UX; fast HMR. | Yet another React app to maintain. |
| **Next.js** | App router, server components. | Server-side machinery wasted on a local-dev tool that needs SSE + REST against an existing FastAPI; routing collisions. |
| **HTMX over FastAPI** | No JS build pipeline; small ops surface. | Timeline scrubber + diff panel are stateful and would fight the model. |
| **Streamlit / Gradio** | Zero frontend code. | Both are opinionated about layout and don't do SSE / live timelines well. |
| **Svelte / SolidJS** | Smaller bundles; finer reactivity. | Team is more familiar with React; the operator UI complexity doesn't justify the switch. |

## Why Vite + React 19 + Tailwind v4

- **Vite dev proxy is the killer feature.** `vite.config.ts` proxies `/runs`, `/events`, `/forks`, `/health` to FastAPI on :8000 with zero ceremony — the SPA uses relative URLs in both dev and prod, only the proxy-vs-direct mode differs. The cross-host case (backend on a VM) is one env var (`VITE_API_BASE_URL`) away.
- **React 19** lets us lean on `use()` for SSE and cleaner suspense; we don't use server components, but React 19 is the modern baseline and the type stories are best.
- **Tailwind v4 with `@tailwindcss/vite`** removes the postcss config pain and gives us CSS-variable-based theming via `src/index.css`. The design surface is small enough (4 tabs + sidebar + scrubber) that utility-first styling is faster than component libraries with strong opinions.
- **Radix UI primitives + shadcn-style local components**: Tabs, ScrollArea, Badge. Local components under `src/components/ui/` so we own the API and styling. Avoids the brittleness of a fully-managed component library.

## Why no data-fetching library

`useRuns` is one `fetch('/runs')` on mount. `useEvents` is `new EventSource(...)`. Adding TanStack Query / SWR would buy retries and caching neither hook actually wants — runs are listed once per session; events are a fresh stream per selected run, and reconnect on overflow is a custom path the library would not understand. 1.5 KB of hook code is the right size.

## Why Bun

- Faster install (`bun install` vs `npm install`) — matters because the justfile recipes call it before every dev session.
- `bun run` matches the script invocation we'd use anyway.
- No lock-file churn drama: `bun.lock` is checked in.
- If someone uses `pnpm` or `npm` instead, it works fine — there's no Bun-specific runtime code, just the package manager.

## Consequences

**Positive**
- Standard Vite setup is well-understood; any frontend dev can ramp instantly.
- Local components mean the design can evolve without library upgrades blocking on third-party releases.
- Smaller dependency graph than a full component library (~10 prod deps vs 30+).
- SPA bundles are static; production deployment is "drop `dist/` behind any HTTP server".

**Negative**
- Tailwind v4 is recent (2025); minor breaking changes are still possible vs v3. The pin in `package.json` insulates us.
- shadcn-style components mean we maintain them; an `npm update` doesn't bring upstream improvements automatically. We accept this — the components are small and stable.
- No SSR / no static site generation. Out of scope for a local-dev tool; would matter only if we shipped this as a public-facing surface.

## Related

- [Chapter 19](../part7-arena-web/19-web-architecture.md) — what the UI does.
- [Chapter 21](../part7-arena-web/21-running-the-ui-locally.md) — dev workflow.
