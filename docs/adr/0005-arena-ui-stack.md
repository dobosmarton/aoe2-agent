# ADR 0005: Arena UI stack

Status: accepted.

## Context

The arena dashboard is an operator tool over an existing FastAPI backend: a run list, a run page with a turn scrubber and four tabs fed by an SSE stream, an experiment overview, and one write path (`POST /forks`). View state such as the selected turn and tab should be linkable and survive a reload. After a fork the run list must refresh before the app navigates to the child.

## Decision

| Concern | Choice |
|---|---|
| Build and dev server | Vite, with a dev proxy to the API on :8000 |
| UI library | React 19, with the React Compiler Babel plugin |
| Styling | Tailwind CSS 4 through `@tailwindcss/vite`, theme tokens as CSS variables in `src/index.css` |
| Routing | TanStack Router, file-based routes, typed search params via `validateSearch` |
| Server state | TanStack Query for every read and for the fork mutation; the SSE stream through `experimental_streamedQuery` |
| Primitives | React Aria Components, wrapped as local shadcn components (style `aria-nova`) in `src/components/ui/` |
| Package manager | Bun, one workspace with `apps/landing` |

## Consequences

- Route loaders call `queryClient.ensureQueryData`, so data loads during navigation, and hierarchical keys let one `invalidateQueries({ queryKey: ["runs"] })` refresh the run list, summaries and series.
- `?turn=` and `?tab=` are parsed once per route; the run page follows the stream with `turn ?? maxTurn` and needs no effects for pinning or resetting.
- `src/routeTree.gen.ts` is generated but committed, because `bun run build` runs `tsc -b` before Vite writes it.
- `tanstackRouter()` must precede `react()` in `vite.config.ts`.
- `/runs` is both an API path and a client route, so the dev proxy needs an `Accept: text/html` bypass.
- The React Compiler skips components it cannot compile without failing the build; the `react-hooks` lint rules report those cases.
- The SSE hook depends on an API TanStack Query marks experimental.
- Local components are owned code: library upgrades do not reach them automatically.

## Alternatives considered

- Next.js: server rendering and its router add nothing to a client that only consumes a separate FastAPI service.
- HTMX on FastAPI: the scrubber and the panels share client state derived from a growing event stream, which fits a client-side component model better.
- Streamlit or Gradio: fixed layouts and weak support for SSE timelines.
- Svelte or SolidJS: smaller bundles, but the UI does not need them and React is the familiar option.
- React Router: search params are untyped strings that every reader would validate itself.
- Hand-written fetch hooks: the fork flow needs cache invalidation across several reads, which Query provides.
- Radix UI primitives: React Aria supplies focus management, keyboard interaction and screen-reader semantics as behaviour, and one primitive family avoids mixing prop conventions.
