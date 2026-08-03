# ADR 0006 — TanStack Router + Query and React Aria for the dashboard

**Status:** Accepted (2026-07). Shipped in the routing/tracker migration.
**Supersedes:** two sub-decisions of [ADR 0005](./0005-vite-react-tailwind-for-arena-ui.md) — "Radix UI primitives" and "no data-fetching library". The rest of 0005 (Vite, React 19, Tailwind v4, Bun) is unchanged.
**Context:** [Chapter 19 — Arena Web Architecture](../part7-arena-web/19-web-architecture.md), [Chapter 24 — Detection Training Tracker](../part3-entity-detection/24-training-tracker.md).

## Decision

Adopt **TanStack Router** (file-based routes, typed search params) for navigation, **TanStack Query** for all server reads and writes, and **React Aria Components** in place of Radix UI as the headless primitive layer. Keep `useEvents` as a hand-written `EventSource` hook — it is not a query and should not pretend to be one. Enable the **React Compiler** babel plugin.

## What changed since ADR 0005

0005 was written when the dashboard was one screen: pick a run, scrub it, fork it. Two things broke its assumptions.

1. **A second surface arrived.** The detection training tracker (Chapter 24) added `/training/coverage` and `/training/images` — a different backend, a different data model, and *write* operations (approve / reclassify / re-box an annotation). The app went from one view to two independent sections.
2. **State stopped fitting in one component.** With a run detail view, an experiment overview, an image table, and a lightbox, "which thing is selected" became something users wanted to link to and reload into.

## What we considered

| Option | Pros | Cons |
|---|---|---|
| **Stay hand-rolled** | Zero new deps; 0005's argument still applies to a single screen. | `App.tsx` was already carrying two effects just to reconcile scrubber position with the stream. Adding a second section and write-invalidations would have meant hand-writing a cache. |
| **React Router** | Ubiquitous; large ecosystem. | Search params are untyped strings; we'd hand-validate `?turn=` and `?tab=` at every read. No loader/cache integration story as tight as Router+Query. |
| **TanStack Router + Query** *(chosen)* | Typed search params validated once per route; loaders prime the Query cache; one invalidation contract for writes. | Two more deps; the generated `routeTree.gen.ts` is a build artifact that must be committed. |
| **Keep Radix, add Router/Query** | Smaller diff. | Doesn't fix the accessibility gap on the new interactive surfaces (see below). |

## Why TanStack Router

- **Typed search params are the feature.** `validateSearch` runs once per route and yields a typed object, so `?turn=17&tab=trace` is parsed and bounds-checked in one place instead of at every `useSearchParams` call site. Invalid values are dropped rather than crashing the view.
- **Pathless layouts express the real hierarchy.** `_arena` groups `/runs/*` and `/experiments/*` under one sidebar instance without appearing in the URL, so cross-section navigation doesn't remount the run list.
- **Loaders remove the fetch-on-mount waterfall.** `loader: ({context}) => context.queryClient.ensureQueryData(...)` starts the request during navigation; the component reads the same key and gets a warm cache.

## Why TanStack Query

0005 said: *"Adding TanStack Query / SWR would buy retries and caching neither hook actually wants."* That was true of `useRuns`. It stopped being true when the tracker added writes.

The concrete need is **invalidation**: approving an annotation changes the open image's detail, the image list's labeled counts, *and* the coverage matrix. Hand-rolled, that is three refetches wired at each of four call sites. With hierarchical query keys it is one line — `invalidateQueries({queryKey: ["tracker"]})` — owned by a single hook (`use-annotation-mutations.ts`), which is why the review rail and the box editor cannot drift apart. The same pattern gives the Operator panel its fork flow: invalidate `["runs"]`, *then* navigate.

Note what did **not** move into Query: `useEvents`. A long-lived `EventSource` that appends over minutes and reconnects on a custom `overflow` signal is not a request/response cache entry. Forcing it into `useQuery` would have meant fighting the library's staleness model for no gain. **Use the cache for things that are fetched; keep the stream hand-written.**

## Why React Aria over Radix

- **Accessibility is the actual deliverable.** The new tracker surfaces are keyboard-heavy — a lightbox with zoom/pan, drag-to-resize annotation boxes, a class picker over 60 entries. React Aria ships focus management, drag interactions, and screen-reader semantics as behaviour hooks rather than leaving them to the consumer.
- **One primitive family, not two.** Mixing Radix's `value`/`onValueChange` with React Aria's `selectedKey`/`onSelectionChange` across sections would be a permanent papercut. Converting the existing handful of components (`Tabs`, `Badge`, `ScrollArea`, `Slider`) was a bounded, one-time cost.
- **Still headless and still local.** Components live under `src/components/ui/` exactly as before, so 0005's "we own the API and styling" property is preserved.

## Why the React Compiler

Panels are re-rendered on every SSE message. Rather than hand-placing `useMemo`/`useCallback` — and reviewing every PR for missing ones — the compiler memoises automatically. It bails out silently when it cannot compile a component, so the `react-hooks` lint rules are kept on as the visibility mechanism.

## Consequences

**Positive**
- View state is linkable, reload-survivable, and resets correctly on navigation for free.
- One invalidation contract for every write; no bespoke cache code.
- Two effects deleted from the run-detail view (pin-to-latest, reset-on-run-change) — both are now consequences of `turn ?? maxTurn`.
- Accessibility on the new interactive surfaces comes from the primitive layer.

**Negative**
- `routeTree.gen.ts` is generated but **must be committed**: `bun run build` runs `tsc -b` before Vite, so a fresh clone would fail typecheck before the router plugin ever wrote the file. It will show up in diffs whenever routes change.
- Plugin ordering is load-bearing — `tanstackRouter()` must precede `react()` in `vite.config.ts`, or the React refresh transform sees unrewritten route modules.
- Three more production dependencies than 0005's "~10 prod deps" boast.
- `/runs` is now both an API path and a client route, requiring an `Accept: text/html` bypass in the dev proxy (Chapter 19).

## Related

- [Chapter 19](../part7-arena-web/19-web-architecture.md) — the route table and the three-way state split.
- [Chapter 20](../part7-arena-web/20-fork-and-diff-ui.md) — the tabs and panels.
- [Chapter 24](../part3-entity-detection/24-training-tracker.md) — the surface that forced the change.
- [ADR 0005](./0005-vite-react-tailwind-for-arena-ui.md) — the scaffold this amends.
