# Chapter 20 — Fork and Diff UI

The four tabs in `apps/dashboard/src/App.tsx` are the operator's main surface:

| Tab | Component | Purpose |
|---|---|---|
| World | `src/panels/world.tsx` | Current resources / population / age / buildings at the scrubber's turn. |
| Trace | `src/panels/trace.tsx` | LLM reasoning + actions for the selected turn, with the live event stream. |
| Diff | `src/panels/diff.tsx` | Side-by-side parent vs child state when viewing a forked run. |
| Operator | `src/panels/operator.tsx` | Form to POST a `/forks` request: pick `parent_t`, optionally edit a `MutationPatch`, submit. |

<aside class="prereqs">

React hooks (`useState`, `useEffect`, `useMemo`). [Chapter 19 — Web Architecture](./19-web-architecture.md) for the SSE backend this UI consumes.

</aside>

The cross-cutting infrastructure is the **Timeline scrubber** (`src/components/timeline.tsx`) at the bottom of `main` and the **event hook** (`src/hooks/use-events.ts`).

## Timeline scrubber

Lives outside the Tabs so the scrubber position survives tab switches. The pin-to-latest behaviour (`App.tsx:42–53`) is what makes the UI feel "live":

- If you're at the latest turn and a new turn arrives, the scrubber auto-advances.
- If you scrubbed back to turn 17 to inspect something, new turns at 18, 19, 20 do not move you.

The scrubber's `selectedTurn` state is also the input to the World and Trace panels. The Diff panel ignores it (it shows parent vs child final states); the Operator panel takes `initialParentT={selectedTurn}` as the default for the form's `parent_t` field, so "scrub to interesting turn, switch to Operator, hit Fork" is a fluid workflow.

## Event utilities

`src/lib/event-utils.ts` does the heavy lifting of turning a flat `events: Event[]` array into per-turn state. Two key derivations consumed by the panels:

- `statesByTurn(events)` — folds `turn_start` snapshots into a `Map<turnNumber, WorldStateSnapshot>`. The World panel renders the entry for `selectedTurn`.
- `lastTurn(events)` — the maximum `t` seen so far. Drives the scrubber's right edge and the pin-to-latest check.

Both functions are pure; the App `useMemo`s them so React only recomputes when `events` changes (which is once per SSE message).

## Diff panel

`src/panels/diff.tsx`. The panel takes:

- `events` — the **current** run's events (whatever's selected in the run list).
- `currentRunId` — for the header.
- `onOpenRun` — callback to swap the selected run.

It looks at the first `fork` event in the current run, extracts `parent_run_id`, fetches the parent's `/events` stream separately, and renders the two final states side-by-side. The two `WorldMutationPayload` summaries (`before_summary` / `after_summary`) are surfaced as the explicit "what the operator changed" annotation between parent and child.

Clicking the parent's run-id in the diff header calls `onOpenRun(parent_run_id)` — the run-list selection updates, the panel rerenders with the parent as "current", and the diff inverts. The roundtrip is what makes fork lineage navigable.

## Operator panel

`src/panels/operator.tsx`. The form fields map directly to `ForkRequest` (`apps/api/src/forks.py:77`):

- `parent_run_id` — pre-filled with the currently selected run.
- `parent_t` — pre-filled with the scrubber's `selectedTurn`.
- `mutation` — seven optional override fields (food, wood, gold, stone, population, pop_cap, age). Unset fields are omitted from the JSON body so the backend treats them as "don't mutate".
- `n_turns` — default 10, bounded `0 ≤ n ≤ 200` by the Pydantic model.
- `reason` — free-text annotation that ends up in the `WorldMutationPayload.reason` field, so post-mortem queries can find the operator's intent.

On submit, the panel POSTs to `/forks`, awaits the `ForkResponse`, and calls `onOpenRun(child_run_id)`. The new run appears in the run-list (the SSE stream keeps `useRuns` up to date) and the timeline starts streaming live as `_replay` publishes events. Watching the same agent replay from a mutated state — and comparing against the parent in the Diff tab — is the workflow the whole arena buildout exists to enable.

## What the SSE stream looks like to the UI

`src/hooks/use-events.ts` wraps `EventSource`. Each line comes in as `event.data = "<payload_json>"`. The hook `JSON.parse`s it, asserts the `kind` discriminator is one of the known nine (Chapter 16), and appends to the events array. On the special `event: overflow` line (named event, not default `data`), the hook reads `available_from`, closes the EventSource, and immediately reopens it with `?from_seq=<available_from>` — the user sees a brief reconnect status, never silently-lost events.

`SseStatus` is a 5-state union (`idle | connecting | open | closed | error`) surfaced via the Badge in the page header (`App.tsx:79–86`). `closed` is the normal end state for a finalized run.

## What's intentionally *not* in the UI

- **No editing of past events.** The event log is append-only. Forking is the supported way to ask "what if".
- **No multi-run overlay.** The Diff panel shows two runs but the World/Trace panels are single-run. Cross-run comparison at the metric level is the `arena rank` table's job.
- **No agent-level controls.** Pause / step / inject — none of those exist. The agent is the synth_game_loop, which runs end-to-end; the operator surface intervenes at fork-time only.

## Component primitives

The UI uses Radix UI primitives wrapped in shadcn-style components under `src/components/ui/`. Three you'll see across panels: `Tabs` (the four-tab nav), `Badge` (status indicators), `ScrollArea` (the trace panel's long event list). Tailwind v4 with `@tailwindcss/vite` provides the styling tokens (CSS variables in `src/index.css`).

If you're adding a new tab or panel, the pattern is: define the component under `src/panels/`, import it in `App.tsx`, add a `TabsTrigger` + `TabsContent` pair. The `selectedTurn` and `events` props are the ambient context — pass them through.

## Related reading

- [Chapter 19](./19-web-architecture.md) — backend wiring and lifespan.
- [Chapter 21](./21-running-the-ui-locally.md) — how to actually run all this.
- [Chapter 16](../part6-evaluation-arena/16-duckdb-persister-and-replay.md) — the `fork()` primitive that `/forks` wraps.
