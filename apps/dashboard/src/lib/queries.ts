// Every server read in the dashboard, as TanStack Query option factories.
//
// `lib/api.ts` takes an optional AbortSignal, which Query hands to each
// queryFn. These are thin wrappers around the arena API.
//
// Key prefixes are hierarchical so a single invalidate can sweep a subtree:
// invalidating ["runs"] also refreshes ["runs","summaries"] and ["runs","series"].

import { queryOptions } from "@tanstack/react-query";

import { fetchRunSeries, fetchRunSummaries, fetchRuns } from "@/lib/api";
import type { RunMetrics } from "@/lib/events";

export const runsQueryOptions = () =>
  queryOptions({
    queryKey: ["runs"] as const,
    queryFn: ({ signal }) => fetchRuns(signal),
  });

/** End-of-run metrics, indexed by run_id for the experiment overview's join.
 * The Map is built in `select` so it is memoised by the cache rather than
 * rebuilt on every render. */
export const runSummariesQueryOptions = () =>
  queryOptions({
    queryKey: ["runs", "summaries"] as const,
    queryFn: ({ signal }) => fetchRunSummaries(signal),
    select: (rows): ReadonlyMap<string, RunMetrics> =>
      new Map(rows.map((m) => [m.run_id, m])),
  });

/** Per-turn trajectories for one operation's DuckDB file. Live operations have
 * no finalized file and pass "", which disables the query instead of firing a
 * request that would 404. */
export const runSeriesQueryOptions = (dbPath: string) =>
  queryOptions({
    queryKey: ["runs", "series", dbPath] as const,
    queryFn: ({ signal }) => fetchRunSeries(dbPath, signal),
    enabled: dbPath !== "",
  });
