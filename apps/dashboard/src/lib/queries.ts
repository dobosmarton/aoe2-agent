// Every server read in the dashboard, as TanStack Query option factories.
//
// The transport layer is untouched: `lib/api.ts` and `lib/training-api.ts`
// already take an optional AbortSignal, which is exactly what Query hands to a
// queryFn — so these are thin wrappers, not a reimplementation.
//
// Key prefixes are hierarchical so a single invalidate can sweep a subtree:
// invalidating ["runs"] also refreshes ["runs","summaries"] and ["runs","series"].

import { queryOptions } from "@tanstack/react-query";

import { fetchRunSeries, fetchRunSummaries, fetchRuns } from "@/lib/api";
import type { RunMetrics } from "@/lib/events";
import {
  fetchClasses,
  fetchDatasets,
  fetchImageDetail,
  fetchImages,
  fetchStats,
  type LabeledFilter,
} from "@/lib/training-api";

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

// -- detection training tracker ---------------------------------------------

export const trackerClassesQueryOptions = () =>
  queryOptions({
    queryKey: ["tracker", "classes"] as const,
    queryFn: ({ signal }) => fetchClasses(signal),
    staleTime: Infinity, // class schema only changes when classes.yaml does
  });

export const trackerStatsQueryOptions = () =>
  queryOptions({
    queryKey: ["tracker", "stats"] as const,
    queryFn: ({ signal }) => fetchStats(signal),
  });

export const trackerDatasetsQueryOptions = () =>
  queryOptions({
    queryKey: ["tracker", "datasets"] as const,
    queryFn: ({ signal }) => fetchDatasets(signal),
  });

export const trackerImagesQueryOptions = (labeled: LabeledFilter, page: number) =>
  queryOptions({
    queryKey: ["tracker", "images", { labeled, page }] as const,
    queryFn: ({ signal }) => fetchImages(labeled, page, signal),
  });

export const trackerImageDetailQueryOptions = (imageId: number) =>
  queryOptions({
    queryKey: ["tracker", "images", imageId] as const,
    queryFn: ({ signal }) => fetchImageDetail(imageId, signal),
  });
