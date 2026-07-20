// Grouping for the run list: reconstruct which runs came from the *same*
// `rank`/`race` operation. A single operation spawns many parallel
// `synth_game_loop`s (one per profile × scenario × round), each with its own
// `run_id`, all written to one DuckDB file. The connection is latent in the
// `/runs` payload — these helpers surface it without any backend change.

import type { RunSummary } from "@/lib/events";

/**
 * Stable key shared by every run from one operation.
 *
 * - Completed runs share their DuckDB file path (`db_path`) — one file per
 *   operation (`server.py:_runs_in_file`).
 * - Live runs carry an empty `db_path`, but the broker surfaces the sink's
 *   single `started_at` identically as `first_ts` for every sibling
 *   (`server.py:_live_summaries`), so `label:first_ts` groups them.
 */
export const groupKey = (run: RunSummary): string => {
  return run.db_path !== "" ? run.db_path : `${run.label}:${run.first_ts}`;
};

export type RunGroup = {
  /** `groupKey` of the member runs; unique per operation. */
  readonly key: string;
  /** Shared operation label (`rank` / `race` / `smoke`). */
  readonly label: string;
  readonly runs: readonly RunSummary[];
  readonly firstTs: string;
  readonly lastTs: string;
  readonly totalEvents: number;
  /** True if any member run is still live on the broker. */
  readonly running: boolean;
}

/**
 * Bucket runs into operations, preserving the server's ordering: groups appear
 * in first-seen order (the `/runs` response is newest-first), and runs keep
 * their incoming order within a group. A `Map` preserves insertion order, so
 * no re-sorting is needed.
 */
export const groupRuns = (runs: readonly RunSummary[]): readonly RunGroup[] => {
  const byKey = new Map<string, RunSummary[]>();
  for (const run of runs) {
    const key = groupKey(run);
    const bucket = byKey.get(key);
    if (bucket) {
      bucket.push(run);
    } else {
      byKey.set(key, [run]);
    }
  }

  const groups: RunGroup[] = [];
  for (const [key, members] of byKey) {
    const first = members[0];
    if (first === undefined) {
      continue; // unreachable: a key only exists once a run was pushed.
    }
    groups.push({
      key,
      label: first.label,
      runs: members,
      firstTs: members.reduce(
        (min, r) => (r.first_ts < min ? r.first_ts : min),
        first.first_ts,
      ),
      lastTs: members.reduce(
        (max, r) => (r.last_ts > max ? r.last_ts : max),
        first.last_ts,
      ),
      totalEvents: members.reduce((sum, r) => sum + r.n_events, 0),
      running: members.some((r) => r.status === "running"),
    });
  }
  return groups;
};

/**
 * Every run from `runId`'s operation, in list order (the selected run
 * included). Empty if the run is unknown or has no parallel siblings — callers
 * use that to decide whether a sibling switcher is worth showing.
 */
export const operationRuns = (
  runs: readonly RunSummary[],
  runId: string,
): readonly RunSummary[] => {
  const target = runs.find((r) => r.run_id === runId);
  if (target === undefined) {
    return [];
  }
  const key = groupKey(target);
  const members = runs.filter((r) => groupKey(r) === key);
  return members.length > 1 ? members : [];
};
