import { useEffect, useState } from "react";

import { fetchRunSummaries } from "@/lib/api";
import type { RunMetrics } from "@/lib/events";

export type SummariesStatus = "loading" | "ready" | "error";

interface UseRunSummariesResult {
  /** Per-run metrics keyed by run_id. Empty until the fetch resolves. */
  metricsByRunId: ReadonlyMap<string, RunMetrics>;
  status: SummariesStatus;
  error: string | null;
}

/**
 * Fetch-on-mount loader for `GET /runs/summaries`, indexed by run_id so the
 * experiment overview can join an operation's runs to their end-of-run
 * metrics. Mirrors `use-runs.ts` (single fetch; reload to refresh).
 */
export function useRunSummaries(): UseRunSummariesResult {
  const [metricsByRunId, setMetricsByRunId] = useState<
    ReadonlyMap<string, RunMetrics>
  >(() => new Map());
  const [status, setStatus] = useState<SummariesStatus>("loading");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    setStatus("loading");
    setError(null);
    fetchRunSummaries(controller.signal)
      .then((result) => {
        setMetricsByRunId(new Map(result.map((m) => [m.run_id, m])));
        setStatus("ready");
      })
      .catch((err: unknown) => {
        if (controller.signal.aborted) {
          return;
        }
        setError(err instanceof Error ? err.message : String(err));
        setStatus("error");
      });
    return () => {
      controller.abort();
    };
  }, []);

  return { metricsByRunId, status, error };
}
