import { useEffect, useState } from "react";

import { fetchRuns } from "@/lib/api";
import type { RunSummary } from "@/lib/events";

export type RunsStatus = "loading" | "ready" | "error";

interface UseRunsResult {
  runs: readonly RunSummary[];
  status: RunsStatus;
  error: string | null;
}

export function useRuns(): UseRunsResult {
  const [runs, setRuns] = useState<readonly RunSummary[]>([]);
  const [status, setStatus] = useState<RunsStatus>("loading");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    setStatus("loading");
    setError(null);
    fetchRuns(controller.signal)
      .then((result) => {
        setRuns(result);
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

  return { runs, status, error };
}
