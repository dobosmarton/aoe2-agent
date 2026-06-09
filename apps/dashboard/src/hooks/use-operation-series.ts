import { useEffect, useState } from "react";

import { fetchRunSeries } from "@/lib/api";
import type { RunSeries } from "@/lib/events";

export type SeriesStatus = "idle" | "loading" | "ready" | "error";

interface UseOperationSeriesResult {
  series: readonly RunSeries[];
  status: SeriesStatus;
  error: string | null;
}

/**
 * Load per-turn resource trajectories for one operation's DuckDB file. Pass the
 * `db_path` shared by the operation's runs (empty for live operations, which
 * have no finalized file — those stay "idle" with no series). Re-fetches when
 * the path changes.
 */
export function useOperationSeries(dbPath: string | null): UseOperationSeriesResult {
  const [series, setSeries] = useState<readonly RunSeries[]>([]);
  const [status, setStatus] = useState<SeriesStatus>("idle");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (dbPath === null || dbPath === "") {
      setSeries([]);
      setStatus("idle");
      setError(null);
      return;
    }
    const controller = new AbortController();
    setStatus("loading");
    setError(null);
    fetchRunSeries(dbPath, controller.signal)
      .then((result) => {
        setSeries(result);
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
  }, [dbPath]);

  return { series, status, error };
}
