import { useEffect, useState } from "react";

import {
  fetchDatasets,
  fetchImageDetail,
  fetchImages,
  fetchStats,
  type CoverageDto,
  type DatasetSummaryDto,
  type ImageDetailDto,
  type ImagePageDto,
  type LabeledFilter,
} from "@/lib/training-api";

export type LoadStatus = "loading" | "ready" | "error";

interface AsyncResult<T> {
  data: T | null;
  status: LoadStatus;
  error: string | null;
}

/** Shared loader: runs `load` on mount and whenever a dep in `deps` changes,
 * aborting the in-flight request on cleanup. Keeps the three tracker hooks from
 * each re-declaring the same fetch/abort/error dance. */
function useAsync<T>(
  load: (signal: AbortSignal) => Promise<T>,
  deps: readonly unknown[],
): AsyncResult<T> {
  const [data, setData] = useState<T | null>(null);
  const [status, setStatus] = useState<LoadStatus>("loading");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    setStatus("loading");
    setError(null);
    load(controller.signal)
      .then((result) => {
        setData(result);
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
  }, deps);

  return { data, status, error };
}

export function useTrackerStats(): AsyncResult<CoverageDto> {
  return useAsync((signal) => fetchStats(signal), []);
}

export function useTrackerImages(labeled: LabeledFilter, page: number): AsyncResult<ImagePageDto> {
  return useAsync((signal) => fetchImages(labeled, page, signal), [labeled, page]);
}

/** `null` id means "nothing selected" — resolves without touching the network,
 * so the lightbox can mount unconditionally and stay a pure function of state. */
export function useImageDetail(imageId: number | null): AsyncResult<ImageDetailDto | null> {
  return useAsync(
    (signal) => (imageId === null ? Promise.resolve(null) : fetchImageDetail(imageId, signal)),
    [imageId],
  );
}

export function useDatasets(): AsyncResult<readonly DatasetSummaryDto[]> {
  return useAsync((signal) => fetchDatasets(signal), []);
}
