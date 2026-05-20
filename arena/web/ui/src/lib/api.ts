import type { RunSummary } from "@/lib/events";

export async function fetchRuns(signal?: AbortSignal): Promise<readonly RunSummary[]> {
  const init = signal === undefined ? {} : { signal };
  const response = await fetch("/runs", init);
  if (!response.ok) {
    throw new Error(`GET /runs failed: ${response.status} ${response.statusText}`);
  }
  return (await response.json()) as readonly RunSummary[];
}

export function eventsUrl(runId: string): string {
  return `/events?run_id=${encodeURIComponent(runId)}`;
}
