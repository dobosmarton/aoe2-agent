import {
  experimental_streamedQuery as streamedQuery,
  queryOptions,
  useQuery,
} from "@tanstack/react-query";

import { eventsUrl } from "@/lib/api";
import type { ArenaEvent } from "@/lib/events";

export type SseStatus = "idle" | "connecting" | "open" | "closed" | "error";

interface UseEventsResult {
  events: readonly ArenaEvent[];
  status: SseStatus;
}

const EMPTY: readonly ArenaEvent[] = [];

/**
 * Bridge an `EventSource` into an AsyncIterable so `streamedQuery` can
 * accumulate it into the cache.
 *
 * EventSource reports *both* a clean end-of-replay and a failed connect through
 * `onerror` — whether the socket ever opened is the only thing that
 * distinguishes them. That distinction becomes `return` (clean) vs `throw`
 * (failed), which Query then surfaces as success vs error.
 */
async function* eventStream(
  runId: string,
  signal: AbortSignal,
): AsyncGenerator<ArenaEvent> {
  const source = new EventSource(eventsUrl(runId));
  const pending: ArenaEvent[] = [];
  let wake: (() => void) | null = null;
  let finished = false;
  let failure: Error | null = null;
  let opened = false;

  const nudge = (): void => {
    const resume = wake;
    wake = null;
    resume?.();
  };

  source.onopen = () => {
    opened = true;
    nudge();
  };
  source.onmessage = (message: MessageEvent<string>) => {
    pending.push(JSON.parse(message.data) as ArenaEvent);
    nudge();
  };
  source.onerror = () => {
    if (!opened) {
      failure = new Error(`Event stream for ${runId} failed to open`);
    }
    finished = true;
    source.close();
    nudge();
  };

  const onAbort = (): void => {
    finished = true;
    nudge();
  };
  signal.addEventListener("abort", onAbort);

  try {
    for (;;) {
      while (pending.length > 0) {
        const next = pending.shift();
        if (next !== undefined) {
          yield next;
        }
      }
      if (finished) {
        if (failure !== null) {
          throw failure;
        }
        return;
      }
      await new Promise<void>((resolve) => {
        wake = resolve;
      });
    }
  } finally {
    signal.removeEventListener("abort", onAbort);
    source.close();
  }
}

/**
 * `staleTime: 0` so revisiting a run re-subscribes rather than replaying a
 * cached array — a finished run would be safe to cache, but a live one would
 * silently stop updating.
 */
export const runEventsQueryOptions = (runId: string | null) =>
  queryOptions({
    queryKey: ["events", runId] as const,
    queryFn: streamedQuery({
      streamFn: ({ signal }) => eventStream(runId ?? "", signal),
    }),
    enabled: runId !== null,
    staleTime: 0,
    gcTime: 0,
  });

/** Derive the five-state SSE status the UI already speaks from Query's state. */
export function useEvents(runId: string | null): UseEventsResult {
  const query = useQuery(runEventsQueryOptions(runId));

  const status: SseStatus =
    runId === null
      ? "idle"
      : query.isError
        ? "error"
        : query.fetchStatus === "fetching"
          ? query.data === undefined
            ? "connecting"
            : "open"
          : query.isSuccess
            ? "closed"
            : "connecting";

  return { events: query.data ?? EMPTY, status };
}
