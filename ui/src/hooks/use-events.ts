import { useEffect, useState } from "react";

import { eventsUrl } from "@/lib/api";
import type { ArenaEvent } from "@/lib/events";

export type SseStatus = "idle" | "connecting" | "open" | "closed" | "error";

interface UseEventsResult {
  events: readonly ArenaEvent[];
  status: SseStatus;
}

export function useEvents(runId: string | null): UseEventsResult {
  const [events, setEvents] = useState<readonly ArenaEvent[]>([]);
  const [status, setStatus] = useState<SseStatus>("idle");

  useEffect(() => {
    if (runId === null) {
      setEvents([]);
      setStatus("idle");
      return;
    }

    setEvents([]);
    setStatus("connecting");
    const source = new EventSource(eventsUrl(runId));

    source.onopen = () => {
      setStatus("open");
    };
    source.onmessage = (message: MessageEvent<string>) => {
      const parsed = JSON.parse(message.data) as ArenaEvent;
      setEvents((prev) => [...prev, parsed]);
    };
    // EventSource fires `error` both on transient hiccups (browser retries
    // automatically) and on the final "no more data, server closed" close.
    // For 7.2's finite replay streams, treat error as end-of-stream and
    // surface the difference between "completed cleanly" and "never opened".
    source.onerror = () => {
      setStatus((current) => (current === "open" ? "closed" : "error"));
      source.close();
    };

    return () => {
      source.close();
    };
  }, [runId]);

  return { events, status };
}
