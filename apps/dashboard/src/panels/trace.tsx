import { useEffect, useMemo, useRef, useState } from "react";

import { ScrollArea } from "@/components/ui/scroll-area";
import { EmptyState } from "@/components/empty-state";
import { TraceRow } from "@/components/trace/trace-row";
import { eventKindMeta } from "@/lib/event-meta";
import { allEventKinds, eventsByTurn } from "@/lib/event-utils";
import type { ArenaEvent, EventKind } from "@/lib/events";
import { cn } from "@/lib/utils";

type TracePanelProps = {
  readonly events: readonly ArenaEvent[];
  readonly selectedTurn: number | null;
}

export function TracePanel({
  events,
  selectedTurn,
}: TracePanelProps): React.ReactElement {
  const allKinds = allEventKinds();
  const [activeKinds, setActiveKinds] = useState<ReadonlySet<EventKind>>(
    () => new Set(allKinds),
  );

  const groups = useMemo(() => eventsByTurn(events), [events]);
  const counts = useMemo(() => {
    const map = new Map<EventKind, number>();
    for (const event of events) {
      map.set(event.kind, (map.get(event.kind) ?? 0) + 1);
    }
    return map;
  }, [events]);

  const visibleCount = useMemo(
    () => events.filter((e) => activeKinds.has(e.kind)).length,
    [events, activeKinds],
  );

  const scrollRootRef = useRef<HTMLDivElement>(null);

  // Scroll the trace list to whichever turn the timeline scrubber lands on.
  // Keyed on `selectedTurn` only: depending on `groups` too would re-scroll on
  // every streamed event, yanking a user who has scrubbed back to read.
  // Debounced: every change clears the pending timer, so a drag coalesces into a
  // single smooth scroll once the slider settles — no competing animations. The
  // delay also lets layout settle, which matters when the tab first mounts.
  useEffect(() => {
    if (selectedTurn === null) {
      return;
    }
    const timer = window.setTimeout(() => {
      const target = scrollRootRef.current?.querySelector(
        `[data-turn="${String(selectedTurn)}"]`,
      );
      target?.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 150);
    return (): void => {
      window.clearTimeout(timer);
    };
  }, [selectedTurn]);

  if (events.length === 0) {
    return <EmptyState title="No events yet" hint="Waiting for the SSE stream…" />;
  }

  function toggleKind(kind: EventKind): void {
    setActiveKinds((current) => {
      const next = new Set(current);
      if (next.has(kind)) {
        next.delete(kind);
      } else {
        next.add(kind);
      }
      return next;
    });
  }

  return (
    <div ref={scrollRootRef} className="flex h-full flex-col">
      {/* Filter bar doubles as a color legend. */}
      <div className="border-border flex flex-wrap items-center gap-1 border-b p-2">
        {allKinds.map((kind) => {
          const active = activeKinds.has(kind);
          const meta = eventKindMeta[kind];
          return (
            <button
              key={kind}
              type="button"
              onClick={() => {
                toggleKind(kind);
              }}
              className={cn(
                "flex items-center gap-1.5 rounded-md border px-2 py-1 font-mono text-[11px] transition-all",
                active
                  ? "border-border bg-accent/60 text-foreground"
                  : "border-transparent text-muted-foreground opacity-50 hover:opacity-100",
              )}
            >
              <span
                className="inline-block size-2 rounded-[2px]"
                style={{ backgroundColor: meta.colorVar }}
              />
              {kind}
              <span className="tabular-nums opacity-70">
                {counts.get(kind) ?? 0}
              </span>
            </button>
          );
        })}
        <span className="text-muted-foreground ml-auto pr-1 font-mono text-[11px] tabular-nums">
          {visibleCount} / {events.length} events
        </span>
      </div>

      {/* Dense, expandable rows grouped by turn. Streams are finite replays, so
          virtualization is unnecessary; swap ScrollArea for @tanstack/react-virtual
          if that ever changes. `min-h-0` lets this flex child shrink below its
          content height so the internal scroll engages instead of overflowing. */}
      <ScrollArea className="min-h-0 flex-1">
        <div className="flex flex-col pb-4">
          {groups.map((group) => {
            const visible = group.events.filter((event) =>
              activeKinds.has(event.kind),
            );
            if (visible.length === 0) {
              return null;
            }
            const highlighted = selectedTurn === group.turn;
            return (
              <div key={group.turn} data-turn={group.turn}>
                <div
                  className={cn(
                    "border-border/50 bg-background/95 sticky top-0 z-10 flex items-center gap-2 border-b px-3 py-1 font-mono text-[11px] font-semibold uppercase tracking-wide backdrop-blur",
                    highlighted ? "text-foreground" : "text-muted-foreground",
                  )}
                >
                  <span>turn {group.turn}</span>
                  <span className="opacity-60">· {visible.length} events</span>
                  {highlighted ? (
                    <span className="bg-ring ml-1 inline-block size-1.5 rounded-full" />
                  ) : null}
                </div>
                <div className={cn("flex flex-col", highlighted && "bg-accent/15")}>
                  {visible.map((event, index) => (
                    <TraceRow key={`${String(group.turn)}-${String(index)}`} event={event} />
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      </ScrollArea>
    </div>
  );
}
