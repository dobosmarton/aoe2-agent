import { useMemo, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { EmptyState } from "@/components/empty-state";
import { allEventKinds, eventsByTurn } from "@/lib/event-utils";
import type { ArenaEvent, EventKind } from "@/lib/events";
import { cn } from "@/lib/utils";

interface TracePanelProps {
  readonly events: readonly ArenaEvent[];
  readonly selectedTurn: number | null;
}

function summarise(event: ArenaEvent): string {
  switch (event.kind) {
    case "turn_start":
      return `turn ${event.turn_num}`;
    case "observation":
      return `${event.entity_count} entities`;
    case "llm_prompt":
      return event.state_summary;
    case "llm_response":
      return `${event.actions.length} actions · $${event.cost_usd.toFixed(5)}`;
    case "action":
      return JSON.stringify(event.action);
    case "action_result":
      return `${event.action_type}${event.state_changed ? "" : " (no-op)"}`;
    case "world_mutation":
      return event.reason;
    case "fork":
      return `from ${event.parent_run_id.slice(0, 8)}@${event.parent_t}`;
    case "metric":
      return `${event.name} = ${event.value}`;
  }
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
    <div className="flex h-full flex-col">
      <div className="border-border flex flex-wrap gap-1 border-b p-3">
        {allKinds.map((kind) => {
          const active = activeKinds.has(kind);
          return (
            <Button
              key={kind}
              variant={active ? "default" : "outline"}
              size="sm"
              className="h-6 px-2 text-[10px]"
              onClick={() => toggleKind(kind)}
            >
              {kind}
            </Button>
          );
        })}
      </div>
      <ScrollArea className="flex-1">
        <ol className="flex flex-col gap-3 p-3">
          {groups.map((group) => {
            const visible = group.events.filter((event) => activeKinds.has(event.kind));
            if (visible.length === 0) {
              return null;
            }
            const highlighted = selectedTurn === group.turn;
            return (
              <li key={group.turn}>
                <Card className={cn(highlighted && "ring-2 ring-ring")}>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-muted-foreground text-xs font-mono">
                      turn {group.turn}
                      <span className="ml-2">({visible.length} events)</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-1.5">
                    {visible.map((event, index) => (
                      <div
                        key={`${group.turn}-${String(index)}`}
                        className="flex items-start gap-2 text-xs"
                      >
                        <Badge variant="outline" className="shrink-0 font-mono">
                          {event.kind}
                        </Badge>
                        <span className="text-foreground/90 break-all">
                          {summarise(event)}
                        </span>
                      </div>
                    ))}
                  </CardContent>
                </Card>
              </li>
            );
          })}
        </ol>
      </ScrollArea>
    </div>
  );
}
