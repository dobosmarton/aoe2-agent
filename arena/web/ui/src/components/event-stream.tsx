import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { EmptyState } from "@/components/empty-state";
import type { SseStatus } from "@/hooks/use-events";
import type { ArenaEvent } from "@/lib/events";

interface EventStreamProps {
  readonly runId: string | null;
  readonly events: readonly ArenaEvent[];
  readonly status: SseStatus;
}

function statusLabel(status: SseStatus): string {
  switch (status) {
    case "idle":
      return "Select a run to start";
    case "connecting":
      return "Connecting…";
    case "open":
      return "Streaming";
    case "closed":
      return "Stream complete";
    case "error":
      return "Stream error";
  }
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
      return `${event.actions.length} actions, $${event.cost_usd.toFixed(5)}`;
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

export function EventStream({
  runId,
  events,
  status,
}: EventStreamProps): React.ReactElement {
  if (runId === null) {
    return (
      <EmptyState
        title="No run selected"
        hint="Pick one from the sidebar to start streaming events."
      />
    );
  }

  return (
    <div className="flex h-full flex-col">
      <header className="border-border bg-card flex items-center justify-between border-b px-4 py-2">
        <div className="font-mono text-xs">{runId}</div>
        <Badge variant={status === "open" ? "default" : "outline"}>
          {statusLabel(status)} · {events.length} events
        </Badge>
      </header>
      <ScrollArea className="flex-1">
        <ol className="flex flex-col gap-2 p-3">
          {events.map((event, index) => (
            <li key={`${runId}-${String(index)}`}>
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="flex items-center justify-between text-xs">
                    <Badge variant="outline">{event.kind}</Badge>
                    <span className="text-muted-foreground font-mono">
                      #{index + 1}
                    </span>
                  </CardTitle>
                </CardHeader>
                <CardContent className="text-foreground/90 text-sm">
                  {summarise(event)}
                </CardContent>
              </Card>
            </li>
          ))}
          {events.length === 0 && status !== "error" ? (
            <EmptyState title="Waiting for events…" />
          ) : null}
        </ol>
      </ScrollArea>
    </div>
  );
}
