import { useEffect, useMemo, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { RunList } from "@/components/run-list";
import { Timeline } from "@/components/timeline";
import { useEvents } from "@/hooks/use-events";
import { useRuns } from "@/hooks/use-runs";
import { lastTurn, statesByTurn } from "@/lib/event-utils";
import { DiffPanel } from "@/panels/diff";
import { TracePanel } from "@/panels/trace";
import { WorldPanel } from "@/panels/world";
import type { SseStatus } from "@/hooks/use-events";

function statusLabel(status: SseStatus): string {
  switch (status) {
    case "idle":
      return "Idle";
    case "connecting":
      return "Connecting…";
    case "open":
      return "Streaming";
    case "closed":
      return "Complete";
    case "error":
      return "Error";
  }
}

export function App(): React.ReactElement {
  const { runs, status: runsStatus, error: runsError } = useRuns();
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const { events, status: sseStatus } = useEvents(selectedRunId);
  const [selectedTurn, setSelectedTurn] = useState<number | null>(null);

  const states = useMemo(() => statesByTurn(events), [events]);
  const maxTurn = useMemo(() => lastTurn(events), [events]);

  // Auto-advance the scrubber as new turns stream in, but only while it's
  // pinned to the latest turn. If the user scrubs back, we don't yank them
  // forward.
  useEffect(() => {
    setSelectedTurn((current) => {
      if (maxTurn === null) {
        return null;
      }
      if (current === null || current === maxTurn - 1) {
        return maxTurn;
      }
      return current;
    });
  }, [maxTurn]);

  // Reset the scrubber whenever the user picks a different run.
  useEffect(() => {
    setSelectedTurn(null);
  }, [selectedRunId]);

  return (
    <div className="grid h-screen grid-cols-[300px_1fr] gap-0">
      <aside className="border-border bg-card flex flex-col overflow-hidden border-r">
        <header className="border-border border-b px-4 py-3">
          <h1 className="text-sm font-semibold">AoE2 Arena</h1>
          <p className="text-muted-foreground text-xs">Event log replay</p>
        </header>
        <div className="min-h-0 flex-1">
          <RunList
            runs={runs}
            status={runsStatus}
            error={runsError}
            selected={selectedRunId}
            onSelect={setSelectedRunId}
          />
        </div>
      </aside>

      <main className="bg-background flex min-h-0 flex-col">
        <header className="border-border bg-card flex items-center justify-between border-b px-4 py-2">
          <div className="font-mono text-xs">
            {selectedRunId ?? "no run selected"}
          </div>
          <Badge variant={sseStatus === "open" ? "default" : "outline"}>
            {statusLabel(sseStatus)} · {events.length} events
          </Badge>
        </header>

        <Tabs defaultValue="world" className="flex min-h-0 flex-1 flex-col">
          <TabsList className="bg-card mx-4 mt-3 self-start">
            <TabsTrigger value="world">World</TabsTrigger>
            <TabsTrigger value="trace">Trace</TabsTrigger>
            <TabsTrigger value="diff">Diff</TabsTrigger>
          </TabsList>

          <TabsContent value="world" className="min-h-0 flex-1 overflow-auto">
            <WorldPanel states={states} selectedTurn={selectedTurn} />
          </TabsContent>
          <TabsContent value="trace" className="min-h-0 flex-1">
            <TracePanel events={events} selectedTurn={selectedTurn} />
          </TabsContent>
          <TabsContent value="diff" className="min-h-0 flex-1">
            <DiffPanel />
          </TabsContent>
        </Tabs>

        <Timeline
          maxTurn={maxTurn}
          selectedTurn={selectedTurn}
          onSelect={setSelectedTurn}
        />
      </main>
    </div>
  );
}
