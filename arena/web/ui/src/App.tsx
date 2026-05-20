import { useState } from "react";

import { EventStream } from "@/components/event-stream";
import { RunList } from "@/components/run-list";
import { useEvents } from "@/hooks/use-events";
import { useRuns } from "@/hooks/use-runs";

export function App(): React.ReactElement {
  const { runs, status: runsStatus, error: runsError } = useRuns();
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const { events, status: sseStatus } = useEvents(selectedRunId);

  return (
    <div className="grid h-screen grid-cols-[300px_1fr] gap-0">
      <aside className="border-border bg-card border-r">
        <header className="border-border border-b px-4 py-3">
          <h1 className="text-sm font-semibold">AoE2 Arena</h1>
          <p className="text-muted-foreground text-xs">Event log replay</p>
        </header>
        <RunList
          runs={runs}
          status={runsStatus}
          error={runsError}
          selected={selectedRunId}
          onSelect={setSelectedRunId}
        />
      </aside>
      <main className="bg-background">
        <EventStream runId={selectedRunId} events={events} status={sseStatus} />
      </main>
    </div>
  );
}
