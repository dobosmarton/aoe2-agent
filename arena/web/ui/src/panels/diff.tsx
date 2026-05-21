import { useMemo, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { EmptyState } from "@/components/empty-state";
import { StateSummary } from "@/components/state-summary";
import { useEvents } from "@/hooks/use-events";
import { forksIn, statesByTurn } from "@/lib/event-utils";
import type { ArenaEvent } from "@/lib/events";

interface DiffPanelProps {
  readonly events: readonly ArenaEvent[];
  readonly currentRunId: string | null;
  readonly onOpenRun: (runId: string) => void;
}

export function DiffPanel({
  events,
  currentRunId,
  onOpenRun,
}: DiffPanelProps): React.ReactElement {
  const forks = useMemo(() => forksIn(events), [events]);
  const [selectedForkIndex, setSelectedForkIndex] = useState(0);

  if (currentRunId === null) {
    return (
      <EmptyState
        title="Select a run"
        hint="Pick a run from the sidebar to inspect its fork relationships."
      />
    );
  }

  if (forks.length === 0) {
    return (
      <EmptyState
        title="No forks in this run"
        hint="A run gets a fork event when created via evaluation.fork.fork(parent_run_id, t, mutation_fn)."
      />
    );
  }

  const safeIndex = Math.min(selectedForkIndex, forks.length - 1);
  const selectedFork = forks[safeIndex];
  if (selectedFork === undefined) {
    return <EmptyState title="No fork at this index" />;
  }

  return (
    <div className="flex h-full flex-col gap-3 p-4">
      {forks.length > 1 ? (
        <div className="flex flex-wrap gap-1">
          {forks.map((fork, index) => (
            <Button
              key={`${fork.parent_run_id}-${String(fork.parent_t)}`}
              variant={index === safeIndex ? "default" : "outline"}
              size="sm"
              onClick={() => setSelectedForkIndex(index)}
            >
              fork {index + 1} ← {fork.parent_run_id.slice(0, 6)}@{fork.parent_t}
            </Button>
          ))}
        </div>
      ) : null}

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Fork metadata</CardTitle>
          <CardDescription className="font-mono text-xs">
            parent {selectedFork.parent_run_id.slice(0, 12)}… @ turn{" "}
            {selectedFork.parent_t}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-2 text-xs">
          <div className="flex items-baseline justify-between gap-2">
            <span className="text-muted-foreground">mutation</span>
            <span className="font-mono">
              {selectedFork.mutation_summary === ""
                ? "(no mutation — clean clone)"
                : selectedFork.mutation_summary}
            </span>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => onOpenRun(selectedFork.parent_run_id)}
          >
            Open parent run
          </Button>
        </CardContent>
      </Card>

      <ForkComparison
        currentEvents={events}
        currentRunId={currentRunId}
        fork={selectedFork}
      />
    </div>
  );
}

interface ForkComparisonProps {
  readonly currentEvents: readonly ArenaEvent[];
  readonly currentRunId: string;
  readonly fork: ReturnType<typeof forksIn>[number];
}

function ForkComparison({
  currentEvents,
  currentRunId,
  fork,
}: ForkComparisonProps): React.ReactElement {
  // Subscribe to the parent run's events in parallel with the current run's
  // stream. The browser handles two concurrent EventSources fine; the
  // backend's read-only DuckDB connection is per-request.
  const { events: parentEvents, status: parentStatus } = useEvents(fork.parent_run_id);

  const parentStates = useMemo(() => statesByTurn(parentEvents), [parentEvents]);
  const currentStates = useMemo(() => statesByTurn(currentEvents), [currentEvents]);

  const parentState = parentStates.get(fork.parent_t) ?? null;
  // The child's first turn_start after fork is turn 1 (synth_game_loop resets
  // the turn counter); fall back to the lowest turn we have for safety.
  const childFirstTurn = [...currentStates.keys()].sort((a, b) => a - b)[0] ?? null;
  const childState = childFirstTurn === null ? null : (currentStates.get(childFirstTurn) ?? null);

  return (
    <div className="flex flex-col gap-2">
      <div className="text-muted-foreground flex items-baseline justify-between text-xs">
        <span>Side-by-side state comparison</span>
        <Badge variant="outline">parent stream: {parentStatus}</Badge>
      </div>
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        <StateSummary
          state={parentState}
          label="Parent"
          sublabel={`${fork.parent_run_id.slice(0, 8)}… @ turn ${fork.parent_t}`}
        />
        <StateSummary
          state={childState}
          label="Child"
          sublabel={`${currentRunId.slice(0, 8)}… @ turn ${
            childFirstTurn ?? "?"
          }`}
        />
      </div>
    </div>
  );
}
