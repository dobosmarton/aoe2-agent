import { useMemo, useState } from "react";
import { ExternalLink, GitBranch } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { EmptyState } from "@/components/empty-state";
import { ForkComparison } from "@/panels/fork-comparison";
import { forksIn } from "@/lib/event-utils";
import { cn } from "@/lib/utils";
import type { ArenaEvent } from "@/lib/events";

const SECTION_TITLE =
  "text-muted-foreground text-xs font-semibold uppercase tracking-wide";

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
              onClick={() => {
                setSelectedForkIndex(index);
              }}
              className={cn(
                index === safeIndex &&
                  "bg-event-fork/15 border-event-fork/50 text-foreground hover:bg-event-fork/25",
              )}
            >
              <GitBranch className="text-event-fork size-3.5" />
              fork {index + 1} ← {fork.parent_run_id.slice(0, 6)}@{fork.parent_t}
            </Button>
          ))}
        </div>
      ) : null}

      <Card className="gap-3 py-4">
        <CardHeader className="px-4">
          <CardTitle className={`${SECTION_TITLE} flex items-center gap-1.5`}>
            <GitBranch className="text-event-fork size-3.5" />
            Fork metadata
          </CardTitle>
          <CardDescription className="font-mono text-xs">
            parent {selectedFork.parent_run_id.slice(0, 12)}… @ turn{" "}
            {selectedFork.parent_t}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3 px-4 text-xs">
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
            onClick={() => {
              onOpenRun(selectedFork.parent_run_id);
            }}
          >
            <ExternalLink className="size-3.5" />
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
