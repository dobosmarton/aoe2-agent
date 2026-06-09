import { useMemo } from "react";
import { Users } from "lucide-react";

import { cn } from "@/lib/utils";
import { operationRuns } from "@/lib/run-grouping";
import { shortRunId } from "@/lib/run-format";
import type { RunSummary } from "@/lib/events";

interface SiblingStripProps {
  readonly runs: readonly RunSummary[];
  readonly selectedRunId: string | null;
  readonly onSelect: (runId: string) => void;
}

/**
 * Quick switcher between the parallel runs of one operation. When the selected
 * run is part of a multi-run `rank`/`race`, this strip lists every sibling as a
 * chip (the current one highlighted) so you can flip between them without
 * leaving the panel. Renders nothing for standalone runs.
 */
export function SiblingStrip({
  runs,
  selectedRunId,
  onSelect,
}: SiblingStripProps): React.ReactElement | null {
  const members = useMemo(
    () => (selectedRunId === null ? [] : operationRuns(runs, selectedRunId)),
    [runs, selectedRunId],
  );

  if (members.length === 0) {
    return null;
  }

  return (
    <div className="border-border bg-card flex min-w-0 items-center gap-2 border-b px-4 py-1.5">
      <span className="text-muted-foreground flex shrink-0 items-center gap-1 text-xs">
        <Users className="size-3" />
        siblings
        <span className="tabular-nums">· {members.length}</span>
      </span>
      {/* Scroll horizontally instead of widening the layout when an operation
          has many parallel runs (a rank can have dozens). */}
      <div className="flex min-w-0 items-center gap-1 overflow-x-auto">
        {members.map((run) => {
          const active = run.run_id === selectedRunId;
          return (
            <button
              key={run.run_id}
              type="button"
              onClick={() => {
                onSelect(run.run_id);
              }}
              title={run.run_id}
              className={cn(
                "shrink-0 rounded-md border px-2 py-0.5 font-mono text-xs transition-colors",
                active
                  ? "border-primary bg-accent text-foreground"
                  : "border-border text-muted-foreground hover:bg-accent/50 hover:text-foreground",
              )}
            >
              {shortRunId(run.run_id)}
            </button>
          );
        })}
      </div>
    </div>
  );
}
