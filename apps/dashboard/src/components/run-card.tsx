import { Activity } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { LiveDot } from "@/components/live-dot";
import { cn } from "@/lib/utils";
import { formatRelative, shortRunId } from "@/lib/run-format";
import type { RunSummary } from "@/lib/events";

type RunCardProps = {
  readonly run: RunSummary;
  readonly selected: boolean;
  /** Trailing label slot — standalone cards pass the operation badge; cards
   * nested under a group header pass nothing (the header already shows it). */
  readonly labelSlot?: React.ReactNode;
  readonly onSelect: (runId: string) => void;
}

export function RunCard({
  run,
  selected,
  labelSlot,
  onSelect,
}: RunCardProps): React.ReactElement {
  return (
    <Card
      onClick={() => {
        onSelect(run.run_id);
      }}
      className={cn(
        "hover:bg-accent cursor-pointer gap-0 border-l-2 border-l-transparent py-0 transition-colors",
        selected && "border-l-primary bg-accent/40",
      )}
    >
      <CardContent className="p-3">
        <div className="flex items-center justify-between gap-2">
          <span
            className="truncate font-mono text-xs font-medium"
            title={run.run_id}
          >
            {shortRunId(run.run_id)}
          </span>
          <span className="flex shrink-0 items-center gap-1">
            {run.status === "running" ? (
              <Badge
                variant="outline"
                className="gap-1 border-emerald-500/40 text-emerald-600 dark:text-emerald-400"
                title="Run in progress — reload to refresh"
              >
                <LiveDot />
                live
              </Badge>
            ) : null}
            {labelSlot}
          </span>
        </div>
        <div className="text-muted-foreground mt-2 flex items-center justify-between text-xs">
          <span title={run.first_ts}>{formatRelative(run.last_ts)}</span>
          <span className="flex items-center gap-1 tabular-nums">
            <Activity className="size-3" />
            {run.n_events}
          </span>
        </div>
      </CardContent>
    </Card>
  );
}
