import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { EmptyState } from "@/components/empty-state";
import type { RunsStatus } from "@/hooks/use-runs";
import { cn } from "@/lib/utils";
import type { RunSummary } from "@/lib/events";

interface RunListProps {
  readonly runs: readonly RunSummary[];
  readonly status: RunsStatus;
  readonly error: string | null;
  readonly selected: string | null;
  readonly onSelect: (runId: string) => void;
}

const _LABEL_VARIANT: Record<string, "default" | "secondary" | "outline"> = {
  race: "default",
  rank: "secondary",
  smoke: "outline",
};

function labelVariant(label: string): "default" | "secondary" | "outline" {
  return _LABEL_VARIANT[label] ?? "outline";
}

function formatTs(iso: string): string {
  // Strip seconds and timezone for compactness; the full ISO is in the tooltip.
  return iso.length >= 16 ? iso.slice(0, 16).replace("T", " ") : iso;
}

export function RunList({
  runs,
  status,
  error,
  selected,
  onSelect,
}: RunListProps): React.ReactElement {
  if (status === "loading") {
    return <EmptyState title="Loading runs…" />;
  }
  if (status === "error") {
    return (
      <EmptyState title="Could not load runs" hint={error ?? "unknown error"} />
    );
  }
  if (runs.length === 0) {
    return (
      <EmptyState
        title="No runs yet"
        hint="Try `just arena-race` or `just arena-smoke`."
      />
    );
  }

  return (
    <ScrollArea className="h-full">
      <ol className="flex flex-col gap-2 p-3">
        {runs.map((run) => {
          const isSelected = run.run_id === selected;
          return (
            <li key={`${run.db_path}::${run.run_id}`}>
              <Card
                onClick={() => onSelect(run.run_id)}
                className={cn(
                  "cursor-pointer transition-colors hover:bg-accent",
                  isSelected && "ring-2 ring-ring",
                )}
              >
                <CardHeader className="pb-2">
                  <CardTitle className="flex items-center justify-between text-xs font-mono">
                    <span className="truncate" title={run.run_id}>
                      {run.run_id.slice(0, 8)}…
                    </span>
                    <Badge variant={labelVariant(run.label)}>{run.label}</Badge>
                  </CardTitle>
                </CardHeader>
                <CardContent className="text-muted-foreground space-y-1 text-xs">
                  <div title={run.first_ts}>{formatTs(run.first_ts)}</div>
                  <div>{run.n_events} events</div>
                </CardContent>
              </Card>
            </li>
          );
        })}
      </ol>
    </ScrollArea>
  );
}
