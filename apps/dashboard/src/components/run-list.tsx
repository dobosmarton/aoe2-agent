import { useMemo, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { EmptyState } from "@/components/empty-state";
import { GroupHeader } from "@/components/group-header";
import { RunCard } from "@/components/run-card";
import { groupRuns } from "@/lib/run-grouping";
import { labelVariant } from "@/lib/run-format";
import type { RunsStatus } from "@/hooks/use-runs";
import type { RunSummary } from "@/lib/events";

interface RunListProps {
  readonly runs: readonly RunSummary[];
  readonly status: RunsStatus;
  readonly error: string | null;
  readonly selected: string | null;
  readonly onSelect: (runId: string) => void;
  /** Open the experiment overview for a multi-run operation (group key). */
  readonly onOpenOperation: (key: string) => void;
  /** Group key whose overview is currently open, for header highlighting. */
  readonly selectedOperation: string | null;
}

export function RunList({
  runs,
  status,
  error,
  selected,
  onSelect,
  onOpenOperation,
  selectedOperation,
}: RunListProps): React.ReactElement {
  const groups = useMemo(() => groupRuns(runs), [runs]);
  // Keys present here are collapsed; groups default to expanded so the parallel
  // runs are visible — that's the whole point of grouping them.
  const [collapsed, setCollapsed] = useState<ReadonlySet<string>>(() => new Set());

  const toggle = (key: string): void => {
    setCollapsed((prev) => {
      const next = new Set(prev);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  };

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
      <ol className="flex flex-col gap-1.5 p-2">
        {groups.map((group) => {
          // A lone run (e.g. a smoke run) needs no header — render it plainly.
          const [soleRun] = group.runs;
          if (group.runs.length === 1 && soleRun !== undefined) {
            const run = soleRun;
            return (
              <li key={group.key}>
                <RunCard
                  run={run}
                  selected={run.run_id === selected}
                  labelSlot={
                    <Badge variant={labelVariant(run.label)}>{run.label}</Badge>
                  }
                  onSelect={onSelect}
                />
              </li>
            );
          }
          const expanded = !collapsed.has(group.key);
          return (
            <li key={group.key} className="flex flex-col gap-1">
              <GroupHeader
                group={group}
                expanded={expanded}
                active={group.key === selectedOperation}
                onToggle={toggle}
                onOpen={onOpenOperation}
              />
              {expanded ? (
                <ol className="border-border/60 ml-3 flex flex-col gap-1 border-l pl-2">
                  {group.runs.map((run) => (
                    <li key={`${group.key}::${run.run_id}`}>
                      <RunCard
                        run={run}
                        selected={run.run_id === selected}
                        onSelect={onSelect}
                      />
                    </li>
                  ))}
                </ol>
              ) : null}
            </li>
          );
        })}
      </ol>
    </ScrollArea>
  );
}
