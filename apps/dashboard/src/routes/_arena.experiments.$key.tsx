import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { createFileRoute, useNavigate } from "@tanstack/react-router";

import { EmptyState } from "@/components/empty-state";
import { toLoadStatus } from "@/lib/load-status";
import { runSummariesQueryOptions, runsQueryOptions } from "@/lib/queries";
import { groupRuns } from "@/lib/run-grouping";
import { ExperimentOverview } from "@/panels/experiment-overview";

export const Route = createFileRoute("/_arena/experiments/$key")({
  loader: ({ context }) =>
    Promise.all([
      context.queryClient.ensureQueryData(runsQueryOptions()),
      context.queryClient.ensureQueryData(runSummariesQueryOptions()),
    ]),
  component: ExperimentRoute,
});

function ExperimentRoute(): React.ReactElement {
  const { key } = Route.useParams();
  const navigate = useNavigate();
  const runsQuery = useQuery(runsQueryOptions());
  const summariesQuery = useQuery(runSummariesQueryOptions());

  const runs = runsQuery.data ?? [];
  const group = useMemo(
    () => groupRuns(runs).find((g) => g.key === key) ?? null,
    [runs, key],
  );

  if (group === null) {
    return (
      <EmptyState
        title="Experiment not found"
        hint={
          runsQuery.isPending
            ? "Loading runs…"
            : `No run group matches "${key}". It may have been pruned.`
        }
      />
    );
  }

  return (
    <>
      <header className="border-border bg-card flex items-center gap-2 border-b px-4 py-2">
        <span className="text-muted-foreground text-xs">experiment</span>
        <code className="text-foreground truncate font-mono text-xs">
          {group.label} · {group.runs.length} runs
        </code>
      </header>
      <ExperimentOverview
        group={group}
        metricsByRunId={summariesQuery.data ?? new Map()}
        metricsStatus={toLoadStatus(summariesQuery.status)}
        onOpenRun={(runId) => {
          void navigate({ to: "/runs/$runId", params: { runId }, search: {} });
        }}
      />
    </>
  );
}
