import { useCallback, useEffect, useMemo, useState } from "react";
import {
  GitCompare,
  Globe2,
  ListTree,
  ScanEye,
  SlidersHorizontal,
  Swords,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { TrainingView } from "@/panels/training/training-view";
import { RunList } from "@/components/run-list";
import { SiblingStrip } from "@/components/sibling-strip";
import { StatusBadge } from "@/components/status-badge";
import { Timeline } from "@/components/timeline";
import { useEvents } from "@/hooks/use-events";
import { useRuns } from "@/hooks/use-runs";
import { useRunSummaries } from "@/hooks/use-run-summaries";
import { lastTurn, statesByTurn } from "@/lib/event-utils";
import { groupRuns } from "@/lib/run-grouping";
import { DiffPanel } from "@/panels/diff";
import { ExperimentOverview } from "@/panels/experiment-overview";
import { OperatorPanel } from "@/panels/operator";
import { TracePanel } from "@/panels/trace";
import { WorldPanel } from "@/panels/world";

/** What the main panel is showing: a single run's detail, or an operation's
 * experiment overview. */
type Selection =
  | { readonly kind: "run"; readonly runId: string }
  | { readonly kind: "operation"; readonly key: string };

export function App(): React.ReactElement {
  const [view, setView] = useState<"arena" | "training">("arena");
  if (view === "training") {
    return (
      <TrainingView
        onExit={() => {
          setView("arena");
        }}
      />
    );
  }
  return (
    <ArenaView
      onOpenTraining={() => {
        setView("training");
      }}
    />
  );
}

function ArenaView(props: { readonly onOpenTraining: () => void }): React.ReactElement {
  const { runs, status: runsStatus, error: runsError } = useRuns();
  const { metricsByRunId, status: summariesStatus } = useRunSummaries();
  const [selection, setSelection] = useState<Selection | null>(null);
  const selectedRunId = selection?.kind === "run" ? selection.runId : null;
  const selectedOperationKey =
    selection?.kind === "operation" ? selection.key : null;
  const selectRun = useCallback((runId: string) => {
    setSelection({ kind: "run", runId });
  }, []);
  const openOperation = useCallback((key: string) => {
    setSelection({ kind: "operation", key });
  }, []);

  const groups = useMemo(() => groupRuns(runs), [runs]);
  const selectedGroup =
    selectedOperationKey === null
      ? null
      : (groups.find((g) => g.key === selectedOperationKey) ?? null);

  const { events, status: sseStatus } = useEvents(selectedRunId);
  const [selectedTurn, setSelectedTurn] = useState<number | null>(null);

  const states = useMemo(() => statesByTurn(events), [events]);
  const maxTurn = useMemo(() => lastTurn(events), [events]);
  const totalCost = useMemo(
    () =>
      events.reduce(
        (sum, e) => (e.kind === "llm_response" ? sum + e.cost_usd : sum),
        0,
      ),
    [events],
  );

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
        <header className="border-border flex items-center gap-2 border-b px-4 py-3">
          <Swords className="text-primary size-4 shrink-0" />
          <div className="min-w-0 flex-1">
            <h1 className="text-sm font-semibold leading-none">AoE2 Arena</h1>
            <p className="text-muted-foreground mt-0.5 text-[11px]">
              Event log replay
            </p>
          </div>
          <Button
            variant="ghost"
            size="sm"
            className="shrink-0"
            title="Detection training tracker"
            onClick={props.onOpenTraining}
          >
            <ScanEye className="size-4" />
            Training
          </Button>
        </header>
        <div className="min-h-0 flex-1">
          <RunList
            runs={runs}
            status={runsStatus}
            error={runsError}
            selected={selectedRunId}
            onSelect={selectRun}
            onOpenOperation={openOperation}
            selectedOperation={selectedOperationKey}
          />
        </div>
      </aside>

      <main className="bg-background flex min-h-0 min-w-0 flex-col">
        <header className="border-border bg-card flex items-center justify-between gap-3 border-b px-4 py-2">
          <div className="flex min-w-0 items-center gap-2">
            {selectedGroup !== null ? (
              <>
                <span className="text-muted-foreground text-xs">experiment</span>
                <code className="text-foreground truncate font-mono text-xs">
                  {selectedGroup.label} · {selectedGroup.runs.length} runs
                </code>
              </>
            ) : (
              <>
                <span className="text-muted-foreground text-xs">run</span>
                <code className="text-foreground truncate font-mono text-xs">
                  {selectedRunId ?? "no run selected"}
                </code>
              </>
            )}
          </div>
          {selectedGroup === null ? (
            <div className="flex shrink-0 items-center gap-3">
              {totalCost > 0 ? (
                <span className="text-muted-foreground font-mono text-xs tabular-nums">
                  ${totalCost.toFixed(4)}
                </span>
              ) : null}
              <span className="text-muted-foreground font-mono text-xs tabular-nums">
                {events.length} events
              </span>
              <StatusBadge status={sseStatus} />
            </div>
          ) : null}
        </header>

        {selectedGroup !== null ? (
          <ExperimentOverview
            group={selectedGroup}
            metricsByRunId={metricsByRunId}
            metricsStatus={summariesStatus}
            onOpenRun={selectRun}
          />
        ) : (
          <>
            <SiblingStrip
              runs={runs}
              selectedRunId={selectedRunId}
              onSelect={selectRun}
            />

            <Tabs defaultValue="world" className="flex min-h-0 flex-1 flex-col">
              <TabsList variant="line" className="mx-4 mt-3 self-start">
                <TabsTrigger value="world">
                  <Globe2 />
                  World
                </TabsTrigger>
                <TabsTrigger value="trace">
                  <ListTree />
                  Trace
                </TabsTrigger>
                <TabsTrigger value="diff">
                  <GitCompare />
                  Diff
                </TabsTrigger>
                <TabsTrigger value="operator">
                  <SlidersHorizontal />
                  Operator
                </TabsTrigger>
              </TabsList>

              <TabsContent value="world" className="min-h-0 flex-1 overflow-auto">
                <WorldPanel states={states} selectedTurn={selectedTurn} />
              </TabsContent>
              <TabsContent value="trace" className="min-h-0 flex-1">
                <TracePanel events={events} selectedTurn={selectedTurn} />
              </TabsContent>
              <TabsContent value="diff" className="min-h-0 flex-1 overflow-auto">
                <DiffPanel
                  events={events}
                  currentRunId={selectedRunId}
                  onOpenRun={selectRun}
                />
              </TabsContent>
              <TabsContent value="operator" className="min-h-0 flex-1 overflow-auto">
                <OperatorPanel
                  currentRunId={selectedRunId}
                  initialParentT={selectedTurn}
                  onOpenRun={selectRun}
                />
              </TabsContent>
            </Tabs>

            <Timeline
              maxTurn={maxTurn}
              selectedTurn={selectedTurn}
              onSelect={setSelectedTurn}
            />
          </>
        )}
      </main>
    </div>
  );
}
