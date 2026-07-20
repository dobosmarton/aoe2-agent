import { useMemo } from "react";
import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { GitCompare, Globe2, ListTree, SlidersHorizontal } from "lucide-react";

import { SiblingStrip } from "@/components/sibling-strip";
import { StatusBadge } from "@/components/status-badge";
import { Timeline } from "@/components/timeline";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useEvents } from "@/hooks/use-events";
import { lastTurn, statesByTurn } from "@/lib/event-utils";
import { runsQueryOptions } from "@/lib/queries";
import { DiffPanel } from "@/panels/diff";
import { OperatorPanel } from "@/panels/operator";
import { TracePanel } from "@/panels/trace";
import { WorldPanel } from "@/panels/world";

import { useQuery } from "@tanstack/react-query";

const TABS = ["world", "trace", "diff", "operator"] as const;
export type TabId = (typeof TABS)[number];

/** Both params are optional so `search: {}` is a valid navigation target and
 * the URL stays clean until the user actually changes something. Defaults are
 * applied at read time, not written into the URL. */
type RunSearch = {
  /** Absent means "pinned to the newest turn" — see selectedTurn below. */
  readonly turn?: number;
  readonly tab?: TabId;
}

export const Route = createFileRoute("/_arena/runs/$runId")({
  validateSearch: (search: Record<string, unknown>): RunSearch => {
    const tab = TABS.find((t) => t === search.tab);
    const turn = Number(search.turn);
    // Keys are omitted rather than set to undefined (exactOptionalPropertyTypes).
    return {
      ...(tab === undefined ? {} : { tab }),
      ...(Number.isFinite(turn) && turn > 0 ? { turn } : {}),
    };
  },
  component: RunDetail,
});

function RunDetail(): React.ReactElement {
  const { runId } = Route.useParams();
  const { tab = "world", turn } = Route.useSearch();
  const navigate = useNavigate({ from: Route.fullPath });
  const runsQuery = useQuery(runsQueryOptions());
  const { events, status: sseStatus } = useEvents(runId);

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

  // An absent `turn` means "follow the stream", so the scrubber auto-advances
  // simply by re-deriving from maxTurn. That replaces the two effects the old
  // App.tsx needed (pin-to-latest, and reset-on-run-change) — and because the
  // turn lives in the URL, switching runs resets it for free.
  const selectedTurn = turn ?? maxTurn;

  return (
    <>
      <header className="border-border bg-card flex items-center justify-between gap-3 border-b px-4 py-2">
        <div className="flex min-w-0 items-center gap-2">
          <span className="text-muted-foreground text-xs">run</span>
          <code className="text-foreground truncate font-mono text-xs">{runId}</code>
        </div>
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
      </header>

      <SiblingStrip
        runs={runsQuery.data ?? []}
        selectedRunId={runId}
        onSelect={(next) => {
          void navigate({ to: "/runs/$runId", params: { runId: next }, search: {} });
        }}
      />

      <Tabs
        selectedKey={tab}
        onSelectionChange={(key) => {
          void navigate({ search: (prev) => ({ ...prev, tab: key as TabId }) });
        }}
        className="flex min-h-0 flex-1 flex-col"
      >
        <TabsList variant="line" className="mx-4 mt-3 self-start">
          <TabsTrigger id="world">
            <Globe2 />
            World
          </TabsTrigger>
          <TabsTrigger id="trace">
            <ListTree />
            Trace
          </TabsTrigger>
          <TabsTrigger id="diff">
            <GitCompare />
            Diff
          </TabsTrigger>
          <TabsTrigger id="operator">
            <SlidersHorizontal />
            Operator
          </TabsTrigger>
        </TabsList>

        <TabsContent id="world" className="min-h-0 flex-1 overflow-auto">
          <WorldPanel states={states} selectedTurn={selectedTurn} />
        </TabsContent>
        <TabsContent id="trace" className="min-h-0 flex-1">
          <TracePanel events={events} selectedTurn={selectedTurn} />
        </TabsContent>
        <TabsContent id="diff" className="min-h-0 flex-1 overflow-auto">
          <DiffPanel
            events={events}
            currentRunId={runId}
            onOpenRun={(next) => {
              void navigate({
                to: "/runs/$runId",
                params: { runId: next },
                search: {},
              });
            }}
          />
        </TabsContent>
        <TabsContent id="operator" className="min-h-0 flex-1 overflow-auto">
          <OperatorPanel currentRunId={runId} initialParentT={selectedTurn} />
        </TabsContent>
      </Tabs>

      <Timeline
        maxTurn={maxTurn}
        selectedTurn={selectedTurn}
        onSelect={(next) => {
          // `replace` so dragging the scrubber doesn't bury the Back button
          // under one history entry per turn.
          void navigate({
            search: (prev) => ({ ...prev, turn: next }),
            replace: true,
          });
        }}
      />
    </>
  );
}
