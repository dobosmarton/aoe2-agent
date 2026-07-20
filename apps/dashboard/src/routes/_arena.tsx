import { useQuery } from "@tanstack/react-query";
import {
  Outlet,
  createFileRoute,
  useNavigate,
  useParams,
} from "@tanstack/react-router";
import { ScanEye, Swords } from "lucide-react";

import { RunList } from "@/components/run-list";
import { Button } from "@/components/ui/button";
import { errorMessage, toLoadStatus } from "@/lib/load-status";
import { runsQueryOptions } from "@/lib/queries";

/**
 * Pathless layout: the run sidebar is shared by /runs/* and /experiments/*,
 * but "_arena" contributes nothing to the URL.
 */
export const Route = createFileRoute("/_arena")({
  loader: ({ context }) => context.queryClient.ensureQueryData(runsQueryOptions()),
  component: ArenaLayout,
});

function ArenaLayout(): React.ReactElement {
  const runsQuery = useQuery(runsQueryOptions());
  const navigate = useNavigate();

  // Which child route is active decides what the sidebar highlights. `strict:
  // false` because this layout is rendered for several different child params.
  const params: Record<string, string | undefined> = useParams({ strict: false });
  const selectedRunId = params.runId ?? null;
  const selectedOperation = params.key ?? null;

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
            onClick={() => {
              void navigate({ to: "/training" });
            }}
          >
            <ScanEye className="size-4" />
            Training
          </Button>
        </header>
        <div className="min-h-0 flex-1">
          <RunList
            runs={runsQuery.data ?? []}
            status={toLoadStatus(runsQuery.status)}
            error={errorMessage(runsQuery.error)}
            selected={selectedRunId}
            onSelect={(runId) => {
              void navigate({ to: "/runs/$runId", params: { runId }, search: {} });
            }}
            onOpenOperation={(key) => {
              void navigate({ to: "/experiments/$key", params: { key } });
            }}
            selectedOperation={selectedOperation}
          />
        </div>
      </aside>

      <main className="bg-background flex min-h-0 min-w-0 flex-col">
        <Outlet />
      </main>
    </div>
  );
}
