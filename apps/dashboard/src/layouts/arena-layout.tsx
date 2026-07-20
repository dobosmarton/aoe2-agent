import { useQuery } from "@tanstack/react-query";
import { Outlet, useNavigate, useParams } from "@tanstack/react-router";
import { ScanEye, Swords } from "lucide-react";

import { RunList } from "@/components/run-list";
import { Button } from "@/components/ui/button";
import { AppShell } from "@/layouts/app-shell";
import { SidebarHeader } from "@/layouts/sidebar-header";
import { errorMessage, toLoadStatus } from "@/lib/load-status";
import { runsQueryOptions } from "@/lib/queries";

const SIDEBAR_WIDTH = 300;

/** Shell for the run/experiment section, sharing one run sidebar across both. */
export function ArenaLayout(): React.ReactElement {
  const runsQuery = useQuery(runsQueryOptions());
  const navigate = useNavigate();

  // Which child route is active decides what the sidebar highlights. `strict:
  // false` because this layout is rendered for several different child params.
  const params: Record<string, string | undefined> = useParams({ strict: false });
  const selectedRunId = params.runId ?? null;
  const selectedOperation = params.key ?? null;

  return (
    <AppShell
      sidebarWidth={SIDEBAR_WIDTH}
      sidebar={
        <>
          <SidebarHeader
            icon={<Swords className="text-primary size-4 shrink-0" />}
            title="AoE2 Arena"
            subtitle="Event log replay"
            action={
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
            }
          />
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
        </>
      }
    >
      <Outlet />
    </AppShell>
  );
}
