import { Outlet, useNavigate } from "@tanstack/react-router";
import { ChevronLeft, ScanEye } from "lucide-react";

import { AppShell } from "@/layouts/app-shell";
import { SidebarHeader } from "@/layouts/sidebar-header";
import { TrainingNav } from "@/layouts/training-nav";
import { Button } from "@/components/ui/button";

const SIDEBAR_WIDTH = 240;

/** Shell for the detection-training section (/training/*). */
export function TrainingLayout(): React.ReactElement {
  const navigate = useNavigate();

  return (
    <AppShell
      sidebarWidth={SIDEBAR_WIDTH}
      sidebar={
        <>
          <SidebarHeader
            icon={<ScanEye className="text-primary size-4 shrink-0" />}
            title="Detection Training"
            subtitle="Annotation tracker"
          />
          <TrainingNav />
          <div className="mt-auto p-2">
            <Button
              variant="ghost"
              size="sm"
              className="w-full justify-start"
              onClick={() => {
                void navigate({ to: "/runs" });
              }}
            >
              <ChevronLeft className="size-4" />
              Back to Arena
            </Button>
          </div>
        </>
      }
    >
      <Outlet />
    </AppShell>
  );
}
