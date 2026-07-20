import { createFileRoute } from "@tanstack/react-router";

import { EmptyState } from "@/components/empty-state";

export const Route = createFileRoute("/_arena/runs/")({
  component: () => (
    <EmptyState
      title="No run selected"
      hint="Pick a run from the sidebar to replay its event log."
    />
  ),
});
