import { createFileRoute } from "@tanstack/react-router";

import { ArenaLayout } from "@/layouts/arena-layout";
import { runsQueryOptions } from "@/lib/queries";

/**
 * Pathless layout: the run sidebar is shared by /runs/* and /experiments/*,
 * but "_arena" contributes nothing to the URL.
 */
export const Route = createFileRoute("/_arena")({
  loader: ({ context }) => context.queryClient.ensureQueryData(runsQueryOptions()),
  component: ArenaLayout,
});
