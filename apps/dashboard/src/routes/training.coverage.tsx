import { createFileRoute } from "@tanstack/react-router";

import { CoverageStats } from "@/panels/training/coverage-stats";
import {
  trackerDatasetsQueryOptions,
  trackerStatsQueryOptions,
} from "@/lib/queries";

export const Route = createFileRoute("/training/coverage")({
  loader: ({ context }) =>
    Promise.all([
      context.queryClient.ensureQueryData(trackerStatsQueryOptions()),
      context.queryClient.ensureQueryData(trackerDatasetsQueryOptions()),
    ]),
  component: CoverageStats,
});
