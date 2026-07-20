import { createFileRoute } from "@tanstack/react-router";

import { TrainingLayout } from "@/layouts/training-layout";

export const Route = createFileRoute("/training")({
  component: TrainingLayout,
});
