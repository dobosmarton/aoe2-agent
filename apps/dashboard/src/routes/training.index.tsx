import { createFileRoute, redirect } from "@tanstack/react-router";

export const Route = createFileRoute("/training/")({
  beforeLoad: () => {
    throw redirect({ to: "/training/coverage" });
  },
});
