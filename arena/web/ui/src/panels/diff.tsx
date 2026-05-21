import { EmptyState } from "@/components/empty-state";

export function DiffPanel(): React.ReactElement {
  return (
    <EmptyState
      title="Diff panel"
      hint="Sibling-fork comparison lands in the next commit (uses fork events from evaluation/fork.py)."
    />
  );
}
