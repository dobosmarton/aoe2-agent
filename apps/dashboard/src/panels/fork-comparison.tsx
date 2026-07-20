import { useMemo } from "react";

import { Badge } from "@/components/ui/badge";
import { StateSummary } from "@/components/state-summary";
import { useEvents } from "@/hooks/use-events";
import type { forksIn} from "@/lib/event-utils";
import { statesByTurn } from "@/lib/event-utils";
import { cn } from "@/lib/utils";
import { SECTION_TITLE } from "@/lib/styles";
import type { ArenaEvent } from "@/lib/events";

type ForkComparisonProps = {
  readonly currentEvents: readonly ArenaEvent[];
  readonly currentRunId: string;
  readonly fork: ReturnType<typeof forksIn>[number];
}

export function ForkComparison({
  currentEvents,
  currentRunId,
  fork,
}: ForkComparisonProps): React.ReactElement {
  // Subscribe to the parent run's events in parallel with the current run's
  // stream. The browser handles two concurrent EventSources fine; the
  // backend's read-only DuckDB connection is per-request.
  const { events: parentEvents, status: parentStatus } = useEvents(fork.parent_run_id);

  const parentStates = useMemo(() => statesByTurn(parentEvents), [parentEvents]);
  const currentStates = useMemo(() => statesByTurn(currentEvents), [currentEvents]);

  const parentState = parentStates.get(fork.parent_t) ?? null;
  // The child's first turn_start after fork is turn 1 (synth_game_loop resets
  // the turn counter); fall back to the lowest turn we have for safety.
  const childFirstTurn = [...currentStates.keys()].sort((a, b) => a - b)[0] ?? null;
  const childState =
    childFirstTurn === null ? null : (currentStates.get(childFirstTurn) ?? null);

  return (
    <div className="flex flex-col gap-2">
      <div className="text-muted-foreground flex items-baseline justify-between text-xs">
        <span className={SECTION_TITLE}>Side-by-side state comparison</span>
        <Badge
          variant="outline"
          className={cn(
            "gap-1.5",
            parentStatus === "open" &&
              "border-emerald-500/40 text-emerald-600 dark:text-emerald-400",
          )}
        >
          parent stream: {parentStatus}
        </Badge>
      </div>
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        <StateSummary
          state={parentState}
          label="Parent"
          sublabel={`${fork.parent_run_id.slice(0, 8)}… @ turn ${String(fork.parent_t)}`}
        />
        <StateSummary
          state={childState}
          label="Child"
          sublabel={`${currentRunId.slice(0, 8)}… @ turn ${
            childFirstTurn === null ? "?" : String(childFirstTurn)
          }`}
        />
      </div>
    </div>
  );
}
