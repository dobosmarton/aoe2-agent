import { Badge } from "@/components/ui/badge";
import { SummaryRow } from "@/components/summary-row";
import type { WorldStateSnapshot } from "@/lib/events";

type StateSummaryProps = {
  readonly state: WorldStateSnapshot | null;
  readonly label: string;
  readonly sublabel?: string;
}

export function StateSummary({
  state,
  label,
  sublabel,
}: StateSummaryProps): React.ReactElement {
  return (
    <div className="border-border bg-card flex flex-col gap-1 rounded-lg border p-3">
      <div className="border-border mb-1 flex items-baseline justify-between border-b pb-2">
        <span className="text-sm font-semibold">{label}</span>
        {sublabel === undefined ? null : (
          <span className="text-muted-foreground font-mono text-xs">{sublabel}</span>
        )}
      </div>
      {state === null ? (
        <div className="text-muted-foreground py-4 text-center text-xs">
          (no state at this turn)
        </div>
      ) : (
        <>
          <SummaryRow label="age" value={state.age} />
          <SummaryRow label="turn" value={state.turn} />
          <SummaryRow label="food" value={Math.round(state.food)} colorVar="var(--food)" />
          <SummaryRow label="wood" value={Math.round(state.wood)} colorVar="var(--wood)" />
          <SummaryRow label="gold" value={Math.round(state.gold)} colorVar="var(--gold)" />
          <SummaryRow label="stone" value={Math.round(state.stone)} colorVar="var(--stone)" />
          <SummaryRow
            label="population"
            value={`${String(state.population)} / ${String(state.pop_cap)}`}
            colorVar="var(--population)"
          />
          <SummaryRow label="queued villagers" value={state.villager_queue.length} />
          <SummaryRow label="age-up ticks" value={state.age_up_ticks_remaining} />
          <div className="mt-2 flex flex-wrap gap-1">
            {state.buildings.length === 0 ? (
              <span className="text-muted-foreground text-xs">no buildings</span>
            ) : (
              state.buildings.map((building, index) => (
                <Badge
                  key={`${building}-${String(index)}`}
                  variant="outline"
                  className="font-mono text-[10px]"
                >
                  {building}
                </Badge>
              ))
            )}
          </div>
        </>
      )}
    </div>
  );
}
