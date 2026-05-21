import { Badge } from "@/components/ui/badge";
import type { WorldStateSnapshot } from "@/lib/events";

interface StateSummaryProps {
  readonly state: WorldStateSnapshot | null;
  readonly label: string;
  readonly sublabel?: string;
}

interface RowProps {
  readonly label: string;
  readonly value: string | number;
  readonly highlight?: boolean;
}

function Row({ label, value, highlight }: RowProps): React.ReactElement {
  return (
    <div className="flex items-baseline justify-between gap-2 py-0.5">
      <span className="text-muted-foreground text-xs">{label}</span>
      <span
        className={`font-mono text-sm tabular-nums ${highlight === true ? "text-emerald-400" : ""}`}
      >
        {value}
      </span>
    </div>
  );
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
        {sublabel !== undefined ? (
          <span className="text-muted-foreground font-mono text-xs">{sublabel}</span>
        ) : null}
      </div>
      {state === null ? (
        <div className="text-muted-foreground py-4 text-center text-xs">
          (no state at this turn)
        </div>
      ) : (
        <>
          <Row label="age" value={state.age} />
          <Row label="turn" value={state.turn} />
          <Row label="food" value={Math.round(state.food)} />
          <Row label="wood" value={Math.round(state.wood)} />
          <Row label="gold" value={Math.round(state.gold)} />
          <Row label="stone" value={Math.round(state.stone)} />
          <Row label="population" value={`${state.population} / ${state.pop_cap}`} />
          <Row label="queued villagers" value={state.villager_queue.length} />
          <Row label="age-up ticks" value={state.age_up_ticks_remaining} />
          <div className="mt-2 flex flex-wrap gap-1">
            {state.buildings.length === 0 ? (
              <span className="text-muted-foreground text-xs">no buildings</span>
            ) : (
              state.buildings.map((building, index) => (
                <Badge key={`${building}-${String(index)}`} variant="outline" className="text-[10px]">
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
