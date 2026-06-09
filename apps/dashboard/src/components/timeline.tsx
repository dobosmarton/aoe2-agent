import { Slider } from "@/components/ui/slider";

interface TimelineProps {
  readonly maxTurn: number | null;
  readonly selectedTurn: number | null;
  readonly onSelect: (turn: number) => void;
}

/** Evenly spaced turn marks for the scale, capped so they never crowd. */
function tickTurns(maxTurn: number): readonly number[] {
  if (maxTurn <= 1) {
    return [1];
  }
  const step = Math.max(1, Math.ceil(maxTurn / 8));
  const ticks: number[] = [];
  for (let t = 1; t < maxTurn; t += step) {
    ticks.push(t);
  }
  ticks.push(maxTurn);
  return ticks;
}

export function Timeline({
  maxTurn,
  selectedTurn,
  onSelect,
}: TimelineProps): React.ReactElement {
  if (maxTurn === null || maxTurn < 1) {
    return (
      <div className="text-muted-foreground border-border border-t px-4 py-3 text-xs">
        Timeline activates once turn events arrive…
      </div>
    );
  }

  const current = selectedTurn ?? maxTurn;
  const ticks = tickTurns(maxTurn);

  return (
    <div className="border-border flex items-center gap-4 border-t px-4 py-3">
      <span className="bg-muted text-foreground rounded-md px-2 py-1 font-mono text-xs tabular-nums whitespace-nowrap">
        turn {current}{" "}
        <span className="text-muted-foreground">/ {maxTurn}</span>
      </span>
      <div className="flex-1">
        <Slider
          min={1}
          max={maxTurn}
          step={1}
          value={[current]}
          onValueChange={(values) => {
            const next = values[0];
            if (next !== undefined) {
              onSelect(next);
            }
          }}
        />
        <div className="relative mt-1.5 h-3">
          {ticks.map((t) => {
            const pct = maxTurn === 1 ? 0 : ((t - 1) / (maxTurn - 1)) * 100;
            const transform =
              pct === 0
                ? "translateX(0)"
                : pct === 100
                  ? "translateX(-100%)"
                  : "translateX(-50%)";
            return (
              <span
                key={t}
                className="text-muted-foreground absolute top-0 font-mono text-[10px] tabular-nums"
                style={{ left: `${String(pct)}%`, transform }}
              >
                {t}
              </span>
            );
          })}
        </div>
      </div>
    </div>
  );
}
