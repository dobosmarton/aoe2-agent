import type { ChartSeriesBase } from "@/components/charts/chart-types";

interface TooltipPayloadEntry {
  readonly dataKey?: string | number;
  readonly value?: number | string;
}

interface TimeSeriesTooltipProps {
  readonly active?: boolean;
  readonly payload?: readonly TooltipPayloadEntry[];
  readonly label?: string | number;
  readonly series: readonly ChartSeriesBase[];
}

/** Themed tooltip for `TimeSeriesChart` (passed to recharts via `content`). */
export function TimeSeriesTooltip({
  active,
  payload,
  label,
  series,
}: TimeSeriesTooltipProps): React.ReactElement | null {
  if (!active || payload === undefined || payload.length === 0) {
    return null;
  }
  const colorByKey = new Map(series.map((s) => [s.key, s] as const));
  return (
    <div className="border-border bg-popover text-popover-foreground rounded-md border px-2.5 py-2 text-xs shadow-md">
      <div className="text-muted-foreground mb-1 font-mono">turn {label}</div>
      <div className="flex flex-col gap-1">
        {payload.map((entry) => {
          const meta =
            typeof entry.dataKey === "string"
              ? colorByKey.get(entry.dataKey)
              : undefined;
          const key = String(entry.dataKey ?? "");
          return (
            <div key={key} className="flex items-center justify-between gap-3">
              <span className="flex items-center gap-1.5">
                <span
                  className="inline-block size-2 rounded-[2px]"
                  style={{ backgroundColor: meta?.colorVar ?? "var(--muted-foreground)" }}
                />
                <span className="text-muted-foreground">{meta?.label ?? key}</span>
              </span>
              <span className="font-mono tabular-nums">{entry.value}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
