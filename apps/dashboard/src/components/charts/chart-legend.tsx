import type { ChartSeriesBase } from "@/components/charts/chart-types";

/** A color-key swatch row. Compose it under a chart (it is not built into
 * `TimeSeriesChart`) so callers opt in by rendering it, not via a flag. */
export function ChartLegend({
  series,
}: {
  readonly series: readonly ChartSeriesBase[];
}): React.ReactElement {
  return (
    <div className="flex flex-wrap items-center justify-center gap-x-4 gap-y-1 px-2">
      {series.map((s) => (
        <span
          key={s.key}
          className="text-muted-foreground flex items-center gap-1.5 text-xs"
        >
          <span
            className="inline-block size-2 rounded-[2px]"
            style={{ backgroundColor: s.colorVar }}
          />
          {s.label}
        </span>
      ))}
    </div>
  );
}
