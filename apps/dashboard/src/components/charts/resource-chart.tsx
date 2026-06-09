import { ChartLegend } from "@/components/charts/chart-legend";
import { TimeSeriesChart } from "@/components/charts/time-series-chart";
import type { ChartSeries } from "@/components/charts/chart-types";
import type { TurnSeriesRow } from "@/lib/event-utils";

// The four resources, in display order, bound to their theme tokens.
// Shared by the resources-over-time chart and the per-resource sparklines.
export const RESOURCE_SERIES: readonly ChartSeries<TurnSeriesRow>[] = [
  { key: "food", label: "Food", colorVar: "var(--food)" },
  { key: "wood", label: "Wood", colorVar: "var(--wood)" },
  { key: "gold", label: "Gold", colorVar: "var(--gold)" },
  { key: "stone", label: "Stone", colorVar: "var(--stone)" },
] as const;

interface ResourceChartProps {
  readonly data: readonly TurnSeriesRow[];
  readonly selectedTurn?: number | null;
  readonly height?: number;
}

export function ResourceChart({
  data,
  selectedTurn,
  height,
}: ResourceChartProps): React.ReactElement {
  return (
    <div className="flex flex-col gap-2">
      <TimeSeriesChart
        data={data}
        series={RESOURCE_SERIES}
        selectedTurn={selectedTurn}
        height={height}
        variant="area"
      />
      <ChartLegend series={RESOURCE_SERIES} />
    </div>
  );
}
