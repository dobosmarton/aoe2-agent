import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { ComparisonDatum } from "@/components/charts/chart-types";
import { ComparisonTooltip } from "@/components/charts/comparison-tooltip";

// ---------------------------------------------------------------------------
// Categorical bar comparison — the sibling of TimeSeriesChart for comparing a
// single metric across runs (not over turns). Same integration boundary:
// callers pass theme tokens (colorVar = "var(--food)") and a flat data array,
// never importing recharts directly. One bar may be highlighted (the winner).
// The tooltip lives in `comparison-tooltip.tsx`.
// ---------------------------------------------------------------------------

interface ComparisonChartProps {
  readonly data: readonly ComparisonDatum[];
  readonly colorVar: string;
  /** Metric name shown in the tooltip header. */
  readonly label: string;
  readonly height?: number | undefined;
  /** Optional value formatter for the tooltip (e.g. cost → "$0.0123"). */
  readonly format?: ((value: number) => string) | undefined;
}

export function ComparisonChart({
  data,
  colorVar,
  label,
  height = 200,
  format,
}: ComparisonChartProps): React.ReactElement {
  const fmt = format ?? ((v: number): string => String(v));
  const writableData = data as ComparisonDatum[];
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={writableData} margin={{ top: 8, right: 12, bottom: 4, left: -8 }}>
        <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" vertical={false} />
        <XAxis
          dataKey="name"
          stroke="var(--muted-foreground)"
          tick={{ fontSize: 10 }}
          tickLine={false}
          axisLine={{ stroke: "var(--border)" }}
          interval={0}
        />
        <YAxis
          stroke="var(--muted-foreground)"
          tick={{ fontSize: 10 }}
          tickLine={false}
          axisLine={false}
          width={44}
          tickFormatter={(v: number) => fmt(v)}
        />
        <Tooltip
          cursor={{ fill: "var(--accent)", fillOpacity: 0.3 }}
          content={
            <ComparisonTooltip metricLabel={label} colorVar={colorVar} format={fmt} />
          }
        />
        <Bar dataKey="value" radius={[3, 3, 0, 0]} isAnimationActive={false}>
          {writableData.map((d, i) => (
            <Cell
              key={`${d.name}-${String(i)}`}
              fill={colorVar}
              fillOpacity={d.highlight === true ? 1 : 0.4}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
