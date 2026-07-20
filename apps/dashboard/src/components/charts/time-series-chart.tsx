import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { ChartSeries } from "@/components/charts/chart-types";
import { TimeSeriesTooltip } from "@/components/charts/time-series-tooltip";

// ---------------------------------------------------------------------------
// The single recharts integration boundary for time-series. Panels describe
// their series in terms of theme tokens (colorVar = "var(--food)") and never
// import recharts directly. The legend (`ChartLegend`) and tooltip
// (`TimeSeriesTooltip`) live in sibling files; compose them as needed.
// ---------------------------------------------------------------------------

type TimeSeriesChartProps<Row> = {
  readonly data: readonly Row[];
  readonly series: readonly ChartSeries<Row>[];
  readonly selectedTurn?: number | null | undefined;
  readonly height?: number | undefined;
  readonly variant?: "area" | "line" | undefined;
  /** Sparkline mode: drop axes/grid/tooltip for inline stat cards. */
  readonly compact?: boolean | undefined;
}

export function TimeSeriesChart<Row>({
  data,
  series,
  selectedTurn = null,
  height = 200,
  variant = "area",
  compact = false,
}: TimeSeriesChartProps<Row>): React.ReactElement {
  const showChrome = !compact;
  const gradientId = (key: string): string => `ts-grad-${key}`;

  const margin = compact
    ? { top: 2, right: 2, bottom: 2, left: 2 }
    : { top: 8, right: 12, bottom: 4, left: -8 };

  const reference =
    showChrome && selectedTurn !== null ? (
      <ReferenceLine
        x={selectedTurn}
        stroke="var(--ring)"
        strokeDasharray="3 3"
        strokeWidth={1}
      />
    ) : null;

  const axes = showChrome ? (
    <>
      <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" vertical={false} />
      <XAxis
        dataKey="turn"
        stroke="var(--muted-foreground)"
        tick={{ fontSize: 10 }}
        tickLine={false}
        axisLine={{ stroke: "var(--border)" }}
        minTickGap={24}
      />
      <YAxis
        stroke="var(--muted-foreground)"
        tick={{ fontSize: 10 }}
        tickLine={false}
        axisLine={false}
        width={36}
      />
      <Tooltip
        content={<TimeSeriesTooltip series={series} />}
        cursor={{ stroke: "var(--ring)", strokeWidth: 1, strokeOpacity: 0.3 }}
      />
    </>
  ) : null;

  // recharts' data prop is loosely typed; cast at this boundary so the public
  // API above stays strongly typed (`readonly Row[]` + key-checked series).
  const writableData = data as unknown as Record<string, number>[];

  return (
    <ResponsiveContainer width="100%" height={height}>
      {variant === "line" ? (
        <LineChart data={writableData} margin={margin}>
          {axes}
          {reference}
          {series.map((s) => (
            <Line
              key={s.key}
              type="monotone"
              dataKey={s.key}
              stroke={s.colorVar}
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      ) : (
        <AreaChart data={writableData} margin={margin}>
          <defs>
            {series.map((s) => (
              <linearGradient
                key={s.key}
                id={gradientId(s.key)}
                x1="0"
                y1="0"
                x2="0"
                y2="1"
              >
                <stop offset="0%" stopColor={s.colorVar} stopOpacity={0.35} />
                <stop offset="100%" stopColor={s.colorVar} stopOpacity={0} />
              </linearGradient>
            ))}
          </defs>
          {axes}
          {reference}
          {series.map((s) => (
            <Area
              key={s.key}
              type="monotone"
              dataKey={s.key}
              stroke={s.colorVar}
              strokeWidth={2}
              fill={`url(#${gradientId(s.key)})`}
              isAnimationActive={false}
            />
          ))}
        </AreaChart>
      )}
    </ResponsiveContainer>
  );
}
