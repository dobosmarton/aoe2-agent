import * as React from "react";

import { ArtifactFrame } from "@/components/ui/artifact-frame";

// Horizontal dot plot, ported from the author's blog: one row per category, one dot per
// value, an optional line that links a row's dots (dumbbell), and an optional reference
// line. Several panels share the same rows and legend and stack vertically. The blog
// draws it with Recharts; this version is plain SVG with the same marks (6 px dots with
// a 2 px surface ring, 2 px link lines, recessive grid), so the landing needs no chart
// library. Hydrate it with `client:visible` for the hover tooltip and the phone layout.

type SeriesColor = "primary" | "secondary" | "neutral";

const seriesFill: Record<SeriesColor, string> = {
  primary: "var(--chart-1)",
  secondary: "var(--chart-2)",
  neutral: "var(--chart-neutral)",
};

// "light" is the series color mixed toward the page background: the quieter of two
// related values. "dark" is the plain series color.
type Shade = "light" | "dark";

const fillOf = (color: SeriesColor, shade: Shade = "dark"): string =>
  shade === "light"
    ? `color-mix(in oklab, ${seriesFill[color]} var(--chart-light-mix), hsl(var(--background)))`
    : seriesFill[color];

type Series = { key: string; label: string; color: SeriesColor; shade?: Shade };

type Point = {
  row: string;
  value: number;
  series: string;
  tip: string;
  label?: string;
  labelSide?: "left" | "right";
  shade?: Shade;
};

type Panel = {
  title?: string;
  domain: [number, number];
  ticks: number[];
  unit?: string;
  decimals?: number;
  axisLabel?: string;
  reference?: { value: number; label: string };
  connect?: boolean;
  points: Point[];
};

type DotPlotProps = {
  title: string;
  rows: string[];
  series: Series[];
  panels: Panel[];
  legendNote?: string;
  shadeLegend?: { light: string; dark: string };
  caption?: string;
  labelWidth?: number;
};

const ROW_HEIGHT = 40;
const RIGHT_MARGIN = 56;
// Narrow screens: the label column shrinks to this share of the chart width, and rows
// get taller so a label can wrap onto a third line.
const NARROW_LABEL_SHARE = 0.36;
const NARROW_ROW_HEIGHT = 52;
// Width used for the server render, before the chart is measured.
const DEFAULT_WIDTH = 640;
const DEFAULT_LABEL_WIDTH = 120;

// Row labels: rough width of one 13 px character, the gap kept clear of the plot, and
// the most lines a label may wrap onto.
const LABEL_CHAR_WIDTH = 7.2;
const LABEL_PADDING = 12;
const MIN_LABEL_CHARS = 6;
const MAX_LABEL_LINES = 3;

// Space above the plot (more when a reference label sits there) and below it for the
// tick labels (more when an axis label sits under them).
const PLOT_TOP = 6;
const PLOT_TOP_WITH_REFERENCE = 22;
const AXIS_HEIGHT = 24;
const AXIS_HEIGHT_WITH_LABEL = 48;

const DOT_RADIUS = 6;
// Invisible circle around each dot that catches the hover.
const DOT_HIT_RADIUS = 14;
const POINT_LABEL_GAP = 12;

// Tooltip: its gap above the dot, and the height of a panel title above the chart.
const TOOLTIP_GAP = 12;
const PANEL_TITLE_HEIGHT = 20;

const muted = "hsl(var(--muted-foreground))";

type Tip = { x: number; y: number; text: string };
type ActiveTip = { panel: number; tip: Tip };

type LabelSide = "left" | "right" | "above";

type LabelPlacement = { dx: number; dy: number; textDy: number | string; textAnchor: "start" | "end" | "middle" };

const LABEL_PLACEMENT: Record<LabelSide, LabelPlacement> = {
  right: { dx: POINT_LABEL_GAP, dy: 0, textDy: "0.35em", textAnchor: "start" },
  left: { dx: -POINT_LABEL_GAP, dy: 0, textDy: "0.35em", textAnchor: "end" },
  above: { dx: 0, dy: -POINT_LABEL_GAP, textDy: 0, textAnchor: "middle" },
};

// On narrow screens a left label moves above its dot, where it has room.
const labelSideOf = (point: Point, compact: boolean): LabelSide => {
  if (point.labelSide === "left") {
    return compact ? "above" : "left";
  }

  return "right";
};

const useWidth = (): readonly [React.RefObject<HTMLDivElement | null>, number | null] => {
  const ref = React.useRef<HTMLDivElement>(null);
  const [width, setWidth] = React.useState<number | null>(null);

  React.useEffect(() => {
    const element = ref.current;

    if (!element) {
      return;
    }

    const observer = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width));

    observer.observe(element);

    return () => observer.disconnect();
  }, []);

  return [ref, width] as const;
};

// Greedy word wrap for row labels: SVG text does not wrap by itself.
const wrapLabel = (label: string, width: number): string[] => {
  const maxChars = Math.max(MIN_LABEL_CHARS, Math.floor((width - LABEL_PADDING) / LABEL_CHAR_WIDTH));
  const lines: string[] = [];

  for (const word of label.split(" ")) {
    const last = lines.at(-1);

    if (last !== undefined && `${last} ${word}`.length <= maxChars) {
      lines[lines.length - 1] = `${last} ${word}`;
    } else {
      lines.push(word);
    }
  }

  return lines.slice(0, MAX_LABEL_LINES);
};

const formatTick = (value: number, panel: Panel): string => {
  const text = panel.decimals === undefined ? value.toLocaleString("en-US") : value.toFixed(panel.decimals);

  return panel.unit ? `${text} ${panel.unit}` : text;
};

type PlacedPoint = Point & { fill: string };

// A row's points in value order, each with the fill of its series and shade.
const rowPoints = (panel: Panel, row: string, seriesByKey: Map<string, Series>): PlacedPoint[] =>
  panel.points
    .filter((point) => point.row === row)
    .sort((a, b) => a.value - b.value)
    .map((point) => {
      const entry = seriesByKey.get(point.series);

      return { ...point, fill: fillOf(entry?.color ?? "neutral", point.shade ?? entry?.shade) };
    });

const RowLabel = ({ row, x, y, width }: { row: string; x: number; y: number; width: number }): React.ReactElement => {
  const lines = wrapLabel(row, width);

  return (
    <text x={x} y={y} textAnchor="end" fontSize={13} style={{ fill: "hsl(var(--foreground))" }}>
      {lines.map((line, lineIndex) => (
        // The first line moves up half a line per extra line, so the block stays centred on the row.
        <tspan key={lineIndex} x={x} dy={lineIndex === 0 ? `${0.35 - (lines.length - 1) * 0.6}em` : "1.2em"}>
          {line}
        </tspan>
      ))}
    </text>
  );
};

type DotMarkProps = {
  point: PlacedPoint;
  cx: number;
  cy: number;
  compact: boolean;
  onTip: (tip: Tip | null) => void;
};

const DotMark = ({ point, cx, cy, compact, onTip }: DotMarkProps): React.ReactElement => {
  const placement = LABEL_PLACEMENT[labelSideOf(point, compact)];
  const show = (): void => onTip({ x: cx, y: cy, text: point.tip });

  return (
    <g
      tabIndex={0}
      role="img"
      aria-label={point.tip}
      onMouseEnter={show}
      onFocus={show}
      onMouseLeave={() => onTip(null)}
      onBlur={() => onTip(null)}
      className="outline-none"
    >
      <circle cx={cx} cy={cy} r={DOT_HIT_RADIUS} style={{ fill: "transparent" }} />
      <circle cx={cx} cy={cy} r={DOT_RADIUS} style={{ fill: point.fill, stroke: "hsl(var(--background))", strokeWidth: 2 }} />
      {point.label ? (
        <text
          x={cx + placement.dx}
          y={cy + placement.dy}
          dy={placement.textDy}
          textAnchor={placement.textAnchor}
          fontSize={12}
          style={{ fill: muted }}
        >
          {point.label}
        </text>
      ) : null}
    </g>
  );
};

type PanelChartProps = {
  panel: Panel;
  rows: string[];
  series: Series[];
  width: number;
  labelWidth: number;
  rowHeight: number;
  onTip: (tip: Tip | null) => void;
};

const PanelChart = ({ panel, rows, series, width, labelWidth, rowHeight, onTip }: PanelChartProps): React.ReactElement => {
  const compact = rowHeight === NARROW_ROW_HEIGHT;
  const seriesByKey = new Map(series.map((entry) => [entry.key, entry]));
  const top = panel.reference ? PLOT_TOP_WITH_REFERENCE : PLOT_TOP;
  const plotBottom = top + rows.length * rowHeight;
  const height = plotBottom + (panel.axisLabel ? AXIS_HEIGHT_WITH_LABEL : AXIS_HEIGHT);
  const [domainMin, domainMax] = panel.domain;
  const plotLeft = labelWidth;
  const plotRight = width - RIGHT_MARGIN;
  const x = (value: number): number => plotLeft + ((value - domainMin) / (domainMax - domainMin)) * (plotRight - plotLeft);
  const y = (rowIndex: number): number => top + rowHeight * (rowIndex + 0.5);

  return (
    <div>
      {panel.title ? <p className="text-sm font-semibold text-foreground">{panel.title}</p> : null}
      <svg
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        className="block h-auto w-full overflow-visible"
        style={{ fontVariantNumeric: "tabular-nums" }}
      >
        {panel.ticks.map((tick) => (
          <g key={tick}>
            <line x1={x(tick)} x2={x(tick)} y1={top} y2={plotBottom} style={{ stroke: "hsl(var(--border))" }} />
            <text x={x(tick)} y={plotBottom + 16} textAnchor="middle" fontSize={12} style={{ fill: muted }}>
              {formatTick(tick, panel)}
            </text>
          </g>
        ))}
        {panel.axisLabel ? (
          <text x={(plotLeft + plotRight) / 2} y={height - 6} textAnchor="middle" fontSize={12} style={{ fill: muted }}>
            {panel.axisLabel}
          </text>
        ) : null}
        {panel.reference ? (
          <g>
            <line
              x1={x(panel.reference.value)}
              x2={x(panel.reference.value)}
              y1={top}
              y2={plotBottom}
              style={{ stroke: muted }}
            />
            <text x={x(panel.reference.value)} y={top - 8} textAnchor="middle" fontSize={12} style={{ fill: muted }}>
              {panel.reference.label}
            </text>
          </g>
        ) : null}
        {rows.map((row, rowIndex) => {
          const points = rowPoints(panel, row, seriesByKey);
          const first = points.at(0);
          const last = points.at(-1);

          return (
            <g key={row}>
              <RowLabel row={row} x={labelWidth - 10} y={y(rowIndex)} width={labelWidth} />
              {panel.connect && first && last && first !== last ? (
                <line
                  x1={x(first.value)}
                  x2={x(last.value)}
                  y1={y(rowIndex)}
                  y2={y(rowIndex)}
                  style={{ stroke: "var(--chart-link)", strokeWidth: 2 }}
                />
              ) : null}
              {points.map((point) => (
                <DotMark
                  key={`${point.series}-${point.value}`}
                  point={point}
                  cx={x(point.value)}
                  cy={y(rowIndex)}
                  compact={compact}
                  onTip={onTip}
                />
              ))}
            </g>
          );
        })}
      </svg>
    </div>
  );
};

const LegendItem = ({ fill, children }: { fill: string; children: React.ReactNode }): React.ReactElement => (
  <span className="inline-flex items-center gap-1.5">
    <span className="inline-block size-2.5 rounded-full" style={{ background: fill }} />
    {children}
  </span>
);

type LegendProps = Pick<DotPlotProps, "series" | "legendNote" | "shadeLegend">;

const Legend = ({ series, legendNote, shadeLegend }: LegendProps): React.ReactElement => {
  // With one series the shades are that series' color; with several they are only a key, in gray.
  const shadeColor: SeriesColor = series.length === 1 ? series[0].color : "neutral";

  return (
    <div className="mb-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground">
      {series.length > 1
        ? series.map((entry) => (
            <LegendItem key={entry.key} fill={fillOf(entry.color, entry.shade)}>
              {entry.label}
            </LegendItem>
        ))
        : null}
      {shadeLegend ? (
        <>
          <LegendItem fill={fillOf(shadeColor, "light")}>{shadeLegend.light}</LegendItem>
          <LegendItem fill={fillOf(shadeColor)}>{shadeLegend.dark}</LegendItem>
        </>
      ) : null}
      {legendNote ? <span>{legendNote}</span> : null}
    </div>
  );
};

const Tooltip = ({ tip, width, hasTitle }: { tip: Tip; width: number; hasTitle: boolean }): React.ReactElement => (
  <div
    role="tooltip"
    className="pointer-events-none absolute z-10 -translate-x-1/2 -translate-y-full rounded bg-foreground px-2 py-1 text-xs font-medium whitespace-nowrap text-background shadow"
    style={{
      left: `${(tip.x / width) * 100}%`,
      top: tip.y - TOOLTIP_GAP + (hasTitle ? PANEL_TITLE_HEIGHT : 0),
    }}
  >
    {tip.text}
  </div>
);

export const DotPlot = ({
  title,
  rows,
  series,
  panels,
  legendNote,
  shadeLegend,
  caption,
  labelWidth = DEFAULT_LABEL_WIDTH,
}: DotPlotProps): React.ReactElement => {
  const [chartRef, measured] = useWidth();
  const width = measured ?? DEFAULT_WIDTH;
  const fittedLabelWidth = Math.min(labelWidth, Math.round(width * NARROW_LABEL_SHARE));
  const rowHeight = fittedLabelWidth < labelWidth ? NARROW_ROW_HEIGHT : ROW_HEIGHT;
  const [activeTip, setActiveTip] = React.useState<ActiveTip | null>(null);
  const hasLegend = series.length > 1 || Boolean(legendNote) || Boolean(shadeLegend);

  return (
    <ArtifactFrame label={title} meta={`${rows.length} rows`} caption={caption}>
      <div className="font-sans" role="group" aria-label={title}>
        {hasLegend ? <Legend series={series} legendNote={legendNote} shadeLegend={shadeLegend} /> : null}
        <div ref={chartRef} className="space-y-4">
          {panels.map((panel, panelIndex) => (
            <div key={panel.title ?? panelIndex} className="relative">
              <PanelChart
                panel={panel}
                rows={rows}
                series={series}
                width={width}
                labelWidth={fittedLabelWidth}
                rowHeight={rowHeight}
                onTip={(next) => setActiveTip(next ? { panel: panelIndex, tip: next } : null)}
              />
              {activeTip?.panel === panelIndex ? <Tooltip tip={activeTip.tip} width={width} hasTitle={Boolean(panel.title)} /> : null}
            </div>
          ))}
        </div>
      </div>
    </ArtifactFrame>
  );
};
