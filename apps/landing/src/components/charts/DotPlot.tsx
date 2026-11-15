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

const fillOf = (color: SeriesColor, shade: Shade = "dark") =>
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

const muted = "hsl(var(--muted-foreground))";

function useWidth() {
  const ref = React.useRef<HTMLDivElement>(null);
  const [width, setWidth] = React.useState<number | null>(null);
  React.useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width));
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  return [ref, width] as const;
}

// Greedy word wrap for row labels: SVG text does not wrap by itself.
function wrapLabel(label: string, width: number): string[] {
  const maxChars = Math.max(6, Math.floor((width - 12) / 7.2));
  const lines: string[] = [];
  for (const word of label.split(" ")) {
    const last = lines[lines.length - 1];
    if (last !== undefined && `${last} ${word}`.length <= maxChars) lines[lines.length - 1] = `${last} ${word}`;
    else lines.push(word);
  }
  return lines.slice(0, 3);
}

function formatTick(value: number, panel: Panel) {
  const text = panel.decimals === undefined ? value.toLocaleString("en-US") : value.toFixed(panel.decimals);
  return panel.unit ? `${text} ${panel.unit}` : text;
}

type Tip = { x: number; y: number; text: string };

function PanelChart({
  panel,
  rows,
  series,
  width,
  labelWidth,
  rowHeight,
  onTip,
}: {
  panel: Panel;
  rows: string[];
  series: Series[];
  width: number;
  labelWidth: number;
  rowHeight: number;
  onTip: (tip: Tip | null) => void;
}) {
  const compact = rowHeight === NARROW_ROW_HEIGHT;
  const seriesOf = new Map(series.map((s) => [s.key, s]));
  const top = panel.reference ? 22 : 6;
  const plotBottom = top + rows.length * rowHeight;
  const height = plotBottom + (panel.axisLabel ? 48 : 24);
  const [d0, d1] = panel.domain;
  const x0 = labelWidth;
  const x1 = width - RIGHT_MARGIN;
  const x = (value: number) => x0 + ((value - d0) / (d1 - d0)) * (x1 - x0);
  const y = (row: number) => top + rowHeight * (row + 0.5);

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
          <text x={(x0 + x1) / 2} y={height - 6} textAnchor="middle" fontSize={12} style={{ fill: muted }}>
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
        {rows.map((row, i) => {
          const lines = wrapLabel(row, labelWidth);
          const points = panel.points
            .filter((p) => p.row === row)
            .sort((a, b) => a.value - b.value)
            .map((p) => {
              const s = seriesOf.get(p.series);
              return { ...p, fill: fillOf(s?.color ?? "neutral", p.shade ?? s?.shade) };
            });
          const first = points[0];
          const last = points[points.length - 1];
          return (
            <g key={row}>
              <text x={labelWidth - 10} y={y(i)} textAnchor="end" fontSize={13} style={{ fill: "hsl(var(--foreground))" }}>
                {lines.map((line, n) => (
                  <tspan key={n} x={labelWidth - 10} dy={n === 0 ? `${0.35 - (lines.length - 1) * 0.6}em` : "1.2em"}>
                    {line}
                  </tspan>
                ))}
              </text>
              {panel.connect && first && last && first !== last ? (
                <line
                  x1={x(first.value)}
                  x2={x(last.value)}
                  y1={y(i)}
                  y2={y(i)}
                  style={{ stroke: "var(--chart-link)", strokeWidth: 2 }}
                />
              ) : null}
              {points.map((point) => {
                const cx = x(point.value);
                const cy = y(i);
                const side = point.labelSide === "left" ? (compact ? "above" : "left") : "right";
                const show = () => onTip({ x: cx, y: cy, text: point.tip });
                return (
                  <g
                    key={`${point.series}-${point.value}`}
                    tabIndex={0}
                    role="img"
                    aria-label={point.tip}
                    onMouseEnter={show}
                    onFocus={show}
                    onMouseLeave={() => onTip(null)}
                    onBlur={() => onTip(null)}
                    className="outline-none"
                  >
                    <circle cx={cx} cy={cy} r={14} style={{ fill: "transparent" }} />
                    <circle
                      cx={cx}
                      cy={cy}
                      r={6}
                      style={{ fill: point.fill, stroke: "hsl(var(--background))", strokeWidth: 2 }}
                    />
                    {point.label ? (
                      <text
                        x={side === "right" ? cx + 12 : side === "left" ? cx - 12 : cx}
                        y={side === "above" ? cy - 12 : cy}
                        dy={side === "above" ? 0 : "0.35em"}
                        textAnchor={side === "right" ? "start" : side === "left" ? "end" : "middle"}
                        fontSize={12}
                        style={{ fill: muted }}
                      >
                        {point.label}
                      </text>
                    ) : null}
                  </g>
                );
              })}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

export default function DotPlot({
  title,
  rows,
  series,
  panels,
  legendNote,
  shadeLegend,
  caption,
  labelWidth = 120,
}: DotPlotProps) {
  // With one series the shades are that series' color; with several they are only a key, in gray.
  const shadeColor: SeriesColor = series.length === 1 ? series[0]!.color : "neutral";
  const [chartRef, measured] = useWidth();
  const width = measured ?? DEFAULT_WIDTH;
  const fittedLabelWidth = Math.min(labelWidth, Math.round(width * NARROW_LABEL_SHARE));
  const rowHeight = fittedLabelWidth < labelWidth ? NARROW_ROW_HEIGHT : ROW_HEIGHT;
  const [tip, setTip] = React.useState<{ panel: number; tip: Tip } | null>(null);
  const swatch = (background: string) => <span className="inline-block size-2.5 rounded-full" style={{ background }} />;

  return (
    <ArtifactFrame label={title} meta={`${rows.length} rows`} caption={caption}>
      <div className="font-sans" role="group" aria-label={title}>
        {series.length > 1 || legendNote || shadeLegend ? (
          <div className="mb-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground">
            {series.length > 1
              ? series.map((s) => (
                  <span key={s.key} className="inline-flex items-center gap-1.5">
                    {swatch(fillOf(s.color, s.shade))}
                    {s.label}
                  </span>
                ))
              : null}
            {shadeLegend ? (
              <>
                <span className="inline-flex items-center gap-1.5">
                  {swatch(fillOf(shadeColor, "light"))}
                  {shadeLegend.light}
                </span>
                <span className="inline-flex items-center gap-1.5">
                  {swatch(fillOf(shadeColor))}
                  {shadeLegend.dark}
                </span>
              </>
            ) : null}
            {legendNote ? <span>{legendNote}</span> : null}
          </div>
        ) : null}
        <div ref={chartRef} className="space-y-4">
          {panels.map((panel, i) => (
            <div key={panel.title ?? i} className="relative">
              <PanelChart
                panel={panel}
                rows={rows}
                series={series}
                width={width}
                labelWidth={fittedLabelWidth}
                rowHeight={rowHeight}
                onTip={(next) => setTip(next ? { panel: i, tip: next } : null)}
              />
              {tip?.panel === i ? (
                <div
                  role="tooltip"
                  className="pointer-events-none absolute z-10 -translate-x-1/2 -translate-y-full rounded bg-foreground px-2 py-1 text-xs font-medium whitespace-nowrap text-background shadow"
                  style={{
                    left: `${(tip.tip.x / width) * 100}%`,
                    top: tip.tip.y - 12 + (panel.title ? 20 : 0),
                  }}
                >
                  {tip.tip.text}
                </div>
              ) : null}
            </div>
          ))}
        </div>
      </div>
    </ArtifactFrame>
  );
}
