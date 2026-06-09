import type { ComparisonDatum } from "@/components/charts/chart-types";

interface TooltipPayloadEntry {
  readonly value?: number | string;
  readonly payload?: ComparisonDatum;
}

interface ComparisonTooltipProps {
  readonly active?: boolean;
  readonly payload?: readonly TooltipPayloadEntry[];
  readonly label?: string | number;
  readonly metricLabel: string;
  readonly colorVar: string;
  readonly format: (value: number) => string;
}

/** Themed tooltip for `ComparisonChart` (passed to recharts via `content`). */
export function ComparisonTooltip({
  active,
  payload,
  label,
  metricLabel,
  colorVar,
  format,
}: ComparisonTooltipProps): React.ReactElement | null {
  if (!active || payload === undefined || payload.length === 0) {
    return null;
  }
  const raw = payload[0]?.value;
  const value = typeof raw === "number" ? format(raw) : String(raw ?? "");
  return (
    <div className="border-border bg-popover text-popover-foreground rounded-md border px-2.5 py-2 text-xs shadow-md">
      <div className="text-muted-foreground mb-1 font-mono">{String(label)}</div>
      <div className="flex items-center justify-between gap-3">
        <span className="flex items-center gap-1.5">
          <span
            className="inline-block size-2 rounded-[2px]"
            style={{ backgroundColor: colorVar }}
          />
          <span className="text-muted-foreground">{metricLabel}</span>
        </span>
        <span className="font-mono tabular-nums">{value}</span>
      </div>
    </div>
  );
}
