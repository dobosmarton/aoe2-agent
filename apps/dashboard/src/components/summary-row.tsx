interface SummaryRowProps {
  readonly label: string;
  readonly value: string | number;
  readonly colorVar?: string;
}

/** One `label: value` line in a `StateSummary`, with an optional color swatch. */
export function SummaryRow({
  label,
  value,
  colorVar,
}: SummaryRowProps): React.ReactElement {
  return (
    <div className="flex items-baseline justify-between gap-2 py-0.5">
      <span className="text-muted-foreground flex items-center gap-1.5 text-xs">
        {colorVar === undefined ? null : (
          <span
            className="inline-block size-2 rounded-[2px]"
            style={{ backgroundColor: colorVar }}
          />
        )}
        {label}
      </span>
      <span className="font-mono text-sm tabular-nums">{value}</span>
    </div>
  );
}
