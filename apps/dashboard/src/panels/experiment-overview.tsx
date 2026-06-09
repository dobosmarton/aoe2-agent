import { useMemo } from "react";
import { Trophy } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { ComparisonChart } from "@/components/charts/comparison-chart";
import { ChartLegend } from "@/components/charts/chart-legend";
import { TimeSeriesChart } from "@/components/charts/time-series-chart";
import type { ChartSeries, ComparisonDatum } from "@/components/charts/chart-types";
import { EmptyState } from "@/components/empty-state";
import { cn } from "@/lib/utils";
import { formatRelative, labelVariant, shortRunId } from "@/lib/run-format";
import {
  aggregateByProfile,
  resourceRows,
  type ResourceKey,
} from "@/lib/series-aggregate";
import { useOperationSeries } from "@/hooks/use-operation-series";
import type { RunMetrics, RunSummary } from "@/lib/events";
import type { RunGroup } from "@/lib/run-grouping";
import type { SummariesStatus } from "@/hooks/use-run-summaries";

// Distinct hues for per-profile lines, independent of the resource color tokens
// (a profile keeps the same color across all four resource charts).
const PALETTE: readonly string[] = [
  "#60a5fa",
  "#f472b6",
  "#34d399",
  "#fbbf24",
  "#a78bfa",
  "#fb7185",
  "#22d3ee",
  "#a3e635",
];

const RESOURCE_CHARTS: ReadonlyArray<{ key: ResourceKey; label: string }> = [
  { key: "food", label: "Food" },
  { key: "wood", label: "Wood" },
  { key: "gold", label: "Gold" },
  { key: "stone", label: "Stone" },
];

interface ExperimentOverviewProps {
  readonly group: RunGroup;
  readonly metricsByRunId: ReadonlyMap<string, RunMetrics>;
  readonly metricsStatus: SummariesStatus;
  readonly onOpenRun: (runId: string) => void;
}

interface Row {
  readonly run: RunSummary;
  readonly metrics: RunMetrics | undefined;
}

/** Lexicographic outcome score (higher is better), mirroring the backend's
 * `composite_score` (age → population → economy). Missing metrics sort last. */
function scoreTuple(m: RunMetrics | undefined): readonly [number, number, number] {
  return [
    m?.final_age_index ?? -1,
    m?.final_population ?? -1,
    m?.final_economy ?? -1,
  ];
}

function compareDesc(a: Row, b: Row): number {
  const [aAge, aPop, aEcon] = scoreTuple(a.metrics);
  const [bAge, bPop, bEcon] = scoreTuple(b.metrics);
  if (aAge !== bAge) {
    return bAge - aAge;
  }
  if (aPop !== bPop) {
    return bPop - aPop;
  }
  return bEcon - aEcon;
}

/** Short, unique-ish label for charts: the profile name when it's unique in
 * the operation (the `race` case), else a short run_id (the `rank` case, where
 * one profile plays many scenarios/rounds). */
function chartLabel(row: Row, profileCounts: ReadonlyMap<string, number>): string {
  const p = row.metrics?.profile_name;
  if (p !== null && p !== undefined && profileCounts.get(p) === 1) {
    return p;
  }
  return shortRunId(row.run.run_id);
}

const COST_FMT = (v: number): string => `$${v.toFixed(4)}`;

export function ExperimentOverview({
  group,
  metricsByRunId,
  metricsStatus,
  onOpenRun,
}: ExperimentOverviewProps): React.ReactElement {
  const rows = useMemo(() => {
    const built: Row[] = group.runs
      .map((run) => ({ run, metrics: metricsByRunId.get(run.run_id) }))
      // Drop non-game runs (e.g. a rank's synthetic "ranking" aggregate, which
      // carries only metric events → n_turns === 0). Runs without metrics yet
      // (live/pending) are kept so the operation still shows them.
      .filter((r) => r.metrics === undefined || r.metrics.n_turns > 0);
    return built.sort(compareDesc);
  }, [group.runs, metricsByRunId]);

  // Per-turn resource curves: all runs of this operation share one DuckDB file.
  const dbPath = group.runs[0]?.db_path ?? "";
  const { series } = useOperationSeries(dbPath);
  const profiles = useMemo(() => aggregateByProfile(series), [series]);
  const resourceSeries: readonly ChartSeries[] = useMemo(
    () =>
      profiles.map((p, i) => ({
        key: p.key,
        label: p.label,
        colorVar: PALETTE[i % PALETTE.length] ?? "var(--muted-foreground)",
      })),
    [profiles],
  );
  // Pivot each resource's rows once per profile change, not per render in JSX.
  const resourceData = useMemo(
    () => RESOURCE_CHARTS.map((rc) => ({ ...rc, rows: resourceRows(profiles, rc.key) })),
    [profiles],
  );

  const withMetrics = rows.filter((r) => r.metrics !== undefined);

  const profileCounts = useMemo(() => {
    const counts = new Map<string, number>();
    for (const r of withMetrics) {
      const p = r.metrics?.profile_name;
      if (p !== null && p !== undefined) {
        counts.set(p, (counts.get(p) ?? 0) + 1);
      }
    }
    return counts;
  }, [withMetrics]);

  const winnerRunId = withMetrics[0]?.run.run_id;
  const cheapestRunId = useMemo(() => {
    let best: { runId: string; cost: number } | null = null;
    for (const r of withMetrics) {
      const cost = r.metrics?.total_cost_usd ?? Infinity;
      if (best === null || cost < best.cost) {
        best = { runId: r.run.run_id, cost };
      }
    }
    return best?.runId;
  }, [withMetrics]);

  const populationData: readonly ComparisonDatum[] = withMetrics.map((r) => ({
    name: chartLabel(r, profileCounts),
    value: r.metrics?.final_population ?? 0,
    highlight: r.run.run_id === winnerRunId,
  }));
  const costData: readonly ComparisonDatum[] = withMetrics.map((r) => ({
    name: chartLabel(r, profileCounts),
    value: r.metrics?.total_cost_usd ?? 0,
    highlight: r.run.run_id === cheapestRunId,
  }));

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-auto p-4">
      {/* Header */}
      <div className="flex items-center gap-3">
        <Badge variant={labelVariant(group.label)}>{group.label}</Badge>
        <h2 className="text-sm font-semibold">Experiment overview</h2>
        <span className="text-muted-foreground text-xs tabular-nums">
          {group.runs.length} parallel runs
        </span>
        <span className="text-muted-foreground ml-auto text-xs" title={group.firstTs}>
          {formatRelative(group.lastTs)}
        </span>
      </div>

      {metricsStatus === "loading" && withMetrics.length === 0 ? (
        <EmptyState title="Loading metrics…" />
      ) : withMetrics.length === 0 ? (
        <EmptyState
          title="No finalized metrics yet"
          hint="Metrics appear once the operation's runs finish and its log file is released."
        />
      ) : (
        <>
          {/* Leaderboard */}
          <div className="border-border overflow-hidden rounded-lg border">
            <table className="w-full text-xs">
              <thead className="bg-muted/40 text-muted-foreground">
                <tr className="[&>th]:px-3 [&>th]:py-2 [&>th]:text-left [&>th]:font-medium">
                  <th className="w-8">#</th>
                  <th>Run</th>
                  <th>Profile</th>
                  <th className="text-right">Age</th>
                  <th className="text-right">Pop</th>
                  <th className="text-right">Economy</th>
                  <th className="text-right">Cost</th>
                  <th className="text-right">Turns</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row, idx) => {
                  const m = row.metrics;
                  const isWinner = row.run.run_id === winnerRunId;
                  return (
                    <tr
                      key={row.run.run_id}
                      onClick={() => {
                        onOpenRun(row.run.run_id);
                      }}
                      className={cn(
                        "border-border hover:bg-accent cursor-pointer border-t transition-colors [&>td]:px-3 [&>td]:py-2",
                        isWinner && "bg-emerald-500/5",
                      )}
                    >
                      <td className="text-muted-foreground tabular-nums">
                        {isWinner ? (
                          <Trophy className="size-3.5 text-emerald-500" />
                        ) : (
                          idx + 1
                        )}
                      </td>
                      <td className="font-mono" title={row.run.run_id}>
                        {shortRunId(row.run.run_id)}
                      </td>
                      <td className="truncate">{m?.profile_name ?? "—"}</td>
                      <td className="text-right">{m?.final_age ?? "—"}</td>
                      <td className="text-right tabular-nums">
                        {m?.final_population ?? "—"}
                      </td>
                      <td className="text-right tabular-nums">
                        {m?.final_economy !== null && m?.final_economy !== undefined
                          ? Math.round(m.final_economy)
                          : "—"}
                      </td>
                      <td className="text-right tabular-nums">
                        {m !== undefined ? COST_FMT(m.total_cost_usd) : "—"}
                      </td>
                      <td className="text-right tabular-nums">{m?.n_turns ?? "—"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Comparison charts */}
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <div className="border-border rounded-lg border p-3">
              <div className="text-muted-foreground mb-2 text-xs font-medium">
                Final population
              </div>
              <ComparisonChart
                data={populationData}
                colorVar="var(--population)"
                label="population"
                height={200}
              />
            </div>
            <div className="border-border rounded-lg border p-3">
              <div className="text-muted-foreground mb-2 text-xs font-medium">
                Total LLM cost
              </div>
              <ComparisonChart
                data={costData}
                colorVar="var(--event-metric)"
                label="cost"
                height={200}
                format={COST_FMT}
              />
            </div>
          </div>

          {/* Per-resource trajectories: one averaged line per profile, so you
              can read each config's characteristic economy over the game. */}
          {profiles.length > 0 ? (
            <div className="border-border rounded-lg border p-3">
              <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                <div className="text-muted-foreground text-xs font-medium">
                  Resources over time
                  <span className="ml-1 normal-case opacity-70">
                    (mean per profile)
                  </span>
                </div>
                <ChartLegend series={resourceSeries} />
              </div>
              <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                {resourceData.map((rc) => (
                  <div key={rc.key}>
                    <div className="text-muted-foreground mb-1 text-xs">
                      {rc.label}
                    </div>
                    <TimeSeriesChart
                      data={rc.rows}
                      series={resourceSeries}
                      variant="line"
                      height={160}
                    />
                  </div>
                ))}
              </div>
            </div>
          ) : null}
        </>
      )}
    </div>
  );
}
