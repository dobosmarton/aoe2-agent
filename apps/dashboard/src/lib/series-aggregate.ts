// Aggregate per-run resource trajectories into one characteristic curve per
// profile, so the overview's per-resource charts show config behaviour rather
// than one line per (possibly dozens of) runs. Runs of the same profile are
// averaged turn-by-turn.

import { shortRunId } from "@/lib/run-format";
import type { RunSeries } from "@/lib/events";

export type ResourceKey = "food" | "wood" | "gold" | "stone" | "population";

export interface ProfileSeries {
  /** Stable key used both as the chart series key and the React key. */
  readonly key: string;
  /** Display label (profile name, or a short run_id when unlabelled). */
  readonly label: string;
  readonly points: ReadonlyArray<{
    readonly turn: number;
    readonly food: number;
    readonly wood: number;
    readonly gold: number;
    readonly stone: number;
    readonly population: number;
  }>;
}

interface Acc {
  food: number;
  wood: number;
  gold: number;
  stone: number;
  population: number;
  count: number;
}

/** Group runs by profile and mean each resource per turn. Profiles appear in
 * first-seen order; turns are sorted ascending. */
export function aggregateByProfile(
  series: readonly RunSeries[],
): readonly ProfileSeries[] {
  const groups = new Map<string, RunSeries[]>();
  for (const run of series) {
    const label = run.profile_name ?? shortRunId(run.run_id);
    const bucket = groups.get(label);
    if (bucket) {
      bucket.push(run);
    } else {
      groups.set(label, [run]);
    }
  }

  const result: ProfileSeries[] = [];
  for (const [label, runs] of groups) {
    const acc = new Map<number, Acc>();
    for (const run of runs) {
      for (const p of run.points) {
        const a = acc.get(p.turn) ?? {
          food: 0,
          wood: 0,
          gold: 0,
          stone: 0,
          population: 0,
          count: 0,
        };
        a.food += p.food;
        a.wood += p.wood;
        a.gold += p.gold;
        a.stone += p.stone;
        a.population += p.population;
        a.count += 1;
        acc.set(p.turn, a);
      }
    }
    const points = [...acc.entries()]
      .sort(([a], [b]) => a - b)
      .map(([turn, a]) => ({
        turn,
        food: a.food / a.count,
        wood: a.wood / a.count,
        gold: a.gold / a.count,
        stone: a.stone / a.count,
        population: a.population / a.count,
      }));
    result.push({ key: label, label, points });
  }
  return result;
}

/** Pivot the per-profile curves into wide rows for one resource, keyed by turn
 * with one column per profile — the shape `TimeSeriesChart` consumes. The
 * `Record<string, number>` is intentional: the columns are profile labels,
 * which are runtime-dynamic, so the key set is genuinely open here. */
export function resourceRows(
  profiles: readonly ProfileSeries[],
  resource: ResourceKey,
): ReadonlyArray<Record<string, number>> {
  const turns = new Set<number>();
  for (const p of profiles) {
    for (const pt of p.points) {
      turns.add(pt.turn);
    }
  }
  const lookup = profiles.map((p) => ({
    key: p.key,
    byTurn: new Map(p.points.map((pt) => [pt.turn, pt[resource]] as const)),
  }));
  return [...turns]
    .sort((a, b) => a - b)
    .map((turn) => {
      const row: Record<string, number> = { turn };
      for (const { key, byTurn } of lookup) {
        const v = byTurn.get(turn);
        if (v !== undefined) {
          row[key] = v;
        }
      }
      return row;
    });
}
