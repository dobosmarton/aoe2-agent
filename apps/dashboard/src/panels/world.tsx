import { useMemo } from "react";

import { ResourceChart } from "@/components/charts/resource-chart";
import { TimeSeriesChart } from "@/components/charts/time-series-chart";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { EmptyState } from "@/components/empty-state";
import { seriesFromStates } from "@/lib/event-utils";
import { SECTION_TITLE } from "@/lib/styles";
import type { WorldStateSnapshot } from "@/lib/events";

type WorldPanelProps = {
  readonly states: ReadonlyMap<number, WorldStateSnapshot>;
  readonly selectedTurn: number | null;
}

export function WorldPanel({
  states,
  selectedTurn,
}: WorldPanelProps): React.ReactElement {
  const rows = useMemo(() => seriesFromStates(states), [states]);

  if (states.size === 0) {
    return (
      <EmptyState
        title="No world state yet"
        hint="Waiting for the first turn_start event."
      />
    );
  }

  const effectiveTurn =
    selectedTurn ?? [...states.keys()].sort((a, b) => b - a)[0] ?? null;
  const state = effectiveTurn === null ? null : (states.get(effectiveTurn) ?? null);

  if (state === null) {
    return (
      <EmptyState
        title={`Turn ${String(effectiveTurn)} has no recorded state`}
        hint="Try scrubbing to a turn whose state was captured."
      />
    );
  }

  const popPercent = state.pop_cap === 0 ? 0 : (state.population / state.pop_cap) * 100;
  const ageUpInProgress = state.age_up_ticks_remaining > 0;
  const buildings = buildingCounts(state.buildings);

  return (
    <div className="flex flex-col gap-3 p-4">
      <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
        <Card className="gap-3 py-4">
          <CardHeader className="px-4">
            <CardTitle className={SECTION_TITLE}>Resources over time</CardTitle>
          </CardHeader>
          <CardContent className="px-2">
            <ResourceChart data={rows} selectedTurn={selectedTurn} height={200} />
          </CardContent>
        </Card>

        <Card className="gap-3 py-4">
          <CardHeader className="px-4">
            <CardTitle className={SECTION_TITLE}>Population</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 px-4">
            <div className="flex items-baseline justify-between">
              <span className="text-muted-foreground text-xs">current / cap</span>
              <span className="font-mono text-2xl tabular-nums">
                <span style={{ color: "var(--population)" }}>{state.population}</span>
                <span className="text-muted-foreground"> / {state.pop_cap}</span>
              </span>
            </div>
            <Progress
              value={popPercent}
              aria-label="population versus cap"
              className="[&>[data-slot=progress-indicator]]:bg-population"
            />
            <div className="text-muted-foreground text-xs">
              {state.villager_queue.length} villager
              {state.villager_queue.length === 1 ? "" : "s"} queued
            </div>
            <div className="-mx-2">
              <TimeSeriesChart
                data={rows}
                series={[
                  { key: "population", label: "Population", colorVar: "var(--population)" },
                  { key: "pop_cap", label: "Cap", colorVar: "var(--muted-foreground)" },
                ]}
                selectedTurn={selectedTurn}
                variant="line"
                height={120}
              />
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
        <Card className="gap-3 py-4">
          <CardHeader className="px-4">
            <CardTitle className={`${SECTION_TITLE} flex items-center justify-between`}>
              <span>Age &amp; progress</span>
              <Badge variant={ageUpInProgress ? "default" : "outline"}>
                {ageUpInProgress
                  ? `Age-up in ${String(state.age_up_ticks_remaining)}`
                  : state.age}
              </Badge>
            </CardTitle>
          </CardHeader>
          <CardContent className="text-muted-foreground px-4 text-xs">
            Turn {state.turn} · age = {state.age}
            {ageUpInProgress
              ? ` · ${String(state.age_up_ticks_remaining)} ticks until next age`
              : ""}
          </CardContent>
        </Card>

        <Card className="gap-3 py-4">
          <CardHeader className="px-4">
            <CardTitle className={SECTION_TITLE}>Buildings</CardTitle>
          </CardHeader>
          <CardContent className="px-4">
            {buildings.length === 0 ? (
              <span className="text-muted-foreground text-xs">No buildings yet.</span>
            ) : (
              <div className="flex flex-wrap gap-2">
                {buildings.map(([name, count]) => (
                  <Badge key={name} variant="secondary" className="font-mono">
                    {name}
                    {count > 1 ? ` ×${String(count)}` : ""}
                  </Badge>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

function buildingCounts(
  buildings: readonly string[],
): ReadonlyArray<readonly [string, number]> {
  const counts = new Map<string, number>();
  for (const building of buildings) {
    counts.set(building, (counts.get(building) ?? 0) + 1);
  }
  return [...counts.entries()].sort(([a], [b]) => a.localeCompare(b));
}
