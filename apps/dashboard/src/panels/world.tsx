import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { EmptyState } from "@/components/empty-state";
import type { WorldStateSnapshot } from "@/lib/events";

interface WorldPanelProps {
  readonly states: ReadonlyMap<number, WorldStateSnapshot>;
  readonly selectedTurn: number | null;
}

interface ResourceRowProps {
  readonly label: string;
  readonly value: number;
  readonly previous: number | null;
}

function ResourceRow({ label, value, previous }: ResourceRowProps): React.ReactElement {
  const delta = previous === null ? null : value - previous;
  const deltaLabel =
    delta === null || delta === 0
      ? null
      : `${delta > 0 ? "+" : ""}${Math.round(delta)}`;
  return (
    <div className="flex items-baseline justify-between gap-2">
      <span className="text-muted-foreground text-xs capitalize">{label}</span>
      <span className="font-mono text-lg tabular-nums">
        {Math.round(value)}
        {deltaLabel === null ? null : (
          <span
            className={`ml-2 text-xs ${
              delta !== null && delta > 0 ? "text-emerald-400" : "text-red-400"
            }`}
          >
            {deltaLabel}
          </span>
        )}
      </span>
    </div>
  );
}

function buildingCounts(buildings: readonly string[]): ReadonlyArray<[string, number]> {
  const counts = new Map<string, number>();
  for (const building of buildings) {
    counts.set(building, (counts.get(building) ?? 0) + 1);
  }
  return [...counts.entries()].sort(([a], [b]) => a.localeCompare(b));
}

export function WorldPanel({
  states,
  selectedTurn,
}: WorldPanelProps): React.ReactElement {
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
  const previous =
    effectiveTurn === null ? null : (states.get(effectiveTurn - 1) ?? null);

  if (state === null) {
    return (
      <EmptyState
        title={`Turn ${effectiveTurn} has no recorded state`}
        hint="Try scrubbing to a turn whose state was captured."
      />
    );
  }

  const popPercent = state.pop_cap === 0 ? 0 : (state.population / state.pop_cap) * 100;
  const ageUpInProgress = state.age_up_ticks_remaining > 0;

  return (
    <div className="grid grid-cols-1 gap-3 p-4 lg:grid-cols-2">
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Resources</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <ResourceRow
            label="food"
            value={state.food}
            previous={previous?.food ?? null}
          />
          <ResourceRow
            label="wood"
            value={state.wood}
            previous={previous?.wood ?? null}
          />
          <ResourceRow
            label="gold"
            value={state.gold}
            previous={previous?.gold ?? null}
          />
          <ResourceRow
            label="stone"
            value={state.stone}
            previous={previous?.stone ?? null}
          />
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Population</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex items-baseline justify-between">
            <span className="text-muted-foreground text-xs">current / cap</span>
            <span className="font-mono text-lg tabular-nums">
              {state.population} / {state.pop_cap}
            </span>
          </div>
          <Progress value={popPercent} />
          <div className="text-muted-foreground text-xs">
            {state.villager_queue.length} villager{state.villager_queue.length === 1 ? "" : "s"} queued
          </div>
        </CardContent>
      </Card>

      <Card className="lg:col-span-2">
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center justify-between text-sm">
            <span>Age &amp; progress</span>
            <Badge variant={ageUpInProgress ? "default" : "outline"}>
              {ageUpInProgress
                ? `Age-up in ${state.age_up_ticks_remaining}`
                : state.age}
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="text-muted-foreground text-xs">
          Turn {state.turn} · age = {state.age}
          {ageUpInProgress
            ? ` · ${state.age_up_ticks_remaining} ticks until next age`
            : ""}
        </CardContent>
      </Card>

      <Card className="lg:col-span-2">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Buildings</CardTitle>
        </CardHeader>
        <CardContent>
          {state.buildings.length === 0 ? (
            <span className="text-muted-foreground text-xs">No buildings yet.</span>
          ) : (
            <div className="flex flex-wrap gap-2">
              {buildingCounts(state.buildings).map(([name, count]) => (
                <Badge key={name} variant="secondary">
                  {name}
                  {count > 1 ? ` x${count}` : ""}
                </Badge>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
