import type { ArenaEvent, EventKind, WorldStateSnapshot } from "@/lib/events";

const _EVENT_KINDS: readonly EventKind[] = [
  "turn_start",
  "observation",
  "llm_prompt",
  "llm_response",
  "action",
  "action_result",
  "world_mutation",
  "fork",
  "metric",
] as const;

export function allEventKinds(): readonly EventKind[] {
  return _EVENT_KINDS;
}

/**
 * `turn_start` events carry the canonical state-at-start-of-turn.
 * We sweep the stream and index by turn_num so the World panel can render
 * any turn the scrubber lands on.
 */
export function statesByTurn(
  events: readonly ArenaEvent[],
): ReadonlyMap<number, WorldStateSnapshot> {
  const result = new Map<number, WorldStateSnapshot>();
  for (const event of events) {
    if (event.kind === "turn_start" && event.state !== null) {
      result.set(event.turn_num, event.state);
    }
  }
  return result;
}

/**
 * Returns the highest turn_num seen on any `turn_start` event.
 * Used by the Timeline to bound the slider; returns null if no turns yet.
 */
export function lastTurn(events: readonly ArenaEvent[]): number | null {
  let maxTurn: number | null = null;
  for (const event of events) {
    if (event.kind === "turn_start") {
      if (maxTurn === null || event.turn_num > maxTurn) {
        maxTurn = event.turn_num;
      }
    }
  }
  return maxTurn;
}

/**
 * Group events into ordered buckets by their turn. Events between
 * `turn_start(t)` (inclusive) and `turn_start(t+1)` (exclusive) belong
 * to turn `t`. Events before the first `turn_start` go to turn 0.
 */
export interface TurnGroup {
  readonly turn: number;
  readonly events: readonly ArenaEvent[];
}

export function eventsByTurn(events: readonly ArenaEvent[]): readonly TurnGroup[] {
  const groups = new Map<number, ArenaEvent[]>();
  let currentTurn = 0;
  for (const event of events) {
    if (event.kind === "turn_start") {
      currentTurn = event.turn_num;
    }
    const bucket = groups.get(currentTurn);
    if (bucket === undefined) {
      groups.set(currentTurn, [event]);
    } else {
      bucket.push(event);
    }
  }
  return [...groups.entries()]
    .sort(([a], [b]) => a - b)
    .map(([turn, items]) => ({ turn, events: items }));
}

/**
 * For the Diff panel: find any `fork` events in the stream.
 * Each tells us this run was created from a parent run at parent_t.
 */
export interface ForkInfo {
  readonly parent_run_id: string;
  readonly parent_t: number;
  readonly mutation_summary: string;
}

export function forksIn(events: readonly ArenaEvent[]): readonly ForkInfo[] {
  return events
    .filter((event): event is Extract<ArenaEvent, { kind: "fork" }> => event.kind === "fork")
    .map((event) => ({
      parent_run_id: event.parent_run_id,
      parent_t: event.parent_t,
      mutation_summary: event.mutation_summary,
    }));
}
