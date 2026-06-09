// Discriminated union mirroring `evaluation/event_log.py`'s payload kinds.
// Each new payload kind on the Python side needs a matching variant here.

export interface WorldStateSnapshot {
  food: number;
  wood: number;
  gold: number;
  stone: number;
  population: number;
  pop_cap: number;
  age: string;
  buildings: readonly string[];
  villager_queue: readonly number[];
  age_up_ticks_remaining: number;
  turn: number;
}

export type ArenaEvent =
  | {
      kind: "turn_start";
      turn_num: number;
      state: WorldStateSnapshot | null;
      // Racing config that produced this run; null for forks / pre-labeling runs.
      profile_name?: string | null;
    }
  | { kind: "observation"; entity_count: number; classes: readonly string[] }
  | { kind: "llm_prompt"; state_summary: string }
  | {
      kind: "llm_response";
      actions: ReadonlyArray<Record<string, unknown>>;
      reasoning: string;
      cost_usd: number;
    }
  | { kind: "action"; index_in_turn: number; action: Record<string, unknown> }
  | {
      kind: "action_result";
      index_in_turn: number;
      action_type: string;
      state_changed: boolean;
    }
  | {
      kind: "world_mutation";
      before_summary: string;
      after_summary: string;
      reason: string;
    }
  | {
      kind: "fork";
      parent_run_id: string;
      parent_t: number;
      mutation_summary: string;
    }
  | { kind: "metric"; name: string; value: number };

export type EventKind = ArenaEvent["kind"];

export interface RunSummary {
  run_id: string;
  db_path: string;
  label: string;
  n_events: number;
  first_ts: string;
  last_ts: string;
  // "running" = a live run from the broker; "complete" = a finalized DuckDB run.
  status: "running" | "complete";
}

// Mirrors `RunMetrics` in apps/api/src/server.py — end-of-run comparable
// metrics for the experiment overview. `final_age_index` is the rank of
// `final_age` in the backend's AGE_SEQUENCE, so we sort by the same
// lexicographic score (age → population → economy) without duplicating the
// age order here. Fields are null for runs missing a final snapshot/profile.
export interface RunMetrics {
  run_id: string;
  profile_name: string | null;
  total_cost_usd: number;
  n_turns: number;
  final_age: string | null;
  final_age_index: number | null;
  final_population: number | null;
  final_economy: number | null;
}

// Mirrors `RunSeriesPoint` / `RunSeries` in apps/api/src/server.py — per-turn
// resource trajectories for the overview's per-resource curves.
export interface RunSeriesPoint {
  turn: number;
  food: number;
  wood: number;
  gold: number;
  stone: number;
  population: number;
}

export interface RunSeries {
  run_id: string;
  profile_name: string | null;
  points: RunSeriesPoint[];
}
