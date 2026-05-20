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
  | { kind: "turn_start"; turn_num: number; state: WorldStateSnapshot | null }
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
}
