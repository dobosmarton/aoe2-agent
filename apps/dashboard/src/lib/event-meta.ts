import {
  Activity,
  CheckCircle2,
  Eye,
  GitBranch,
  MessageSquare,
  Play,
  Sparkles,
  Wand2,
  Zap,
  type LucideIcon,
} from "lucide-react";

import type { ArenaEvent, EventKind } from "@/lib/events";

// ---------------------------------------------------------------------------
// Shared event-kind presentation: color token, label, icon. Used by the Trace
// rows, the filter-bar legend, and anywhere else that color-codes by kind.
// `textClass` is a literal so Tailwind's scanner emits the utility; `colorVar`
// is for inline SVG/border styling where a token reference is more convenient.
// ---------------------------------------------------------------------------

export type EventKindMeta = {
  readonly label: string;
  readonly icon: LucideIcon;
  readonly colorVar: string;
  readonly textClass: string;
}

export const eventKindMeta: Record<EventKind, EventKindMeta> = {
  turn_start: {
    label: "turn_start",
    icon: Play,
    colorVar: "var(--event-turn-start)",
    textClass: "text-event-turn-start",
  },
  observation: {
    label: "observation",
    icon: Eye,
    colorVar: "var(--event-observation)",
    textClass: "text-event-observation",
  },
  llm_prompt: {
    label: "llm_prompt",
    icon: MessageSquare,
    colorVar: "var(--event-llm-prompt)",
    textClass: "text-event-llm-prompt",
  },
  llm_response: {
    label: "llm_response",
    icon: Sparkles,
    colorVar: "var(--event-llm-response)",
    textClass: "text-event-llm-response",
  },
  action: {
    label: "action",
    icon: Zap,
    colorVar: "var(--event-action)",
    textClass: "text-event-action",
  },
  action_result: {
    label: "action_result",
    icon: CheckCircle2,
    colorVar: "var(--event-action-result)",
    textClass: "text-event-action-result",
  },
  world_mutation: {
    label: "world_mutation",
    icon: Wand2,
    colorVar: "var(--event-world-mutation)",
    textClass: "text-event-world-mutation",
  },
  fork: {
    label: "fork",
    icon: GitBranch,
    colorVar: "var(--event-fork)",
    textClass: "text-event-fork",
  },
  metric: {
    label: "metric",
    icon: Activity,
    colorVar: "var(--event-metric)",
    textClass: "text-event-metric",
  },
};

/** One-line human summary of an event for the collapsed trace row. */
export const summarise = (event: ArenaEvent): string => {
  switch (event.kind) {
    case "turn_start":
      return `turn ${event.turn_num}`;
    case "observation":
      return `${event.entity_count} entities`;
    case "llm_prompt":
      return event.state_summary;
    case "llm_response":
      return `${event.actions.length} actions · $${event.cost_usd.toFixed(5)}`;
    case "action":
      return JSON.stringify(event.action);
    case "action_result":
      return `${event.action_type}${event.state_changed ? "" : " (no-op)"}`;
    case "world_mutation":
      return event.reason;
    case "fork":
      return `from ${event.parent_run_id.slice(0, 8)}@${event.parent_t}`;
    case "metric":
      return `${event.name} = ${event.value}`;
  }
};

/**
 * Optional right-aligned meta shown on the collapsed row (cost, action count).
 * Returns null when the kind carries no such signal.
 */
export const eventMetaTag = (event: ArenaEvent): string | null => {
  switch (event.kind) {
    case "llm_response":
      return `$${event.cost_usd.toFixed(5)}`;
    case "metric":
      return event.name === "cost_usd" ? `$${event.value.toFixed(5)}` : null;
    default:
      return null;
  }
};
