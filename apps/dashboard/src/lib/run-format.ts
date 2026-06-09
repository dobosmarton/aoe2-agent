// Small presentation helpers shared by the run list and the experiment
// overview, so the label/time formatting stays consistent across both.

export type LabelVariant = "default" | "secondary" | "outline";

const _LABEL_VARIANT: Record<string, LabelVariant> = {
  race: "default",
  rank: "secondary",
  smoke: "outline",
};

export function labelVariant(label: string): LabelVariant {
  return _LABEL_VARIANT[label] ?? "outline";
}

/** Number of leading hex chars shown for an abbreviated run_id. */
const SHORT_RUN_ID_LEN = 8;

/** Abbreviate a run_id for display (e.g. "fd509730…"). Single source of the
 * truncation length, shared by the run list, sibling strip, and overview. */
export function shortRunId(runId: string): string {
  return `${runId.slice(0, SHORT_RUN_ID_LEN)}…`;
}

/** Compact relative time ("3m ago"); keep the full ISO in a tooltip. */
export function formatRelative(iso: string): string {
  const then = Date.parse(iso);
  if (Number.isNaN(then)) {
    return iso;
  }
  const sec = Math.max(0, Math.round((Date.now() - then) / 1000));
  if (sec < 60) {
    return `${String(sec)}s ago`;
  }
  const min = Math.round(sec / 60);
  if (min < 60) {
    return `${String(min)}m ago`;
  }
  const hr = Math.round(min / 60);
  if (hr < 24) {
    return `${String(hr)}h ago`;
  }
  return `${String(Math.round(hr / 24))}d ago`;
}
