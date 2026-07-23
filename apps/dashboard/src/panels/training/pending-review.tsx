import { useQuery } from "@tanstack/react-query";
import { Check, Pencil, Trash2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { trackerClassesQueryOptions } from "@/lib/queries";
import { type AnnotationDto, type ImageDetailDto } from "@/lib/training-api";
import { classColor } from "@/panels/training/class-color";
import type { AnnotationMutations } from "@/panels/training/use-annotation-mutations";

/**
 * Right rail for reviewing model-proposed boxes: approve, reclassify, reject, or
 * enter geometry-edit (the pencil) on each pending annotation. Model predictions
 * arrive as `model`/`pending` (see prelabel_pending.py); a decision here is what
 * turns one into training data.
 *
 * Write actions come from a shared `useAnnotationMutations` owned by the lightbox,
 * so the rail and the box editor invalidate the cache the same way. Rendered only
 * when the image has annotations — a truly unlabeled image has nothing to review.
 */
export function PendingReview(props: {
  readonly detail: ImageDetailDto;
  readonly mutations: AnnotationMutations;
  /** Annotation id whose box is in geometry-edit mode, if any. */
  readonly selectedId: number | null;
  readonly onSelect: (id: number | null) => void;
  readonly onFocusChange: (className: string | null) => void;
}): React.ReactElement | null {
  const { detail, mutations, selectedId, onSelect, onFocusChange } = props;
  const classesQuery = useQuery(trackerClassesQueryOptions());
  const classes = classesQuery.data ?? [];
  const busy = mutations.isPending;

  if (detail.annotations.length === 0) {
    return null;
  }
  const pending = detail.annotations.filter(
    (ann): ann is AnnotationDto & { id: number } => ann.status === "pending" && ann.id !== null,
  );
  const approved = detail.annotations.length - pending.length;

  // Approving/rejecting the box being edited must drop it out of edit mode.
  const decideAndDeselect = (id: number, decide: (id: number) => void): void => {
    if (selectedId === id) {
      onSelect(null);
    }
    decide(id);
  };

  return (
    <aside className="border-border flex w-72 shrink-0 flex-col border-l">
      <header className="border-border flex items-baseline justify-between border-b px-3 py-2">
        <span className="text-xs font-medium">Review</span>
        <span className="text-muted-foreground font-mono text-[11px] tabular-nums">
          {pending.length} pending · {approved} approved
        </span>
      </header>

      {pending.length === 0 ? (
        <p className="text-muted-foreground px-3 py-4 text-center text-xs">
          All boxes reviewed.
        </p>
      ) : (
        <ul className="min-h-0 flex-1 overflow-y-auto">
          {pending.map((ann) => {
            const editing = selectedId === ann.id;
            return (
              <li
                key={ann.id}
                className={`flex items-center gap-1.5 border-b px-2 py-1.5 ${
                  editing ? "bg-muted border-border" : "border-border/60"
                }`}
                onMouseEnter={() => {
                  onFocusChange(ann.class_name);
                }}
                onMouseLeave={() => {
                  onFocusChange(null);
                }}
              >
                <span
                  className="size-2.5 shrink-0 rounded-full"
                  style={{ backgroundColor: classColor(ann.class_id) }}
                  aria-hidden
                />
                <select
                  className="border-border bg-background min-w-0 flex-1 rounded border px-1 py-0.5 font-mono text-[11px]"
                  value={ann.class_id}
                  disabled={busy}
                  aria-label={`Class for box ${String(ann.id)}`}
                  onChange={(event) => {
                    mutations.reclassify(ann.id, Number(event.target.value));
                  }}
                >
                  {classes.map((cls) => (
                    <option key={cls.id} value={cls.id}>
                      {cls.name}
                    </option>
                  ))}
                </select>
                <Button
                  variant={editing ? "secondary" : "ghost"}
                  size="icon-xs"
                  aria-label={editing ? "Stop editing box" : "Edit box"}
                  onClick={() => {
                    onSelect(editing ? null : ann.id);
                  }}
                >
                  <Pencil className="size-3.5" />
                </Button>
                <Button
                  variant="ghost"
                  size="icon-xs"
                  aria-label="Approve"
                  isDisabled={busy}
                  onClick={() => {
                    decideAndDeselect(ann.id, mutations.approve);
                  }}
                >
                  <Check className="size-3.5 text-emerald-500" />
                </Button>
                <Button
                  variant="ghost"
                  size="icon-xs"
                  aria-label="Reject"
                  isDisabled={busy}
                  onClick={() => {
                    decideAndDeselect(ann.id, mutations.remove);
                  }}
                >
                  <Trash2 className="text-destructive size-3.5" />
                </Button>
              </li>
            );
          })}
        </ul>
      )}
    </aside>
  );
}
