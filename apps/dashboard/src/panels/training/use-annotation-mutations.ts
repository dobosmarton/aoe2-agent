import { useMutation, useQueryClient } from "@tanstack/react-query";

import { deleteAnnotation, updateAnnotation } from "@/lib/training-api";
import type { BBox } from "@/panels/training/box-geometry";

export type AnnotationMutations = {
  readonly approve: (id: number) => void;
  readonly reclassify: (id: number, classId: number) => void;
  readonly setGeometry: (id: number, box: BBox) => void;
  readonly remove: (id: number) => void;
  readonly isPending: boolean;
};

/**
 * Annotation write actions shared by the review rail and the box editor, so the
 * cache-invalidation contract has one owner: each mutation sweeps the `["tracker", …]`
 * subtree (open detail + image-list counts + coverage stats). A geometry or class
 * edit flips the box to `source="human"` server-side; a bare approve keeps provenance.
 */
export function useAnnotationMutations(): AnnotationMutations {
  const queryClient = useQueryClient();
  const invalidate = (): Promise<void> =>
    queryClient.invalidateQueries({ queryKey: ["tracker"] });

  const patch = useMutation({
    mutationFn: (vars: {
      readonly id: number;
      readonly body: Parameters<typeof updateAnnotation>[1];
    }) => updateAnnotation(vars.id, vars.body),
    onSuccess: invalidate,
  });
  const remove = useMutation({
    mutationFn: (id: number) => deleteAnnotation(id),
    onSuccess: invalidate,
  });

  return {
    approve: (id): void => {
      patch.mutate({ id, body: { status: "approved" } });
    },
    reclassify: (id, classId): void => {
      patch.mutate({ id, body: { class_id: classId } });
    },
    setGeometry: (id, box): void => {
      patch.mutate({
        id,
        body: { geom_type: "bbox", coords: [box.x, box.y, box.w, box.h] },
      });
    },
    remove: (id): void => {
      remove.mutate(id);
    },
    isPending: patch.isPending || remove.isPending,
  };
}
