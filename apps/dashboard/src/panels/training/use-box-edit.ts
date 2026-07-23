import { useRef, useState } from "react";

import {
  clientToImagePoint,
  resizeBox,
  type BBox,
  type Handle,
} from "@/panels/training/box-geometry";

type Dimensions = { readonly width: number; readonly height: number };

type BoxEdit = {
  /** The in-progress geometry while a handle is held; null when idle. */
  readonly draft: BBox | null;
  readonly startDrag: (handle: Handle, event: React.PointerEvent<HTMLElement>) => void;
};

const sameBox = (a: BBox, b: BBox): boolean =>
  a.x === b.x && a.y === b.y && a.w === b.w && a.h === b.h;

/**
 * Pointer-drag editing for one bounding box, mirroring the capture + native-listener
 * shape of `useZoomPan.onPointerDown`. `startDrag` `stopPropagation`s so the drag
 * never reaches the pan handler; `onCommit` fires once on release, only if the box
 * moved. Handlers stay plain functions so React Compiler can memoise them (a
 * useCallback made it bail in the zoom-pan hook).
 */
export function useBoxEdit(params: {
  readonly content: React.RefObject<HTMLDivElement | null>;
  readonly record: Dimensions;
  readonly bbox: BBox;
  readonly onCommit: (next: BBox) => void;
}): BoxEdit {
  const { content, record, bbox, onCommit } = params;
  const [draft, setDraft] = useState<BBox | null>(null);
  // The freshest draft, read by `onUp` — draft state is async and can't be
  // observed reliably from inside the pointerup handler.
  const latest = useRef<BBox | null>(null);

  const startDrag = (handle: Handle, event: React.PointerEvent<HTMLElement>): void => {
    const rect = content.current?.getBoundingClientRect();
    if (rect === undefined || event.button !== 0) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    const node = event.currentTarget;
    const originBox = bbox;
    const originPoint = clientToImagePoint(event.clientX, event.clientY, rect, record);
    latest.current = null;
    node.setPointerCapture(event.pointerId);

    const onMove = (move: PointerEvent): void => {
      const liveRect = content.current?.getBoundingClientRect();
      if (liveRect === undefined) {
        return;
      }
      const point = clientToImagePoint(move.clientX, move.clientY, liveRect, record);
      const next = resizeBox(
        originBox,
        handle,
        point.x - originPoint.x,
        point.y - originPoint.y,
        record,
      );
      latest.current = next;
      setDraft(next);
    };
    const onUp = (): void => {
      node.releasePointerCapture(event.pointerId);
      node.removeEventListener("pointermove", onMove);
      node.removeEventListener("pointerup", onUp);
      node.removeEventListener("pointercancel", onUp);
      const committed = latest.current;
      latest.current = null;
      setDraft(null);
      if (committed !== null && !sameBox(committed, originBox)) {
        onCommit(committed);
      }
    };
    node.addEventListener("pointermove", onMove);
    node.addEventListener("pointerup", onUp);
    node.addEventListener("pointercancel", onUp);
  };

  return { draft, startDrag };
}
