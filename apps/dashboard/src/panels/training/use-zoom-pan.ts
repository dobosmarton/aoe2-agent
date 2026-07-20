import { useRef, useState } from "react";

export const MIN_ZOOM = 1;
export const MAX_ZOOM = 12;

type Transform = {
  readonly z: number;
  readonly x: number;
  readonly y: number;
};

const IDENTITY: Transform = { z: 1, x: 0, y: 0 };

const clamp = (value: number, lo: number, hi: number): number =>
  Math.min(hi, Math.max(lo, value));

type ZoomPan = {
  readonly zoom: number;
  readonly offset: { readonly x: number; readonly y: number };
  readonly reset: () => void;
  readonly zoomBy: (factor: number) => void;
  readonly onWheel: (event: WheelEvent) => void;
  readonly onPointerDown: (event: React.PointerEvent<HTMLDivElement>) => void;
};

/**
 * Pan/zoom transform plus the handlers that drive it.
 *
 * The transform is `translate(x, y) scale(z)` with a top-left origin. The content
 * is flex-centred, so its unpanned origin sits at a layout point `C` that we read
 * back from the DOM as `rect.left - x`; a content point `p` therefore lands on
 * screen at `C + x + z * p`.
 *
 * Every update is computed synchronously from `ref.current` rather than inside a
 * `setState` updater: updaters must be pure, and nesting one `setState` inside
 * another applies the offset twice under StrictMode's double-invoke.
 *
 * The handlers are deliberately plain functions. React Compiler memoises them,
 * and hand-written useCallbacks here made it bail out of compiling this hook
 * entirely — losing more than they bought.
 */
export function useZoomPan(
  viewport: React.RefObject<HTMLDivElement | null>,
  content: React.RefObject<HTMLDivElement | null>,
): ZoomPan {
  const [transform, setTransform] = useState<Transform>(IDENTITY);
  const ref = useRef<Transform>(IDENTITY);

  /** Current geometry in client coordinates, undoing the live transform. */
  const measure = ():
    | { view: DOMRect; cx: number; cy: number; natW: number; natH: number }
    | null => {
    const view = viewport.current?.getBoundingClientRect();
    const box = content.current?.getBoundingClientRect();
    const { z, x, y } = ref.current;
    if (view === undefined || box === undefined || box.width === 0) {
      return null;
    }
    return {
      view,
      cx: box.left - x,
      cy: box.top - y,
      natW: box.width / z,
      natH: box.height / z,
    };
  };

  /** Keep the image glued to the viewport: fill it when zoomed past fit, centre
   * it on the axes where it is still smaller. Without this, panning and
   * cursor-anchored zoom can strand the image in a corner of empty space. */
  const commit = (next: Transform): void => {
    const m = measure();
    let { x, y } = next;
    if (m !== null) {
      const w = m.natW * next.z;
      const h = m.natH * next.z;
      x =
        w <= m.view.width
          ? m.view.left + (m.view.width - w) / 2 - m.cx
          : clamp(x, m.view.right - w - m.cx, m.view.left - m.cx);
      y =
        h <= m.view.height
          ? m.view.top + (m.view.height - h) / 2 - m.cy
          : clamp(y, m.view.bottom - h - m.cy, m.view.top - m.cy);
    }
    const settled = { z: next.z, x, y };
    ref.current = settled;
    setTransform(settled);
  };

  const reset = (): void => {
    ref.current = IDENTITY;
    setTransform(IDENTITY);
  };

  /** Zoom about a fixed client point; defaults to the viewport centre. */
  const zoomAbout = (factor: number, clientX?: number, clientY?: number): void => {
    const prev = ref.current;
    const z = clamp(prev.z * factor, MIN_ZOOM, MAX_ZOOM);
    if (z === prev.z) {
      return;
    }
    const m = measure();
    if (m === null) {
      commit({ z, x: 0, y: 0 });
      return;
    }
    // Solve `C + x' + z'*p = s` for x', where p = (s - C - x) / z.
    const sx = clientX ?? m.view.left + m.view.width / 2;
    const sy = clientY ?? m.view.top + m.view.height / 2;
    const ratio = z / prev.z;
    commit({
      z,
      x: sx - m.cx - ratio * (sx - m.cx - prev.x),
      y: sy - m.cy - ratio * (sy - m.cy - prev.y),
    });
  };

  const zoomBy = (factor: number): void => {
    zoomAbout(factor);
  };

  /** Attached manually by the caller: React registers `wheel` passively at the
   * root, so preventDefault only works on a listener we add ourselves. A
   * trackpad pinch arrives as ctrl+wheel. */
  const onWheel = (event: WheelEvent): void => {
    event.preventDefault();
    const step = event.ctrlKey ? 0.01 : 0.002;
    zoomAbout(Math.exp(-event.deltaY * step), event.clientX, event.clientY);
  };

  const onPointerDown = (event: React.PointerEvent<HTMLDivElement>): void => {
    if (ref.current.z === 1 || event.button !== 0) {
      return;
    }
    event.preventDefault();
    const node = event.currentTarget;
    const origin = { px: event.clientX, py: event.clientY, ...ref.current };
    node.setPointerCapture(event.pointerId);

    const onMove = (move: PointerEvent): void => {
      commit({
        z: origin.z,
        x: origin.x + (move.clientX - origin.px),
        y: origin.y + (move.clientY - origin.py),
      });
    };
    const onUp = (): void => {
      node.releasePointerCapture(event.pointerId);
      node.removeEventListener("pointermove", onMove);
      node.removeEventListener("pointerup", onUp);
      node.removeEventListener("pointercancel", onUp);
    };
    node.addEventListener("pointermove", onMove);
    node.addEventListener("pointerup", onUp);
    node.addEventListener("pointercancel", onUp);
  };

  return {
    zoom: transform.z,
    offset: { x: transform.x, y: transform.y },
    reset,
    zoomBy,
    onWheel,
    onPointerDown,
  };
}
