// Geometry for the box overlay/editor — no React, no DOM writes, so the math is
// unit-testable. Absolute pixels, top-left origin: same convention as the DB
// `coords` and the detector's bbox (see geometry.py). The only app import is the
// AnnotationDto *type*, erased at build, so this stays runtime-dependency-free.

import type { AnnotationDto } from "@/lib/training-api";

/** Axis-aligned box, `coords = [x, y, w, h]` order. */
export type BBox = {
  readonly x: number;
  readonly y: number;
  readonly w: number;
  readonly h: number;
};

/** `move` drags the whole box; the rest drag one edge (n/s/e/w) or corner. */
export type Handle = "move" | "n" | "s" | "e" | "w" | "ne" | "nw" | "se" | "sw";

type Dimensions = { readonly width: number; readonly height: number };

/** Smallest box a resize may collapse to, in natural pixels. */
export const MIN_BOX_SIZE = 4;

/** The 8 resize handles: fractional anchor within the box + the cursor to show. */
export const HANDLES: readonly {
  readonly id: Handle;
  readonly fx: number;
  readonly fy: number;
  readonly cursor: string;
}[] = [
  { id: "nw", fx: 0, fy: 0, cursor: "nwse-resize" },
  { id: "n", fx: 0.5, fy: 0, cursor: "ns-resize" },
  { id: "ne", fx: 1, fy: 0, cursor: "nesw-resize" },
  { id: "e", fx: 1, fy: 0.5, cursor: "ew-resize" },
  { id: "se", fx: 1, fy: 1, cursor: "nwse-resize" },
  { id: "s", fx: 0.5, fy: 1, cursor: "ns-resize" },
  { id: "sw", fx: 0, fy: 1, cursor: "nesw-resize" },
  { id: "w", fx: 0, fy: 0.5, cursor: "ew-resize" },
];

const clamp = (value: number, lo: number, hi: number): number =>
  Math.min(Math.max(value, lo), hi);

const pct = (value: number, extent: number): string => `${String((value / extent) * 100)}%`;

/** Counter-scale for a box's border/handles/label: grows sublinearly with zoom and
 * caps, so chrome stays legible without ever swallowing the object it marks. */
export const chromeScale = (zoom: number): number => Math.min(zoom ** 0.5, 2.2) / zoom;

/** Absolute-pixel box for any annotation; a polygon collapses to its bounding extent. */
export const annotationBBox = (annotation: AnnotationDto): BBox => {
  if (annotation.geom_type === "bbox") {
    const [x, y, w, h] = annotation.coords;
    return { x, y, w, h };
  }
  const xs = annotation.coords.map(([px]) => px);
  const ys = annotation.coords.map(([, py]) => py);
  const minX = Math.min(...xs);
  const minY = Math.min(...ys);
  return { x: minX, y: minY, w: Math.max(...xs) - minX, h: Math.max(...ys) - minY };
};

/** A box as CSS `%` offsets of the image's natural size, so the overlay tracks
 * whatever size the browser renders — no measuring, no resize listener. */
export const bboxToPercent = (
  box: BBox,
  dims: Dimensions,
): { readonly left: string; readonly top: string; readonly width: string; readonly height: string } => ({
  left: pct(box.x, dims.width),
  top: pct(box.y, dims.height),
  width: pct(box.w, dims.width),
  height: pct(box.h, dims.height),
});

/**
 * Client (viewport) point → natural image pixels. `rect` is
 * `content.getBoundingClientRect()`, which already bakes in the zoom/pan
 * transform, so a fraction across it maps straight onto natural-pixel space.
 */
export const clientToImagePoint = (
  clientX: number,
  clientY: number,
  rect: DOMRect,
  record: Dimensions,
): { x: number; y: number } => {
  const fx = rect.width === 0 ? 0 : (clientX - rect.left) / rect.width;
  const fy = rect.height === 0 ? 0 : (clientY - rect.top) / rect.height;
  return {
    x: clamp(fx, 0, 1) * record.width,
    y: clamp(fy, 0, 1) * record.height,
  };
};

/**
 * Apply a pointer delta (image pixels) to `origin` for one handle. Each moved edge
 * is clamped to the frame and stops `MIN_BOX_SIZE` short of its opposite, so the box
 * can't invert or escape — no handle flip, `w`/`h` stay positive by construction.
 */
export const resizeBox = (
  origin: BBox,
  handle: Handle,
  dImgX: number,
  dImgY: number,
  record: Dimensions,
  minSize: number = MIN_BOX_SIZE,
): BBox => {
  if (handle === "move") {
    return {
      x: clamp(origin.x + dImgX, 0, record.width - origin.w),
      y: clamp(origin.y + dImgY, 0, record.height - origin.h),
      w: origin.w,
      h: origin.h,
    };
  }

  let left = origin.x;
  let top = origin.y;
  let right = origin.x + origin.w;
  let bottom = origin.y + origin.h;

  // A handle never carries opposing letters (no "nw"+"e"), so these are independent.
  if (handle.includes("w")) left = clamp(left + dImgX, 0, right - minSize);
  if (handle.includes("e")) right = clamp(right + dImgX, left + minSize, record.width);
  if (handle.includes("n")) top = clamp(top + dImgY, 0, bottom - minSize);
  if (handle.includes("s")) bottom = clamp(bottom + dImgY, top + minSize, record.height);

  return { x: left, y: top, w: right - left, h: bottom - top };
};
