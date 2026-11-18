import * as THREE from "three";

// Text in the scene is real DOM in screen space: each label follows a point in the 3D
// scene, is placed on whole pixels at a fixed type size (so it stays sharp at every
// zoom), and can hang off its point on a hairline leader. Styles: `.wt3d-*` in
// globals.css.

export type LabelOptions = {
  /** Writes the label's anchor in world coordinates. */
  at: (out: THREE.Vector3) => void;
  /** Where the label's reference point sits relative to the anchor, CSS px. */
  offset?: [number, number];
  /** Which point of the label sits there: 0 left/top, 0.5 centre, 1 right/bottom. */
  align?: [number, number];
  /** Draw a hairline from the anchor to the label. */
  leader?: boolean;
  /** Keep the whole label inside the view, this many CSS px from its edges. */
  keepInView?: number;
  /** Leader colour (CSS). */
  leaderColor?: string;
};

export type Label = {
  element: HTMLElement;
  opacity: number;
  /** Extra CSS px offset, for motion. */
  shift: [number, number];
  /** Placement; may change at any time. */
  options: { offset: [number, number]; align: [number, number] };
  /** CSS px size, once measured (null until the label has been placed). */
  readonly size: [number, number] | null;
};

type Item = Label & {
  options: Required<Omit<LabelOptions, "leaderColor" | "keepInView">> & { leaderColor: string; keepInView: number | null };
  line: SVGLineElement | null;
  shown: number;
  size: [number, number] | null;
  /** Last written position and leader, to skip unchanged writes. */
  x: number;
  y: number;
  leaderKey: string;
};

const SVG = "http://www.w3.org/2000/svg";
const DEFAULT_LEADER_COLOR = "rgb(124 196 255 / 0.7)";

export type Labels = {
  add: (element: HTMLElement, options: LabelOptions) => Label;
  /** Places every label for this camera and view size (CSS px). */
  update: (camera: THREE.Camera, width: number, height: number) => void;
  dispose: () => void;
  /** Styles change label sizes (breakpoints, rotation): measure again after a resize. */
  remeasure: () => void;
  /** Labels kept in view also stay below this line (the site header), CSS px. */
  setInsetTop: (px: number) => void;
  layer: HTMLDivElement;
};

const hide = (item: Item): void => {
  if (item.shown === 0) {
    return;
  }

  item.element.style.visibility = "hidden";

  if (item.line) {
    item.line.style.visibility = "hidden";
  }

  item.shown = 0;
};

const show = (item: Item, opacity: number): void => {
  if (opacity === item.shown) {
    return;
  }

  item.element.style.opacity = String(opacity);
  item.element.style.visibility = "visible";

  if (item.line) {
    item.line.style.opacity = String(opacity);
    item.line.style.visibility = "visible";
  }

  item.shown = opacity;
};

/** Draws the leader from the anchor to the nearest point of the label's box. */
const drawLeader = (line: SVGLineElement, item: Item, anchorX: number, anchorY: number, size: [number, number]): void => {
  const x1 = anchorX.toFixed(1);
  const y1 = anchorY.toFixed(1);
  const x2 = THREE.MathUtils.clamp(anchorX, item.x, item.x + size[0]).toFixed(1);
  const y2 = THREE.MathUtils.clamp(anchorY, item.y, item.y + size[1]).toFixed(1);
  const key = `${x1} ${y1} ${x2} ${y2}`;

  if (key === item.leaderKey) {
    return;
  }

  line.setAttribute("x1", x1);
  line.setAttribute("y1", y1);
  line.setAttribute("x2", x2);
  line.setAttribute("y2", y2);
  item.leaderKey = key;
};

export const createLabels = (container: HTMLElement): Labels => {
  const layer = document.createElement("div");

  layer.className = "wt-labels";

  const svg = document.createElementNS(SVG, "svg");

  svg.classList.add("wt-leaders");
  layer.append(svg);
  container.append(layer);

  const items: Item[] = [];
  const point = new THREE.Vector3();

  const add = (element: HTMLElement, options: LabelOptions): Label => {
    element.classList.add("wt3d");
    layer.append(element);

    let line: SVGLineElement | null = null;

    if (options.leader) {
      line = document.createElementNS(SVG, "line");
      line.setAttribute("stroke", options.leaderColor ?? DEFAULT_LEADER_COLOR);
      svg.append(line);
    }

    const item: Item = {
      element,
      opacity: 0,
      shift: [0, 0],
      options: { offset: [0, 0], align: [0.5, 0.5], leader: false, leaderColor: "", keepInView: null, ...options },
      line,
      shown: -1,
      size: null,
      x: Number.NaN,
      y: Number.NaN,
      leaderKey: "",
    };

    items.push(item);

    return item;
  };

  let viewW = 0;
  let viewH = 0;
  let insetTop = 0;

  const setInsetTop = (px: number): void => {
    insetTop = px;
  };

  const remeasure = (): void => {
    for (const item of items) {
      item.size = null;
    }
  };

  const place = (item: Item, opacity: number, camera: THREE.Camera, width: number, height: number): void => {
    item.options.at(point);
    point.project(camera);

    if (point.z > 1 || point.z < -1) {
      hide(item);

      return;
    }

    // Measured once per resize (a hidden label still has its size).
    const size = (item.size ??= [item.element.offsetWidth, item.element.offsetHeight]);
    const anchorX = (point.x * 0.5 + 0.5) * width;
    const anchorY = (-point.y * 0.5 + 0.5) * height;
    const [offsetX, offsetY] = item.options.offset;
    const [alignX, alignY] = item.options.align;
    let x = Math.round(anchorX + offsetX + item.shift[0] - alignX * size[0]);
    let y = Math.round(anchorY + offsetY + item.shift[1] - alignY * size[1]);
    const margin = item.options.keepInView;

    if (margin != null) {
      x = Math.round(THREE.MathUtils.clamp(x, margin, Math.max(margin, width - margin - size[0])));
      y = Math.round(THREE.MathUtils.clamp(y, insetTop + margin, Math.max(insetTop + margin, height - margin - size[1])));
    }

    if (x !== item.x || y !== item.y) {
      item.element.style.transform = `translate3d(${x}px, ${y}px, 0)`;
      item.x = x;
      item.y = y;
    }

    show(item, opacity);

    if (item.line) {
      drawLeader(item.line, item, anchorX, anchorY, size);
    }
  };

  const update = (camera: THREE.Camera, width: number, height: number): void => {
    // Only touch the DOM when something changed: an unchanged frame writes nothing.
    if (width !== viewW || height !== viewH) {
      svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
      viewW = width;
      viewH = height;
      remeasure();
    }

    for (const item of items) {
      const opacity = Math.round(THREE.MathUtils.clamp(item.opacity, 0, 1) * 100) / 100;

      if (opacity <= 0) {
        hide(item);
      } else {
        place(item, opacity, camera, width, height);
      }
    }
  };

  const dispose = (): void => layer.remove();

  return { add, update, dispose, remeasure, setInsetTop, layer };
};

/** A detached `<div>` with this class and (trusted) inner HTML, ready for `labels.add`. */
export const el = (className: string, html: string): HTMLDivElement => {
  const div = document.createElement("div");

  div.className = className;
  div.innerHTML = html;

  return div;
};

/** The first element under `parent` that matches `selector`; throws when there is none. */
export const childOf = (parent: ParentNode, selector: string): HTMLElement => {
  const child = parent.querySelector<HTMLElement>(selector);

  if (!child) {
    throw new Error(`No element matches ${selector}`);
  }

  return child;
};
