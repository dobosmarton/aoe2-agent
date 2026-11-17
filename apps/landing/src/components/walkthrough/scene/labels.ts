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
  /** Leader colour (CSS). */
  leaderColor?: string;
};

export type Label = {
  element: HTMLElement;
  opacity: number;
  /** Extra CSS px offset, for motion. */
  shift: [number, number];
};

type Item = Label & {
  options: Required<Omit<LabelOptions, "leaderColor">> & { leaderColor: string };
  line: SVGLineElement | null;
  shown: number;
  size: [number, number];
};

const SVG = "http://www.w3.org/2000/svg";

export function createLabels(container: HTMLElement) {
  const layer = document.createElement("div");
  layer.className = "wt-labels";
  const svg = document.createElementNS(SVG, "svg");
  svg.classList.add("wt-leaders");
  layer.append(svg);
  container.append(layer);

  const items: Item[] = [];
  const point = new THREE.Vector3();

  function add(element: HTMLElement, options: LabelOptions): Label {
    element.classList.add("wt3d");
    layer.append(element);
    let line: SVGLineElement | null = null;
    if (options.leader) {
      line = document.createElementNS(SVG, "line");
      line.setAttribute("stroke", options.leaderColor ?? "rgb(124 196 255 / 0.7)");
      svg.append(line);
    }
    const item: Item = {
      element,
      opacity: 0,
      shift: [0, 0],
      options: { offset: [0, 0], align: [0.5, 0.5], leader: false, leaderColor: "", ...options },
      line,
      shown: -1,
      size: [0, 0],
    };
    items.push(item);
    return item;
  }

  function update(camera: THREE.Camera, width: number, height: number) {
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    for (const item of items) {
      const o = Math.round(THREE.MathUtils.clamp(item.opacity, 0, 1) * 100) / 100;
      if (o <= 0) {
        if (item.shown !== 0) {
          item.element.style.visibility = "hidden";
          if (item.line) item.line.style.visibility = "hidden";
          item.shown = 0;
        }
        continue;
      }
      item.options.at(point);
      point.project(camera);
      if (point.z > 1 || point.z < -1) {
        item.element.style.visibility = "hidden";
        if (item.line) item.line.style.visibility = "hidden";
        item.shown = 0;
        continue;
      }
      if (item.size[0] === 0) item.size = [item.element.offsetWidth, item.element.offsetHeight];
      const ax = (point.x * 0.5 + 0.5) * width;
      const ay = (-point.y * 0.5 + 0.5) * height;
      const [ox, oy] = item.options.offset;
      const [alx, aly] = item.options.align;
      const x = Math.round(ax + ox + item.shift[0] - alx * item.size[0]);
      const y = Math.round(ay + oy + item.shift[1] - aly * item.size[1]);
      item.element.style.transform = `translate3d(${x}px, ${y}px, 0)`;
      if (o !== item.shown) {
        item.element.style.opacity = String(o);
        item.element.style.visibility = "visible";
        if (item.line) {
          item.line.style.opacity = String(o);
          item.line.style.visibility = "visible";
        }
        item.shown = o;
      }
      if (item.line) {
        // From the anchor to the nearest point of the label's box.
        const bx = THREE.MathUtils.clamp(ax, x, x + item.size[0]);
        const by = THREE.MathUtils.clamp(ay, y, y + item.size[1]);
        item.line.setAttribute("x1", ax.toFixed(1));
        item.line.setAttribute("y1", ay.toFixed(1));
        item.line.setAttribute("x2", bx.toFixed(1));
        item.line.setAttribute("y2", by.toFixed(1));
      }
    }
  }

  function dispose() {
    layer.remove();
  }

  return { add, update, dispose, layer };
}

export type Labels = ReturnType<typeof createLabels>;

export function el(className: string, html: string): HTMLDivElement {
  const div = document.createElement("div");
  div.className = className;
  div.innerHTML = html;
  return div;
}
