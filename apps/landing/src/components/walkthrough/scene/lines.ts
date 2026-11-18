import * as THREE from "three";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";

// Anti-aliased lines with a width in CSS pixels (three's LineSegments2). Every
// material is registered so one call keeps their resolution in step with the view.

const materials = new Set<LineMaterial>();

export const setLineResolution = (width: number, height: number): void => {
  for (const m of materials) {
    m.resolution.set(width, height);
  }
};

export type FatLineOptions = {
  color: THREE.ColorRepresentation;
  /** CSS pixels. */
  width?: number;
  opacity?: number;
  dashed?: boolean;
  dashSize?: number;
  gapSize?: number;
};

export const lineMaterial = ({ color, width = 1.5, opacity = 1, dashed = false, dashSize = 0.3, gapSize = 0.2 }: FatLineOptions): LineMaterial => {
  const material = new LineMaterial({
    color: color instanceof THREE.Color ? color.getHex() : new THREE.Color(color).getHex(),
    linewidth: width,
    transparent: true,
    opacity,
    dashed,
    dashSize,
    gapSize,
    depthWrite: false,
    toneMapped: false,
  });

  // Colours above 1 make a line bloom; LineMaterial stores colour as a uniform.
  if (color instanceof THREE.Color) {
    material.color.copy(color);
  }

  materials.add(material);

  return material;
};

/** Segments as flat [ax, ay, az, bx, by, bz, ...]. */
export const fatSegments = (points: number[], options: FatLineOptions | LineMaterial): LineSegments2 => {
  const geometry = new LineSegmentsGeometry();

  geometry.setPositions(points);

  const material = options instanceof LineMaterial ? options : lineMaterial(options);
  const line = new LineSegments2(geometry, material);

  if (material.dashed) {
    line.computeLineDistances();
  }

  line.frustumCulled = false;

  return line;
};

/** A closed polyline as segments. */
export const loop = (points: readonly THREE.Vector3[]): number[] => {
  const out: number[] = [];

  points.forEach((p, i) => {
    const q = points[(i + 1) % points.length];

    out.push(p.x, p.y, p.z, q.x, q.y, q.z);
  });

  return out;
};

/** An open polyline as segments. */
export const path = (points: readonly THREE.Vector3[]): number[] => {
  const out: number[] = [];

  for (let i = 0; i + 1 < points.length; i++) {
    const p = points[i];
    const q = points[i + 1];

    out.push(p.x, p.y, p.z, q.x, q.y, q.z);
  }

  return out;
};

/** Corner brackets of a w x h rectangle centred on the origin in the XY plane. */
export const brackets = (w: number, h: number, arm: number, z = 0): number[] => {
  const out: number[] = [];

  for (const [sx, sy] of [[-1, -1], [1, -1], [1, 1], [-1, 1]] as const) {
    const x = (sx * w) / 2;
    const y = (sy * h) / 2;

    out.push(x, y, z, x - sx * arm, y, z, x, y, z, x, y - sy * arm, z);
  }

  return out;
};
