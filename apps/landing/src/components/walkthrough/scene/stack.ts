import * as THREE from "three";

import { categoryColor } from "./board";
import { el, type Label, type Labels } from "./labels";
import { fatSegments, lineMaterial } from "./lines";
import { GLSL_OUTPUT_COLOR } from "./shaders";
import type { TurnData } from "./types";

// Detect: the detector as an exploded stack of glass plates beside the frame. Each
// plate shows this frame as that stage of the network sees it: the letterboxed input,
// three feature maps at strides 8, 16 and 32 (coarser grids, drawn from the frame's
// own edges and brightness), and the head, whose cells light up where a detection is
// centred, in its category colour. A pulse runs through the stack; then the boxes land.

// The model's input is 1280 x 736 (letterboxed); stride s gives a (736/s) x (1280/s) map.
const LAYERS = [
  { name: "input", note: "letterbox 736 × 1280", grid: [0, 0], mode: 0 },
  { name: "stride 8", note: "92 × 160", grid: [160, 92], mode: 1 },
  { name: "stride 16", note: "46 × 80", grid: [80, 46], mode: 2 },
  { name: "stride 32", note: "23 × 40", grid: [40, 23], mode: 3 },
  { name: "head", note: "65 classes · NMS 0.5", grid: [40, 23], mode: 4 },
] as const;

type Layer = (typeof LAYERS)[number];

/** Each plate's size as a fraction of the frame's: deeper stages are smaller. */
const PLATE_SCALE = [0.46, 0.4, 0.35, 0.3, 0.26];

/** `pulse` runs from 0 to this, one unit per plate. */
const PULSE_END = LAYERS.length;

const PLATE_FRAGMENT = /* glsl */ `
  uniform sampler2D uMap, uHead;
  uniform float uMode, uOn, uActive, uAspect;
  uniform vec2 uGrid;
  varying vec2 vUv;
  float lum(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }
  vec3 colormap(float t) {
    // Navy to blue to cyan to white, like an activation heatmap.
    vec3 a = vec3(0.02, 0.05, 0.14);
    vec3 b = vec3(0.1, 0.32, 0.95);
    vec3 c = vec3(0.35, 0.9, 1.2);
    vec3 d = vec3(1.4, 1.5, 1.6);
    return t < 0.4 ? mix(a, b, t / 0.4) : (t < 0.75 ? mix(b, c, (t - 0.4) / 0.35) : mix(c, d, (t - 0.75) / 0.25));
  }
  void main() {
    vec3 c;
    float gridLine = 0.0;
    if (uMode < 0.5) {
      // The letterboxed input: a thin bar above and below the picture.
      float bar = (1.0 - (1280.0 / 736.0) / (3024.0 / 1672.0)) * 0.5;
      float y = (vUv.y - bar) / (1.0 - 2.0 * bar);
      c = (y < 0.0 || y > 1.0) ? vec3(0.22, 0.24, 0.3) : texture2D(uMap, vec2(vUv.x, y)).rgb;
    } else {
      vec2 cell = floor(vUv * uGrid);
      vec2 uvc = (cell + 0.5) / uGrid;
      float bias = log2(2048.0 / uGrid.x);
      vec2 d = 1.0 / uGrid;
      if (uMode < 3.5) {
        float l = lum(texture2D(uMap, uvc, bias).rgb);
        float lx = lum(texture2D(uMap, uvc + vec2(d.x, 0.0), bias).rgb);
        float ly = lum(texture2D(uMap, uvc + vec2(0.0, d.y), bias).rgb);
        float edge = clamp(length(vec2(lx - l, ly - l)) * 4.0, 0.0, 1.0);
        // Early maps respond to edges; deeper ones to larger structures.
        float t = uMode < 1.5 ? edge : (uMode < 2.5 ? mix(edge, l, 0.35) : smoothstep(0.1, 0.9, mix(edge, 1.0 - l, 0.5)));
        c = colormap(clamp(t * 1.1, 0.0, 1.0));
      } else {
        vec4 h = texture2D(uHead, vUv);
        c = mix(vec3(0.03, 0.06, 0.15), h.rgb * 1.6, h.a);
      }
      // Cell lines, one pixel wide; they fade out where cells get too small to draw.
      vec2 g = vUv * uGrid;
      vec2 fw = max(fwidth(g), vec2(1e-5));
      vec2 toLine = abs(fract(g - 0.5) - 0.5) / fw;
      gridLine = (1.0 - smoothstep(0.0, 1.0, min(toLine.x, toLine.y))) * (1.0 - smoothstep(0.12, 0.3, max(fw.x, fw.y)));
    }
    c = mix(c, vec3(0.5, 0.75, 1.4), gridLine * 0.25);
    // Glass: a crisp lit rim and a soft inner falloff.
    vec2 px = min(vUv, 1.0 - vUv) / max(fwidth(vUv), vec2(1e-5));
    float rim = 1.0 - smoothstep(0.6, 1.8, min(px.x, px.y));
    float bright = 0.5 + 0.5 * uActive;
    c = c * bright + vec3(0.55, 0.85, 1.8) * rim * (0.6 + 0.8 * uActive);
    float a = uOn * mix(0.82, 1.0, rim);
    gl_FragColor = vec4(c, a);
    ${GLSL_OUTPUT_COLOR}
  }
`;

type PlateUniforms = {
  uMap: { value: THREE.Texture };
  uHead: { value: THREE.Texture };
  uMode: { value: number };
  uGrid: { value: THREE.Vector2 };
  uOn: { value: number };
  uActive: { value: number };
  uAspect: { value: number };
};

type Plate = { mesh: THREE.Mesh; uniforms: PlateUniforms; w: number; h: number; label: Label };

export type Stack = {
  group: THREE.Group;
  /** `on` shows the stack; `pulse` runs 0..5 through the plates. */
  update: (on: number, pulse: number) => void;
};

/** The head's cells (its 40 x 23 grid), lit in the category colour where a detection's centre falls. */
const createHeadTexture = (data: TurnData): THREE.DataTexture => {
  const [columns, rows] = LAYERS[LAYERS.length - 1].grid;
  const head = new Uint8Array(columns * rows * 4);
  for (const e of data.entities) {
    const cx = Math.min(columns - 1, Math.floor((e.x + e.w / 2) * columns));
    const cy = Math.min(rows - 1, Math.floor((1 - (e.y + e.h / 2)) * rows));
    const c = new THREE.Color(categoryColor(e.cat));
    head.set([c.r * 255, c.g * 255, c.b * 255, 255], (cy * columns + cx) * 4);
  }
  const texture = new THREE.DataTexture(head, columns, rows, THREE.RGBAFormat);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.magFilter = texture.minFilter = THREE.NearestFilter;
  texture.needsUpdate = true;
  return texture;
};

// A deck rising behind the frame to the upper right: each stage deeper into the
// network sits further back and smaller.
const platePosition = (i: number): THREE.Vector3 => new THREE.Vector3(5.6 + i * 2.1, 2.7 + i * 1.75, -2 - i * 1.5);

const createPlate = (layer: Layer, i: number, frame: THREE.Texture, headTexture: THREE.Texture, boardSize: THREE.Vector2, labels: Labels): Plate => {
  const w = boardSize.x * PLATE_SCALE[i];
  const h = boardSize.y * PLATE_SCALE[i];
  const uniforms: PlateUniforms = {
    uMap: { value: frame },
    uHead: { value: headTexture },
    uMode: { value: layer.mode },
    uGrid: { value: new THREE.Vector2(layer.grid[0], layer.grid[1]) },
    uOn: { value: 0 },
    uActive: { value: 0 },
    uAspect: { value: w / h },
  };
  const mesh = new THREE.Mesh(
    new THREE.PlaneGeometry(w, h),
    new THREE.ShaderMaterial({
      uniforms,
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      vertexShader: /* glsl */ `
        varying vec2 vUv;
        void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }
      `,
      fragmentShader: PLATE_FRAGMENT,
    }),
  );
  mesh.position.copy(platePosition(i));
  mesh.renderOrder = 4;

  const label = labels.add(el("wt3d-layer", `<span>${layer.name}</span><i>${layer.note}</i>`), {
    at: (out) => out.set(-w / 2, h / 2, 0).applyMatrix4(mesh.matrixWorld),
    offset: [-10, 0],
    align: [1, 0.5],
  });
  return { mesh, uniforms, w, h, label };
};

/** Each plate's corners to the next plate's, a frustum through the stack. Needs the plates' matrices. */
const connectorSegments = (plates: readonly Plate[]): number[] => {
  const corner = (plate: Plate, sx: number, sy: number): THREE.Vector3 =>
    new THREE.Vector3((sx * plate.w) / 2, (sy * plate.h) / 2, 0).applyMatrix4(plate.mesh.matrix);
  const segments: number[] = [];
  for (let k = 0; k + 1 < plates.length; k++) {
    for (const [sx, sy] of [[-1, -1], [1, -1], [1, 1], [-1, 1]] as const) {
      const a = corner(plates[k], sx, sy);
      const b = corner(plates[k + 1], sx, sy);
      segments.push(a.x, a.y, a.z, b.x, b.y, b.z);
    }
  }
  return segments;
};

export const createStack = (data: TurnData, frame: THREE.Texture, labels: Labels, origin: THREE.Vector3, boardW: number, boardH: number): Stack => {
  const group = new THREE.Group();
  group.position.copy(origin);

  const headTexture = createHeadTexture(data);
  const boardSize = new THREE.Vector2(boardW, boardH);
  const plates = LAYERS.map((layer, i) => {
    const plate = createPlate(layer, i, frame, headTexture, boardSize, labels);
    group.add(plate.mesh);
    return plate;
  });

  group.updateMatrixWorld(true);
  const connectorMaterial = lineMaterial({ color: new THREE.Color(0.45, 0.75, 1.6), width: 1, opacity: 0 });
  group.add(fatSegments(connectorSegments(plates), connectorMaterial));

  const update = (on: number, pulse: number): void => {
    group.visible = on > 0.002;
    plates.forEach((p, i) => {
      // Plates appear one after another, slightly staggered.
      const appear = THREE.MathUtils.clamp(on * 1.6 - i * 0.15, 0, 1);
      const passing = 1 - THREE.MathUtils.smoothstep(Math.abs(pulse - (i + 0.5)), 0, 1.1);
      const passed = pulse > i + 0.5;
      const done = passed ? 0.45 : 0;
      p.uniforms.uOn.value = appear;
      p.uniforms.uActive.value = Math.max(passing, done);
      p.mesh.scale.setScalar(0.92 + 0.08 * appear);
      p.label.opacity = appear * (0.55 + 0.45 * Math.max(passing, passed ? 1 : 0));
    });
    connectorMaterial.opacity = on * (0.25 + 0.35 * THREE.MathUtils.clamp(pulse / PULSE_END, 0, 1));
  };

  return { group, update };
};
