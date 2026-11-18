import * as THREE from "three";
import type { LineSegments2 } from "three/addons/lines/LineSegments2.js";

import { createFlatBoxes, LAND_DELAY_SPAN, LAND_DURATION } from "./boxes";
import { fatSegments, loop } from "./lines";
import { GLSL_OUTPUT_COLOR, GLSL_UV_VERTEX } from "./shaders";
import type { SceneState } from "./timeline";
import type { TurnData, TurnEntity } from "./types";

// The board: the captured frame as a plane in the XY plane (normal +Z), with the
// detections raised from it as glowing wireframe volumes, the HUD strip as its own
// layer. One board unit is 1/16 of the frame width.

export const BOARD_W = 16;

// Category colours of apps/agent/src/overlay.py, lifted in lightness so they glow on
// the dark stage.
const UNIT_COLOR = "#ff4f6a";
const CATEGORY_COLOR = new Map<string, string>([
  ["resource", "#3fd27a"],
  ["animal", "#ffae42"],
  ["building", "#5b8dff"],
  ["defense", "#b36bff"],
  ["unit", UNIT_COLOR],
]);

/** A detection category's colour; an unknown category draws as a unit. */
export const categoryColor = (category: string): string => CATEGORY_COLOR.get(category) ?? UNIT_COLOR;

// HUD strip crop of scripts/build_walkthrough_data.py, in frame pixels.
const HUD_STRIP = { x0: 0, y0: 0, x1: 1400, y1: 92 };

export type Board = {
  group: THREE.Group;
  /** Frame size in board units. */
  W: number;
  H: number;
  update: (S: SceneState) => void;
  /** Normalised image point (top-left origin) to board coordinates. */
  toBoard: (nx: number, ny: number) => THREE.Vector2;
  /** Top-left corner of an entity's box on the frame, in board coordinates; it rides the box as it lands. */
  entityTop: (e: TurnEntity, out: THREE.Vector3) => THREE.Vector3;
  /** Number of boxes on the frame at the current landing. */
  landedCount: () => number;
  /** Centre of the top edge of each OCR field, in the HUD layer's coordinates. */
  fieldCenters: Record<string, THREE.Vector3>;
  /** The HUD strip as its own layer, lifting off the frame. */
  hud: THREE.Group;
  hudSize: THREE.Vector2;
};

type FrameUniforms = {
  uMap: { value: THREE.Texture };
  uScan: { value: number };
  uScanOn: { value: number };
  uDim: { value: number };
  uFlash: { value: number };
  uGrid: { value: number };
  uHudHole: { value: number };
  uHudRect: { value: THREE.Vector4 };
};

type HudLayer = {
  group: THREE.Group;
  material: THREE.MeshBasicMaterial;
  fields: LineSegments2;
  fieldCenters: Record<string, THREE.Vector3>;
  size: THREE.Vector2;
};

type ToBoard = (nx: number, ny: number) => THREE.Vector2;

/** The captured frame: scan line, detector grid, the HUD's hole and the dimming behind the loops. */
const createFramePlane = (
  data: TurnData,
  frameTexture: THREE.Texture,
  size: THREE.Vector2,
): { mesh: THREE.Mesh; uniforms: FrameUniforms } => {
  const hudRect = new THREE.Vector4(
    HUD_STRIP.x0 / data.frame.width,
    1 - HUD_STRIP.y1 / data.frame.height,
    HUD_STRIP.x1 / data.frame.width,
    1 - HUD_STRIP.y0 / data.frame.height,
  );
  const uniforms: FrameUniforms = {
    uMap: { value: frameTexture },
    uScan: { value: 1 },
    uScanOn: { value: 0 },
    uDim: { value: 0 },
    uFlash: { value: 0 },
    uGrid: { value: 0 },
    uHudHole: { value: 0 },
    uHudRect: { value: hudRect },
  };
  const mesh = new THREE.Mesh(
    new THREE.PlaneGeometry(size.x, size.y),
    new THREE.ShaderMaterial({
      uniforms,
      vertexShader: GLSL_UV_VERTEX,
      fragmentShader: /* glsl */ `
        uniform sampler2D uMap;
        uniform float uScan, uScanOn, uDim, uFlash, uGrid, uHudHole;
        uniform vec4 uHudRect;
        varying vec2 vUv;
        void main() {
          vec3 c = texture2D(uMap, vUv).rgb;
          // Scan line, top to bottom: rows above it are captured, rows below wait dimmed.
          float edge = 1.0 - uScan;
          float captured = step(edge, vUv.y);
          c *= mix(1.0, mix(0.28, 1.0, captured), uScanOn);
          float line = exp(-pow((vUv.y - edge) * 140.0, 2.0));
          c += vec3(0.45, 0.7, 1.6) * line * uScanOn * 1.6;
          // Detector grid while capturing.
          vec2 cell = abs(fract(vUv * vec2(42.0, 23.2)) - 0.5);
          float grid = 1.0 - smoothstep(0.0, 0.035, 0.5 - max(cell.x, cell.y));
          c += vec3(0.25, 0.45, 1.0) * grid * uGrid * 0.22 * captured;
          // The resource bar leaves a dark hole when it lifts off.
          vec2 inHud = step(uHudRect.xy, vUv) * step(vUv, uHudRect.zw);
          c *= 1.0 - inHud.x * inHud.y * uHudHole * 0.85;
          // Dimmed, cooled frame behind the decision loops.
          float lum = dot(c, vec3(0.299, 0.587, 0.114));
          c = mix(c, vec3(lum) * vec3(0.5, 0.62, 0.95), uDim * 0.75) * (1.0 - uDim * 0.55);
          c += vec3(0.6, 0.75, 1.0) * uFlash * 0.5;
          // Soft edge, so the plane does not end on a hard line.
          vec2 e = smoothstep(vec2(0.0), vec2(0.012, 0.02), vUv) * smoothstep(vec2(0.0), vec2(0.012, 0.02), 1.0 - vUv);
          gl_FragColor = vec4(c * (0.35 + 0.65 * e.x * e.y), 1.0);
          ${GLSL_OUTPUT_COLOR}
        }
      `,
    }),
  );

  return { mesh, uniforms };
};

/** The HUD strip as a plane of its own, with the OCR field rectangles in the overlay's gold. */
const createHudLayer = (data: TurnData, hudTexture: THREE.Texture, toBoard: ToBoard, size: THREE.Vector2): HudLayer => {
  const hudW = ((HUD_STRIP.x1 - HUD_STRIP.x0) / data.frame.width) * size.x;
  const hudH = ((HUD_STRIP.y1 - HUD_STRIP.y0) / data.frame.height) * size.y;
  const hudBase = toBoard(
    (HUD_STRIP.x0 + HUD_STRIP.x1) / 2 / data.frame.width,
    (HUD_STRIP.y0 + HUD_STRIP.y1) / 2 / data.frame.height,
  );
  const group = new THREE.Group();
  const material = new THREE.MeshBasicMaterial({ map: hudTexture, transparent: true, opacity: 0 });

  group.add(new THREE.Mesh(new THREE.PlaneGeometry(hudW, hudH), material));

  const fieldCenters: Record<string, THREE.Vector3> = {};
  const fieldPoints: number[] = [];

  for (const [name, rect] of Object.entries(data.hud.fields)) {
    const [x0 = 0, y0 = 0, x1 = 0, y1 = 0] = rect;
    const a = toBoard(x0, y0).sub(hudBase);
    const b = toBoard(x1, y1).sub(hudBase);

    fieldPoints.push(...loop([a, new THREE.Vector2(b.x, a.y), b, new THREE.Vector2(a.x, b.y)].map((v) => new THREE.Vector3(v.x, v.y, 0.01))));
    fieldCenters[name] = new THREE.Vector3((a.x + b.x) / 2, Math.max(a.y, b.y), 0.05);
  }

  const fields = fatSegments(fieldPoints, { color: new THREE.Color(2.2, 1.75, 0.25), width: 1.75, opacity: 0 });

  group.add(fields);
  group.position.set(hudBase.x, hudBase.y, 0.02);

  return { group, material, fields, fieldCenters, size: new THREE.Vector2(hudW, hudH) };
};

export const createBoard = (data: TurnData, frameTexture: THREE.Texture, hudTexture: THREE.Texture): Board => {
  const W = BOARD_W;
  const H = (W * data.frame.height) / data.frame.width;
  const size = new THREE.Vector2(W, H);
  const group = new THREE.Group();

  const toBoard: ToBoard = (nx, ny) => new THREE.Vector2((nx - 0.5) * W, (0.5 - ny) * H);

  const frame = createFramePlane(data, frameTexture, size);

  group.add(frame.mesh);

  // Detection boxes land on the frame outwards from the Town Center, each with a small overshoot.
  const townCenter = data.entities.find((e) => e.cls === "town_center") ?? data.entities[0];
  const townCenterPoint = toBoard(townCenter.x + townCenter.w / 2, townCenter.y + townCenter.h / 2);
  const maxDist = Math.hypot(W, H) / 2;

  const delayOf = (e: TurnEntity, i: number): number => {
    const c = toBoard(e.x + e.w / 2, e.y + e.h / 2);

    return Math.min(1, (c.distanceTo(townCenterPoint) / maxDist) * 0.9 + (((i * 7919) % 97) / 97) * 0.1);
  };

  const delays = new Map(data.entities.map((e, i) => [e, delayOf(e, i)]));
  const boxes = createFlatBoxes(
    data.entities.map((e) => ({
      x: e.x,
      y: e.y,
      w: e.w,
      h: e.h,
      color: new THREE.Color(categoryColor(e.cat)),
      delay: delays.get(e) ?? 0,
      unit: e.cat === "unit",
    })),
    size,
  );

  group.add(boxes.mesh);

  const landed = (e: TurnEntity): number => {
    const progress = (boxes.uniforms.uLand.value - (delays.get(e) ?? 0) * LAND_DELAY_SPAN) / LAND_DURATION;

    return THREE.MathUtils.clamp(progress, 0, 1);
  };

  const landedCount = (): number => data.entities.filter((e) => landed(e) > 0.5).length;

  const entityTop = (e: TurnEntity, out: THREE.Vector3): THREE.Vector3 => {
    const c = toBoard(e.x, e.y);

    return out.set(c.x, c.y, 0.05 + (1 - landed(e)) * 0.5);
  };

  const hud = createHudLayer(data, hudTexture, toBoard, size);

  group.add(hud.group);

  const update = (S: SceneState): void => {
    frame.uniforms.uScan.value = S.frame.scan;
    frame.uniforms.uScanOn.value = S.frame.scanOn;
    frame.uniforms.uDim.value = S.frame.dim;
    frame.uniforms.uFlash.value = S.frame.flash;
    frame.uniforms.uGrid.value = S.frame.grid;
    frame.uniforms.uHudHole.value = S.frame.hudHole;

    boxes.uniforms.uLand.value = S.boxes.rise;
    boxes.uniforms.uAlpha.value = S.boxes.alpha;
    boxes.uniforms.uFocus.value = S.boxes.track;
    boxes.mesh.visible = S.boxes.alpha > 0.001 && S.boxes.rise > 0.001;

    const lift = S.hud.lift;

    hud.group.position.z = 0.02 + lift * 1.5;
    hud.group.scale.setScalar(1 + lift * 0.35);
    hud.material.opacity = Math.min(1, lift * 4);
    hud.group.visible = lift > 0.001;
    hud.fields.material.opacity = S.hud.fields;
    hud.fields.visible = S.hud.fields > 0.001;
  };

  return { group, W, H, update, toBoard, entityTop, landedCount, fieldCenters: hud.fieldCenters, hud: hud.group, hudSize: hud.size };
};
