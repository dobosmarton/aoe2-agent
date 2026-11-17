import * as THREE from "three";

import { createFlatBoxes } from "./boxes";
import { fatSegments, loop } from "./lines";
import type { SceneState } from "./timeline";
import type { TurnData, TurnEntity } from "./types";

// The board: the captured frame as a plane in the XY plane (normal +Z), with the
// detections raised from it as glowing wireframe volumes, the HUD strip as its own
// layer. One board unit is 1/16 of the frame width.

export const BOARD_W = 16;

export { CATEGORY_COLOR };

// Category colours of apps/agent/src/overlay.py, lifted in lightness so they glow on
// the dark stage.
const CATEGORY_COLOR: Record<string, string> = {
  resource: "#3fd27a",
  animal: "#ffae42",
  building: "#5b8dff",
  defense: "#b36bff",
  unit: "#ff4f6a",
};

// HUD strip crop of scripts/build_walkthrough_data.py, in frame pixels.
const HUD_STRIP = { x0: 0, y0: 0, x1: 1400, y1: 92 };

const GLSL_COLOR = /* glsl */ `
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
`;

export type Board = ReturnType<typeof createBoard>;

export function createBoard(data: TurnData, frameTexture: THREE.Texture, hudTexture: THREE.Texture) {
  const W = BOARD_W;
  const H = (W * data.frame.height) / data.frame.width;
  const group = new THREE.Group();

  // Normalised image rect (top-left origin) to board coordinates.
  const toBoard = (nx: number, ny: number) => new THREE.Vector2((nx - 0.5) * W, (0.5 - ny) * H);

  // ---- Frame plane -------------------------------------------------------------------
  const hudRect = new THREE.Vector4(
    HUD_STRIP.x0 / data.frame.width,
    1 - HUD_STRIP.y1 / data.frame.height,
    HUD_STRIP.x1 / data.frame.width,
    1 - HUD_STRIP.y0 / data.frame.height,
  );
  const frameUniforms = {
    uMap: { value: frameTexture },
    uScan: { value: 1 },
    uScanOn: { value: 0 },
    uDim: { value: 0 },
    uFlash: { value: 0 },
    uGrid: { value: 0 },
    uHudHole: { value: 0 },
    uHudRect: { value: hudRect },
  };
  const frame = new THREE.Mesh(
    new THREE.PlaneGeometry(W, H),
    new THREE.ShaderMaterial({
      uniforms: frameUniforms,
      vertexShader: /* glsl */ `
        varying vec2 vUv;
        void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }
      `,
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
          ${GLSL_COLOR}
        }
      `,
    }),
  );
  group.add(frame);

  // ---- Detection boxes ----------------------------------------------------------------
  // They land on the frame outwards from the Town Center, each with a small overshoot.
  const tc = data.entities.find((e) => e.cls === "town_center") ?? data.entities[0]!;
  const tcCenter = toBoard(tc.x + tc.w / 2, tc.y + tc.h / 2);
  const maxDist = Math.hypot(W, H) / 2;
  const delayOf = (e: TurnEntity, i: number) => {
    const c = toBoard(e.x + e.w / 2, e.y + e.h / 2);
    return Math.min(1, (c.distanceTo(tcCenter) / maxDist) * 0.9 + (((i * 7919) % 97) / 97) * 0.1);
  };
  const boxes = createFlatBoxes(
    data.entities.map((e, i) => ({
      x: e.x,
      y: e.y,
      w: e.w,
      h: e.h,
      color: new THREE.Color(CATEGORY_COLOR[e.cat] ?? CATEGORY_COLOR.unit!),
      delay: delayOf(e, i),
      unit: e.cat === "unit",
    })),
    new THREE.Vector2(W, H),
  );
  group.add(boxes.mesh);
  const landed = (e: TurnEntity) => {
    const t = (boxes.uniforms.uLand.value - delayOf(e, data.entities.indexOf(e)) * 0.7) / 0.3;
    return THREE.MathUtils.clamp(t, 0, 1);
  };
  /** Number of boxes on the frame at the current landing. */
  const landedCount = () => data.entities.filter((e) => landed(e) > 0.5).length;

  /** Top-left corner of an entity's box on the frame, in board coordinates. */
  const entityTop = (e: TurnEntity, out: THREE.Vector3) => {
    const c = toBoard(e.x, e.y);
    return out.set(c.x, c.y, 0.05 + (1 - landed(e)) * 0.5);
  };

  // ---- HUD layer ---------------------------------------------------------------------
  const hudW = ((HUD_STRIP.x1 - HUD_STRIP.x0) / data.frame.width) * W;
  const hudH = ((HUD_STRIP.y1 - HUD_STRIP.y0) / data.frame.height) * H;
  const hudBase = toBoard(
    (HUD_STRIP.x0 + HUD_STRIP.x1) / 2 / data.frame.width,
    (HUD_STRIP.y0 + HUD_STRIP.y1) / 2 / data.frame.height,
  );
  const hud = new THREE.Group();
  const hudMaterial = new THREE.MeshBasicMaterial({ map: hudTexture, transparent: true, opacity: 0 });
  const hudPlane = new THREE.Mesh(new THREE.PlaneGeometry(hudW, hudH), hudMaterial);
  hud.add(hudPlane);

  // OCR field rectangles, in the overlay's gold.
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
  hud.add(fields);
  hud.position.set(hudBase.x, hudBase.y, 0.02);
  group.add(hud);

  function update(S: SceneState, time: number) {
    frameUniforms.uScan.value = S.frame.scan;
    frameUniforms.uScanOn.value = S.frame.scanOn;
    frameUniforms.uDim.value = S.frame.dim;
    frameUniforms.uFlash.value = S.frame.flash;
    frameUniforms.uGrid.value = S.frame.grid;
    frameUniforms.uHudHole.value = S.frame.hudHole;

    boxes.uniforms.uLand.value = S.boxes.rise;
    boxes.uniforms.uAlpha.value = S.boxes.alpha;
    boxes.uniforms.uFocus.value = S.boxes.track;
    boxes.mesh.visible = S.boxes.alpha > 0.001 && S.boxes.rise > 0.001;

    const lift = S.hud.lift;
    hud.position.z = 0.02 + lift * 1.5;
    hud.scale.setScalar(1 + lift * 0.35);
    hudMaterial.opacity = Math.min(1, lift * 4);
    hud.visible = lift > 0.001;
    fields.material.opacity = S.hud.fields;
    fields.visible = S.hud.fields > 0.001;

  }

  return { group, W, H, update, toBoard, entityTop, landedCount, fieldCenters, hud, hudSize: new THREE.Vector2(hudW, hudH) };
}
