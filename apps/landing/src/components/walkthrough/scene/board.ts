import * as THREE from "three";

import type { SceneState } from "./timeline";
import type { TurnData, TurnEntity } from "./types";

// The board: the captured frame as a plane in the XY plane (normal +Z), with the
// detections raised from it as glowing wireframe volumes, the HUD strip as its own
// layer, and the house footprints of the Act chapter. One board unit is 1/16 of the
// frame width.

export const BOARD_W = 16;

// Category colours of apps/agent/src/overlay.py, lifted in lightness so they glow on
// the dark stage.
const CATEGORY_COLOR: Record<string, string> = {
  resource: "#3fd27a",
  animal: "#ffae42",
  building: "#5b8dff",
  defense: "#b36bff",
  unit: "#ff4f6a",
};
const TRACK_COLOR = new THREE.Color("#7cc4ff");

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

  // ---- Detection volumes -------------------------------------------------------------
  const tc = data.entities.find((e) => e.cls === "town_center") ?? data.entities[0]!;
  const tcCenter = toBoard(tc.x + tc.w / 2, tc.y + tc.h / 2);
  const maxDist = Math.hypot(W, H) / 2;

  const heightOf = (e: TurnEntity) => {
    const area = e.w * W * e.h * H;
    const base = e.cat === "unit" || e.cat === "animal" ? 0.32 : Math.sqrt(area) * 0.7;
    return THREE.MathUtils.clamp(base, 0.22, 1.6);
  };
  const delayOf = (e: TurnEntity, i: number) => {
    const c = toBoard(e.x + e.w / 2, e.y + e.h / 2);
    return (c.distanceTo(tcCenter) / maxDist) * 0.38 + ((i * 7919) % 97) / 97 * 0.04;
  };

  // Twelve edges of the unit cube: start corner and axis.
  const EDGES: [number, number, number, number][] = [];
  for (const [y, z] of [[0, 0], [1, 0], [0, 1], [1, 1]]) EDGES.push([0, y!, z!, 0]);
  for (const [x, z] of [[0, 0], [1, 0], [0, 1], [1, 1]]) EDGES.push([x!, 0, z!, 1]);
  for (const [x, y] of [[0, 0], [1, 0], [0, 1], [1, 1]]) EDGES.push([x!, y!, 0, 2]);

  const n = data.entities.length;
  const edgeCount = n * EDGES.length;
  const aStart = new Float32Array(edgeCount * 4);
  const aBox = new Float32Array(edgeCount * 4);
  const aColor = new Float32Array(edgeCount * 3);
  const aMeta = new Float32Array(edgeCount * 4);
  const fillBox = new Float32Array(n * 2 * 4);
  const fillColor = new Float32Array(n * 2 * 3);
  const fillMeta = new Float32Array(n * 2 * 4);
  const color = new THREE.Color();

  data.entities.forEach((e, i) => {
    color.set(CATEGORY_COLOR[e.cat] ?? CATEGORY_COLOR.unit!);
    const meta = [delayOf(e, i), heightOf(e), e.cat === "unit" ? 1 : 0, 0];
    EDGES.forEach((edge, k) => {
      const j = i * EDGES.length + k;
      aStart.set(edge, j * 4);
      aBox.set([e.x, e.y, e.w, e.h], j * 4);
      aColor.set([color.r, color.g, color.b], j * 3);
      aMeta.set(meta, j * 4);
    });
    for (let cap = 0; cap < 2; cap++) {
      const j = i * 2 + cap;
      fillBox.set([e.x, e.y, e.w, e.h], j * 4);
      fillColor.set([color.r, color.g, color.b], j * 3);
      fillMeta.set([meta[0]!, meta[1]!, meta[2]!, cap], j * 4);
    }
  });

  const boxUniforms = {
    uRise: { value: 0 },
    uAlpha: { value: 1 },
    uFill: { value: 0.08 },
    uTrack: { value: 0 },
    uShimmer: { value: 1 },
    uTime: { value: 0 },
    uSize: { value: new THREE.Vector2(W, H) },
    uThick: { value: 0.022 },
    uTrackColor: { value: TRACK_COLOR },
  };

  // Shared GLSL: a box's world rect and its rise at the current uRise.
  const BOX_GLSL = /* glsl */ `
    uniform float uRise, uAlpha, uTrack, uShimmer, uTime, uThick, uFill;
    uniform vec2 uSize;
    uniform vec3 uTrackColor;
    float riseOf(float delay) {
      float r = clamp((uRise * 1.4 - delay) / 1.0, 0.0, 1.0);
      return r * r * (3.0 - 2.0 * r);
    }
    vec3 cornerOf(vec4 box, vec3 c, float height) {
      float x0 = (box.x - 0.5) * uSize.x;
      float x1 = (box.x + box.z - 0.5) * uSize.x;
      float yTop = (0.5 - box.y) * uSize.y;
      float yBot = (0.5 - box.y - box.w) * uSize.y;
      return vec3(mix(x0, x1, c.x), mix(yBot, yTop, c.y), c.z * height);
    }
    vec3 tint(vec3 base, float isUnit, float delay) {
      vec3 col = mix(base, uTrackColor, uTrack * isUnit);
      float pulse = 1.0 + 0.35 * sin(uTime * 2.4 - delay * 16.0) * uShimmer;
      return col * pulse;
    }
    float focus(float isUnit) { return mix(1.0, mix(0.22, 1.0, isUnit), uTrack); }
  `;

  const beam = new THREE.BoxGeometry(1, 1, 1);
  beam.translate(0.5, 0, 0);
  const beamGeo = new THREE.InstancedBufferGeometry();
  beamGeo.index = beam.index;
  beamGeo.setAttribute("position", beam.getAttribute("position"));
  beamGeo.setAttribute("aStart", new THREE.InstancedBufferAttribute(aStart, 4));
  beamGeo.setAttribute("aBox", new THREE.InstancedBufferAttribute(aBox, 4));
  beamGeo.setAttribute("aColor", new THREE.InstancedBufferAttribute(aColor, 3));
  beamGeo.setAttribute("aMeta", new THREE.InstancedBufferAttribute(aMeta, 4));
  beamGeo.instanceCount = edgeCount;

  const beams = new THREE.Mesh(
    beamGeo,
    new THREE.ShaderMaterial({
      uniforms: boxUniforms,
      transparent: true,
      depthWrite: false,
      vertexShader: /* glsl */ `
        ${BOX_GLSL}
        attribute vec4 aStart, aBox, aMeta;
        attribute vec3 aColor;
        varying vec3 vColor;
        varying float vAlpha;
        void main() {
          float r = riseOf(aMeta.x);
          float h = aMeta.y * r;
          vec3 axis = aStart.w < 0.5 ? vec3(1.0, 0.0, 0.0) : (aStart.w < 1.5 ? vec3(0.0, 1.0, 0.0) : vec3(0.0, 0.0, 1.0));
          vec3 u = aStart.w < 0.5 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
          vec3 v = aStart.w < 1.5 ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0);
          vec3 a = cornerOf(aBox, aStart.xyz, h);
          vec3 b = cornerOf(aBox, aStart.xyz + axis, h);
          vec3 p = mix(a, b, position.x) + (u * position.y + v * position.z) * uThick;
          p.z += 0.01;
          vColor = tint(aColor, aMeta.z, aMeta.x) * 2.2;
          vAlpha = uAlpha * smoothstep(0.0, 0.12, r) * focus(aMeta.z);
          gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
        }
      `,
      fragmentShader: /* glsl */ `
        varying vec3 vColor;
        varying float vAlpha;
        void main() {
          gl_FragColor = vec4(vColor, vAlpha);
          ${GLSL_COLOR}
        }
      `,
    }),
  );
  beams.frustumCulled = false;
  beams.renderOrder = 2;
  group.add(beams);

  const quad = new THREE.PlaneGeometry(1, 1);
  quad.translate(0.5, 0.5, 0);
  const fillGeo = new THREE.InstancedBufferGeometry();
  fillGeo.index = quad.index;
  fillGeo.setAttribute("position", quad.getAttribute("position"));
  fillGeo.setAttribute("aBox", new THREE.InstancedBufferAttribute(fillBox, 4));
  fillGeo.setAttribute("aColor", new THREE.InstancedBufferAttribute(fillColor, 3));
  fillGeo.setAttribute("aMeta", new THREE.InstancedBufferAttribute(fillMeta, 4));
  fillGeo.instanceCount = n * 2;

  const fills = new THREE.Mesh(
    fillGeo,
    new THREE.ShaderMaterial({
      uniforms: boxUniforms,
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      vertexShader: /* glsl */ `
        ${BOX_GLSL}
        attribute vec4 aBox, aMeta;
        attribute vec3 aColor;
        varying vec3 vColor;
        varying float vAlpha;
        varying vec2 vUv;
        void main() {
          float r = riseOf(aMeta.x);
          vec3 p = cornerOf(aBox, vec3(position.xy, aMeta.w), aMeta.y * r);
          p.z += 0.008;
          vUv = position.xy;
          vColor = tint(aColor, aMeta.z, aMeta.x);
          vAlpha = uAlpha * uFill * smoothstep(0.0, 0.15, r) * (aMeta.w > 0.5 ? 0.7 : 1.0) * focus(aMeta.z);
          gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
        }
      `,
      fragmentShader: /* glsl */ `
        varying vec3 vColor;
        varying float vAlpha;
        varying vec2 vUv;
        void main() {
          // Brighter towards the edges, like a lit glass cap.
          vec2 d = min(vUv, 1.0 - vUv);
          float edge = 1.0 - smoothstep(0.0, 0.35, min(d.x, d.y));
          gl_FragColor = vec4(vColor * (1.0 + edge), vAlpha * (0.45 + edge));
          ${GLSL_COLOR}
        }
      `,
    }),
  );
  fills.frustumCulled = false;
  fills.renderOrder = 1;
  group.add(fills);

  /** Top centre of an entity's volume at the current rise, in board coordinates. */
  const entityTop = (e: TurnEntity, out: THREE.Vector3) => {
    const c = toBoard(e.x + e.w / 2, e.y + e.h / 2);
    const r = THREE.MathUtils.clamp(boxUniforms.uRise.value * 1.4 - delayOf(e, data.entities.indexOf(e)), 0, 1);
    return out.set(c.x, c.y, heightOf(e) * r * r * (3 - 2 * r) + 0.12);
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
  const fieldLines: THREE.LineSegments[] = [];
  const fieldCenters: Record<string, THREE.Vector3> = {};
  const fieldMaterial = new THREE.LineBasicMaterial({ color: new THREE.Color(2.4, 1.9, 0.2), transparent: true, opacity: 0, toneMapped: false });
  for (const [name, rect] of Object.entries(data.hud.fields)) {
    const [x0 = 0, y0 = 0, x1 = 0, y1 = 0] = rect;
    const a = toBoard(x0, y0).sub(hudBase);
    const b = toBoard(x1, y1).sub(hudBase);
    const corners = [
      [a.x, a.y], [b.x, a.y],
      [b.x, a.y], [b.x, b.y],
      [b.x, b.y], [a.x, b.y],
      [a.x, b.y], [a.x, a.y],
    ];
    const geo = new THREE.BufferGeometry().setFromPoints(corners.map(([x, y]) => new THREE.Vector3(x, y, 0.01)));
    const line = new THREE.LineSegments(geo, fieldMaterial);
    hud.add(line);
    fieldLines.push(line);
    fieldCenters[name] = new THREE.Vector3((a.x + b.x) / 2, Math.max(a.y, b.y) + 0.12, 0.05);
  }
  hud.position.set(hudBase.x, hudBase.y, 0.02);
  group.add(hud);

  // ---- House footprints (Act chapter) --------------------------------------------------
  // A house is 2 tiles on an edge; a Town Center is 4, and its detected width sets the
  // scale, as in executor.py `_footprint`. Candidate spots ring the Town Center; the
  // first two that overlap a building, tree or mine box show the red preview, the first
  // clear one is clicked.
  const px = data.frame.width;
  const py = data.frame.height;
  const tileW = (tc.w * px) / 4;
  const fw = 2 * tileW;
  const fh = fw / 2;
  const blocking = data.entities.filter(
    (e) => e.cat === "building" || e.cat === "defense" || e.cls === "tree" || e.cls.endsWith("_mine"),
  );
  const tcPx = { x: (tc.x + tc.w / 2) * px, y: (tc.y + tc.h / 2) * py };
  const overlaps = (cx: number, cy: number) =>
    blocking.some(
      (b) =>
        cx - fw / 2 < (b.x + b.w) * px && cx + fw / 2 > b.x * px && cy - fh / 2 < (b.y + b.h) * py && cy + fh / 2 > b.y * py,
    );
  const inPlay = (cx: number, cy: number) =>
    cx - fw / 2 > 0.02 * px && cx + fw / 2 < 0.78 * px && cy - fh / 2 > 0.08 * py && cy + fh / 2 < 0.8 * py;
  const red: { x: number; y: number }[] = [];
  let green: { x: number; y: number } | null = null;
  for (const radius of [280, 400, 520, 640]) {
    for (let k = 0; k < 8; k++) {
      const angle = (k / 8) * Math.PI * 2 + 0.3;
      const cx = tcPx.x + Math.cos(angle) * radius;
      const cy = tcPx.y + Math.sin(angle) * radius * 0.6;
      if (!inPlay(cx, cy)) continue;
      if (overlaps(cx, cy)) {
        if (red.length < 2 && !green) red.push({ x: cx, y: cy });
      } else if (!green && red.length === 2) {
        green = { x: cx, y: cy };
      }
    }
  }
  green ??= { x: tcPx.x, y: tcPx.y + 420 };
  const spots = [...red, green].map((s) => toBoard(s.x / px, s.y / py));
  while (spots.length < 3) spots.unshift(spots[0]!.clone());

  const diamond = new THREE.Shape();
  const dw = (fw / px) * W;
  const dh = (fh / py) * H;
  diamond.moveTo(0, dh / 2);
  diamond.lineTo(dw / 2, 0);
  diamond.lineTo(0, -dh / 2);
  diamond.lineTo(-dw / 2, 0);
  diamond.closePath();
  const footprints = spots.map((spot, i) => {
    const ok = i === 2;
    const fill = new THREE.Mesh(
      new THREE.ShapeGeometry(diamond),
      new THREE.MeshBasicMaterial({
        color: ok ? new THREE.Color(0.2, 1.4, 0.45) : new THREE.Color(1.6, 0.18, 0.15),
        transparent: true,
        opacity: 0,
        depthWrite: false,
        toneMapped: false,
      }),
    );
    const outline = new THREE.LineLoop(
      new THREE.BufferGeometry().setFromPoints(diamond.getPoints().map((p) => new THREE.Vector3(p.x, p.y, 0.01))),
      new THREE.LineBasicMaterial({
        color: ok ? new THREE.Color(0.5, 3, 1) : new THREE.Color(3, 0.4, 0.35),
        transparent: true,
        opacity: 0,
        toneMapped: false,
      }),
    );
    const g = new THREE.Group();
    g.add(fill, outline);
    g.position.set(spot.x, spot.y, 0.03);
    group.add(g);
    return { group: g, fill, outline };
  });

  const ripple = new THREE.Mesh(
    new THREE.RingGeometry(0.92, 1, 64),
    new THREE.MeshBasicMaterial({ color: new THREE.Color(0.6, 2.6, 1.2), transparent: true, opacity: 0, toneMapped: false }),
  );
  ripple.position.set(spots[2]!.x, spots[2]!.y, 0.05);
  group.add(ripple);

  function update(S: SceneState, time: number) {
    frameUniforms.uScan.value = S.frame.scan;
    frameUniforms.uScanOn.value = S.frame.scanOn;
    frameUniforms.uDim.value = S.frame.dim;
    frameUniforms.uFlash.value = S.frame.flash;
    frameUniforms.uGrid.value = S.frame.grid;
    frameUniforms.uHudHole.value = S.frame.hudHole;

    boxUniforms.uRise.value = S.boxes.rise;
    boxUniforms.uAlpha.value = S.boxes.alpha;
    boxUniforms.uFill.value = S.boxes.fill;
    boxUniforms.uTrack.value = S.boxes.track;
    boxUniforms.uShimmer.value = S.boxes.shimmer;
    boxUniforms.uTime.value = time;
    beams.visible = fills.visible = S.boxes.alpha > 0.001;

    const lift = S.hud.lift;
    hud.position.z = 0.02 + lift * 1.5;
    hud.scale.setScalar(1 + lift * 0.35);
    hudMaterial.opacity = Math.min(1, lift * 4);
    hud.visible = lift > 0.001;
    fieldMaterial.opacity = S.hud.fields;

    footprints.forEach((f, i) => {
      const hover = THREE.MathUtils.clamp(1.6 - Math.abs(S.act.cursor - (i + 1)) * 2.4, 0, 1) * S.act.on;
      const a = i === 2 ? Math.max(hover, S.act.placed * S.act.on) : hover;
      (f.fill.material as THREE.MeshBasicMaterial).opacity = a * 0.45;
      (f.outline.material as THREE.LineBasicMaterial).opacity = a;
      f.group.visible = a > 0.001;
    });
    const click = S.act.click;
    ripple.visible = click > 0.001 && click < 0.999 && S.act.on > 0.01;
    ripple.scale.setScalar(0.2 + click * 1.4);
    (ripple.material as THREE.MeshBasicMaterial).opacity = (1 - click) * S.act.on;
  }

  return { group, W, H, update, toBoard, entityTop, fieldCenters, hud, spots, footprintSize: { w: dw, h: dh } };
}
