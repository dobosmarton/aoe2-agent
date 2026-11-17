import * as THREE from "three";

import { CATEGORY_COLOR } from "./board";
import type { TurnData } from "./types";

// Publish: three sheets lift off the frame (the boxes, the track IDs, the HUD readings),
// shrink and stack into one sealed Perception plate: a glass card that shows the box
// layout and the readings, stamped with the capture it came from. The plate then rides
// the FramePipe rail to the decision loops.

const GLSL_COLOR = /* glsl */ `
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
`;

const ACCENT = "#7cc4ff";
const MUTED = "#8a9ab3";
const INK = "#e6edf7";
const GOLD = "#ffd75a";

function canvas(width: number, height: number) {
  const c = document.createElement("canvas");
  c.width = width;
  c.height = height;
  return [c, c.getContext("2d")!] as const;
}

function texture(c: HTMLCanvasElement) {
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  return t;
}

function roundRect(g: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  g.beginPath();
  g.roundRect(x, y, w, h, r);
}

/** A plane that shows a canvas texture, fades and never writes depth. */
function sheet(map: THREE.Texture, w: number, h: number) {
  const uniforms = { uMap: { value: map }, uOpacity: { value: 0 }, uGlow: { value: 1 } };
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
      fragmentShader: /* glsl */ `
        uniform sampler2D uMap;
        uniform float uOpacity, uGlow;
        varying vec2 vUv;
        void main() {
          vec4 c = texture2D(uMap, vUv);
          gl_FragColor = vec4(c.rgb * uGlow, c.a * uOpacity);
          if (gl_FragColor.a < 0.003) discard;
          ${GLSL_COLOR}
        }
      `,
    }),
  );
  mesh.renderOrder = 7;
  return { mesh, uniforms };
}

export function createPublish(data: TurnData, mono: string, board: { W: number; H: number; hudCentre: THREE.Vector3; hudSize: THREE.Vector2 }) {
  const group = new THREE.Group();
  const r = data.hud.readings;
  const units = data.entities.filter((e) => e.cat === "unit");
  const captureId = Number(data.frame.run.split("_").pop()) || 1;

  // ---- The three sheets -------------------------------------------------------------------
  const SW = 1024;
  const SH = Math.round((SW * board.H) / board.W);
  const [boxCanvas, bg] = canvas(SW, SH);
  bg.lineWidth = 3;
  for (const e of data.entities) {
    bg.strokeStyle = CATEGORY_COLOR[e.cat] ?? CATEGORY_COLOR.unit!;
    bg.strokeRect(e.x * SW, e.y * SH, e.w * SW, e.h * SH);
  }
  const [trackCanvas, tg] = canvas(SW, SH);
  tg.font = `600 15px ${mono}`;
  tg.textBaseline = "middle";
  units.forEach((e, i) => {
    const id = String(3 + ((i * 37) % 97));
    const x = (e.x + e.w / 2) * SW;
    const y = e.y * SH - 4;
    const w = tg.measureText(id).width + 10;
    tg.fillStyle = "rgba(4, 10, 24, 0.85)";
    roundRect(tg, x - w / 2, y - 10, w, 20, 4);
    tg.fill();
    tg.strokeStyle = ACCENT;
    tg.lineWidth = 1.5;
    tg.stroke();
    tg.fillStyle = ACCENT;
    tg.textAlign = "center";
    tg.fillText(id, x, y + 1);
  });
  const HW = 1024;
  const HH = Math.round((HW * board.hudSize.y) / board.hudSize.x);
  const [hudCanvas, hg] = canvas(HW, HH);
  hg.font = `700 ${Math.round(HH * 0.42)}px ${mono}`;
  hg.textBaseline = "middle";
  hg.fillStyle = GOLD;
  const values = [r.wood, r.food, r.gold, r.stone, r.population, r.age];
  values.forEach((v, i) => hg.fillText(String(v), 20 + i * (HW / values.length), HH / 2));

  const sheets = [
    { ...sheet(texture(boxCanvas), board.W, board.H), from: new THREE.Vector3(0, 0, 0.06) },
    { ...sheet(texture(trackCanvas), board.W, board.H), from: new THREE.Vector3(0, 0, 0.08) },
    { ...sheet(texture(hudCanvas), board.hudSize.x, board.hudSize.y), from: board.hudCentre.clone().setZ(0.1) },
  ];
  for (const s of sheets) group.add(s.mesh);

  // ---- The Perception plate ----------------------------------------------------------------
  const PW = 1200;
  const PH = 720;
  const [plateCanvas, g] = canvas(PW, PH);
  g.fillStyle = "rgba(5, 10, 24, 0.94)";
  roundRect(g, 4, 4, PW - 8, PH - 8, 28);
  g.fill();
  g.strokeStyle = "rgba(124, 196, 255, 0.9)";
  g.lineWidth = 3;
  g.stroke();
  g.textBaseline = "alphabetic";
  g.font = `600 30px ${mono}`;
  g.fillStyle = ACCENT;
  g.fillText("PERCEPTION", 48, 76);
  g.font = `400 26px ${mono}`;
  g.fillStyle = MUTED;
  g.textAlign = "right";
  g.fillText(`capture ${captureId} · frozen`, PW - 48, 76);
  g.textAlign = "left";
  // The box layout, as a small map.
  const mx = 48;
  const my = 112;
  const mw = 520;
  const mh = Math.round((mw * board.H) / board.W);
  g.fillStyle = "rgba(12, 22, 44, 1)";
  roundRect(g, mx, my, mw, mh, 10);
  g.fill();
  g.lineWidth = 2;
  for (const e of data.entities) {
    g.strokeStyle = CATEGORY_COLOR[e.cat] ?? CATEGORY_COLOR.unit!;
    g.strokeRect(mx + e.x * mw, my + e.y * mh, Math.max(3, e.w * mw), Math.max(3, e.h * mh));
  }
  // The readings.
  const rows: [string, string][] = [
    ["entities", `${data.entities.length} boxes`],
    ["tracked", `${units.length} villagers`],
    ["wood", String(r.wood)],
    ["food", String(r.food)],
    ["gold · stone", `${r.gold} · ${r.stone}`],
    ["population", String(r.population)],
    ["age", String(r.age)],
    ["alarm", "none"],
  ];
  rows.forEach(([k, v], i) => {
    const y = my + 30 + i * 46;
    g.font = `400 26px ${mono}`;
    g.fillStyle = MUTED;
    g.fillText(k, 620, y);
    g.font = `600 28px ${mono}`;
    g.fillStyle = INK;
    g.textAlign = "right";
    g.fillText(v, PW - 48, y);
    g.textAlign = "left";
  });
  g.font = `400 24px ${mono}`;
  g.fillStyle = MUTED;
  g.fillText("FramePipe → act loop · deliberate loop", 48, PH - 52);

  const PLATE_W = 7.2;
  const plate = sheet(texture(plateCanvas), PLATE_W, (PLATE_W * PH) / PW);
  plate.mesh.renderOrder = 8;
  group.add(plate.mesh);

  const tmp = new THREE.Vector3();
  const lifted = new THREE.Vector3();
  const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
  const backOut = (t: number) => 1 + 2.7 * Math.pow(t - 1, 3) + 1.7 * Math.pow(t - 1, 2);

  /**
   * `board` is the frame's centre, `plateAt` where the plate seals; `lift` runs the
   * sheets 0..1, `seal` shows the plate, `dock`/`dockAt` carry it along the rail.
   */
  function update(
    s: { lift: number; seal: number; travel: number; on: number },
    at: { board: THREE.Vector3; plate: THREE.Vector3; rail: THREE.Curve<THREE.Vector3> },
  ) {
    group.visible = s.on > 0.002 && (s.lift > 0.001 || s.seal > 0.001);
    sheets.forEach((sh, i) => {
      const t = THREE.MathUtils.clamp(s.lift * 1.5 - i * 0.22, 0, 1);
      const e = ease(t);
      const start = tmp.copy(sh.from).add(at.board);
      // Lift straight off the frame, then fly to the plate.
      lifted.copy(start).add(new THREE.Vector3(0, 0, 1.6 + i * 0.5));
      const p = sh.mesh.position;
      if (e < 0.3) p.lerpVectors(start, lifted, e / 0.3);
      else p.lerpVectors(lifted, at.plate, (e - 0.3) / 0.7);
      const target = i < 2 ? (PLATE_W * 0.45) / board.W : (PLATE_W * 0.4) / board.hudSize.x;
      sh.mesh.scale.setScalar(THREE.MathUtils.lerp(1, target, Math.max(0, (e - 0.25) / 0.75)));
      sh.uniforms.uOpacity.value = s.on * THREE.MathUtils.smoothstep(t, 0, 0.12) * (1 - THREE.MathUtils.smoothstep(t, 0.82, 1));
      sh.uniforms.uGlow.value = 1.2 + 0.6 * Math.sin(t * Math.PI);
    });

    const seal = THREE.MathUtils.clamp(s.seal, 0, 1);
    const travel = ease(THREE.MathUtils.clamp(s.travel, 0, 1));
    if (travel > 0) at.rail.getPointAt(travel, plate.mesh.position);
    else plate.mesh.position.copy(at.plate);
    plate.mesh.scale.setScalar((0.82 + 0.18 * backOut(seal)) * (1 - 0.3 * travel));
    plate.uniforms.uOpacity.value = s.on * THREE.MathUtils.smoothstep(seal, 0, 0.4);
    // A flash on the rim as it seals.
    plate.uniforms.uGlow.value = 1 + 0.8 * Math.max(0, 1 - Math.abs(seal - 0.5) * 3);
    plate.mesh.visible = seal > 0.001;
  }

  return { group, update, plate: plate.mesh };
}
