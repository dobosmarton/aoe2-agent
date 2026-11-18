import * as THREE from "three";

import { categoryColor } from "./board";
import { ACCENT, canvasTexture, createCanvas, createSheet, GOLD, INK, MUTED, roundRectPath, type Sheet } from "./card";
import { easeInOutCubic, easeOutBack } from "./easing";
import type { TurnData, TurnEntity } from "./types";

// Publish: three sheets lift off the frame (the boxes, the track IDs, the HUD readings),
// shrink and stack into one sealed Perception plate: a glass card that shows the box
// layout and the readings, stamped with the capture it came from. The plate then rides
// the FramePipe rail to the decision loops.

/** Texture width of the three sheets, px. */
const SHEET_PX = 1024;
const PLATE_PX = { w: 1200, h: 720 };
const PLATE_W = 7.2;
/** How far each sheet lifts straight off the frame before it flies to the plate. */
const LIFT = 1.6;
const LIFT_STEP = 0.5;
/** Each sheet starts this much later than the one before (in `lift` units). */
const SHEET_STAGGER = 0.22;

export type PublishBoard = { W: number; H: number; hudCentre: THREE.Vector3; hudSize: THREE.Vector2 };

export type PublishState = {
  /** Runs the three sheets, 0..1. */
  lift: number;
  /** Shows the plate. */
  seal: number;
  /** Carries the plate along the rail. */
  travel: number;
  on: number;
};

export type PublishAnchors = {
  /** The frame's centre. */
  board: THREE.Vector3;
  /** Where the plate seals. */
  plate: THREE.Vector3;
  rail: THREE.Curve<THREE.Vector3>;
};

export type Publish = {
  group: THREE.Group;
  update: (s: PublishState, at: PublishAnchors) => void;
  plate: THREE.Mesh;
};

type FlyingSheet = Sheet & { from: THREE.Vector3; targetScale: number };

const drawBoxSheet = (entities: readonly TurnEntity[], w: number, h: number): HTMLCanvasElement => {
  const [c, g] = createCanvas(w, h);
  g.lineWidth = 3;
  for (const e of entities) {
    g.strokeStyle = categoryColor(e.cat);
    g.strokeRect(e.x * w, e.y * h, e.w * w, e.h * h);
  }
  return c;
};

const drawTrackSheet = (units: readonly TurnEntity[], w: number, h: number, mono: string): HTMLCanvasElement => {
  const [c, g] = createCanvas(w, h);
  g.font = `600 15px ${mono}`;
  g.textBaseline = "middle";
  units.forEach((e, i) => {
    const id = String(3 + ((i * 37) % 97));
    const x = (e.x + e.w / 2) * w;
    const y = e.y * h - 4;
    const tagW = g.measureText(id).width + 10;
    g.fillStyle = "rgba(4, 10, 24, 0.85)";
    roundRectPath(g, x - tagW / 2, y - 10, tagW, 20, 4);
    g.fill();
    g.strokeStyle = ACCENT;
    g.lineWidth = 1.5;
    g.stroke();
    g.fillStyle = ACCENT;
    g.textAlign = "center";
    g.fillText(id, x, y + 1);
  });
  return c;
};

const drawHudSheet = (values: readonly (string | number)[], w: number, h: number, mono: string): HTMLCanvasElement => {
  const [c, g] = createCanvas(w, h);
  g.font = `700 ${Math.round(h * 0.42)}px ${mono}`;
  g.textBaseline = "middle";
  g.fillStyle = GOLD;
  values.forEach((v, i) => g.fillText(String(v), 20 + i * (w / values.length), h / 2));
  return c;
};

const drawPlate = (data: TurnData, units: readonly TurnEntity[], board: PublishBoard, mono: string): HTMLCanvasElement => {
  const { w: PW, h: PH } = PLATE_PX;
  const r = data.hud.readings;
  const captureId = Number(data.frame.run.split("_").pop()) || 1;
  const [c, g] = createCanvas(PW, PH);
  g.fillStyle = "rgba(5, 10, 24, 0.94)";
  roundRectPath(g, 4, 4, PW - 8, PH - 8, 28);
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
  roundRectPath(g, mx, my, mw, mh, 10);
  g.fill();
  g.lineWidth = 2;
  for (const e of data.entities) {
    g.strokeStyle = categoryColor(e.cat);
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
  return c;
};

export const createPublish = (data: TurnData, mono: string, board: PublishBoard): Publish => {
  const group = new THREE.Group();
  const r = data.hud.readings;
  const units = data.entities.filter((e) => e.cat === "unit");

  // ---- The three sheets -------------------------------------------------------------------
  const sheetH = Math.round((SHEET_PX * board.H) / board.W);
  const hudH = Math.round((SHEET_PX * board.hudSize.y) / board.hudSize.x);
  const frameScale = (PLATE_W * 0.45) / board.W;
  const sheets: FlyingSheet[] = [
    {
      ...createSheet(canvasTexture(drawBoxSheet(data.entities, SHEET_PX, sheetH)), board.W, board.H),
      from: new THREE.Vector3(0, 0, 0.06),
      targetScale: frameScale,
    },
    {
      ...createSheet(canvasTexture(drawTrackSheet(units, SHEET_PX, sheetH, mono)), board.W, board.H),
      from: new THREE.Vector3(0, 0, 0.08),
      targetScale: frameScale,
    },
    {
      ...createSheet(
        canvasTexture(drawHudSheet([r.wood, r.food, r.gold, r.stone, r.population, r.age], SHEET_PX, hudH, mono)),
        board.hudSize.x,
        board.hudSize.y,
      ),
      from: board.hudCentre.clone().setZ(0.1),
      targetScale: (PLATE_W * 0.4) / board.hudSize.x,
    },
  ];
  for (const s of sheets) group.add(s.mesh);

  // ---- The Perception plate ----------------------------------------------------------------
  const plate = createSheet(canvasTexture(drawPlate(data, units, board, mono)), PLATE_W, (PLATE_W * PLATE_PX.h) / PLATE_PX.w);
  plate.mesh.renderOrder = 8;
  group.add(plate.mesh);

  const start = new THREE.Vector3();
  const lifted = new THREE.Vector3();

  const updateSheet = (sheet: FlyingSheet, i: number, s: PublishState, at: PublishAnchors): void => {
    const t = THREE.MathUtils.clamp(s.lift * 1.5 - i * SHEET_STAGGER, 0, 1);
    const e = easeInOutCubic(t);
    start.copy(sheet.from).add(at.board);
    // Lift straight off the frame, then fly to the plate.
    lifted.copy(start).setZ(start.z + LIFT + i * LIFT_STEP);
    const p = sheet.mesh.position;
    if (e < 0.3) p.lerpVectors(start, lifted, e / 0.3);
    else p.lerpVectors(lifted, at.plate, (e - 0.3) / 0.7);
    sheet.mesh.scale.setScalar(THREE.MathUtils.lerp(1, sheet.targetScale, Math.max(0, (e - 0.25) / 0.75)));
    sheet.uniforms.uOpacity.value = s.on * THREE.MathUtils.smoothstep(t, 0, 0.12) * (1 - THREE.MathUtils.smoothstep(t, 0.82, 1));
    sheet.uniforms.uGlow.value = 1.2 + 0.6 * Math.sin(t * Math.PI);
  };

  const updatePlate = (s: PublishState, at: PublishAnchors): void => {
    const seal = THREE.MathUtils.clamp(s.seal, 0, 1);
    const travel = easeInOutCubic(s.travel);
    if (travel > 0) at.rail.getPointAt(travel, plate.mesh.position);
    else plate.mesh.position.copy(at.plate);
    plate.mesh.scale.setScalar((0.82 + 0.18 * easeOutBack(seal)) * (1 - 0.3 * travel));
    plate.uniforms.uOpacity.value = s.on * THREE.MathUtils.smoothstep(seal, 0, 0.4);
    // A flash on the rim as it seals.
    plate.uniforms.uGlow.value = 1 + 0.8 * Math.max(0, 1 - Math.abs(seal - 0.5) * 3);
    plate.mesh.visible = seal > 0.001;
  };

  const update = (s: PublishState, at: PublishAnchors): void => {
    group.visible = s.on > 0.002 && (s.lift > 0.001 || s.seal > 0.001);
    sheets.forEach((sheet, i) => updateSheet(sheet, i, s, at));
    updatePlate(s, at);
  };

  return { group, update, plate: plate.mesh };
};
