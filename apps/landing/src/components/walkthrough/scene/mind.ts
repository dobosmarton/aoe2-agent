import * as THREE from "three";
import { CSS3DSprite } from "three/examples/jsm/renderers/CSS3DRenderer.js";

import type { Board } from "./board";
import type { SceneState } from "./timeline";
import type { TurnData } from "./types";

// The mind: the agent's pipeline as objects in space, on the viewer's side of the
// pane. Local units match the pane (16 wide); +x runs along the pipeline, +z towards
// the viewer. The frame slab itself is the board (board.ts), placed here by the scene.

export const MIND = {
  slabStart: new THREE.Vector3(0, 0, 0.15),
  slabPeel: new THREE.Vector3(0, 0, 3),
  slab: new THREE.Vector3(30, 0, 3),
  gate: 18,
  crystal: new THREE.Vector3(43, 0, 5),
  decision: new THREE.Vector3(58, 0, 5),
};

const glowColor = (r: number, g: number, b: number) => new THREE.Color(r, g, b);

export function createMind(data: TurnData, board: Board, frameTexture: THREE.Texture) {
  const group = new THREE.Group();
  group.name = "mind";

  // ---- Detector gate -------------------------------------------------------------------
  // The network as five hairline frames the slab slides through: the letterboxed
  // input, three feature maps (strides 8, 16, 32: each grid coarser than the last) and
  // the head. Corner brackets, ruler ticks and dashed rails between the frames; a frame
  // lights up and runs a scan line while the slab passes it.
  const GATE_H = 10.4;
  const GATE_D = 7.2;
  const GATE_STEP = 1.25;
  const LAYERS = [
    { label: "letterbox · 736 × 1280", cells: 18 },
    { label: "stride 8 · 92 × 160", cells: 13 },
    { label: "stride 16 · 46 × 80", cells: 9 },
    { label: "stride 32 · 23 × 40", cells: 5 },
    { label: "head · 65 classes · NMS 0.5", cells: 0 },
  ];
  const gateX = (i: number) => MIND.gate - 2.5 + i * GATE_STEP;
  const gate = new THREE.Group();
  const lineColor = glowColor(0.55, 0.85, 1.6);
  const segments = (points: number[]) => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(points, 3));
    return g;
  };
  const layers = LAYERS.map((layer, i) => {
    const x = gateX(i);
    const y0 = -GATE_H / 2;
    const y1 = GATE_H / 2;
    const z0 = MIND.slab.z - GATE_D / 2;
    const z1 = MIND.slab.z + GATE_D / 2;
    const frame: number[] = [];
    const seg = (ya: number, za: number, yb: number, zb: number) => frame.push(x, ya, za, x, yb, zb);
    // Outline, then corner brackets drawn just outside it.
    seg(y0, z0, y1, z0);
    seg(y1, z0, y1, z1);
    seg(y1, z1, y0, z1);
    seg(y0, z1, y0, z0);
    const b = 0.6;
    const o = 0.18;
    for (const [cy, cz, sy, sz] of [[y0, z0, 1, 1], [y0, z1, 1, -1], [y1, z0, -1, 1], [y1, z1, -1, -1]] as const) {
      seg(cy - sy * o, cz - sz * o, cy - sy * o + sy * b, cz - sz * o);
      seg(cy - sy * o, cz - sz * o, cy - sy * o, cz - sz * o + sz * b);
    }
    // A ruler on every edge, ticks pointing out of the frame: evenly spaced so the
    // first and last land on the corners, every third one longer.
    const ticks = (length: number) => Math.max(2, Math.round(length / 0.6));
    const tick = (k: number) => (k % 3 === 0 ? 0.32 : 0.18);
    for (let k = 0, n = ticks(GATE_D); k <= n; k++) {
      const z = z0 + (k / n) * GATE_D;
      seg(y1 + 0.08, z, y1 + 0.08 + tick(k), z);
      seg(y0 - 0.08, z, y0 - 0.08 - tick(k), z);
    }
    for (let k = 0, n = ticks(GATE_H); k <= n; k++) {
      const y = y0 + (k / n) * GATE_H;
      seg(y, z0 - 0.08, y, z0 - 0.08 - tick(k));
      seg(y, z1 + 0.08, y, z1 + 0.08 + tick(k));
    }
    const frameMaterial = new THREE.LineBasicMaterial({ color: lineColor, transparent: true, opacity: 0, toneMapped: false });
    const outline = new THREE.LineSegments(segments(frame), frameMaterial);

    // The feature map: a grid, coarser through the network.
    const grid: number[] = [];
    if (layer.cells > 0) {
      const rows = layer.cells;
      const cols = Math.max(2, Math.round((layer.cells * GATE_D) / GATE_H));
      for (let r = 1; r < rows; r++) {
        const y = y0 + (r / rows) * GATE_H;
        grid.push(x, y, z0, x, y, z1);
      }
      for (let c = 1; c < cols; c++) {
        const z = z0 + (c / cols) * GATE_D;
        grid.push(x, y0, z, x, y1, z);
      }
    } else {
      // The head: a column of class slots across the frame instead of a grid.
      const slots = 13;
      for (let k = 0; k < slots; k++) {
        const ya = y0 + 0.5 + k * ((GATE_H - 1) / slots);
        const yb = ya + ((GATE_H - 1) / slots) * 0.62;
        for (const y of [ya, yb]) grid.push(x, y, z0 + 0.45, x, y, z1 - 0.45);
        grid.push(x, ya, z0 + 0.45, x, yb, z0 + 0.45, x, ya, z1 - 0.45, x, yb, z1 - 0.45);
      }
    }
    const gridMaterial = new THREE.LineBasicMaterial({ color: lineColor, transparent: true, opacity: 0, toneMapped: false });
    const lattice = new THREE.LineSegments(segments(grid), gridMaterial);

    // A scan line that sweeps the frame while the slab is inside it.
    const scanMaterial = new THREE.MeshBasicMaterial({ color: glowColor(0.8, 1.6, 3.2), transparent: true, opacity: 0, toneMapped: false, side: THREE.DoubleSide });
    const scan = new THREE.Mesh(new THREE.PlaneGeometry(GATE_D, 0.05), scanMaterial);
    scan.rotation.y = Math.PI / 2;
    scan.position.set(x, 0, MIND.slab.z);

    const label = document.createElement("div");
    label.className = "wt3d-gate-label";
    label.innerHTML = `<span>L${i + 1}</span>${layer.label}`;
    const tag = new CSS3DSprite(label);
    // Labels step up through the network, so neighbours never overlap.
    tag.scale.setScalar(0.026);
    tag.position.set(x, y1 + 0.9 + i * 0.7, z1 - 0.4);

    gate.add(outline, lattice, scan, tag);
    return { x, frameMaterial, gridMaterial, scanMaterial, scan, label };
  });
  // Dashed rails from the first frame's corners to the last one's.
  const rails: number[] = [];
  for (const [y, z] of [[-GATE_H / 2, -GATE_D / 2], [-GATE_H / 2, GATE_D / 2], [GATE_H / 2, -GATE_D / 2], [GATE_H / 2, GATE_D / 2]] as const) {
    rails.push(gateX(0) - 0.8, y, MIND.slab.z + z, gateX(LAYERS.length - 1) + 0.8, y, MIND.slab.z + z);
  }
  const railGeometry = segments(rails);
  const railLines = new THREE.LineSegments(
    railGeometry,
    new THREE.LineDashedMaterial({ color: lineColor, dashSize: 0.18, gapSize: 0.14, transparent: true, opacity: 0, toneMapped: false }),
  );
  railLines.computeLineDistances();
  gate.add(railLines);
  group.add(gate);

  // ---- Earlier frames behind the slab, joined by track tubes -------------------------
  const ghosts = new THREE.Group();
  const ghostMaterials: THREE.MeshBasicMaterial[] = [];
  for (let k = 1; k <= 3; k++) {
    const material = new THREE.MeshBasicMaterial({ map: frameTexture, transparent: true, opacity: 0, depthWrite: false });
    ghostMaterials.push(material);
    const plane = new THREE.Mesh(new THREE.PlaneGeometry(board.W, board.H), material);
    plane.position.set(MIND.slab.x - k * 0.6, MIND.slab.y + k * 0.25, MIND.slab.z - k * 3.2);
    ghosts.add(plane);
  }
  group.add(ghosts);

  // Tubes follow villagers back through the earlier frames. The drift between frames
  // is illustrative: no recorded run stores track histories.
  const villagers = data.entities.filter((e) => e.cls === "villager").slice(0, 14);
  const tubeMaterial = new THREE.MeshBasicMaterial({ color: glowColor(0.5, 1.2, 2.8), transparent: true, opacity: 0, toneMapped: false });
  const tubes = villagers.map((e, i) => {
    const c = board.toBoard(e.x + e.w / 2, e.y + e.h / 2);
    const points = [0, 1, 2, 3].map((k) => {
      const wobble = k === 0 ? 0 : Math.sin(i * 1.9 + k * 1.3) * 0.25 * k;
      return new THREE.Vector3(MIND.slab.x - k * 0.6 + c.x + wobble, MIND.slab.y + k * 0.25 + c.y + Math.cos(i + k) * 0.15 * k, MIND.slab.z - k * 3.2 + 0.25);
    });
    const geometry = new THREE.TubeGeometry(new THREE.CatmullRomCurve3(points), 32, 0.045, 6);
    const mesh = new THREE.Mesh(geometry, tubeMaterial);
    group.add(mesh);
    return mesh;
  });

  // ---- Perception crystal and the FramePipe rail -------------------------------------
  const crystal = new THREE.Group();
  const crystalMaterial = new THREE.MeshStandardMaterial({
    color: "#1e3a8a",
    emissive: new THREE.Color("#3b82f6"),
    emissiveIntensity: 1.4,
    roughness: 0.15,
    metalness: 0.2,
    transparent: true,
    opacity: 0,
    flatShading: true,
  });
  const core = new THREE.Mesh(new THREE.OctahedronGeometry(1.3, 0), crystalMaterial);
  const cage = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.IcosahedronGeometry(2.1, 0)),
    new THREE.LineBasicMaterial({ color: glowColor(0.6, 1.3, 3), transparent: true, opacity: 0, toneMapped: false }),
  );
  crystal.add(core, cage);
  crystal.position.copy(MIND.crystal);
  group.add(crystal);

  const railCurve = new THREE.CatmullRomCurve3([
    MIND.slab.clone().add(new THREE.Vector3(board.W / 2 + 0.5, 0, 0)),
    MIND.crystal.clone(),
    MIND.decision.clone().add(new THREE.Vector3(-8, 0, 0)),
  ]);
  const railMaterial = new THREE.MeshBasicMaterial({ color: glowColor(0.35, 0.8, 2), transparent: true, opacity: 0, toneMapped: false });
  const rail = new THREE.Mesh(new THREE.TubeGeometry(railCurve, 64, 0.06, 6), railMaterial);
  group.add(rail);

  // ---- Action pulse: from the policy ring back to the pane ---------------------------
  const pulseCurve = new THREE.CatmullRomCurve3([
    MIND.decision.clone().add(new THREE.Vector3(-3, 0, 0)),
    new THREE.Vector3(MIND.crystal.x, 6, 9),
    new THREE.Vector3(14, 5, 6),
    new THREE.Vector3(0, 0, 0.4),
  ]);
  const pulse = new THREE.Mesh(
    new THREE.SphereGeometry(0.45, 20, 14),
    new THREE.MeshBasicMaterial({ color: glowColor(1.2, 2.6, 1.4), transparent: true, toneMapped: false }),
  );
  const trailGeometry = new THREE.TubeGeometry(pulseCurve, 120, 0.1, 6);
  const trailMaterial = new THREE.MeshBasicMaterial({ color: glowColor(0.6, 2, 1), transparent: true, opacity: 0, toneMapped: false });
  const trail = new THREE.Mesh(trailGeometry, trailMaterial);
  group.add(pulse, trail);
  const trailIndexCount = trailGeometry.index!.count;

  const tmp = new THREE.Vector3();

  function update(S: SceneState, time: number) {
    // Slab path: pane -> peel -> through the gate -> detection station.
    const peel = S.slab.peel;
    const travel = S.slab.travel;
    tmp.lerpVectors(MIND.slabStart, MIND.slabPeel, peel);
    tmp.lerp(MIND.slab, travel);
    board.group.position.copy(tmp);
    board.group.visible = S.slab.on > 0.01;

    // A frame lights up while the slab is inside it and scans it top to bottom.
    gate.visible = S.mind.gate > 0.01;
    const front = tmp.x + board.W / 2;
    const back = tmp.x - board.W / 2;
    layers.forEach((layer, i) => {
      const inside = front > layer.x && back < layer.x;
      const through = THREE.MathUtils.clamp((front - layer.x) / board.W, 0, 1);
      const active = inside ? 1 : 0;
      const g = S.mind.gate;
      layer.frameMaterial.opacity = g * (0.45 + 0.55 * active);
      layer.gridMaterial.opacity = g * (0.1 + 0.4 * active + 0.04 * Math.sin(time * 2 + i));
      layer.scanMaterial.opacity = g * active * 0.9;
      layer.scan.position.y = (0.5 - through) * GATE_H;
      layer.label.style.opacity = String(g * (0.55 + 0.45 * active));
      layer.label.classList.toggle("is-active", inside);
    });
    (railLines.material as THREE.LineDashedMaterial).opacity = S.mind.gate * 0.35;

    ghosts.visible = S.mind.track > 0.01;
    ghostMaterials.forEach((m, i) => (m.opacity = S.mind.track * (0.42 - i * 0.1)));
    tubeMaterial.opacity = S.mind.track;
    tubes.forEach((t, i) => {
      t.visible = S.mind.track > 0.01;
      const grow = THREE.MathUtils.clamp(S.mind.track * 1.6 - i * 0.04, 0, 1);
      t.geometry.setDrawRange(0, Math.floor(t.geometry.index!.count * grow));
    });

    crystal.visible = S.card.on > 0.01;
    crystalMaterial.opacity = S.card.on * 0.9;
    (cage.material as THREE.LineBasicMaterial).opacity = S.card.on;
    crystal.rotation.set(time * 0.3, time * 0.5, 0);
    crystal.scale.setScalar(0.6 + 0.4 * S.card.on);

    rail.visible = S.mind.rail > 0.01;
    railMaterial.opacity = S.mind.rail * (0.7 + 0.3 * Math.sin(time * 4));

    const p = S.mind.pulse;
    pulse.visible = p > 0.001 && p < 0.999;
    pulseCurve.getPointAt(Math.min(0.999, p), pulse.position);
    trail.visible = p > 0.001;
    trailMaterial.opacity = (1 - Math.max(0, p - 0.85) / 0.15) * 0.8;
    trailGeometry.setDrawRange(0, Math.floor(trailIndexCount * p / 6) * 6);
  }

  return { group, update, railCurve };
}
