import * as THREE from "three";

import type { Board } from "./board";
import type { SceneState } from "./timeline";
import type { TurnData } from "./types";

// The mind: the agent's pipeline as objects in space, on the viewer's side of the
// pane. Local units match the pane (16 wide); +x runs along the pipeline, +z towards
// the viewer. The frame slab itself is the board (board.ts), placed here by the scene.

export const MIND = {
  slabStart: new THREE.Vector3(0, 0, 0.15),
  slabPeel: new THREE.Vector3(0, 0, 3),
  slab: new THREE.Vector3(27, 0, 3),
  gate: 15,
  crystal: new THREE.Vector3(40, 0, 5),
  decision: new THREE.Vector3(55, 0, 5),
};

const glowColor = (r: number, g: number, b: number) => new THREE.Color(r, g, b);

export function createMind(data: TurnData, board: Board, frameTexture: THREE.Texture) {
  const group = new THREE.Group();
  group.name = "mind";

  // ---- Detector gate: five glowing frames the slab slides through --------------------
  const gate = new THREE.Group();
  const gateMaterials: THREE.MeshBasicMaterial[] = [];
  for (let i = 0; i < 5; i++) {
    const material = new THREE.MeshBasicMaterial({ color: glowColor(0.5, 1.1, 2.6), transparent: true, opacity: 0, toneMapped: false, side: THREE.DoubleSide });
    gateMaterials.push(material);
    const ring = new THREE.Mesh(new THREE.RingGeometry(6.2, 6.35, 4, 1), material);
    ring.rotation.set(0, Math.PI / 2, Math.PI / 4);
    ring.position.set(MIND.gate - 2 + i, 0, MIND.slab.z);
    ring.scale.set(1, 1.15, 1);
    gate.add(ring);
    const fill = new THREE.Mesh(
      new THREE.PlaneGeometry(8.6, 10.2),
      new THREE.MeshBasicMaterial({ color: "#1d4ed8", transparent: true, opacity: 0, depthWrite: false, side: THREE.DoubleSide }),
    );
    fill.rotation.y = Math.PI / 2;
    fill.position.copy(ring.position);
    gate.add(fill);
  }
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

    // Gate frames light up as the slab's leading edge passes them.
    const lead = tmp.x + board.W / 2;
    gate.visible = S.mind.gate > 0.01;
    gateMaterials.forEach((m, i) => {
      const x = MIND.gate - 2 + i;
      const hit = Math.max(0, 1 - Math.abs(lead - x - board.W * 0.5) / (board.W * 0.6));
      m.opacity = S.mind.gate * (0.35 + 0.65 * hit + 0.15 * Math.sin(time * 3 + i));
      const fill = gate.children[i * 2 + 1] as THREE.Mesh;
      (fill.material as THREE.MeshBasicMaterial).opacity = S.mind.gate * 0.06 * (1 + hit * 2);
    });

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
