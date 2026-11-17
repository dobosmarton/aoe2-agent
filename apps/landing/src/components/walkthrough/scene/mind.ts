import * as THREE from "three";

import type { Board } from "./board";
import type { Labels } from "./labels";
import { fatSegments, lineMaterial } from "./lines";
import { createPacket } from "./packet";
import { createPublish } from "./publish";
import { createStack } from "./stack";
import type { SceneState } from "./timeline";
import type { TurnData } from "./types";

// The mind: the agent's pipeline as objects in space, on the viewer's side of the
// pane. Local units match the pane (16 wide); +x runs along the pipeline, +z towards
// the viewer. The frame itself is the board (board.ts), placed here by the scene.

export const MIND = {
  slabStart: new THREE.Vector3(0, 0, 0.15),
  slabPeel: new THREE.Vector3(0, 0, 3),
  /** Where the frame stops: detection, the HUD, publishing. */
  slab: new THREE.Vector3(24, 0, 3),
  /** Where the Perception plate seals. */
  plate: new THREE.Vector3(35.5, 2.2, 7),
  decision: new THREE.Vector3(60, 0, 5),
};

/** Where the plate docks at the decision loops (left of the policy ring). */
export const DOCK = MIND.decision.clone().add(new THREE.Vector3(-8.8, 0.3, 0));

const GLSL_COLOR = /* glsl */ `
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
`;

/** Drifting dust in the mind's space and a dotted floor below it, for depth. */
function createAtmosphere() {
  const group = new THREE.Group();
  const COUNT = 420;
  const dust = new Float32Array(COUNT * 3);
  const seed = new Float32Array(COUNT);
  for (let i = 0; i < COUNT; i++) {
    const h = (k: number) => {
      const s = Math.sin((i + 1) * 12.9898 + k * 78.233) * 43758.5453;
      return s - Math.floor(s);
    };
    dust.set([-6 + h(1) * 80, -7 + h(2) * 15, -6 + h(3) * 20], i * 3);
    seed[i] = h(4);
  }
  const dustGeometry = new THREE.BufferGeometry();
  dustGeometry.setAttribute("position", new THREE.BufferAttribute(dust, 3));
  dustGeometry.setAttribute("aSeed", new THREE.BufferAttribute(seed, 1));
  const uniforms = { uTime: { value: 0 }, uOn: { value: 0 }, uScale: { value: 1 } };
  const points = new THREE.Points(
    dustGeometry,
    new THREE.ShaderMaterial({
      uniforms,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      vertexShader: /* glsl */ `
        uniform float uTime, uScale;
        attribute float aSeed;
        varying float vA;
        void main() {
          vec3 p = position;
          p.y += sin(uTime * 0.15 + aSeed * 40.0) * 0.6;
          p.x += cos(uTime * 0.11 + aSeed * 23.0) * 0.5;
          vec4 mv = modelViewMatrix * vec4(p, 1.0);
          gl_PointSize = uScale * (1.5 + aSeed * 2.5) / -mv.z;
          vA = 0.25 + 0.5 * aSeed;
          gl_Position = projectionMatrix * mv;
        }
      `,
      fragmentShader: /* glsl */ `
        uniform float uOn;
        varying float vA;
        void main() {
          float d = length(gl_PointCoord - 0.5);
          float a = smoothstep(0.5, 0.0, d) * vA * uOn;
          gl_FragColor = vec4(vec3(0.5, 0.7, 1.2) * a, a);
          ${GLSL_COLOR}
        }
      `,
    }),
  );
  points.frustumCulled = false;
  group.add(points);

  // Floor: dots on a grid, fading away from the pipeline's axis.
  const floor: number[] = [];
  for (let x = -6; x <= 74; x += 1.25) for (let z = -8; z <= 16; z += 1.25) floor.push(x, -7, z);
  const floorGeometry = new THREE.BufferGeometry();
  floorGeometry.setAttribute("position", new THREE.Float32BufferAttribute(floor, 3));
  const floorUniforms = { uOn: { value: 0 }, uScale: { value: 1 } };
  const floorPoints = new THREE.Points(
    floorGeometry,
    new THREE.ShaderMaterial({
      uniforms: floorUniforms,
      transparent: true,
      depthWrite: false,
      vertexShader: /* glsl */ `
        uniform float uScale;
        varying float vA;
        void main() {
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = max(1.0, uScale * 1.6 / -mv.z);
          vA = 1.0 - smoothstep(4.0, 12.0, abs(position.z - 4.0));
          gl_Position = projectionMatrix * mv;
        }
      `,
      fragmentShader: /* glsl */ `
        uniform float uOn;
        varying float vA;
        void main() {
          float d = length(gl_PointCoord - 0.5);
          float a = smoothstep(0.5, 0.2, d) * vA * uOn * 0.22;
          gl_FragColor = vec4(vec3(0.45, 0.6, 1.0), a);
          ${GLSL_COLOR}
        }
      `,
    }),
  );
  floorPoints.frustumCulled = false;
  group.add(floorPoints);

  return {
    group,
    update(on: number, time: number, scale: number) {
      group.visible = on > 0.002;
      uniforms.uTime.value = time;
      uniforms.uOn.value = on;
      uniforms.uScale.value = scale;
      floorUniforms.uOn.value = on;
      floorUniforms.uScale.value = scale;
    },
  };
}

export function createMind(data: TurnData, board: Board, frameTexture: THREE.Texture, labels: Labels, mono: string) {
  const group = new THREE.Group();
  group.name = "mind";

  // ---- Detector stack ------------------------------------------------------------------
  const stack = createStack(data, frameTexture, labels, MIND.slab, board.W, board.H);
  group.add(stack.group);

  // ---- Publishing: the sheets and the Perception plate ----------------------------------
  const publish = createPublish(data, mono, {
    W: board.W,
    H: board.H,
    hudCentre: board.hud.position.clone(),
    hudSize: board.hudSize,
  });
  group.add(publish.group);

  // ---- FramePipe rail: from the plate to the decision loops, dashes flowing along -----
  const railCurve = new THREE.CatmullRomCurve3([
    MIND.plate.clone(),
    MIND.plate.clone().lerp(DOCK, 0.5).add(new THREE.Vector3(0, 0.8, 0.6)),
    DOCK.clone(),
  ]);
  const railPoints: number[] = [];
  const samples = railCurve.getSpacedPoints(60);
  for (let i = 0; i + 1 < samples.length; i++) railPoints.push(...samples[i]!.toArray(), ...samples[i + 1]!.toArray());
  const railMaterial = lineMaterial({ color: new THREE.Color(0.45, 0.95, 2.2), width: 2, opacity: 0, dashed: true, dashSize: 0.5, gapSize: 0.32 });
  const rail = fatSegments(railPoints, railMaterial);
  group.add(rail);

  // ---- Action pulse: from the policy ring back through the pane ------------------------
  const pulseCurve = new THREE.CatmullRomCurve3([
    MIND.decision.clone().add(new THREE.Vector3(-3.6, 0, 0)),
    new THREE.Vector3(MIND.plate.x, 7, 9),
    new THREE.Vector3(14, 5, 6),
    new THREE.Vector3(0, 0, 0.4),
  ]);
  const packet = createPacket(pulseCurve, labels, (local, out) => group.localToWorld(out.copy(local)));
  group.add(packet.group);

  const atmosphere = createAtmosphere();
  group.add(atmosphere.group);

  const tmp = new THREE.Vector3();
  const boardCentre = new THREE.Vector3();

  function update(S: SceneState, time: number, pointScale: number) {
    // Frame path: pane -> peel -> the station.
    tmp.lerpVectors(MIND.slabStart, MIND.slabPeel, S.slab.peel);
    tmp.lerp(MIND.slab, S.slab.travel);
    board.group.position.copy(tmp);
    board.group.visible = S.slab.on > 0.01;
    boardCentre.copy(tmp);

    stack.update(S.stack.on, S.stack.pulse);
    publish.update(
      { lift: S.sheets.lift, seal: S.sheets.seal, travel: S.card.travel, on: S.card.on },
      { board: boardCentre, plate: MIND.plate, rail: railCurve },
    );

    rail.visible = S.mind.rail > 0.01;
    railMaterial.opacity = S.mind.rail * 0.9;
    railMaterial.dashOffset = -time * 1.2;

    packet.update(S.mind.pulse, time, pointScale);

    atmosphere.update(S.mind.space, time, pointScale);
  }

  /** The action packet's position in world coordinates. */
  function packetHead(p: number, out: THREE.Vector3) {
    pulseCurve.getPointAt(THREE.MathUtils.clamp(p, 0, 0.999), out);
    return group.localToWorld(out);
  }

  return { group, update, railCurve, plate: publish.plate, packetHead };
}
