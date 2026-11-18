import * as THREE from "three";

import { createAction } from "./action";
import type { Board } from "./board";
import { RING_X } from "./flow";
import { el, type Labels } from "./labels";
import { fatSegments, lineMaterial, path } from "./lines";
import { createPublish } from "./publish";
import { GLSL_OUTPUT_COLOR } from "./shaders";
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

const DUST_COUNT = 420;
const RAIL_SEGMENTS = 60;

/** A stable pseudo-random value in 0..1 for dust particle `i`, channel `k`. */
const hash01 = (i: number, k: number): number => {
  const s = Math.sin((i + 1) * 12.9898 + k * 78.233) * 43758.5453;
  return s - Math.floor(s);
};

type PointsUniforms = { uOn: { value: number }; uScale: { value: number } };

type Atmosphere = {
  group: THREE.Group;
  update: (on: number, time: number, scale: number) => void;
};

export type Mind = {
  group: THREE.Group;
  update: (S: SceneState, time: number, pointScale: number) => void;
  /** Where the action card is on its rail at `travel`, in the mind's local coordinates. */
  actionAt: (travel: number, out: THREE.Vector3) => THREE.Vector3;
};

/** Drifting dust in the mind's space. */
const createDust = (): { points: THREE.Points; uniforms: PointsUniforms & { uTime: { value: number } } } => {
  const dust = new Float32Array(DUST_COUNT * 3);
  const seed = new Float32Array(DUST_COUNT);
  for (let i = 0; i < DUST_COUNT; i++) {
    dust.set([-6 + hash01(i, 1) * 80, -7 + hash01(i, 2) * 15, -6 + hash01(i, 3) * 20], i * 3);
    seed[i] = hash01(i, 4);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(dust, 3));
  geometry.setAttribute("aSeed", new THREE.BufferAttribute(seed, 1));
  const uniforms = { uTime: { value: 0 }, uOn: { value: 0 }, uScale: { value: 1 } };
  const points = new THREE.Points(
    geometry,
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
          ${GLSL_OUTPUT_COLOR}
        }
      `,
    }),
  );
  points.frustumCulled = false;
  return { points, uniforms };
};

/** A dotted floor below the pipeline, fading away from its axis. */
const createFloor = (): { points: THREE.Points; uniforms: PointsUniforms } => {
  const floor: number[] = [];
  for (let x = -6; x <= 74; x += 1.25) for (let z = -8; z <= 16; z += 1.25) floor.push(x, -7, z);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(floor, 3));
  const uniforms = { uOn: { value: 0 }, uScale: { value: 1 } };
  const points = new THREE.Points(
    geometry,
    new THREE.ShaderMaterial({
      uniforms,
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
          ${GLSL_OUTPUT_COLOR}
        }
      `,
    }),
  );
  points.frustumCulled = false;
  return { points, uniforms };
};

/** Drifting dust in the mind's space and a dotted floor below it, for depth. */
const createAtmosphere = (): Atmosphere => {
  const group = new THREE.Group();
  const dust = createDust();
  const floor = createFloor();
  group.add(dust.points, floor.points);

  const update = (on: number, time: number, scale: number): void => {
    group.visible = on > 0.002;
    dust.uniforms.uTime.value = time;
    dust.uniforms.uOn.value = on;
    dust.uniforms.uScale.value = scale;
    floor.uniforms.uOn.value = on;
    floor.uniforms.uScale.value = scale;
  };

  return { group, update };
};

/** The FramePipe rail, from where the plate seals to the dock at the decision loops. */
const framePipeCurve = (): THREE.CatmullRomCurve3 =>
  new THREE.CatmullRomCurve3([MIND.plate.clone(), MIND.plate.clone().lerp(DOCK, 0.5).add(new THREE.Vector3(0, 0.8, 0.6)), DOCK.clone()]);

/** The action's return rail, from the policy ring back to the glass, under the pipeline. */
const actionCurve = (): THREE.CatmullRomCurve3 =>
  new THREE.CatmullRomCurve3([
    MIND.decision.clone().add(new THREE.Vector3(RING_X.policy, -1.55, 0)),
    MIND.decision.clone().add(new THREE.Vector3(RING_X.policy - 4, -6.2, 5)),
    new THREE.Vector3(MIND.plate.x - 4, -6.8, 7),
    new THREE.Vector3(MIND.slab.x - 8, -6.4, 6),
    new THREE.Vector3(5, -3.2, 2.5),
    new THREE.Vector3(0, 0, 0.35),
  ]);

export const createMind = (data: TurnData, board: Board, frameTexture: THREE.Texture, labels: Labels, mono: string): Mind => {
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
  const railCurve = framePipeCurve();
  const railPoints = path(railCurve.getSpacedPoints(RAIL_SEGMENTS));
  const railMaterial = lineMaterial({ color: new THREE.Color(0.45, 0.95, 2.2), width: 2, opacity: 0, dashed: true, dashSize: 0.5, gapSize: 0.32 });
  const rail = fatSegments(railPoints, railMaterial);
  group.add(rail);

  const railLabel = labels.add(el("wt3d-rail", "FramePipe →"), {
    at: (out) => group.localToWorld(railCurve.getPointAt(0.5, out)),
    offset: [0, -12],
    align: [0.5, 1],
  });

  // ---- The action: from the policy ring back to the glass, under the pipeline ---------
  const action = createAction(actionCurve(), labels, mono, (local, out) => group.localToWorld(out.copy(local)));
  group.add(action.group);

  const atmosphere = createAtmosphere();
  group.add(atmosphere.group);

  const boardCentre = new THREE.Vector3();

  const update = (S: SceneState, time: number, pointScale: number): void => {
    // Frame path: pane -> peel -> the station.
    boardCentre.lerpVectors(MIND.slabStart, MIND.slabPeel, S.slab.peel);
    boardCentre.lerp(MIND.slab, S.slab.travel);
    board.group.position.copy(boardCentre);
    board.group.visible = S.slab.on > 0.01;

    stack.update(S.stack.on, S.stack.pulse);
    publish.update(
      { lift: S.sheets.lift, seal: S.sheets.seal, travel: S.card.travel, on: S.card.on },
      { board: boardCentre, plate: MIND.plate, rail: railCurve },
    );

    rail.visible = S.mind.rail > 0.01;
    railMaterial.opacity = S.mind.rail * 0.9;
    railMaterial.dashOffset = -time * 1.2;

    railLabel.opacity = S.mind.rail * S.action.on;
    action.update(S.action, time);

    atmosphere.update(S.mind.space, time, pointScale);
  };

  return { group, update, actionAt: action.cardAt };
};
