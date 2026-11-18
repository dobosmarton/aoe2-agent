import * as THREE from "three";
import type { LineSegments2 } from "three/examples/jsm/lines/LineSegments2.js";

import type { Board } from "./board";
import { fatSegments } from "./lines";
import { GLSL_OUTPUT_COLOR } from "./shaders";
import type { SceneState } from "./timeline";
import type { TurnData } from "./types";

// Publish chapter: a light accent under the sheets. Sparks leave every detection and
// every HUD field and gather where the Perception plate seals.

const SPARKS_PER_SOURCE = 2;

export type Flow = {
  /** The sparks, in board coordinates. */
  object: THREE.Points;
  update: (S: SceneState, pointScale: number) => void;
};

/** Where sparks start: every detection's centre and every HUD field, in board coordinates. */
const sparkSources = (data: TurnData, board: Board): THREE.Vector3[] => [
  ...data.entities.map((e) => {
    const c = board.toBoard(e.x + e.w / 2, e.y + e.h / 2);
    return new THREE.Vector3(c.x, c.y, 0.25);
  }),
  ...Object.values(board.fieldCenters).map((center) => center.clone().add(board.hud.position)),
];

const sparkGeometry = (sources: readonly THREE.Vector3[]): THREE.BufferGeometry => {
  const count = sources.length * SPARKS_PER_SOURCE;
  const aStart = new Float32Array(count * 3);
  const aSeed = new Float32Array(count * 2);
  sources.forEach((s, i) => {
    for (let k = 0; k < SPARKS_PER_SOURCE; k++) {
      const spark = i * SPARKS_PER_SOURCE + k;
      aStart.set([s.x, s.y, s.z], spark * 3);
      aSeed.set([((spark * 7919) % 1000) / 1000, ((spark * 104729) % 1000) / 1000], spark * 2);
    }
  });
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(aStart, 3));
  geometry.setAttribute("aSeed", new THREE.BufferAttribute(aSeed, 2));
  return geometry;
};

export const createFlow = (data: TurnData, board: Board, cardPosition: THREE.Vector3): Flow => {

  const uniforms = {
    uT: { value: 0 },
    uOn: { value: 0 },
    uEnd: { value: cardPosition },
    uScale: { value: 1 },
  };
  const points = new THREE.Points(
    sparkGeometry(sparkSources(data, board)),
    new THREE.ShaderMaterial({
      uniforms,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      vertexShader: /* glsl */ `
        uniform float uT, uScale;
        uniform vec3 uEnd;
        attribute vec2 aSeed;
        varying float vAlpha;
        void main() {
          float t = clamp((uT * 1.5 - aSeed.x * 0.5), 0.0, 1.0);
          float e = t * t * (3.0 - 2.0 * t);
          vec3 mid = mix(position, uEnd, 0.5) + vec3((aSeed.y - 0.5) * 3.0, 0.0, 2.5 + aSeed.y * 1.5);
          vec3 p = mix(mix(position, mid, e), mix(mid, uEnd, e), e);
          vAlpha = smoothstep(0.0, 0.08, t) * (1.0 - smoothstep(0.88, 1.0, t));
          vec4 mv = modelViewMatrix * vec4(p, 1.0);
          gl_PointSize = uScale * (6.0 + aSeed.y * 6.0) / -mv.z;
          gl_Position = projectionMatrix * mv;
        }
      `,
      fragmentShader: /* glsl */ `
        uniform float uOn;
        varying float vAlpha;
        void main() {
          float d = length(gl_PointCoord - 0.5);
          float a = smoothstep(0.5, 0.0, d);
          gl_FragColor = vec4(vec3(0.55, 0.8, 1.8) * 2.0 * a, a * vAlpha * uOn);
          ${GLSL_OUTPUT_COLOR}
        }
      `,
    }),
  );
  points.frustumCulled = false;
  points.renderOrder = 4;

  const update = (S: SceneState, pointScale: number): void => {
    uniforms.uT.value = S.flow.t;
    uniforms.uOn.value = S.flow.on;
    uniforms.uScale.value = pointScale;
    points.visible = S.flow.on > 0.001;
  };

  return { object: points, update };
};

// Decide chapter: three rings for the three roles. A comet runs round each one at the
// rate its role thinks: the policy on every frame, the strategist on a slow clock, the
// executor idle until an alarm. Ticks mark the ring like a dial.
export const RING_X = { policy: -3.6, strategist: 1.3, executor: 6.0 };

const CIRCLE_SEGMENTS = 128;
const TICK_INNER_RADIUS = 1.1;

type RingSpec = {
  x: number;
  color: THREE.Color;
  /** Comet speed, radians per second; 0 is idle. */
  speed: number;
  ticks: number;
  /** Comet arc length, in half-turns. */
  arc: number;
};

type Ring = {
  ring: THREE.Group;
  spin: THREE.Group;
  track: LineSegments2;
  dial: LineSegments2;
  arcMaterial: THREE.ShaderMaterial;
  headMaterial: THREE.MeshBasicMaterial;
  speed: number;
  idle: boolean;
};

export type Rings = {
  group: THREE.Group;
  update: (S: SceneState, time: number) => void;
};

const RING_SPECS: readonly RingSpec[] = [
  { x: RING_X.policy, color: new THREE.Color(0.55, 1.35, 3.0), speed: 2.4, ticks: 36, arc: 0.9 },
  { x: RING_X.strategist, color: new THREE.Color(2.6, 1.55, 0.45), speed: 0.3, ticks: 12, arc: 0.6 },
  { x: RING_X.executor, color: new THREE.Color(0.5, 0.56, 0.7), speed: 0, ticks: 12, arc: 0.6 },
];

/** A unit circle as segments. */
const circleSegments = (): number[] => {
  const circle: number[] = [];
  for (let k = 0; k < CIRCLE_SEGMENTS; k++) {
    const a = (k / CIRCLE_SEGMENTS) * Math.PI * 2;
    const b = ((k + 1) / CIRCLE_SEGMENTS) * Math.PI * 2;
    circle.push(Math.cos(a), Math.sin(a), 0, Math.cos(b), Math.sin(b), 0);
  }
  return circle;
};

/** Dial ticks just outside the unit circle; every quarter turn gets a long one. */
const tickSegments = (count: number): number[] => {
  const ticks: number[] = [];
  for (let k = 0; k < count; k++) {
    const a = (k / count) * Math.PI * 2;
    const long = k % (count / 4) === 0;
    const outer = long ? 1.24 : 1.16;
    ticks.push(Math.cos(a) * TICK_INNER_RADIUS, Math.sin(a) * TICK_INNER_RADIUS, 0, Math.cos(a) * outer, Math.sin(a) * outer, 0);
  }
  return ticks;
};

/** The comet's arc, brightening towards its head. */
const cometArcMaterial = (color: THREE.Color): THREE.ShaderMaterial =>
  new THREE.ShaderMaterial({
    uniforms: { uColor: { value: color }, uOpacity: { value: 0 } },
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    vertexShader: /* glsl */ `
      varying vec2 vUv;
      void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }
    `,
    fragmentShader: /* glsl */ `
      uniform vec3 uColor;
      uniform float uOpacity;
      varying vec2 vUv;
      void main() {
        float tail = pow(vUv.x, 2.2);
        gl_FragColor = vec4(uColor * (0.4 + 1.2 * tail), tail * uOpacity);
      }
    `,
  });

const createRing = (spec: RingSpec): Ring => {
  const ring = new THREE.Group();
  ring.position.x = spec.x;
  // The track: a fine circle, and ticks just outside it.
  const track = fatSegments(circleSegments(), { color: spec.color.clone().multiplyScalar(0.55), width: 1.25, opacity: 0 });
  const dial = fatSegments(tickSegments(spec.ticks), { color: spec.color.clone().multiplyScalar(0.7), width: 1.25, opacity: 0 });
  // The comet: an arc that brightens towards its head, and the head itself.
  const arcAngle = Math.PI * spec.arc;
  const arcMaterial = cometArcMaterial(spec.color);
  const arc = new THREE.Mesh(new THREE.TorusGeometry(1, 0.04, 8, 96, arcAngle), arcMaterial);
  const headMaterial = new THREE.MeshBasicMaterial({ color: spec.color.clone().multiplyScalar(1.4), transparent: true, opacity: 0, toneMapped: false });
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.075, 16, 12), headMaterial);
  const spin = new THREE.Group();
  head.position.set(Math.cos(arcAngle), Math.sin(arcAngle), 0);
  spin.add(arc, head);
  ring.add(track, dial, spin);
  return { ring, spin, track, dial, arcMaterial, headMaterial, speed: spec.speed, idle: spec.speed === 0 };
};

/** Longest step the comets take in one frame, seconds, so a stalled tab does not jump them. */
const MAX_FRAME_STEP = 0.1;

export const createRings = (): Rings => {
  const group = new THREE.Group();
  const rings = RING_SPECS.map((spec) => {
    const ring = createRing(spec);
    group.add(ring.ring);
    return ring;
  });

  let last = 0;
  const update = (S: SceneState, time: number): void => {
    const dt = Math.min(MAX_FRAME_STEP, Math.max(0, time - last));
    last = time;
    const on = S.decide.on;
    group.visible = on > 0.001;
    rings.forEach((r, i) => {
      const appear = THREE.MathUtils.clamp(on * 1.5 - i * 0.2, 0, 1);
      r.ring.scale.setScalar(0.85 + 0.15 * appear);
      // The policy speeds up and flares as it commits to a choice.
      const isPolicy = i === 0;
      const flare = isPolicy ? 1 + 1.5 * S.decide.pick * (1 - S.decide.pick) * 4 : 1;
      r.spin.rotation.z -= dt * r.speed * flare;
      const strength = r.idle ? 0.35 : 1;
      r.track.material.opacity = appear * (r.idle ? 0.35 : 0.8);
      r.dial.material.opacity = appear * (r.idle ? 0.3 : 0.65);
      r.arcMaterial.uniforms.uOpacity.value = appear * strength * (r.idle ? 0.4 : Math.min(1.6, flare));
      r.headMaterial.opacity = appear * strength;
    });
  };

  return { group, update };
};
