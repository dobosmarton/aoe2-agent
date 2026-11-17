import * as THREE from "three";

import type { Board } from "./board";
import { fatSegments } from "./lines";
import type { SceneState } from "./timeline";
import type { TurnData } from "./types";

const GLSL_COLOR = /* glsl */ `
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
`;

// Publish chapter: a light accent under the sheets. Sparks leave every detection and
// every HUD field and gather where the Perception plate seals.
export function createFlow(data: TurnData, board: Board, cardPosition: THREE.Vector3) {
  const sources: THREE.Vector3[] = [];
  const tmp = new THREE.Vector3();
  for (const e of data.entities) {
    const c = board.toBoard(e.x + e.w / 2, e.y + e.h / 2);
    sources.push(tmp.set(c.x, c.y, 0.25).clone());
  }
  for (const center of Object.values(board.fieldCenters)) {
    sources.push(center.clone().add(board.hud.position));
  }

  const perSource = 2;
  const count = sources.length * perSource;
  const aStart = new Float32Array(count * 3);
  const aSeed = new Float32Array(count * 2);
  sources.forEach((s, i) => {
    for (let k = 0; k < perSource; k++) {
      const j = i * perSource + k;
      aStart.set([s.x, s.y, s.z], j * 3);
      aSeed.set([((j * 7919) % 1000) / 1000, ((j * 104729) % 1000) / 1000], j * 2);
    }
  });
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(aStart, 3));
  geo.setAttribute("aSeed", new THREE.BufferAttribute(aSeed, 2));

  const uniforms = {
    uT: { value: 0 },
    uOn: { value: 0 },
    uEnd: { value: cardPosition },
    uScale: { value: 1 },
  };
  const points = new THREE.Points(
    geo,
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
          ${GLSL_COLOR}
        }
      `,
    }),
  );
  points.frustumCulled = false;
  points.renderOrder = 4;

  function update(S: SceneState, pointScale: number) {
    uniforms.uT.value = S.flow.t;
    uniforms.uOn.value = S.flow.on;
    uniforms.uScale.value = pointScale;
    points.visible = S.flow.on > 0.001;
  }

  return { object: points, update };
}

// Decide chapter: three rings for the three roles. A comet runs round each one at the
// rate its role thinks: the policy on every frame, the strategist on a slow clock, the
// executor idle until an alarm. Ticks mark the ring like a dial.
export const RING_X = { policy: -3.6, strategist: 1.3, executor: 6.0 };

export function createRings() {
  const group = new THREE.Group();
  const specs = [
    { x: RING_X.policy, color: new THREE.Color(0.55, 1.35, 3.0), speed: 2.4, ticks: 36 },
    { x: RING_X.strategist, color: new THREE.Color(2.6, 1.55, 0.45), speed: 0.3, ticks: 12 },
    { x: RING_X.executor, color: new THREE.Color(0.5, 0.56, 0.7), speed: 0, ticks: 12 },
  ];
  const rings = specs.map((spec, i) => {
    const ring = new THREE.Group();
    ring.position.x = spec.x;
    // The track: a fine circle, and ticks just outside it.
    const circle: number[] = [];
    const N = 128;
    for (let k = 0; k < N; k++) {
      const a = (k / N) * Math.PI * 2;
      const b = ((k + 1) / N) * Math.PI * 2;
      circle.push(Math.cos(a), Math.sin(a), 0, Math.cos(b), Math.sin(b), 0);
    }
    const track = fatSegments(circle, { color: spec.color.clone().multiplyScalar(0.55), width: 1.25, opacity: 0 });
    const ticks: number[] = [];
    for (let k = 0; k < spec.ticks; k++) {
      const a = (k / spec.ticks) * Math.PI * 2;
      const long = k % (spec.ticks / 4) === 0;
      const r0 = 1.1;
      const r1 = long ? 1.24 : 1.16;
      ticks.push(Math.cos(a) * r0, Math.sin(a) * r0, 0, Math.cos(a) * r1, Math.sin(a) * r1, 0);
    }
    const dial = fatSegments(ticks, { color: spec.color.clone().multiplyScalar(0.7), width: 1.25, opacity: 0 });
    // The comet: an arc that brightens towards its head, and the head itself.
    const arcMaterial = new THREE.ShaderMaterial({
      uniforms: { uColor: { value: spec.color }, uOpacity: { value: 0 } },
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
    const arc = new THREE.Mesh(new THREE.TorusGeometry(1, 0.04, 8, 96, Math.PI * (i === 0 ? 0.9 : 0.6)), arcMaterial);
    const headMaterial = new THREE.MeshBasicMaterial({ color: spec.color.clone().multiplyScalar(1.4), transparent: true, opacity: 0, toneMapped: false });
    const head = new THREE.Mesh(new THREE.SphereGeometry(0.075, 16, 12), headMaterial);
    const spin = new THREE.Group();
    head.position.set(Math.cos(Math.PI * (i === 0 ? 0.9 : 0.6)), Math.sin(Math.PI * (i === 0 ? 0.9 : 0.6)), 0);
    spin.add(arc, head);
    ring.add(track, dial, spin);
    group.add(ring);
    return { ring, spin, track, dial, arcMaterial, headMaterial, speed: spec.speed, idle: spec.speed === 0 };
  });

  let last = 0;
  function update(S: SceneState, time: number) {
    const dt = Math.min(0.1, Math.max(0, time - last));
    last = time;
    const on = S.decide.on;
    group.visible = on > 0.001;
    rings.forEach((r, i) => {
      const appear = THREE.MathUtils.clamp(on * 1.5 - i * 0.2, 0, 1);
      r.ring.scale.setScalar(0.85 + 0.15 * appear);
      // The policy speeds up and flares as it commits to a choice.
      const flare = i === 0 ? 1 + 1.5 * S.decide.pick * (1 - S.decide.pick) * 4 : 1;
      r.spin.rotation.z -= dt * r.speed * flare;
      const strength = r.idle ? 0.35 : 1;
      r.track.material.opacity = appear * (r.idle ? 0.35 : 0.8);
      r.dial.material.opacity = appear * (r.idle ? 0.3 : 0.65);
      r.arcMaterial.uniforms.uOpacity.value = appear * strength * (r.idle ? 0.4 : Math.min(1.6, flare));
      r.headMaterial.opacity = appear * strength;
    });
  }

  return { group, update };
}
