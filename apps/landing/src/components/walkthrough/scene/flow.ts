import * as THREE from "three";

import type { Board } from "./board";
import type { SceneState } from "./timeline";
import type { TurnData } from "./types";

const GLSL_COLOR = /* glsl */ `
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
`;

// Publish chapter: sparks leave every detection and every HUD field and gather in the
// frozen Perception frame above the board.
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

  const perSource = 5;
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

// Decide chapter: three rings for the three roles. Each spins at its own rate: the
// policy on every frame, the strategist on a slow clock, the executor idle.
export const RING_X = { policy: -2.8, strategist: 0.9, executor: 4.1 };

export function createRings() {
  const group = new THREE.Group();
  const specs = [
    { x: RING_X.policy, color: new THREE.Color(0.6, 1.6, 3.2), speed: 2.6 },
    { x: RING_X.strategist, color: new THREE.Color(2.6, 1.6, 0.5), speed: 0.35 },
    { x: RING_X.executor, color: new THREE.Color(0.5, 0.55, 0.7), speed: 0 },
  ];
  const rings = specs.map((spec) => {
    const ring = new THREE.Group();
    const baseMaterial = new THREE.MeshBasicMaterial({ color: spec.color, transparent: true, opacity: 0, toneMapped: false });
    const base = new THREE.Mesh(new THREE.TorusGeometry(1, 0.012, 8, 128), baseMaterial);
    const arcMaterial = new THREE.MeshBasicMaterial({
      color: spec.color.clone().multiplyScalar(1.6),
      transparent: true,
      opacity: 0,
      toneMapped: false,
    });
    const arc = new THREE.Mesh(new THREE.TorusGeometry(1, 0.035, 8, 64, Math.PI * 0.55), arcMaterial);
    const dot = new THREE.Mesh(new THREE.SphereGeometry(0.075, 16, 12), arcMaterial);
    dot.position.set(1, 0, 0);
    arc.add(dot);
    ring.add(base, arc);
    ring.position.x = spec.x;
    group.add(ring);
    return { ring, arc, baseMaterial, arcMaterial, speed: spec.speed };
  });

  function update(S: SceneState, time: number) {
    const on = S.decide.on;
    group.visible = on > 0.001;
    group.scale.setScalar(0.7 + 0.3 * on);
    rings.forEach((r, i) => {
      r.arc.rotation.z = -time * r.speed;
      // The policy ring flares as it commits to a choice.
      const flare = i === 0 ? 1 + S.decide.pick * 0.6 * (1 - S.decide.pick) * 4 : 1;
      r.baseMaterial.opacity = on * (i === 2 ? 0.35 : 0.7);
      r.arcMaterial.opacity = on * (i === 2 ? 0.25 : 1) * Math.min(flare, 2);
    });
  }

  return { group, update };
}
