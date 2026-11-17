import * as THREE from "three";

import { el, type Labels } from "./labels";

// Act: the chosen action travels from the policy ring back to the glass as a packet.
// A bright core in a soft halo carries its name; behind it a comet tail fades out over
// the last stretch of its path, with a few sparks shed along the way.

const GLSL_COLOR = /* glsl */ `
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
`;

const CORE = new THREE.Color(0.75, 2.5, 1.35);
const TAIL = 0.32; // how much of the path the tail covers
const SPARKS = 56;

function glowTexture() {
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const g = c.getContext("2d")!;
  const r = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  r.addColorStop(0, "rgba(255,255,255,1)");
  r.addColorStop(0.12, "rgba(255,255,255,0.9)");
  r.addColorStop(0.3, "rgba(255,255,255,0.28)");
  r.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = r;
  g.fillRect(0, 0, 128, 128);
  return new THREE.CanvasTexture(c);
}

/**
 * A comet tail: one continuous tube, so nothing overlaps and doubles up. It is
 * brightest along its axis and fades to its edges (facing the camera) and towards its
 * end, TAIL behind the head; it is drawn only up to the head.
 */
function tail(curve: THREE.Curve<THREE.Vector3>, radius: number, strength: number) {
  const SEGMENTS = 240;
  const RADIAL = 10;
  const geometry = new THREE.TubeGeometry(curve, SEGMENTS, radius, RADIAL, false);
  const uniforms = { uHead: { value: 0 }, uStrength: { value: strength }, uFade: { value: 1 }, uColor: { value: CORE } };
  const mesh = new THREE.Mesh(
    geometry,
    new THREE.ShaderMaterial({
      uniforms,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      vertexShader: /* glsl */ `
        varying float vAlong;
        varying float vFacing;
        void main() {
          vAlong = uv.x;
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          vec3 n = normalize(normalMatrix * normal);
          vFacing = abs(dot(n, normalize(-mv.xyz)));
          gl_Position = projectionMatrix * mv;
        }
      `,
      fragmentShader: /* glsl */ `
        uniform float uHead, uStrength, uFade;
        uniform vec3 uColor;
        varying float vAlong;
        varying float vFacing;
        void main() {
          if (vAlong > uHead) discard;
          float k = smoothstep(uHead - ${TAIL.toFixed(2)}, uHead, vAlong);
          float a = k * k * pow(vFacing, 1.6) * uStrength * uFade;
          gl_FragColor = vec4(uColor * a, a);
          ${GLSL_COLOR}
        }
      `,
    }),
  );
  mesh.renderOrder = 9;
  mesh.frustumCulled = false;
  return { mesh, uniforms };
}

export function createPacket(curve: THREE.Curve<THREE.Vector3>, labels: Labels, toWorld: (local: THREE.Vector3, out: THREE.Vector3) => void) {
  const group = new THREE.Group();
  // A wide soft glow around a thin bright core.
  const tails = [tail(curve, 0.32, 0.35), tail(curve, 0.06, 1.4)];
  for (const t of tails) group.add(t.mesh);

  const glow = glowTexture();
  const halo = new THREE.Sprite(
    new THREE.SpriteMaterial({ map: glow, color: CORE, blending: THREE.AdditiveBlending, transparent: true, depthWrite: false, toneMapped: false }),
  );
  const core = new THREE.Sprite(
    new THREE.SpriteMaterial({ map: glow, color: new THREE.Color(2.2, 2.6, 2.2), blending: THREE.AdditiveBlending, transparent: true, depthWrite: false, toneMapped: false }),
  );
  halo.renderOrder = core.renderOrder = 10;
  group.add(halo, core);

  // Sparks: each one leaves the head, drifts outwards and fades, then starts again.
  const sparkPositions = new Float32Array(SPARKS * 3);
  const sparkAlpha = new Float32Array(SPARKS);
  const sparkGeometry = new THREE.BufferGeometry();
  sparkGeometry.setAttribute("position", new THREE.BufferAttribute(sparkPositions, 3).setUsage(THREE.DynamicDrawUsage));
  sparkGeometry.setAttribute("aAlpha", new THREE.BufferAttribute(sparkAlpha, 1).setUsage(THREE.DynamicDrawUsage));
  const sparkUniforms = { uMap: { value: glow }, uSize: { value: 1 } };
  const sparks = new THREE.Points(
    sparkGeometry,
    new THREE.ShaderMaterial({
      uniforms: sparkUniforms,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      vertexShader: /* glsl */ `
        uniform float uSize;
        attribute float aAlpha;
        varying float vAlpha;
        void main() {
          vAlpha = aAlpha;
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = uSize * (0.5 + aAlpha) / -mv.z;
          gl_Position = projectionMatrix * mv;
        }
      `,
      fragmentShader: /* glsl */ `
        uniform sampler2D uMap;
        varying float vAlpha;
        void main() {
          float a = texture2D(uMap, gl_PointCoord).a * vAlpha;
          gl_FragColor = vec4(vec3(0.75, 2.4, 1.3) * a, a);
          ${GLSL_COLOR}
        }
      `,
    }),
  );
  sparks.frustumCulled = false;
  sparks.renderOrder = 10;
  group.add(sparks);
  const seeds = Array.from({ length: SPARKS }, (_, k) => {
    const h = (n: number) => {
      const s = Math.sin((k + 1) * 91.17 + n * 13.3) * 43758.5453;
      return s - Math.floor(s);
    };
    return { phase: h(1), dir: new THREE.Vector3(h(2) - 0.5, h(3) - 0.5, h(4) - 0.5).normalize(), speed: 0.4 + h(5) * 0.8 };
  });

  // The action's name rides with it.
  const head = new THREE.Vector3();
  const label = labels.add(el("wt3d-packet", `<span>act</span>build_house`), {
    at: (out) => toWorld(head, out),
    offset: [18, -14],
    align: [0, 1],
  });

  const point = new THREE.Vector3();
  function update(p: number, time: number, pointScale: number) {
    const on = p > 0.001 && p < 0.999;
    const fade = 1 - THREE.MathUtils.smoothstep(p, 0.97, 1);
    group.visible = p > 0.001 && fade > 0.001;
    if (!group.visible) {
      label.opacity = 0;
      return;
    }
    curve.getPointAt(Math.min(0.999, p), head);

    for (const t of tails) {
      t.uniforms.uHead.value = p;
      t.uniforms.uFade.value = fade;
    }

    const throb = 1 + 0.12 * Math.sin(time * 18);
    halo.position.copy(head);
    core.position.copy(head);
    halo.scale.setScalar(1.5 * throb);
    core.scale.setScalar(0.6);
    halo.material.opacity = 0.5 * fade;
    core.material.opacity = fade;
    halo.visible = core.visible = on;

    seeds.forEach((s, k) => {
      const age = (time * s.speed + s.phase) % 1;
      curve.getPointAt(THREE.MathUtils.clamp(p - age * 0.08, 0, 0.999), point);
      point.addScaledVector(s.dir, age * 0.9);
      sparkPositions.set([point.x, point.y, point.z], k * 3);
      sparkAlpha[k] = on ? (1 - age) * (1 - age) * fade : 0;
    });
    sparkGeometry.attributes.position!.needsUpdate = true;
    sparkGeometry.attributes.aAlpha!.needsUpdate = true;
    sparkUniforms.uSize.value = pointScale * 2.4;

    label.opacity = on ? THREE.MathUtils.smoothstep(p, 0.02, 0.1) * (1 - THREE.MathUtils.smoothstep(p, 0.88, 0.97)) : 0;
  }

  return { group, update };
}
