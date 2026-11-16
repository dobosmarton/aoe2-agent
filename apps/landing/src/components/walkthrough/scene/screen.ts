import * as THREE from "three";

// The glass pane between the world and the mind: everything the agent gets is this
// flat picture. It first shows the game camera's render of the village, then fades to
// the real recorded frame. Local units: the pane is 16 wide, centred on the origin,
// facing +z (towards the mind and the viewer).

const GLSL_COLOR = /* glsl */ `
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
`;

export const PANE_W = 16;

export function createScreen(real: THREE.Texture, rendered: THREE.Texture, aspect: number) {
  const PANE_H = PANE_W / aspect;
  const group = new THREE.Group();
  group.name = "screen";

  const uniforms = {
    uReal: { value: real },
    uRendered: { value: rendered },
    uMix: { value: 0 },
    uOpacity: { value: 0 },
    uFlash: { value: 0 },
    uTime: { value: 0 },
  };
  const pane = new THREE.Mesh(
    new THREE.PlaneGeometry(PANE_W, PANE_H),
    new THREE.ShaderMaterial({
      uniforms,
      transparent: true,
      depthWrite: false,
      vertexShader: /* glsl */ `
        varying vec2 vUv;
        void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }
      `,
      fragmentShader: /* glsl */ `
        uniform sampler2D uReal, uRendered;
        uniform float uMix, uOpacity, uFlash, uTime;
        varying vec2 vUv;
        void main() {
          vec3 a = texture2D(uRendered, vUv).rgb;
          vec3 b = texture2D(uReal, vUv).rgb;
          // The real frame wipes in from the top, row by row, like a capture.
          float wipe = smoothstep(uMix * 1.2 - 0.1, uMix * 1.2 - 0.02, 1.0 - vUv.y);
          vec3 c = mix(b, a, wipe);
          float edge = exp(-pow((1.0 - vUv.y - (uMix * 1.2 - 0.06)) * 60.0, 2.0)) * step(0.001, uMix) * step(uMix, 0.999);
          c += vec3(0.4, 0.7, 1.6) * edge;
          // Glass: a faint diagonal sheen.
          float sheen = smoothstep(0.0, 0.02, 0.02 - abs(fract(vUv.x * 0.6 + vUv.y * 0.4 - uTime * 0.03) - 0.5) * 0.04);
          c += vec3(0.08) * sheen;
          c += vec3(0.7, 0.85, 1.0) * uFlash;
          gl_FragColor = vec4(c, uOpacity);
          ${GLSL_COLOR}
        }
      `,
    }),
  );
  pane.renderOrder = 5;
  group.add(pane);

  // A thin dark bezel with glowing inner edge.
  const bezel = new THREE.Group();
  const bezelMaterial = new THREE.MeshStandardMaterial({ color: "#0d1526", roughness: 0.4, metalness: 0.6, transparent: true });
  const glowMaterial = new THREE.MeshBasicMaterial({ color: new THREE.Color(0.5, 0.9, 2.2), transparent: true, toneMapped: false });
  const T = 0.22;
  for (const [w, h, x, y] of [
    [PANE_W + T * 2, T, 0, PANE_H / 2 + T / 2],
    [PANE_W + T * 2, T, 0, -PANE_H / 2 - T / 2],
    [T, PANE_H, -PANE_W / 2 - T / 2, 0],
    [T, PANE_H, PANE_W / 2 + T / 2, 0],
  ] as const) {
    const bar = new THREE.Mesh(new THREE.BoxGeometry(w, h, 0.3), bezelMaterial);
    bar.position.set(x, y, -0.1);
    bezel.add(bar);
    const glow = new THREE.Mesh(new THREE.BoxGeometry(Math.max(w - T * 1.4, T * 0.15), Math.max(h - T * 1.4, T * 0.15), 0.32), glowMaterial);
    glow.scale.set(w > h ? 1 : 0.18, w > h ? 0.18 : 1, 1);
    glow.position.set(x + (w > h ? 0 : x > 0 ? -T * 0.4 : T * 0.4), y + (w > h ? (y > 0 ? -T * 0.4 : T * 0.4) : 0), -0.1);
    bezel.add(glow);
  }
  group.add(bezel);

  function update(s: { on: number; mix: number; flash: number }, time: number) {
    uniforms.uOpacity.value = s.on;
    uniforms.uMix.value = s.mix;
    uniforms.uFlash.value = s.flash;
    uniforms.uTime.value = time;
    group.visible = s.on > 0.002;
    bezelMaterial.opacity = Math.min(1, s.on * 1.5);
    glowMaterial.opacity = Math.min(1, s.on * 1.5);
  }

  return { group, update, width: PANE_W, height: PANE_H };
}
