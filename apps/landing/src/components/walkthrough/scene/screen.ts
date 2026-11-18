import * as THREE from "three";

import { easeOutCubic } from "./easing";
import { brackets, fatSegments, lineMaterial } from "./lines";
import { GLSL_OUTPUT_COLOR, GLSL_UV_VERTEX } from "./shaders";

// The glass pane between the world and the mind: everything the agent gets is this
// flat picture. It first shows the game camera's render of the village, then the real
// recorded frame wipes in row by row, like a capture. Viewfinder brackets close on it
// as the shot is taken. Local units: the pane is 16 wide, centred on the origin,
// facing +z (towards the mind and the viewer).

export const PANE_W = 16;

/** Cap on the exposure pulse when the shot is taken: never a full white flash. */
const MAX_FLASH = 0.16;

export type ScreenState = {
  on: number;
  /** Wipes the real frame in over the rendered one. */
  mix: number;
  flash: number;
  /** Closes the viewfinder brackets on the pane. */
  finder: number;
  /** The action arriving at the glass. */
  impact: number;
};

export type Screen = {
  group: THREE.Group;
  update: (s: ScreenState, time: number) => void;
  height: number;
};

export const createScreen = (real: THREE.Texture, rendered: THREE.Texture, aspect: number): Screen => {
  const paneH = PANE_W / aspect;
  const group = new THREE.Group();

  group.name = "screen";

  const uniforms = {
    uReal: { value: real },
    uRendered: { value: rendered },
    uMix: { value: 0 },
    uOpacity: { value: 0 },
    uFlash: { value: 0 },
    uImpact: { value: 0 },
    uTime: { value: 0 },
    uSize: { value: new THREE.Vector2(PANE_W, paneH) },
  };
  const pane = new THREE.Mesh(
    new THREE.PlaneGeometry(PANE_W, paneH),
    new THREE.ShaderMaterial({
      uniforms,
      transparent: true,
      depthWrite: false,
      vertexShader: GLSL_UV_VERTEX,
      fragmentShader: /* glsl */ `
        uniform sampler2D uReal, uRendered;
        uniform float uMix, uOpacity, uFlash, uImpact, uTime;
        uniform vec2 uSize;
        varying vec2 vUv;
        void main() {
          vec3 a = texture2D(uRendered, vUv).rgb;
          vec3 b = texture2D(uReal, vUv).rgb;
          // The real frame wipes in from the top, row by row, like a capture.
          float front = uMix * 1.1 - 0.05;
          float wipe = smoothstep(front - 0.015, front, 1.0 - vUv.y);
          vec3 c = mix(b, a, wipe);
          float scan = exp(-pow((1.0 - vUv.y - front) * 90.0, 2.0)) * step(0.001, uMix) * step(uMix, 0.999);
          c += vec3(0.35, 0.65, 1.5) * scan;
          // Rounded glass: corner radius 0.35 units, a lit rim one pixel wide.
          vec2 p = (vUv - 0.5) * uSize;
          vec2 q = abs(p) - (uSize * 0.5 - 0.35);
          float d = length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - 0.35;
          float px = fwidth(d);
          float inside = 1.0 - smoothstep(-px, 0.0, d);
          float rim = 1.0 - smoothstep(0.0, 1.5 * px, abs(d + px));
          // A faint sheen drifting across the glass.
          float band = abs(fract(vUv.x * 0.5 + vUv.y * 0.3 - uTime * 0.02) - 0.5);
          c += vec3(0.05) * (1.0 - smoothstep(0.0, 0.03, band));
          c = mix(c, vec3(1.0), uFlash);
          // The action arriving: the rim flares and a faint blue wash fades off the glass.
          float hit = (1.0 - uImpact) * step(0.001, uImpact);
          c += vec3(0.25, 0.45, 1.0) * hit * 0.18;
          rim *= 1.0 + 3.0 * hit;
          c += vec3(0.5, 0.85, 1.9) * rim;
          gl_FragColor = vec4(c, uOpacity * max(inside, rim));
          if (gl_FragColor.a < 0.003) discard;
          ${GLSL_OUTPUT_COLOR}
        }
      `,
    }),
  );

  pane.renderOrder = 5;
  group.add(pane);

  // Viewfinder: corner brackets a little outside the pane, closing in on the shot.
  const finderMaterial = lineMaterial({ color: new THREE.Color(0.7, 1.2, 2.4), width: 2, opacity: 0 });
  const finder = fatSegments(brackets(PANE_W + 1.2, paneH + 1.2, 1.4, 0.05), finderMaterial);

  group.add(finder);

  const update = (s: ScreenState, time: number): void => {
    uniforms.uImpact.value = s.impact;
    uniforms.uOpacity.value = s.on;
    uniforms.uMix.value = s.mix;
    uniforms.uFlash.value = Math.min(MAX_FLASH, s.flash * MAX_FLASH);
    uniforms.uTime.value = time;
    group.visible = s.on > 0.002 || s.finder > 0.002;

    const finderProgress = THREE.MathUtils.clamp(s.finder, 0, 1);

    // In from 14% larger, settling as the shutter fires; then out with the glass.
    finder.scale.setScalar(1.14 - 0.14 * easeOutCubic(finderProgress));
    finderMaterial.opacity = Math.min(finderProgress * 2, 1) * s.on;
  };

  return { group, update, height: paneH };
};
