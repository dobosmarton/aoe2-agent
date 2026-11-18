import * as THREE from "three";

import { GLSL_OUTPUT_COLOR, GLSL_UV_VERTEX } from "./shaders";

// Glass cards drawn on a 2D canvas and shown on a plane in the scene: the Perception
// plate, the sheets that seal into it and the action card.

export const ACCENT = "#7cc4ff";
export const MUTED = "#8a9ab3";
export const INK = "#e6edf7";
export const GOLD = "#ffd75a";

export type CanvasPair = readonly [HTMLCanvasElement, CanvasRenderingContext2D];

export const createCanvas = (width: number, height: number): CanvasPair => {
  const c = document.createElement("canvas");

  c.width = width;
  c.height = height;

  const g = c.getContext("2d");

  if (!g) {
    throw new Error("2D canvas is not available");
  }

  return [c, g];
};

export const canvasTexture = (c: HTMLCanvasElement): THREE.CanvasTexture => {
  const t = new THREE.CanvasTexture(c);

  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;

  return t;
};

/** Starts a rounded-rectangle path; fill or stroke it after. */
export const roundRectPath = (g: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number): void => {
  g.beginPath();
  g.roundRect(x, y, w, h, r);
};

export type Sheet = {
  mesh: THREE.Mesh<THREE.PlaneGeometry, THREE.ShaderMaterial>;
  uniforms: { uMap: { value: THREE.Texture }; uOpacity: { value: number }; uGlow: { value: number } };
};

/** A plane that shows a canvas texture, fades and never writes depth. */
export const createSheet = (map: THREE.Texture, w: number, h: number): Sheet => {
  const uniforms = { uMap: { value: map }, uOpacity: { value: 0 }, uGlow: { value: 1 } };
  const mesh = new THREE.Mesh(
    new THREE.PlaneGeometry(w, h),
    new THREE.ShaderMaterial({
      uniforms,
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      vertexShader: GLSL_UV_VERTEX,
      fragmentShader: /* glsl */ `
        uniform sampler2D uMap;
        uniform float uOpacity, uGlow;
        varying vec2 vUv;
        void main() {
          vec4 c = texture2D(uMap, vUv);
          gl_FragColor = vec4(c.rgb * uGlow, c.a * uOpacity);
          if (gl_FragColor.a < 0.003) discard;
          ${GLSL_OUTPUT_COLOR}
        }
      `,
    }),
  );

  mesh.renderOrder = 7;

  return { mesh, uniforms };
};
