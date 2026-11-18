import * as THREE from "three";

import { GLSL_OUTPUT_COLOR } from "./shaders";

// Detection boxes drawn the way an overlay draws them: one quad per box, its outline
// computed per pixel (crisp and anti-aliased at any zoom), heavier corner brackets and
// a faint fill. Two placements share the fragment shader: flat on the frame (the board)
// and facing the camera around a villager (tracking in the village).

const BOX_FRAGMENT = /* glsl */ `
  uniform float uLine, uBracket, uFill;
  varying vec2 vUv;
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    // Distance to the nearest edge, in screen pixels.
    vec2 px = min(vUv, 1.0 - vUv) / max(fwidth(vUv), vec2(1e-5));
    float edge = min(px.x, px.y);
    float line = 1.0 - smoothstep(uLine - 0.6, uLine + 0.6, edge);
    // Corner brackets: thicker along the first quarter of each side.
    vec2 corner = step(min(vUv, 1.0 - vUv), vec2(0.24));
    float arm = corner.x * corner.y;
    float bracket = arm * (1.0 - smoothstep(uBracket - 0.6, uBracket + 0.6, edge));
    float a = max(max(line * 0.75, bracket), uFill);
    vec3 c = vColor * (1.0 + bracket * 0.6);
    gl_FragColor = vec4(c, a * vAlpha);
    if (gl_FragColor.a < 0.004) discard;
    ${GLSL_OUTPUT_COLOR}
  }
`;

/** Outline and corner-bracket widths in screen pixels; fill as an opacity. */
export type BoxStyle = { line?: number; bracket?: number; fill?: number };

type StyleUniforms = { uLine: { value: number }; uBracket: { value: number }; uFill: { value: number } };

type BoxMesh = THREE.Mesh<THREE.InstancedBufferGeometry, THREE.ShaderMaterial>;

const uniformsFor = ({ line = 1.25, bracket = 2.4, fill = 0.06 }: BoxStyle): StyleUniforms => {
  return { uLine: { value: line }, uBracket: { value: bracket }, uFill: { value: fill } };
};

const quadGeometry = (count: number): THREE.InstancedBufferGeometry => {
  const quad = new THREE.PlaneGeometry(1, 1);
  const geometry = new THREE.InstancedBufferGeometry();
  geometry.index = quad.index;
  geometry.setAttribute("position", quad.getAttribute("position"));
  geometry.setAttribute("uv", quad.getAttribute("uv"));
  geometry.instanceCount = count;
  return geometry;
};

export type FlatBox = {
  /** Normalised frame rect, top-left origin. */
  x: number;
  y: number;
  w: number;
  h: number;
  color: THREE.Color;
  /** 0..1: when it lands, as a fraction of the landing. */
  delay: number;
  unit: boolean;
};

export type FlatBoxUniforms = StyleUniforms & {
  /** The frame's size in board units. */
  uSize: { value: THREE.Vector2 };
  /** Landing progress 0..1. */
  uLand: { value: number };
  uAlpha: { value: number };
  /** 1 keeps units at full strength and steps everything else back. */
  uFocus: { value: number };
  uGlow: { value: number };
};

export type FlatBoxes = { mesh: BoxMesh; uniforms: FlatBoxUniforms };

export type BillboardBoxes = {
  mesh: BoxMesh;
  /** Starts a frame's worth of boxes. */
  begin: () => void;
  /** Adds a box; ignored past capacity or when nearly transparent. */
  set: (centre: THREE.Vector3, w: number, h: number, color: THREE.Color, alpha: number) => void;
  /** Uploads the boxes added since `begin`. */
  end: () => void;
};

/**
 * Boxes lying on a frame of size `size` (board units) in its XY plane. They land: drop
 * from above with a small overshoot, staggered by `delay`.
 */
export const createFlatBoxes = (boxes: readonly FlatBox[], size: THREE.Vector2, style: BoxStyle = {}): FlatBoxes => {
  const geometry = quadGeometry(boxes.length);
  const aRect = new Float32Array(boxes.length * 4);
  const aColor = new Float32Array(boxes.length * 3);
  const aMeta = new Float32Array(boxes.length * 2);
  boxes.forEach((b, i) => {
    aRect.set([b.x, b.y, b.w, b.h], i * 4);
    aColor.set([b.color.r, b.color.g, b.color.b], i * 3);
    aMeta.set([b.delay, b.unit ? 1 : 0], i * 2);
  });
  geometry.setAttribute("aRect", new THREE.InstancedBufferAttribute(aRect, 4));
  geometry.setAttribute("aColor", new THREE.InstancedBufferAttribute(aColor, 3));
  geometry.setAttribute("aMeta", new THREE.InstancedBufferAttribute(aMeta, 2));

  const uniforms: FlatBoxUniforms = {
    ...uniformsFor(style),
    uSize: { value: size },
    uLand: { value: 0 },
    uAlpha: { value: 1 },
    uFocus: { value: 0 },
    uGlow: { value: 1.8 },
  };
  const material = new THREE.ShaderMaterial({
    uniforms,
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    vertexShader: /* glsl */ `
      uniform vec2 uSize;
      uniform float uLand, uAlpha, uFocus, uGlow;
      attribute vec4 aRect;
      attribute vec3 aColor;
      attribute vec2 aMeta;
      varying vec2 vUv;
      varying vec3 vColor;
      varying float vAlpha;
      float backOut(float t) {
        float s = 1.6;
        t -= 1.0;
        return t * t * ((s + 1.0) * t + s) + 1.0;
      }
      void main() {
        float t = clamp((uLand - aMeta.x * 0.7) / 0.3, 0.0, 1.0);
        float s = mix(1.35, 1.0, backOut(t));
        vec2 centre = vec2((aRect.x + aRect.z * 0.5 - 0.5) * uSize.x, (0.5 - aRect.y - aRect.w * 0.5) * uSize.y);
        vec2 dims = vec2(aRect.z * uSize.x, aRect.w * uSize.y) * s;
        vec3 p = vec3(centre + position.xy * dims, 0.02 + (1.0 - t) * 0.5);
        vUv = uv;
        // In the Track chapter units keep full strength and everything else steps back.
        float focus = mix(1.0, mix(0.25, 1.0, aMeta.y), uFocus);
        vColor = aColor * uGlow;
        vAlpha = uAlpha * smoothstep(0.0, 0.35, t) * focus;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
      }
    `,
    fragmentShader: BOX_FRAGMENT,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.frustumCulled = false;
  mesh.renderOrder = 3;
  return { mesh, uniforms };
};

/**
 * Camera-facing boxes around points in the world, like a tracker's boxes on screen.
 * Instances are written each frame with `set`.
 */
export const createBillboardBoxes = (capacity: number, style: BoxStyle = {}): BillboardBoxes => {
  const geometry = quadGeometry(0);
  const aCentre = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 3), 3);
  const aSize = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 2), 2);
  const aColor = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 4), 4);
  for (const a of [aCentre, aSize, aColor]) a.setUsage(THREE.DynamicDrawUsage);
  geometry.setAttribute("aCentre", aCentre);
  geometry.setAttribute("aSize", aSize);
  geometry.setAttribute("aColor", aColor);
  const uniforms = { ...uniformsFor(style) };
  const material = new THREE.ShaderMaterial({
    uniforms,
    transparent: true,
    depthWrite: false,
    depthTest: false,
    vertexShader: /* glsl */ `
      attribute vec3 aCentre;
      attribute vec2 aSize;
      attribute vec4 aColor;
      varying vec2 vUv;
      varying vec3 vColor;
      varying float vAlpha;
      void main() {
        vec4 view = modelViewMatrix * vec4(aCentre, 1.0);
        view.xy += position.xy * aSize;
        vUv = uv;
        vColor = aColor.rgb;
        vAlpha = aColor.a;
        gl_Position = projectionMatrix * view;
      }
    `,
    fragmentShader: BOX_FRAGMENT,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.frustumCulled = false;
  mesh.renderOrder = 6;

  let count = 0;
  return {
    mesh,
    begin: (): void => {
      count = 0;
    },
    set: (centre: THREE.Vector3, w: number, h: number, color: THREE.Color, alpha: number): void => {
      if (count >= capacity || alpha <= 0.002) return;
      aCentre.setXYZ(count, centre.x, centre.y, centre.z);
      aSize.setXY(count, w, h);
      aColor.setXYZW(count, color.r, color.g, color.b, alpha);
      count++;
    },
    end: (): void => {
      geometry.instanceCount = count;
      for (const a of [aCentre, aSize, aColor]) {
        a.clearUpdateRanges();
        a.addUpdateRange(0, count * a.itemSize);
        a.needsUpdate = true;
      }
    },
  };
};
