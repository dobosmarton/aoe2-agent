import * as THREE from "three";

import { GLSL_OUTPUT_COLOR } from "./shaders";

// AoE2:DE sprites in the 3D scene. Each sprite is a quad that faces the camera,
// standing on its ground anchor (the frame's hotspot), at the game's own scale: a
// sprite pixel covers as much of the view as a screen pixel does in the game at the
// recorded zoom. Sprites draw back to front, like the game, and are tested against
// depth from a little in front of their anchor, so terrain never cuts their base but
// objects in front of the world (the glass, the mind) still hide them.

export type SpriteMeta = { size: [number, number]; sprites: Record<string, number[][]> };

export type SpriteHandle = {
  anchor: THREE.Vector3;
  /** Sprite name in the atlas and the frame to show. */
  name: string;
  frame: number;
  tint: THREE.Color;
  alpha: number;
  /** 0..1 of the sprite shown, from the bottom up (construction). */
  reveal: number;
  visible: boolean;
};

/** Most sprites one layer draws. */
const DEFAULT_CAPACITY = 1200;

/** Shared ground uniforms: the recorded frame's fog of war, where the frame shows the
 * map, drifting cloud shadows and how far the map has opened (the hero intro). */
export type Fog = {
  map: THREE.Texture;
  /** (frame width, frame height, tile width) in px. */
  frame: THREE.Vector3;
  /** Top and bottom of the map area in the frame (resource bar and panel excluded), px. */
  play: THREE.Vector2;
  time: { value: number };
  open: { value: number };
};

export type FogUniforms = {
  uFog: { value: THREE.Texture };
  uFrame: { value: THREE.Vector3 };
  uPlay: { value: THREE.Vector2 };
  uTime: { value: number };
  uOpen: { value: number };
};

export type SpriteLayer = {
  mesh: THREE.Mesh;
  /** Places a sprite; past the layer's capacity the handle is returned but never drawn. */
  add: (name: string, x: number, y: number, z: number, frame?: number) => SpriteHandle;
  /** Writes the visible sprites, farthest first, into the instance buffers. */
  update: (camera: THREE.Camera, light: number) => void;
};

/** `time` and `open` are shared by reference, so one write moves every ground material. */
export const fogUniforms = (fog: Fog): FogUniforms => ({
  uFog: { value: fog.map },
  uFrame: { value: fog.frame },
  uPlay: { value: fog.play },
  uTime: fog.time,
  uOpen: fog.open,
});

/** GLSL for ground points: the fog of war, the map edge and cloud shadows. */
export const FOG_GLSL = /* glsl */ `
  uniform sampler2D uFog;
  uniform vec3 uFrame;
  uniform vec2 uPlay;
  uniform float uTime, uOpen;
  vec2 framePx(vec3 p) {
    return vec2(uFrame.x * 0.5 + (p.x - p.z) * uFrame.z * 0.5, uFrame.y * 0.5 + (p.x + p.z) * uFrame.z * 0.25);
  }
  float fogAt(vec3 p) {
    vec2 s = framePx(p);
    float lit = texture2D(uFog, vec2(s.x / uFrame.x, 1.0 - s.y / uFrame.y)).r;
    // Grass within 15% of the brightest counts as in full view.
    return max(pow(clamp(lit / 0.85, 0.0, 1.0), 1.4), 0.04);
  }
  float hash21(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
  float vnoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash21(i), hash21(i + vec2(1.0, 0.0)), f.x), mix(hash21(i + vec2(0.0, 1.0)), hash21(i + vec2(1.0, 1.0)), f.x), f.y);
  }
  // 1 inside the map, 0 past its edge. The edge is ragged, like the border of explored
  // ground, and recedes as the map opens.
  float mapEdge(vec3 p) {
    vec2 s = framePx(p);
    float e = min(min(s.x, uFrame.x - s.x), min(s.y - uPlay.x, uPlay.y - s.y)) / uFrame.z;
    float n = vnoise(p.xz * 0.45) * 1.6 + vnoise(p.xz * 1.7 + 7.0) * 0.6;
    return smoothstep(-0.5, 1.6, e + n - 1.1 + (1.0 - uOpen) * 5.0 * (n - 1.6));
  }
  // Slow cloud shadows drifting over the map.
  float clouds(vec3 p) {
    vec2 q = p.xz * 0.06 + uTime * vec2(0.016, 0.007);
    float n = vnoise(q) * 0.65 + vnoise(q * 2.3 + 5.0) * 0.35;
    return 1.0 - 0.17 * smoothstep(0.48, 0.78, n);
  }
`;

type SpriteAttributes = {
  geometry: THREE.InstancedBufferGeometry;
  aAnchor: THREE.InstancedBufferAttribute;
  aRect: THREE.InstancedBufferAttribute;
  aPivot: THREE.InstancedBufferAttribute;
  aTint: THREE.InstancedBufferAttribute;
  aReveal: THREE.InstancedBufferAttribute;
};

type SpriteUniforms = FogUniforms & {
  uAtlas: { value: THREE.Texture };
  uAtlasSize: { value: THREE.Vector2 };
  uScale: { value: number };
  uLight: { value: number };
};

/** A sprite queued for drawing, with its view depth; reused from frame to frame. */
type DepthEntry = { handle: SpriteHandle; depth: number };

const prepareAtlas = (atlas: THREE.Texture): void => {
  atlas.flipY = false;
  atlas.colorSpace = THREE.SRGBColorSpace;
  atlas.generateMipmaps = true;
  atlas.minFilter = THREE.LinearMipmapLinearFilter;
  atlas.needsUpdate = true;
};

/** A unit quad anchored at its bottom-left corner, instanced with per-sprite buffers. */
const createSpriteGeometry = (capacity: number): SpriteAttributes => {
  const quad = new THREE.PlaneGeometry(1, 1);
  quad.translate(0.5, 0.5, 0);
  const geometry = new THREE.InstancedBufferGeometry();
  geometry.index = quad.index;
  geometry.setAttribute("position", quad.getAttribute("position"));
  const aAnchor = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 3), 3);
  const aRect = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 4), 4);
  const aPivot = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 2), 2);
  const aTint = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 4), 4);
  const aReveal = new THREE.InstancedBufferAttribute(new Float32Array(capacity), 1);
  for (const a of [aAnchor, aRect, aPivot, aTint, aReveal]) a.setUsage(THREE.DynamicDrawUsage);
  geometry.setAttribute("aAnchor", aAnchor);
  geometry.setAttribute("aRect", aRect);
  geometry.setAttribute("aPivot", aPivot);
  geometry.setAttribute("aTint", aTint);
  geometry.setAttribute("aReveal", aReveal);
  geometry.instanceCount = 0;
  return { geometry, aAnchor, aRect, aPivot, aTint, aReveal };
};

const createSpriteMaterial = (uniforms: SpriteUniforms): THREE.ShaderMaterial =>
  new THREE.ShaderMaterial({
    uniforms,
    transparent: true,
    depthWrite: false,
    vertexShader: /* glsl */ `
      ${FOG_GLSL}
      uniform vec2 uAtlasSize;
      uniform float uScale;
      attribute vec3 aAnchor;
      attribute vec4 aRect;
      attribute vec2 aPivot;
      attribute vec4 aTint;
      attribute float aReveal;
      varying vec2 vUv;
      varying vec4 vTint;
      varying float vRow;
      varying float vReveal;
      varying float vFog;
      varying float vEdge;
      void main() {
        vFog = fogAt(aAnchor) * clouds(aAnchor);
        vEdge = mapEdge(aAnchor);
        // Sprite pixel of this corner, y down from the top of the sprite.
        vec2 px = vec2(position.x * aRect.z, (1.0 - position.y) * aRect.w);
        vec2 offset = vec2(px.x - aPivot.x, aPivot.y - px.y) * uScale;
        vec4 view = modelViewMatrix * vec4(aAnchor, 1.0);
        vec4 clipAnchor = projectionMatrix * vec4(view.xyz + vec3(0.0, 0.0, 4.0), 1.0);
        vec4 clip = projectionMatrix * vec4(view.xy + offset, view.z, 1.0);
        // Depth from 4 units in front of the anchor, shared by the whole quad.
        clip.z = clipAnchor.z / clipAnchor.w * clip.w;
        gl_Position = clip;
        vUv = (aRect.xy + px) / uAtlasSize;
        vTint = aTint;
        vRow = px.y / aRect.w;
        vReveal = aReveal;
      }
    `,
    fragmentShader: /* glsl */ `
      uniform sampler2D uAtlas;
      uniform float uLight;
      varying vec2 vUv;
      varying vec4 vTint;
      varying float vRow;
      varying float vReveal;
      varying float vFog;
      varying float vEdge;
      void main() {
        if (vRow < 1.0 - vReveal) discard;
        vec4 c = texture2D(uAtlas, vUv);
        if (c.a < 0.01 || vEdge < 0.01) discard;
        gl_FragColor = vec4(c.rgb * vTint.rgb * uLight * vFog, c.a * vTint.a * vEdge);
        ${GLSL_OUTPUT_COLOR}
      }
    `,
  });

export const createSpriteLayer = (atlas: THREE.Texture, meta: SpriteMeta, worldPerPixel: number, fog: Fog, capacity = DEFAULT_CAPACITY): SpriteLayer => {
  prepareAtlas(atlas);
  const { geometry, aAnchor, aRect, aPivot, aTint, aReveal } = createSpriteGeometry(capacity);
  const uniforms: SpriteUniforms = {
    uAtlas: { value: atlas },
    uAtlasSize: { value: new THREE.Vector2(meta.size[0], meta.size[1]) },
    uScale: { value: worldPerPixel },
    uLight: { value: 1 },
    ...fogUniforms(fog),
  };
  const mesh = new THREE.Mesh(geometry, createSpriteMaterial(uniforms));
  mesh.frustumCulled = false;
  mesh.renderOrder = 1;

  const handles: SpriteHandle[] = [];
  const frames = (name: string): number[][] => meta.sprites[name] ?? [];

  const add = (name: string, x: number, y: number, z: number, frame = 0): SpriteHandle => {
    const handle: SpriteHandle = {
      anchor: new THREE.Vector3(x, y, z),
      name,
      frame,
      tint: new THREE.Color(1, 1, 1),
      alpha: 1,
      reveal: 1,
      visible: true,
    };
    if (handles.length < capacity) handles.push(handle);
    return handle;
  };

  const view = new THREE.Vector3();
  const pool: DepthEntry[] = [];
  const order: DepthEntry[] = [];
  const byDepth = (a: DepthEntry, b: DepthEntry): number => a.depth - b.depth;

  const sortVisible = (camera: THREE.Camera): void => {
    order.length = 0;
    for (const handle of handles) {
      if (!handle.visible || handle.alpha <= 0.001) continue;
      view.copy(handle.anchor).applyMatrix4(camera.matrixWorldInverse);
      const entry = (pool[order.length] ??= { handle, depth: 0 });
      entry.handle = handle;
      entry.depth = view.z;
      order.push(entry);
    }
    order.sort(byDepth);
  };

  const update = (camera: THREE.Camera, light: number): void => {
    uniforms.uLight.value = light;
    sortVisible(camera);
    let count = 0;
    for (const { handle } of order) {
      const list = frames(handle.name);
      const rect = list.at(Math.max(0, Math.floor(handle.frame)) % Math.max(1, list.length));
      if (!rect) continue;
      aAnchor.setXYZ(count, handle.anchor.x, handle.anchor.y, handle.anchor.z);
      aRect.setXYZW(count, rect[0], rect[1], rect[2], rect[3]);
      aPivot.setXY(count, rect[4], rect[5]);
      aTint.setXYZW(count, handle.tint.r, handle.tint.g, handle.tint.b, handle.alpha);
      aReveal.setX(count, handle.reveal);
      count++;
    }
    geometry.instanceCount = count;
    for (const a of [aAnchor, aRect, aPivot, aTint, aReveal]) {
      a.clearUpdateRanges();
      a.addUpdateRange(0, count * a.itemSize);
      a.needsUpdate = true;
    }
  };

  return { mesh, add, update };
};
