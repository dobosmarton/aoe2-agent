import * as THREE from "three";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";

import type { Layout, Placement } from "./layout";
import { GLSL_OUTPUT_COLOR } from "./shaders";
import { createSpriteLayer, FOG_GLSL, type Fog, fogUniforms, type FogUniforms, type SpriteLayer, type SpriteMeta } from "./sprites";
import { createVillagers, type TrackedVillager, type VillagerCheck } from "./villagers";

// The world: the recorded frame rebuilt with the game's own art. Terrain uses the
// game's grass, dirt, forest-floor and farm textures; buildings, trees, mines, animals
// and villagers are AoE2:DE sprites (scripts/build_walkthrough_sprites.py) standing
// where the detector found them. Villagers work and carry loads (villagers.ts).

export type WorldTextures = {
  atlas: THREE.Texture;
  grass: THREE.Texture;
  patchy: THREE.Texture;
  dirt: THREE.Texture;
  forest: THREE.Texture;
  farm: THREE.Texture;
  /** The recorded frame's fog of war (scripts/build_walkthrough_sprites.py). */
  visibility: THREE.Texture;
};

export type WorldState = {
  /** Scene light, 1 at full brightness. */
  light: number;
  /** How far the map has opened (the hero intro). */
  open: number;
  /** The house placement preview: its strength, its ground point and whether it fits. */
  ghost: number;
  ghostX: number;
  ghostZ: number;
  ghostOk: boolean;
  /** Construction: 0 nothing, 1 foundation, then the house rises bottom-up by 5. */
  build: number;
};

export type World = {
  group: THREE.Group;
  update: (time: number, camera: THREE.Camera, s: WorldState) => void;
  tracked: TrackedVillager[];
  /** Debug read-out of the village economy. */
  villagerCheck: () => VillagerCheck;
};

/** The terrain grid reaches this many tiles past the frame, so the ragged edge (FOG_GLSL mapEdge) has ground to fade. */
const TERRAIN_MARGIN_TILES = 3;
const TERRAIN_SEGMENTS = 140;
/** Ground-cover sprites to place on open grass, and how many random spots to try for them. */
const COVER_COUNT = 140;
const COVER_ATTEMPTS = 900;
/** Edge of a farm square, tiles. */
const FARM_SIZE = 2.9;

const GHOST_FITS = new THREE.Color("#7dff9a");
const GHOST_BLOCKED = new THREE.Color("#ff5a4a");

const BUILDING_SPRITE: Record<string, string> = {
  town_center: "town_center",
  mill: "mill",
  lumber_camp: "lumber_camp",
  mining_camp: "mining_camp",
  tower: "tower",
};

// The game lays its terrain textures out on screen axes; one texture spans about ten
// tiles across at the recorded zoom.
const TERRAIN_UV = /* glsl */ `vec2 terrainUv(vec3 p) { return vec2(p.x - p.z, (p.x + p.z) * 0.5) / 10.0; }`;

// Value noise for terrain variation.
const hash = (x: number, y: number): number => {
  const s = Math.sin(x * 127.1 + y * 311.7) * 43758.5453;

  return s - Math.floor(s);
};

const noise = (x: number, y: number): number => {
  const xi = Math.floor(x);
  const yi = Math.floor(y);
  const xf = x - xi;
  const yf = y - yi;
  const u = xf * xf * (3 - 2 * xf);
  const v = yf * yf * (3 - 2 * yf);
  const a = hash(xi, yi);
  const b = hash(xi + 1, yi);
  const c = hash(xi, yi + 1);
  const d = hash(xi + 1, yi + 1);

  return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
};

const pick = <T>(list: readonly T[], seed: number): T => list[Math.abs(Math.floor(seed)) % list.length];

/** Texture weights at a ground point: (dirt around buildings, forest floor under trees, patchy grass). */
const terrainMask = (x: number, z: number, buildings: readonly Placement[], trees: readonly Placement[]): [number, number, number] => {
  let dirt = 0;

  for (const b of buildings) {
    const d = Math.hypot(x - b.x, z - b.z) / (b.size * 0.9);

    dirt = Math.max(dirt, 1 - THREE.MathUtils.smoothstep(d, 0.6, 1.3));
  }

  let forest = 0;

  for (const tree of trees) {
    const d = Math.hypot(x - tree.x, z - tree.z);

    forest = Math.max(forest, 1 - THREE.MathUtils.smoothstep(d, 0.5, 1.8));
  }

  const patchy = THREE.MathUtils.smoothstep(noise(x * 0.09 + 4, z * 0.09), 0.45, 0.75);

  return [dirt * (0.7 + 0.3 * noise(x * 0.8, z * 0.8)), forest, patchy];
};

/** A grid laid out in frame pixels, reaching a few tiles past the frame, with the texture mask per vertex. */
const terrainGeometry = (layout: Layout): THREE.PlaneGeometry => {
  const margin = layout.tw * TERRAIN_MARGIN_TILES;
  const top = new THREE.PlaneGeometry(1, 1, TERRAIN_SEGMENTS, Math.round(TERRAIN_SEGMENTS * 0.6));
  const pos = top.getAttribute("position");
  const mask = new Float32Array(pos.count * 3);
  const buildings = layout.placements.filter((p) => p.kind === "building");
  const trees = layout.placements.filter((p) => p.kind === "tree");

  for (let i = 0; i < pos.count; i++) {
    const px = -margin + (pos.getX(i) + 0.5) * (layout.W + 2 * margin);
    const py = layout.playTop - margin / 2 + (0.5 - pos.getY(i)) * (layout.playBottom - layout.playTop + margin);
    const { x, z } = layout.toGround(px, py);

    pos.setXYZ(i, x, 0, z);
    mask.set(terrainMask(x, z, buildings, trees), i * 3);
  }

  top.setAttribute("aMask", new THREE.BufferAttribute(mask, 3));

  return top;
};

type TerrainUniforms = FogUniforms & {
  uGrass: { value: THREE.Texture };
  uPatchy: { value: THREE.Texture };
  uDirt: { value: THREE.Texture };
  uForest: { value: THREE.Texture };
  uLight: { value: number };
};

const createTerrain = (layout: Layout, textures: WorldTextures, fog: Fog): { mesh: THREE.Mesh; uniforms: TerrainUniforms } => {
  for (const texture of [textures.grass, textures.patchy, textures.dirt, textures.forest, textures.farm]) {
    texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.anisotropy = 8;
    texture.needsUpdate = true;
  }

  const uniforms: TerrainUniforms = {
    uGrass: { value: textures.grass },
    uPatchy: { value: textures.patchy },
    uDirt: { value: textures.dirt },
    uForest: { value: textures.forest },
    uLight: { value: 1 },
    ...fogUniforms(fog),
  };
  const mesh = new THREE.Mesh(
    terrainGeometry(layout),
    new THREE.ShaderMaterial({
      uniforms,
      transparent: true,
      vertexShader: /* glsl */ `
        ${TERRAIN_UV}
        attribute vec3 aMask;
        varying vec3 vMask;
        varying vec2 vUv;
        varying vec3 vPos;
        void main() {
          vMask = aMask;
          vPos = position;
          vUv = terrainUv(position);
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: /* glsl */ `
        ${FOG_GLSL}
        uniform sampler2D uGrass, uPatchy, uDirt, uForest;
        uniform float uLight;
        varying vec3 vMask;
        varying vec2 vUv;
        varying vec3 vPos;
        void main() {
          float edge = mapEdge(vPos);
          if (edge < 0.002) discard;
          vec3 c = texture2D(uGrass, vUv).rgb;
          c = mix(c, texture2D(uPatchy, vUv * 0.9).rgb, vMask.z * 0.8);
          c = mix(c, texture2D(uForest, vUv * 1.3).rgb, vMask.y);
          c = mix(c, texture2D(uDirt, vUv * 1.1).rgb, vMask.x);
          // Fog of war, cloud shadows, and the edge darkening into unexplored ground.
          c *= uLight * fogAt(vPos) * clouds(vPos) * mix(0.35, 1.0, edge);
          gl_FragColor = vec4(c, edge);
          ${GLSL_OUTPUT_COLOR}
        }
      `,
    }),
  );

  return { mesh, uniforms };
};

// why: mergeGeometries returns null when the parts' attributes differ; its type leaves that out.
const mergeOrNull = (parts: THREE.BufferGeometry[]): THREE.BufferGeometry | null => mergeGeometries(parts);

/** The game's farm texture on 3x3-tile squares, sharing the terrain's light. */
const createFarms = (layout: Layout, farmTexture: THREE.Texture, fog: Fog, light: { value: number }): THREE.Mesh | null => {
  const farmParts = layout.placements
    .filter((p) => p.kind === "farm")
    .map((f) => new THREE.PlaneGeometry(FARM_SIZE, FARM_SIZE).rotateX(-Math.PI / 2).translate(f.x, 0.01, f.z));

  if (farmParts.length === 0) {
    return null;
  }

  const merged = mergeOrNull(farmParts);

  if (!merged) {
    throw new Error("Farm geometries could not be merged");
  }

  const material = new THREE.ShaderMaterial({
    uniforms: { uFarm: { value: farmTexture }, uLight: light, ...fogUniforms(fog) },
    vertexShader: /* glsl */ `
      ${TERRAIN_UV}
      ${FOG_GLSL}
      varying vec2 vUv;
      varying float vFog;
      void main() {
        vUv = terrainUv(position) * 2.5;
        vFog = fogAt(position) * clouds(position);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: /* glsl */ `
      uniform sampler2D uFarm;
      uniform float uLight;
      varying vec2 vUv;
      varying float vFog;
      void main() { gl_FragColor = vec4(texture2D(uFarm, vUv).rgb * uLight * vFog, 1.0); ${GLSL_OUTPUT_COLOR} }
    `,
    polygonOffset: true,
    polygonOffsetFactor: -2,
  });

  return new THREE.Mesh(merged, material);
};

/** Sprite names in the atlas that are `prefix` or one of its numbered variants. */
const variantsOf = (meta: SpriteMeta, prefix: string): string[] =>
  Object.keys(meta.sprites).filter((n) => n === prefix || n.startsWith(`${prefix}_`));

/** Buildings, trees and mines where the detector found them. */
const placeLandmarks = (layout: Layout, sprites: SpriteLayer, variants: (prefix: string) => string[]): void => {
  const houses = variants("house");
  // Trees without a placement from the build (the forest hidden under the resource
  // bar) are pines, like the stands along the map's edges.
  const treeNames = variants("pine");
  let houseIndex = 0;

  for (const p of layout.placements) {
    if (p.kind === "building") {
      const name = p.sprite ?? (p.entity.cls === "house" ? houses[houseIndex++ % houses.length] : BUILDING_SPRITE[p.entity.cls]);

      if (name) {
        sprites.add(name, p.x, 0, p.z);
      }
    } else if (p.kind === "tree") {
      sprites.add(p.sprite ?? pick(treeNames, p.x * 13.7 + p.z * 7.1), p.x, 0, p.z);
    } else if (p.kind === "mine") {
      const list = variants(p.entity.cls === "gold_mine" ? "gold" : "stone");

      sprites.add(p.sprite ?? pick(list, p.x * 3), p.x, 0, p.z);
    }
  }
};

/** Ground cover on open grass, away from buildings, fields and forest: mostly small grass tufts, with bushes and shrubs. */
const scatterGroundCover = (layout: Layout, sprites: SpriteLayer, variants: (prefix: string) => string[]): void => {
  const [c0, c1, , c3] = layout.corners;
  const cover = [...variants("tuft"), ...variants("tuft"), ...variants("bush"), ...variants("shrub")];
  const blockers = layout.placements.filter((p) => p.kind === "building" || p.kind === "farm" || p.kind === "tree" || p.kind === "mine");

  for (let k = 0, placed = 0; k < COVER_ATTEMPTS && placed < COVER_COUNT; k++) {
    // A random point of the play area, as fractions along its top and left edges.
    const u = layout.rand();
    const v = layout.rand();
    const x = c0.x + (c1.x - c0.x) * u + (c3.x - c0.x) * v;
    const z = c0.z + (c1.z - c0.z) * u + (c3.z - c0.z) * v;

    if (Math.min(u, 1 - u, v, 1 - v) < 0.04) {
      continue;
    }

    if (blockers.some((b) => Math.hypot(x - b.x, z - b.z) < (b.kind === "tree" ? 1.2 : b.size * 0.85 + 0.6))) {
      continue;
    }

    sprites.add(pick(cover, layout.rand() * 1000), x, 0, z);
    placed++;
  }
};

export const createWorld = (layout: Layout, textures: WorldTextures, meta: SpriteMeta): World => {
  const group = new THREE.Group();

  group.name = "world";

  const worldPerPixel = Math.SQRT2 / layout.tw;
  // Explored, fogged and unexplored ground darken as they do in the recorded frame; the
  // map fades out past the frame's edges like unexplored ground.
  const fog: Fog = {
    map: textures.visibility,
    frame: new THREE.Vector3(layout.W, layout.H, layout.tw),
    play: new THREE.Vector2(layout.playTop, layout.playBottom),
    time: { value: 0 },
    open: { value: 1 },
  };

  const terrain = createTerrain(layout, textures, fog);

  group.add(terrain.mesh);

  const farms = createFarms(layout, textures.farm, fog, terrain.uniforms.uLight);

  if (farms) {
    group.add(farms);
  }

  const sprites = createSpriteLayer(textures.atlas, meta, worldPerPixel, fog);

  group.add(sprites.mesh);

  const variants = (prefix: string): string[] => variantsOf(meta, prefix);

  placeLandmarks(layout, sprites, variants);
  scatterGroundCover(layout, sprites, variants);

  const villagers = createVillagers(layout.placements, layout.corners, sprites);
  const animals = layout.placements
    .filter((p) => p.kind === "animal")
    .map((a, i) => {
      const kind = a.entity.cls === "boar" ? "boar" : "deer";

      return { handle: sprites.add(`${kind}_d${i % 2 ? 2 : 10}`, a.x, 0, a.z, i * 2), phase: i };
    });

  // The new house of the Act chapter: its placement preview, foundation and building.
  const house = variants("house")[0];
  const green = layout.spots[2];
  const foundation = sprites.add("foundation", green.x, 0, green.z);
  const built = sprites.add(house, green.x, 0, green.z);
  const ghost = sprites.add(house, green.x, 0, green.z);

  foundation.visible = built.visible = ghost.visible = false;

  const update = (time: number, camera: THREE.Camera, s: WorldState): void => {
    terrain.uniforms.uLight.value = s.light;
    fog.time.value = time;
    fog.open.value = s.open;

    villagers.update(time);

    for (const a of animals) {
      a.handle.frame = Math.floor(time * 8 + a.phase * 5);
    }

    ghost.visible = s.ghost > 0.01;
    ghost.anchor.set(s.ghostX, 0, s.ghostZ);
    ghost.alpha = 0.7 * s.ghost;
    ghost.tint.copy(s.ghostOk ? GHOST_FITS : GHOST_BLOCKED);
    foundation.visible = s.build >= 0.5 && s.build < 4.8;
    built.visible = s.build >= 1;
    built.reveal = THREE.MathUtils.clamp((s.build - 1) / 4, 0, 1);

    sprites.update(camera, s.light);
  };

  return { group, update, tracked: villagers.tracked, villagerCheck: villagers.check };
};
