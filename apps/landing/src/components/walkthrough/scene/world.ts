import * as THREE from "three";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";

import type { Layout } from "./layout";
import { createSpriteLayer, FOG_GLSL, type SpriteMeta } from "./sprites";
import { createVillagers } from "./villagers";

// The world: the recorded frame rebuilt with the game's own art. Terrain uses the
// game's grass, dirt, forest-floor and farm textures; buildings, trees, mines, animals
// and villagers are AoE2:DE sprites (scripts/build_walkthrough_sprites.py) standing
// where the detector found them. Villagers work and carry loads (villagers.ts).

const SOIL_TOP = new THREE.Color("#5b4128");
const SOIL_BOTTOM = new THREE.Color("#1d140c");

const GLSL_COLOR = /* glsl */ `
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
`;

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

// Value noise for terrain variation.
function hash(x: number, y: number) {
  const s = Math.sin(x * 127.1 + y * 311.7) * 43758.5453;
  return s - Math.floor(s);
}
function noise(x: number, y: number) {
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
}

export type World = ReturnType<typeof createWorld>;

export function createWorld(layout: Layout, textures: WorldTextures, meta: SpriteMeta) {
  const group = new THREE.Group();
  group.name = "world";
  const worldPerPixel = Math.SQRT2 / layout.tw;
  // Explored, fogged and unexplored ground darken as they do in the recorded frame.
  const fog = { map: textures.visibility, frame: new THREE.Vector3(layout.W, layout.H, layout.tw) };

  // ---- Terrain: textured island --------------------------------------------------------
  const [c0, c1, c2, c3] = layout.corners as [{ x: number; z: number }, { x: number; z: number }, { x: number; z: number }, { x: number; z: number }];
  const SEG = 120;
  const top = new THREE.PlaneGeometry(1, 1, SEG, SEG);
  const pos = top.getAttribute("position") as THREE.BufferAttribute;
  const mask = new Float32Array(pos.count * 4);
  const buildings = layout.placements.filter((p) => p.kind === "building");
  const trees = layout.placements.filter((p) => p.kind === "tree");
  for (let i = 0; i < pos.count; i++) {
    const s = pos.getX(i) + 0.5;
    const t = 0.5 - pos.getY(i);
    const x = c0.x + (c1.x - c0.x) * s + (c3.x - c0.x) * t;
    const z = c0.z + (c1.z - c0.z) * s + (c3.z - c0.z) * t;
    pos.setXYZ(i, x, 0, z);
    let dirt = 0;
    for (const b of buildings) {
      const d = Math.hypot(x - b.x, z - b.z) / (b.size * 0.9);
      dirt = Math.max(dirt, 1 - THREE.MathUtils.smoothstep(d, 0.6, 1.3));
    }
    let forest = 0;
    for (const tr of trees) {
      const d = Math.hypot(x - tr.x, z - tr.z);
      forest = Math.max(forest, 1 - THREE.MathUtils.smoothstep(d, 0.5, 1.8));
    }
    const patchy = THREE.MathUtils.smoothstep(noise(x * 0.09 + 4, z * 0.09), 0.45, 0.75);
    const edge = Math.min(s, 1 - s, t, 1 - t);
    mask.set([dirt * (0.7 + 0.3 * noise(x * 0.8, z * 0.8)), forest, patchy, THREE.MathUtils.smoothstep(edge, 0, 0.05)], i * 4);
  }
  top.setAttribute("aMask", new THREE.BufferAttribute(mask, 4));

  for (const tex of [textures.grass, textures.patchy, textures.dirt, textures.forest, textures.farm]) {
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.anisotropy = 8;
    tex.needsUpdate = true;
  }
  const terrainUniforms = {
    uGrass: { value: textures.grass },
    uPatchy: { value: textures.patchy },
    uDirt: { value: textures.dirt },
    uForest: { value: textures.forest },
    uLight: { value: 1 },
    uFog: { value: fog.map },
    uFrame: { value: fog.frame },
  };
  // The game lays its terrain textures out on screen axes; one texture spans about ten
  // tiles across at the recorded zoom.
  const TERRAIN_UV = /* glsl */ `vec2 terrainUv(vec3 p) { return vec2(p.x - p.z, (p.x + p.z) * 0.5) / 10.0; }`;
  const ground = new THREE.Mesh(
    top,
    new THREE.ShaderMaterial({
      uniforms: terrainUniforms,
      vertexShader: /* glsl */ `
        ${TERRAIN_UV}
        ${FOG_GLSL}
        attribute vec4 aMask;
        varying vec4 vMask;
        varying vec2 vUv;
        varying float vFog;
        void main() {
          vMask = aMask;
          vFog = fogAt(position);
          vUv = terrainUv(position);
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: /* glsl */ `
        uniform sampler2D uGrass, uPatchy, uDirt, uForest;
        uniform float uLight;
        varying vec4 vMask;
        varying vec2 vUv;
        varying float vFog;
        void main() {
          vec3 c = texture2D(uGrass, vUv).rgb;
          c = mix(c, texture2D(uPatchy, vUv * 0.9).rgb, vMask.z * 0.8);
          c = mix(c, texture2D(uForest, vUv * 1.3).rgb, vMask.y);
          c = mix(c, texture2D(uDirt, vUv * 1.1).rgb, vMask.x);
          // The rim fades to the black of unexplored map.
          c *= mix(0.15, 1.0, vMask.w) * uLight * vFog;
          gl_FragColor = vec4(c, 1.0);
          ${GLSL_COLOR}
        }
      `,
    }),
  );
  group.add(ground);

  // Soil walls under the rim, so the island reads as a cut-out diorama.
  const DEPTH = 2.2;
  const wallParts: THREE.BufferGeometry[] = [];
  const ring = [c0, c1, c2, c3, c0];
  for (let k = 0; k < 4; k++) {
    const a = ring[k]!;
    const b = ring[k + 1]!;
    const g = new THREE.BufferGeometry();
    g.setAttribute(
      "position",
      new THREE.BufferAttribute(new Float32Array([a.x, 0, a.z, b.x, 0, b.z, b.x, -DEPTH, b.z, a.x, 0, a.z, b.x, -DEPTH, b.z, a.x, -DEPTH, a.z]), 3),
    );
    const cols = new Float32Array(18);
    [SOIL_TOP, SOIL_TOP, SOIL_BOTTOM, SOIL_TOP, SOIL_BOTTOM, SOIL_BOTTOM].forEach((c, j) => cols.set([c.r, c.g, c.b], j * 3));
    g.setAttribute("color", new THREE.BufferAttribute(cols, 3));
    wallParts.push(g);
  }
  const wallMaterial = new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide });
  group.add(new THREE.Mesh(mergeGeometries(wallParts)!, wallMaterial));

  // ---- Farms: the game's farm texture on 3x3-tile squares ------------------------------
  const farmMaterial = new THREE.ShaderMaterial({
    uniforms: { uFarm: { value: textures.farm }, uLight: terrainUniforms.uLight, uFog: terrainUniforms.uFog, uFrame: terrainUniforms.uFrame },
    vertexShader: /* glsl */ `
      ${TERRAIN_UV}
      ${FOG_GLSL}
      varying vec2 vUv;
      varying float vFog;
      void main() {
        vUv = terrainUv(position) * 2.5;
        vFog = fogAt(position);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: /* glsl */ `
      uniform sampler2D uFarm;
      uniform float uLight;
      varying vec2 vUv;
      varying float vFog;
      void main() { gl_FragColor = vec4(texture2D(uFarm, vUv).rgb * uLight * vFog, 1.0); ${GLSL_COLOR} }
    `,
    polygonOffset: true,
    polygonOffsetFactor: -2,
  });
  const farmParts = layout.placements
    .filter((p) => p.kind === "farm")
    .map((f) => new THREE.PlaneGeometry(2.9, 2.9).rotateX(-Math.PI / 2).translate(f.x, 0.01, f.z));
  if (farmParts.length) group.add(new THREE.Mesh(mergeGeometries(farmParts)!, farmMaterial));

  // ---- Sprites --------------------------------------------------------------------------
  const sprites = createSpriteLayer(textures.atlas, meta, worldPerPixel, fog);
  group.add(sprites.mesh);
  const variants = (prefix: string) => Object.keys(meta.sprites).filter((n) => n === prefix || n.startsWith(`${prefix}_`));
  const pick = <T,>(list: T[], seed: number) => list[Math.abs(Math.floor(seed)) % list.length]!;

  const BUILDING_SPRITE: Record<string, string> = {
    town_center: "town_center",
    mill: "mill",
    lumber_camp: "lumber_camp",
    mining_camp: "mining_camp",
    tower: "tower",
  };
  const houses = variants("house");
  // Trees without a placement from the build (the forest hidden under the resource
  // bar) are pines, like the stands along the map's edges.
  const treeNames = variants("pine");
  let houseIndex = 0;
  for (const p of layout.placements) {
    if (p.kind === "building") {
      const name = p.sprite ?? (p.entity.cls === "house" ? houses[houseIndex++ % houses.length] : BUILDING_SPRITE[p.entity.cls]);
      if (name) sprites.add(name, p.x, 0, p.z);
    } else if (p.kind === "tree") {
      sprites.add(p.sprite ?? pick(treeNames, p.x * 13.7 + p.z * 7.1), p.x, 0, p.z);
    } else if (p.kind === "mine") {
      const list = variants(p.entity.cls === "gold_mine" ? "gold" : "stone");
      sprites.add(p.sprite ?? pick(list, p.x * 3), p.x, 0, p.z);
    }
  }

  // Ground cover on open grass, away from buildings, fields and forest.
  // Mostly small grass tufts, with bushes and shrubs.
  const cover = [...variants("tuft"), ...variants("tuft"), ...variants("bush"), ...variants("shrub")];
  const blockers = layout.placements.filter((p) => p.kind === "building" || p.kind === "farm" || p.kind === "tree" || p.kind === "mine");
  for (let k = 0, placed = 0; k < 900 && placed < 140; k++) {
    const s = layout.rand();
    const t = layout.rand();
    const x = c0.x + (c1.x - c0.x) * s + (c3.x - c0.x) * t;
    const z = c0.z + (c1.z - c0.z) * s + (c3.z - c0.z) * t;
    if (Math.min(s, 1 - s, t, 1 - t) < 0.04) continue;
    if (blockers.some((b) => Math.hypot(x - b.x, z - b.z) < (b.kind === "tree" ? 1.2 : b.size * 0.85 + 0.6))) continue;
    sprites.add(pick(cover, layout.rand() * 1000), x, 0, z);
    placed++;
  }

  const villagers = createVillagers(layout.placements, layout.corners, sprites);
  const animals = layout.placements
    .filter((p) => p.kind === "animal")
    .map((a, i) => {
      const kind = a.entity.cls === "boar" ? "boar" : "deer";
      return { handle: sprites.add(`${kind}_d${i % 2 ? 2 : 10}`, a.x, 0, a.z, i * 2), phase: i };
    });

  // ---- The new house (Act chapter) ------------------------------------------------------
  const green = layout.spots[2]!;
  const foundation = sprites.add("foundation", green.x, 0, green.z);
  const built = sprites.add(houses[0]!, green.x, 0, green.z);
  const ghost = sprites.add(houses[0]!, green.x, 0, green.z);
  foundation.visible = built.visible = ghost.visible = false;

  const FPS = 12;
  function update(
    time: number,
    camera: THREE.Camera,
    s: { light: number; ghost: number; ghostX: number; ghostZ: number; ghostOk: boolean; build: number },
  ) {
    terrainUniforms.uLight.value = s.light;
    wallMaterial.color.setScalar(s.light);

    villagers.update(time);
    for (const a of animals) a.handle.frame = Math.floor(time * 8 + a.phase * 5);

    ghost.visible = s.ghost > 0.01;
    ghost.anchor.set(s.ghostX, 0, s.ghostZ);
    ghost.alpha = 0.7 * s.ghost;
    ghost.tint.set(s.ghostOk ? "#7dff9a" : "#ff5a4a");
    // Construction: 0 nothing, 1 foundation, then the house rises bottom-up by 5.
    foundation.visible = s.build >= 0.5 && s.build < 4.8;
    built.visible = s.build >= 1;
    built.reveal = THREE.MathUtils.clamp((s.build - 1) / 4, 0, 1);

    sprites.update(camera, s.light);
  }

  return { group, update, center: layout.center, tracked: villagers.tracked, villagerCheck: villagers.check };
}
