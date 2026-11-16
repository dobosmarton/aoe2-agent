import * as THREE from "three";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";

import type { Layout, Placement } from "./layout";

// The world: the recorded frame rebuilt as a low-poly diorama island. Buildings,
// farms, trees and mines stand where the detector found them; villagers and animals
// are small procedural figures. Models: KayKit Medieval Hexagon Pack (CC0).

const GRASS_A = new THREE.Color("#5f9e3a");
const GRASS_B = new THREE.Color("#9bcc52");
const GRASS_DARK = new THREE.Color("#3f7a2c");
const GRASS_DRY = new THREE.Color("#b5b85a");
const DIRT = new THREE.Color("#b89a66");
const SOIL_TOP = new THREE.Color("#6b4f33");
const SOIL_BOTTOM = new THREE.Color("#2e2117");
const PLAYER_BLUE = new THREE.Color("#3f74e0");

// Value noise, enough for terrain colour and a little relief.
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
const fbm = (x: number, y: number) => noise(x, y) * 0.6 + noise(x * 2.3, y * 2.3) * 0.3 + noise(x * 5.1, y * 5.1) * 0.1;

/** Clones a model node from the village GLB, with shadows on. */
function cloneModel(models: Map<string, THREE.Object3D>, name: string) {
  const source = models.get(name);
  if (!source) return new THREE.Group();
  const clone = source.clone(true);
  clone.traverse((o) => {
    if ((o as THREE.Mesh).isMesh) {
      o.castShadow = true;
      o.receiveShadow = true;
    }
  });
  return clone;
}

/** Scales an object so its larger horizontal side spans `tiles`, base on y = 0. */
function fitToTiles(object: THREE.Object3D, tiles: number) {
  const box = new THREE.Box3().setFromObject(object);
  const size = box.getSize(new THREE.Vector3());
  const scale = tiles / Math.max(size.x, size.z, 0.001);
  object.scale.setScalar(scale);
  object.position.y -= box.min.y * scale;
  return scale;
}

/** First mesh of a model, for instancing. */
function firstMesh(models: Map<string, THREE.Object3D>, name: string): THREE.Mesh | null {
  let mesh: THREE.Mesh | null = null;
  models.get(name)?.traverse((o) => {
    if (!mesh && (o as THREE.Mesh).isMesh) mesh = o as THREE.Mesh;
  });
  return mesh;
}

function coloured(geometry: THREE.BufferGeometry, color: THREE.Color) {
  const g = geometry.index ? geometry.toNonIndexed() : geometry;
  const colors = new Float32Array(g.getAttribute("position").count * 3);
  for (let i = 0; i < colors.length; i += 3) colors.set([color.r, color.g, color.b], i);
  g.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  g.deleteAttribute("uv");
  return g;
}

function villagerGeometry() {
  const legs = coloured(new THREE.CylinderGeometry(0.1, 0.11, 0.22, 6).translate(0, 0.11, 0), new THREE.Color("#4a3a2c"));
  const body = coloured(new THREE.CylinderGeometry(0.12, 0.17, 0.32, 6).translate(0, 0.38, 0), PLAYER_BLUE);
  const belt = coloured(new THREE.CylinderGeometry(0.155, 0.155, 0.04, 6).translate(0, 0.3, 0), new THREE.Color("#6b4a2a"));
  const head = coloured(new THREE.IcosahedronGeometry(0.1, 0).translate(0, 0.62, 0), new THREE.Color("#e2b48c"));
  const hat = coloured(new THREE.ConeGeometry(0.13, 0.12, 6).translate(0, 0.73, 0), new THREE.Color("#c9a25a"));
  const tool = coloured(new THREE.BoxGeometry(0.03, 0.3, 0.03).translate(0.17, 0.38, 0.05), new THREE.Color("#8a6a45"));
  return mergeGeometries([legs, body, belt, head, hat, tool])!;
}

function animalGeometry(color: THREE.Color, stubby: boolean) {
  const legH = stubby ? 0.12 : 0.22;
  const parts = [
    new THREE.BoxGeometry(0.46, 0.2, 0.2).translate(0, legH + 0.1, 0),
    new THREE.BoxGeometry(0.16, stubby ? 0.14 : 0.2, 0.14).translate(0.27, legH + (stubby ? 0.1 : 0.22), 0),
  ];
  for (const [x, z] of [[0.16, 0.07], [0.16, -0.07], [-0.16, 0.07], [-0.16, -0.07]]) {
    parts.push(new THREE.BoxGeometry(0.05, legH, 0.05).translate(x!, legH / 2, z!));
  }
  return mergeGeometries(parts.map((p) => coloured(p, color)))!;
}

export type World = Awaited<ReturnType<typeof createWorld>>;

export async function createWorld(layout: Layout, models: Map<string, THREE.Object3D>) {
  const group = new THREE.Group();
  group.name = "world";

  // ---- Island ------------------------------------------------------------------------
  const [c0, c1, c2, c3] = layout.corners as [{ x: number; z: number }, { x: number; z: number }, { x: number; z: number }, { x: number; z: number }];
  const SEG = 96;
  const top = new THREE.PlaneGeometry(1, 1, SEG, SEG);
  const pos = top.getAttribute("position") as THREE.BufferAttribute;
  const colors = new Float32Array(pos.count * 3);
  const buildings = layout.placements.filter((p) => p.kind === "building");
  const color = new THREE.Color();
  for (let i = 0; i < pos.count; i++) {
    // Plane coords in [-0.5, 0.5] become a bilinear point inside the four corners.
    const s = pos.getX(i) + 0.5;
    const t = 0.5 - pos.getY(i);
    const ax = c0.x + (c1.x - c0.x) * s;
    const az = c0.z + (c1.z - c0.z) * s;
    const bx = c3.x + (c2.x - c3.x) * s;
    const bz = c3.z + (c2.z - c3.z) * s;
    const x = ax + (bx - ax) * t;
    const z = az + (bz - az) * t;
    const n = fbm(x * 0.12, z * 0.12);
    const edge = Math.min(s, 1 - s, t, 1 - t);
    const y = (fbm(x * 0.3 + 9, z * 0.3) - 0.5) * 0.18 * Math.min(1, edge * 20);
    pos.setXYZ(i, x, y, z);
    color.copy(GRASS_A).lerp(GRASS_B, THREE.MathUtils.smoothstep(n, 0.25, 0.8));
    color.lerp(GRASS_DARK, THREE.MathUtils.smoothstep(fbm(x * 0.07 + 3, z * 0.07), 0.55, 0.75) * 0.7);
    color.lerp(GRASS_DRY, THREE.MathUtils.smoothstep(fbm(x * 0.09 - 7, z * 0.09 + 2), 0.62, 0.82) * 0.55);
    let dirt = 0;
    for (const b of buildings) {
      const d = Math.hypot(x - b.x, z - b.z) / (b.size * 0.95);
      dirt = Math.max(dirt, 1 - THREE.MathUtils.smoothstep(d, 0.55, 1.15));
    }
    color.lerp(DIRT, dirt * (0.55 + 0.35 * noise(x * 1.7, z * 1.7)));
    // The rim darkens, like the edge of the explored map.
    color.multiplyScalar(0.55 + 0.45 * THREE.MathUtils.smoothstep(edge, 0, 0.06));
    colors.set([color.r, color.g, color.b], i * 3);
  }
  top.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  top.computeVertexNormals();
  const ground = new THREE.Mesh(
    top,
    new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.95, flatShading: true }),
  );
  ground.receiveShadow = true;
  group.add(ground);

  // Soil walls under the rim, so the island reads as a cut-out diorama.
  const DEPTH = 2.2;
  const wallParts: THREE.BufferGeometry[] = [];
  const ring = [c0, c1, c2, c3, c0];
  for (let k = 0; k < 4; k++) {
    const a = ring[k]!;
    const b = ring[k + 1]!;
    const g = new THREE.BufferGeometry();
    const verts = new Float32Array([
      a.x, 0, a.z, b.x, 0, b.z, b.x, -DEPTH, b.z,
      a.x, 0, a.z, b.x, -DEPTH, b.z, a.x, -DEPTH, a.z,
    ]);
    const cols = new Float32Array(18);
    [SOIL_TOP, SOIL_TOP, SOIL_BOTTOM, SOIL_TOP, SOIL_BOTTOM, SOIL_BOTTOM].forEach((c, j) => cols.set([c.r, c.g, c.b], j * 3));
    g.setAttribute("position", new THREE.BufferAttribute(verts, 3));
    g.setAttribute("color", new THREE.BufferAttribute(cols, 3));
    wallParts.push(g);
  }
  const walls = new THREE.Mesh(
    mergeGeometries(wallParts)!,
    new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 1, side: THREE.DoubleSide }),
  );
  walls.geometry.computeVertexNormals();
  group.add(walls);

  // ---- Buildings and farms -----------------------------------------------------------
  // Buildings face the game camera, which sits on the +x +z side.
  const FACING = Math.PI / 4;
  let windmillFan: THREE.Object3D | null = null;
  for (const p of layout.placements) {
    if (!p.model) continue;
    const object = cloneModel(models, p.model);
    fitToTiles(object, p.model === "castle" ? p.size * 1.05 : p.size * 0.92);
    // The castle stands in for a Town Center, which is wide and low.
    if (p.model === "castle") object.scale.y *= 0.62;
    object.position.x = p.x;
    object.position.z = p.z;
    object.rotation.y = p.kind === "farm" ? Math.PI / 4 : FACING;
    if (p.model === "windmill") {
      object.traverse((o) => {
        if (/fan/i.test(o.name)) windmillFan = o;
      });
    }
    group.add(object);
  }

  // ---- Farms: square fields of crop rows, 3 tiles on an edge, aligned with the grid.
  const farms = layout.placements.filter((p) => p.kind === "farm");
  const soilMaterial = new THREE.MeshStandardMaterial({ color: "#7a5a3a", roughness: 1 });
  const cropMaterial = new THREE.MeshStandardMaterial({ color: "#ffffff", roughness: 0.9, flatShading: true });
  // Each row is a line of small plants, so a field reads as crops, not planks.
  const ROWS = 6;
  const PLANTS = 9;
  const plant = new THREE.ConeGeometry(0.14, 0.34, 5).translate(0, 0.17, 0);
  const crops = new THREE.InstancedMesh(plant, cropMaterial, farms.length * ROWS * PLANTS);
  crops.castShadow = crops.receiveShadow = true;
  const holder = new THREE.Object3D();
  let crop = 0;
  farms.forEach((f) => {
    const soil = new THREE.Mesh(new THREE.BoxGeometry(2.85, 0.12, 2.85), soilMaterial);
    soil.position.set(f.x, 0.04, f.z);
    soil.receiveShadow = true;
    group.add(soil);
    const ripe = (((f.x * 3.7 + f.z * 1.3) % 1) + 1) % 1;
    const tint = new THREE.Color().setHSL(0.13 + (1 - ripe) * 0.11, 0.6, 0.42 + ripe * 0.08);
    for (let r = 0; r < ROWS; r++) {
      for (let k = 0; k < PLANTS; k++) {
        holder.position.set(f.x - 1.15 + (k * 2.3) / (PLANTS - 1), 0.1, f.z - 1.15 + (r * 2.3) / (ROWS - 1));
        holder.rotation.set(0, (k * 1.7 + r) % 3, 0);
        holder.scale.set(1, 0.6 + ripe * 0.7 + ((k * 7 + r * 3) % 5) * 0.06, 1);
        holder.updateMatrix();
        crops.setMatrixAt(crop, holder.matrix);
        crops.setColorAt(crop, tint);
        crop++;
      }
    }
  });
  if (farms.length) group.add(crops);

  // ---- Grass tufts and flowers, scattered over open ground.
  // A tuft is three thin blades leaning apart.
  const blade = () => new THREE.ConeGeometry(0.035, 0.22, 3).translate(0, 0.11, 0);
  const tuftGeometry = mergeGeometries([
    blade().rotateZ(0.25),
    blade().rotateZ(-0.25).rotateY(2.1),
    blade().rotateX(0.2).rotateY(4.2),
  ])!;
  const tuftMaterial = new THREE.MeshStandardMaterial({ color: "#ffffff", roughness: 1, flatShading: true });
  const TUFTS = 1400;
  const tufts = new THREE.InstancedMesh(tuftGeometry, tuftMaterial, TUFTS);
  let placed = 0;
  const occupied = layout.placements.filter((p) => p.kind === "building" || p.kind === "farm");
  for (let k = 0; k < TUFTS * 3 && placed < TUFTS; k++) {
    const s = layout.rand();
    const t = layout.rand();
    const x = c0.x + (c1.x - c0.x) * s + (c3.x - c0.x) * t;
    const z = c0.z + (c1.z - c0.z) * s + (c3.z - c0.z) * t;
    if (occupied.some((b) => Math.hypot(x - b.x, z - b.z) < b.size * 0.8)) continue;
    holder.position.set(x, 0, z);
    holder.rotation.set((layout.rand() - 0.5) * 0.5, layout.rand() * 6, (layout.rand() - 0.5) * 0.5);
    holder.scale.setScalar(0.6 + layout.rand() * 0.9);
    holder.updateMatrix();
    tufts.setMatrixAt(placed, holder.matrix);
    tufts.setColorAt(placed, new THREE.Color().setHSL(0.24 + layout.rand() * 0.06, 0.55, 0.28 + layout.rand() * 0.12));
    placed++;
  }
  tufts.count = placed;
  tufts.receiveShadow = true;
  group.add(tufts);

  // A few props around the Town Center and the camps.
  const tc = layout.tc;
  const props: [string, number, number, number][] = [
    ["flag", tc.x - 2.3, tc.z - 1.2, 0.9],
    ["wheelbarrow", tc.x + 2.4, tc.z - 1.6, 0.7],
    ["crates", tc.x - 2.2, tc.z + 1.9, 0.7],
    ["sack", tc.x - 1.6, tc.z + 2.3, 0.45],
  ];
  for (const p of layout.placements) {
    if (p.model === "lumbermill") props.push(["lumber", p.x + 1.4, p.z - 0.6, 0.9]);
    if (p.model === "mine") props.push(["stone", p.x - 1.2, p.z + 1.1, 0.8]);
  }
  for (const [name, x, z, tiles] of props) {
    const object = cloneModel(models, name);
    fitToTiles(object, tiles);
    object.position.x = x;
    object.position.z = z;
    object.rotation.y = FACING + (x * 13.1) % 1;
    group.add(object);
  }

  // ---- Trees (instanced) ---------------------------------------------------------------
  const trees = layout.placements.filter((p) => p.kind === "tree");
  const dummy = new THREE.Object3D();
  for (const [k, name] of ["tree_A", "tree_B"].entries()) {
    const mesh = firstMesh(models, name);
    if (!mesh) continue;
    const mine = trees.filter((_, i) => i % 2 === k);
    const geo = mesh.geometry.clone();
    geo.computeBoundingBox();
    const bb = geo.boundingBox!;
    const base = 1 / Math.max(bb.max.x - bb.min.x, bb.max.z - bb.min.z);
    const inst = new THREE.InstancedMesh(geo, mesh.material, mine.length);
    mine.forEach((t, i) => {
      const s = base * t.size * 1.25;
      dummy.position.set(t.x, -bb.min.y * s, t.z);
      dummy.rotation.set(0, (t.x * 7.3 + t.z * 3.1) % (Math.PI * 2), 0);
      dummy.scale.set(s, s * (1 + ((t.x * 5.7) % 0.35)), s);
      dummy.updateMatrix();
      inst.setMatrixAt(i, dummy.matrix);
    });
    inst.castShadow = true;
    inst.receiveShadow = true;
    group.add(inst);
  }

  // ---- Mines: gold or grey rock clusters --------------------------------------------
  const rockMeshes = ["rock_A", "rock_B", "rock_C"].map((n) => firstMesh(models, n)).filter(Boolean) as THREE.Mesh[];
  for (const p of layout.placements.filter((q) => q.kind === "mine")) {
    const gold = p.entity.cls === "gold_mine";
    for (let i = 0; i < 4; i++) {
      const src = rockMeshes[i % rockMeshes.length];
      if (!src) break;
      const material = (src.material as THREE.MeshStandardMaterial).clone();
      material.color = gold ? new THREE.Color("#ffd24a") : new THREE.Color("#c9ccd2");
      if (gold) {
        material.emissive = new THREE.Color("#4a3500");
        material.metalness = 0.35;
        material.roughness = 0.45;
      }
      const rock = new THREE.Mesh(src.geometry, material);
      rock.castShadow = rock.receiveShadow = true;
      const angle = (i / 4) * Math.PI * 2 + p.x;
      rock.position.set(p.x + Math.cos(angle) * 0.45, 0, p.z + Math.sin(angle) * 0.45);
      fitToTiles(rock, 0.7 + (i % 2) * 0.25);
      rock.position.x = p.x + Math.cos(angle) * 0.45;
      rock.position.z = p.z + Math.sin(angle) * 0.45;
      rock.rotation.y = angle;
      group.add(rock);
    }
  }

  // ---- Villagers and animals (instanced, animated) ----------------------------------
  const villagers = layout.placements.filter((p) => p.kind === "villager");
  const animals = layout.placements.filter((p) => p.kind === "animal");
  const figureMaterial = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.8, flatShading: true });
  const villagerMesh = new THREE.InstancedMesh(villagerGeometry(), figureMaterial, villagers.length);
  villagerMesh.castShadow = true;
  group.add(villagerMesh);
  const deer = animals.filter((a) => a.entity.cls !== "boar");
  const boars = animals.filter((a) => a.entity.cls === "boar");
  const deerMesh = new THREE.InstancedMesh(animalGeometry(new THREE.Color("#a0703f"), false), figureMaterial, Math.max(1, deer.length));
  const boarMesh = new THREE.InstancedMesh(animalGeometry(new THREE.Color("#4d3b2f"), true), figureMaterial, Math.max(1, boars.length));
  deerMesh.count = deer.length;
  boarMesh.count = boars.length;
  deerMesh.castShadow = boarMesh.castShadow = true;
  group.add(deerMesh, boarMesh);

  // Each villager works in place or walks a short beat; the work is chosen from what
  // stands next to it.
  const near = (p: Placement, kind: Placement["kind"], radius: number) =>
    layout.placements.some((q) => q.kind === kind && Math.hypot(q.x - p.x, q.z - p.z) < radius);
  const routines = villagers.map((v, i) => ({
    x: v.x,
    z: v.z,
    heading: (i * 2.399) % (Math.PI * 2),
    work: near(v, "tree", 1.6) || near(v, "mine", 1.6) || near(v, "farm", 1.6),
    phase: i * 0.77,
  }));
  const grazing = animals.map((a, i) => ({ x: a.x, z: a.z, heading: (i * 1.7) % (Math.PI * 2), phase: i * 1.3 }));

  // ---- The new house (Act chapter): ghost, scaffold, stages, finished home -----------
  const site = new THREE.Group();
  const ghost = cloneModel(models, "home_B");
  fitToTiles(ghost, 2 * 0.92);
  ghost.rotation.y = FACING;
  const ghostMaterial = new THREE.MeshBasicMaterial({ color: "#3dff7a", transparent: true, opacity: 0.45, depthWrite: false });
  ghost.traverse((o) => {
    if ((o as THREE.Mesh).isMesh) {
      (o as THREE.Mesh).material = ghostMaterial;
      o.castShadow = false;
    }
  });
  const stages = ["scaffolding", "stage_A", "stage_B", "stage_C", "home_B"].map((name) => {
    const object = cloneModel(models, name);
    fitToTiles(object, 2 * 0.92);
    object.rotation.y = FACING;
    object.visible = false;
    site.add(object);
    return object;
  });
  group.add(ghost);
  const green = layout.spots[2]!;
  site.position.set(green.x, 0, green.z);
  group.add(site);

  function update(time: number, s: { ghost: number; ghostX: number; ghostZ: number; ghostOk: boolean; build: number }) {
    if (windmillFan) (windmillFan as THREE.Object3D).rotation.z = time * 1.2;

    routines.forEach((r, i) => {
      let x = r.x;
      let z = r.z;
      let yaw = r.heading;
      let lean = 0;
      let bob = 0;
      if (r.work) {
        lean = Math.max(0, Math.sin(time * 4 + r.phase)) * 0.35;
      } else {
        const t = Math.sin(time * 0.6 + r.phase);
        x += Math.cos(r.heading) * t * 0.8;
        z += Math.sin(r.heading) * t * 0.8;
        yaw = r.heading + (Math.cos(time * 0.6 + r.phase) > 0 ? 0 : Math.PI);
        bob = Math.abs(Math.sin(time * 7 + r.phase)) * 0.05;
      }
      dummy.position.set(x, bob, z);
      dummy.rotation.set(0, -yaw + Math.PI / 2, lean, "YXZ");
      dummy.scale.setScalar(1);
      dummy.updateMatrix();
      villagerMesh.setMatrixAt(i, dummy.matrix);
    });
    villagerMesh.instanceMatrix.needsUpdate = true;

    grazing.forEach((a, i) => {
      const isBoar = animals[i]!.entity.cls === "boar";
      const t = time * 0.3 + a.phase;
      dummy.position.set(a.x + Math.cos(t) * 0.3, 0, a.z + Math.sin(t) * 0.3);
      dummy.rotation.set(0, -t - Math.PI / 2, 0);
      dummy.scale.setScalar(isBoar ? 1.1 : 1.25);
      dummy.updateMatrix();
      const list = isBoar ? boars : deer;
      const mesh = isBoar ? boarMesh : deerMesh;
      mesh.setMatrixAt(list.indexOf(animals[i]!), dummy.matrix);
    });
    deerMesh.instanceMatrix.needsUpdate = true;
    boarMesh.instanceMatrix.needsUpdate = true;

    // Ghost: red until the preview accepts the spot, then green.
    ghost.visible = s.ghost > 0.01;
    ghost.position.set(s.ghostX, 0.05 + Math.sin(time * 6) * 0.04, s.ghostZ);
    ghostMaterial.opacity = 0.55 * s.ghost;
    ghostMaterial.color.set(s.ghostOk ? "#3dff7a" : "#ff3b3b");
    // Construction: 0 nothing, 1 scaffold, 2-4 stages, 5 finished.
    const stage = Math.floor(s.build) - 1;
    stages.forEach((object, i) => {
      object.visible = i === Math.min(stage, stages.length - 1) && s.build >= 1;
      if (object.visible) {
        const grow = THREE.MathUtils.clamp((s.build - Math.floor(s.build)) * 3, 0, 1);
        object.scale.y = object.scale.x * (i === stages.length - 1 ? 0.6 + 0.4 * Math.min(1, grow + 0.6) : 1);
      }
    });
  }

  return { group, update, center: layout.center };
}
