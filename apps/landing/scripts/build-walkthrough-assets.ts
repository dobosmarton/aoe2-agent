/**
 * Build public/walkthrough/village.glb: the low-poly models of the home page village,
 * merged into one meshopt-compressed file with one named root node per model.
 *
 * The models are from the KayKit Medieval Hexagon Pack by Kay Lousberg (CC0 1.0,
 * https://github.com/KayKit-Game-Assets/KayKit-Medieval-Hexagon-Pack-1.0). Clone it and
 * point KAYKIT_DIR at the repository root:
 *
 *   git clone --depth 1 https://github.com/KayKit-Game-Assets/KayKit-Medieval-Hexagon-Pack-1.0.git
 *   KAYKIT_DIR=./KayKit-Medieval-Hexagon-Pack-1.0 bun run build:walkthrough-assets
 */
import { Document, NodeIO } from "@gltf-transform/core";
import { ALL_EXTENSIONS } from "@gltf-transform/extensions";
import { dedup, meshopt, mergeDocuments, prune, unpartition, weld } from "@gltf-transform/functions";
import { MeshoptDecoder, MeshoptEncoder } from "meshoptimizer";
import { copyFile, mkdir, stat } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const LANDING = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const KAYKIT = process.env.KAYKIT_DIR;
if (!KAYKIT) throw new Error("Set KAYKIT_DIR to a clone of KayKit-Medieval-Hexagon-Pack-1.0");
const GLTF = resolve(KAYKIT, "addons/kaykit_medieval_hexagon_pack/Assets/gltf");
const OUT_DIR = resolve(LANDING, "public/walkthrough");

// Node name in village.glb -> source file. The scene picks models by these names.
const MODELS: Record<string, string> = {
  castle: "buildings/blue/building_castle_blue.gltf",
  home_A: "buildings/blue/building_home_A_blue.gltf",
  home_B: "buildings/blue/building_home_B_blue.gltf",
  windmill: "buildings/blue/building_windmill_blue.gltf",
  lumbermill: "buildings/blue/building_lumbermill_blue.gltf",
  mine: "buildings/blue/building_mine_blue.gltf",
  tower: "buildings/blue/building_tower_A_blue.gltf",
  well: "buildings/blue/building_well_blue.gltf",
  grain: "buildings/neutral/building_grain.gltf",
  scaffolding: "buildings/neutral/building_scaffolding.gltf",
  stage_A: "buildings/neutral/building_stage_A.gltf",
  stage_B: "buildings/neutral/building_stage_B.gltf",
  stage_C: "buildings/neutral/building_stage_C.gltf",
  fence: "buildings/neutral/fence_wood_straight.gltf",
  tree_A: "decoration/nature/tree_single_A.gltf",
  tree_B: "decoration/nature/tree_single_B.gltf",
  trees_A: "decoration/nature/trees_A_medium.gltf",
  trees_B: "decoration/nature/trees_B_medium.gltf",
  trees_A_large: "decoration/nature/trees_A_large.gltf",
  stump: "decoration/nature/tree_single_A_cut.gltf",
  rock_A: "decoration/nature/rock_single_A.gltf",
  rock_B: "decoration/nature/rock_single_B.gltf",
  rock_C: "decoration/nature/rock_single_C.gltf",
  hills: "decoration/nature/hills_A_trees.gltf",
  mountain: "decoration/nature/mountain_A_grass_trees.gltf",
  cloud: "decoration/nature/cloud_big.gltf",
  lumber: "decoration/props/resource_lumber.gltf",
  stone: "decoration/props/resource_stone.gltf",
  wheelbarrow: "decoration/props/wheelbarrow.gltf",
  flag: "decoration/props/flag_blue.gltf",
  crates: "decoration/props/crate_A_big.gltf",
  sack: "decoration/props/sack.gltf",
};

await MeshoptEncoder.ready;
await MeshoptDecoder.ready;
const io = new NodeIO()
  .registerExtensions(ALL_EXTENSIONS)
  .registerDependencies({ "meshopt.encoder": MeshoptEncoder, "meshopt.decoder": MeshoptDecoder });

const target = new Document();
target.createScene("village");

for (const [name, file] of Object.entries(MODELS)) {
  const source = await io.read(resolve(GLTF, file));
  // Wrap each model in one root named after its key, so the scene can clone it whole.
  const scene = source.getRoot().getDefaultScene() ?? source.getRoot().listScenes()[0]!;
  const wrapper = source.createNode(name);
  for (const child of scene.listChildren()) wrapper.addChild(child);
  scene.addChild(wrapper);
  mergeDocuments(target, source);
}

// Gather every merged scene's roots into the first scene.
const [main, ...rest] = target.getRoot().listScenes();
for (const scene of rest) {
  for (const node of scene.listChildren()) main!.addChild(node);
  scene.dispose();
}
target.getRoot().setDefaultScene(main!);

await target.transform(
  unpartition(),
  dedup(),
  weld(),
  prune({ keepLeaves: true }),
  meshopt({ encoder: MeshoptEncoder, level: "medium" }),
);

await mkdir(OUT_DIR, { recursive: true });
const out = resolve(OUT_DIR, "village.glb");
await io.write(out, target);
await copyFile(resolve(KAYKIT, "LICENSE.txt"), resolve(OUT_DIR, "village-LICENSE.txt")).catch(() =>
  copyFile(resolve(KAYKIT, "addons/kaykit_medieval_hexagon_pack/LICENSE.txt"), resolve(OUT_DIR, "village-LICENSE.txt")),
);
const { size } = await stat(out);
console.log(`village.glb: ${Object.keys(MODELS).length} models, ${(size / 1024).toFixed(0)} KB`);
