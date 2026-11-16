import type { TurnData, TurnEntity } from "./types";

// The recorded frame turned back into a ground plan. AoE2 draws the map in a 2:1
// dimetric projection: a tile is `tw` pixels wide and tw/2 tall on screen, and the
// Town Center is 4 tiles wide (the rule executor.py `_footprint` uses), so its
// detected width gives tw. Screen offsets from the frame centre in half-tile units
// (u, v) map to ground tiles as x = (u + v) / 2, z = (v - u) / 2. World units are
// tiles; Y is up. Screen right is +x -z and screen down is +x +z, so the game
// camera sits on the +x +z side at 30 degrees elevation, looking towards -x -z.

export type Placement = {
  entity: TurnEntity;
  kind: "building" | "farm" | "tree" | "mine" | "villager" | "animal" | "other";
  model?: string;
  x: number;
  z: number;
  /** Footprint edge in tiles (buildings) or a size hint. */
  size: number;
};

// Footprint edge in tiles per class, and the village model that stands in for it.
const BUILDINGS: Record<string, { tiles: number; model: string }> = {
  town_center: { tiles: 4, model: "castle" },
  house: { tiles: 2, model: "home_A" },
  mill: { tiles: 2, model: "windmill" },
  lumber_camp: { tiles: 2, model: "lumbermill" },
  mining_camp: { tiles: 2, model: "mine" },
  tower: { tiles: 1, model: "tower" },
  farm: { tiles: 3, model: "" },
};

export type Layout = ReturnType<typeof buildLayout>;

export function buildLayout(data: TurnData) {
  const W = data.frame.width;
  const H = data.frame.height;
  const tc = data.entities.find((e) => e.cls === "town_center")!;
  const tw = (tc.w * W) / 4;

  /** Frame pixel to ground tiles. */
  const toGround = (sx: number, sy: number) => {
    const u = (sx - W / 2) / (tw / 2);
    const v = (sy - H / 2) / (tw / 4);
    return { x: (u + v) / 2, z: (v - u) / 2 };
  };
  /** Ground tiles to frame pixel (for checks and the screen overlay). */
  const toScreen = (x: number, z: number) => {
    const u = x - z;
    const v = x + z;
    return { sx: W / 2 + (u * tw) / 2, sy: H / 2 + (v * tw) / 4 };
  };

  // Seeded random, so the village is the same on every load.
  let seed = 7;
  const rand = () => {
    seed = (seed * 16807) % 2147483647;
    return (seed - 1) / 2147483646;
  };

  const placements: Placement[] = [];
  let houseIndex = 0;
  for (const e of data.entities) {
    const left = e.x * W;
    const top = e.y * H;
    const width = e.w * W;
    const height = e.h * H;
    const bottom = top + height;
    const cx = left + width / 2;
    const building = BUILDINGS[e.cls];
    if (building) {
      // The footprint diamond's lowest corner sits at the box bottom.
      const g = toGround(cx, bottom - (building.tiles * tw) / 4);
      const model = e.cls === "house" ? (houseIndex++ % 2 ? "home_B" : "home_A") : building.model || undefined;
      placements.push({ entity: e, kind: e.cls === "farm" ? "farm" : "building", model, ...g, size: building.tiles });
    } else if (e.cls === "tree") {
      // A tree box covers a stand of trees: scatter trunks over its lower part.
      const count = Math.max(1, Math.min(26, Math.round((width * height) / (tw * tw * 0.55))));
      for (let i = 0; i < count; i++) {
        const sx = left + width * (0.08 + rand() * 0.84);
        const sy = top + height * (0.35 + rand() * 0.62);
        if (sy > 0.83 * H - tw * 0.3 || sy < 0.06 * H + tw * 0.2) continue;
        placements.push({ entity: e, kind: "tree", ...toGround(sx, sy), size: 0.8 + rand() * 0.5 });
      }
    } else if (e.cls === "gold_mine" || e.cls === "stone_mine") {
      const g = toGround(cx, bottom - tw * 0.3);
      placements.push({ entity: e, kind: "mine", ...g, size: Math.max(1, width / tw) });
    } else if (e.cat === "unit") {
      placements.push({ entity: e, kind: "villager", ...toGround(cx, bottom - tw * 0.08), size: 1 });
    } else if (e.cat === "animal") {
      placements.push({ entity: e, kind: "animal", ...toGround(cx, bottom - tw * 0.12), size: 1 });
    } else {
      placements.push({ entity: e, kind: "other", ...toGround(cx, bottom - tw * 0.3), size: 1 });
    }
  }

  // The part of the map the frame shows (between the resource bar and the panel).
  const playTop = 0.06 * H;
  const playBottom = 0.83 * H;
  const corners = [
    toGround(0, playTop),
    toGround(W, playTop),
    toGround(W, playBottom),
    toGround(0, playBottom),
  ];
  const center = toGround(W / 2, (playTop + playBottom) / 2);
  const tcPlacement = placements.find((p) => p.entity === tc)!;

  // House spots for the Act chapter: ring points around the Town Center, as in
  // executor.py `placement_candidates`. A 2x2 footprint that overlaps a building,
  // field, tree or mine shows the red preview; the first clear one is clicked.
  const blockers = placements.filter((p) => p.kind === "building" || p.kind === "farm" || p.kind === "tree" || p.kind === "mine");
  const blocked = (x: number, z: number) =>
    blockers.some((b) => {
      const reach = 1 + (b.kind === "tree" ? 0.45 : b.kind === "mine" ? 0.8 : b.size / 2);
      return Math.abs(x - b.x) < reach && Math.abs(z - b.z) < reach;
    });
  const inPlay = (x: number, z: number) => {
    const { sx, sy } = toScreen(x, z);
    return sx > 0.05 * W && sx < 0.78 * W && sy > playTop + tw && sy < playBottom - tw;
  };
  const clear: { x: number; z: number }[] = [];
  const taken: { x: number; z: number }[] = [];
  for (const radius of [4.5, 6, 7.5, 9]) {
    for (let k = 0; k < 16; k++) {
      const angle = (k / 16) * Math.PI * 2 + 0.4;
      const x = tcPlacement.x + Math.cos(angle) * radius;
      const z = tcPlacement.z + Math.sin(angle) * radius;
      if (!inPlay(x, z)) continue;
      (blocked(x, z) ? taken : clear).push({ x, z });
    }
  }
  // The clicked spot is the first clear one; the two refused hovers before it are the
  // taken candidates nearest to it, so all three fit in one shot.
  const green = clear[0] ?? { x: tcPlacement.x + 5, z: tcPlacement.z + 5 };
  const near = taken
    .map((s) => ({ s, d: Math.hypot(s.x - green.x, s.z - green.z) }))
    .filter((c) => c.d > 2.2)
    .sort((a, b) => a.d - b.d);
  const red: { x: number; z: number }[] = [];
  for (const c of near) {
    if (red.length < 2 && red.every((r) => Math.hypot(r.x - c.s.x, r.z - c.s.z) > 2.2)) red.push(c.s);
  }
  while (red.length < 2) red.push({ x: green.x - 3 + red.length * 6, z: green.z - 3 });
  const spots = [red[0]!, red[1]!, green];

  return { tw, toGround, toScreen, placements, corners, center, tc: tcPlacement, rand, playTop, playBottom, W, H, spots };
}
