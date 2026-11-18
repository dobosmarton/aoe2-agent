import type { TurnData, TurnEntity } from "./types";

// The recorded frame turned back into a ground plan. AoE2 draws the map in a 2:1
// dimetric projection: a tile is `tw` pixels wide and tw/2 tall on screen. Matching
// the game's sprites against the frame (scripts/build_walkthrough_sprites.py) gives
// each building, mine and tree its exact anchor and measures tw from the tile grid
// those anchors sit on; without that, the Town Center's detected width (4 tiles)
// estimates it. Screen offsets from the frame centre in half-tile units (u, v) map to
// ground tiles as x = (u + v) / 2, z = (v - u) / 2. World units are tiles; Y is up.
// Screen right is +x -z and screen down is +x +z, so the game camera sits on the
// +x +z side at 30 degrees elevation, looking towards -x -z.

export type Placement = {
  entity: TurnEntity;
  kind: "building" | "farm" | "tree" | "mine" | "villager" | "animal" | "other";
  x: number;
  z: number;
  /** Footprint edge in tiles (buildings) or a size hint. */
  size: number;
  /** The game sprite matched at this spot in the frame, when there is one. */
  sprite?: string;
};

// Footprint edge in tiles per class.
const BUILDINGS: Record<string, number> = {
  town_center: 4,
  house: 2,
  mill: 2,
  lumber_camp: 2,
  mining_camp: 2,
  tower: 1,
  farm: 3,
};

// The part of the frame that shows the map, as fractions of its height: below the
// resource bar and above the bottom panel.
const PLAY_TOP = 0.06;
const PLAY_BOTTOM = 0.83;
const RANDOM_SEED = 7;
// House spots: candidates on rings around the Town Center, this many per ring.
const SPOT_RADII = [4.5, 6, 7.5, 9];
const SPOTS_PER_RING = 16;
/** Least distance between the clicked spot and each refused one, and between the refused two. */
const SPOT_SEPARATION = 2.2;

export type GroundPoint = { x: number; z: number };
export type ScreenPoint = { sx: number; sy: number };

export type Layout = {
  /** Tile width in frame pixels. */
  tw: number;
  /** Frame pixel to ground tiles. */
  toGround: (sx: number, sy: number) => GroundPoint;
  /** Ground tiles to frame pixel (for checks and the screen overlay). */
  toScreen: (x: number, z: number) => ScreenPoint;
  placements: Placement[];
  /** The ground under the frame's play area: top-left, top-right, bottom-right, bottom-left. */
  corners: [GroundPoint, GroundPoint, GroundPoint, GroundPoint];
  center: GroundPoint;
  /** The Town Center's placement. */
  tc: Placement;
  /** Seeded random 0..1; the village is the same on every load. */
  rand: () => number;
  /** Frame pixel rows that bound the play area. */
  playTop: number;
  playBottom: number;
  /** Frame size in pixels. */
  W: number;
  H: number;
  /** House spots for the Act chapter: the two refused hovers, then the clicked spot. */
  spots: [GroundPoint, GroundPoint, GroundPoint];
};

/** Park-Miller minimal standard generator: 0..1, the same sequence for the same seed. */
const createRandom = (seed: number): (() => number) => {
  let state = seed;

  return (): number => {
    state = (state * 16807) % 2147483647;

    return (state - 1) / 2147483646;
  };
};

type Frame = {
  data: TurnData;
  W: number;
  H: number;
  tw: number;
  playTop: number;
  playBottom: number;
  toGround: Layout["toGround"];
  rand: () => number;
};

type MatchedSprite = { sprite: string; x: number; y: number };

// Buildings sit on whole tiles: an even footprint is anchored on a tile corner, an
// odd one (tower, farm) on a tile centre.
const snap = (g: GroundPoint, tiles: number): GroundPoint => {
  const half = tiles % 2 ? 0.5 : 0;

  return { x: Math.round(g.x - half) + half, z: Math.round(g.z - half) + half };
};

const matchedSprites = (data: TurnData): Map<string, MatchedSprite[]> => {
  const matched = new Map<string, MatchedSprite[]>();

  for (const item of data.world?.items ?? []) {
    matched.set(item.id, [...(matched.get(item.id) ?? []), item]);
  }

  return matched;
};

/** Where each detection stands on the ground (a tree box becomes a stand of trees). */
const placeEntities = (frame: Frame): Placement[] => {
  const { data, W, H, tw, playTop, playBottom, toGround, rand } = frame;
  const matched = matchedSprites(data);
  const placements: Placement[] = [];

  for (const e of data.entities) {
    const left = e.x * W;
    const top = e.y * H;
    const width = e.w * W;
    const height = e.h * H;
    const bottom = top + height;
    const cx = left + width / 2;
    const tiles: number | undefined = BUILDINGS[e.cls];
    const hits = matched.get(e.id);

    if (tiles) {
      // A matched anchor is exact; otherwise the footprint diamond's lowest corner
      // sits at the box bottom (farm boxes also hold the crops above it).
      const hit = hits?.[0];
      const g = hit ? toGround(hit.x, hit.y) : toGround(cx, bottom - (tiles * tw) / 4);

      placements.push({ entity: e, kind: e.cls === "farm" ? "farm" : "building", ...snap(g, tiles), size: tiles, sprite: hit?.sprite });
    } else if (e.cls === "tree" && hits?.length) {
      // Trees under the bottom panel are off the island.
      for (const hit of hits) {
        if (hit.y > playBottom - tw * 0.2 || hit.y < playTop) {
          continue;
        }

        placements.push({ entity: e, kind: "tree", ...toGround(hit.x, hit.y), size: 1, sprite: hit.sprite });
      }
    } else if (e.cls === "tree") {
      // A tree box covers a stand of trees: fill its lower part densely, as forests are.
      const count = Math.max(1, Math.min(70, Math.round((width * height) / (tw * tw * 0.3))));

      for (let i = 0; i < count; i++) {
        const sx = left + width * (0.08 + rand() * 0.84);
        const sy = top + height * (0.35 + rand() * 0.62);

        if (sy > playBottom - tw * 0.3 || sy < playTop + tw * 0.2) {
          continue;
        }

        placements.push({ entity: e, kind: "tree", ...toGround(sx, sy), size: 0.8 + rand() * 0.5 });
      }
    } else if (e.cls === "gold_mine" || e.cls === "stone_mine") {
      const hit = hits?.[0];

      // A duplicate detection of a matched pile.
      if (hit === undefined && matched.size > 0) {
        continue;
      }

      const g = hit ? toGround(hit.x, hit.y) : toGround(cx, bottom - tw * 0.3);

      placements.push({ entity: e, kind: "mine", ...g, size: Math.max(1, width / tw), sprite: hit?.sprite });
    } else if (e.cat === "unit") {
      placements.push({ entity: e, kind: "villager", ...toGround(cx, bottom - tw * 0.08), size: 1 });
    } else if (e.cat === "animal") {
      placements.push({ entity: e, kind: "animal", ...toGround(cx, bottom - tw * 0.12), size: 1 });
    } else {
      placements.push({ entity: e, kind: "other", ...toGround(cx, bottom - tw * 0.3), size: 1 });
    }
  }

  return placements;
};

// Forest the detector has no box for (mostly under the resource bar): trees along
// its visible edge, as a stand fills a tree box.
const placeHiddenForests = (frame: Frame): Placement[] => {
  const { data, W, H, tw, toGround, rand } = frame;
  const placements: Placement[] = [];

  for (const [k, [x0, y0, x1, y1]] of (data.world?.forests ?? []).entries()) {
    const entity = { id: `forest_hidden_${k}`, cls: "tree", cat: "resource", conf: null, x: x0 / W, y: y0 / H, w: (x1 - x0) / W, h: (y1 - y0) / H };
    const count = Math.round(((x1 - x0) * (y1 - y0)) / (tw * tw * 0.12));

    for (let i = 0; i < count; i++) {
      placements.push({ entity, kind: "tree", ...toGround(x0 + (x1 - x0) * rand(), y0 + (y1 - y0) * rand()), size: 0.8 + rand() * 0.5 });
    }
  }

  return placements;
};

// House spots for the Act chapter: ring points around the Town Center, as in
// executor.py `placement_candidates`. A 2x2 footprint that overlaps a building,
// field, tree or mine shows the red preview; the first clear one is clicked.
const houseSpots = (frame: Frame, placements: readonly Placement[], tc: Placement, toScreen: Layout["toScreen"]): Layout["spots"] => {
  const { W, tw, playTop, playBottom } = frame;
  const blockers = placements.filter((p) => p.kind === "building" || p.kind === "farm" || p.kind === "tree" || p.kind === "mine");
  const blocked = (x: number, z: number): boolean =>
    blockers.some((b) => {
      const reach = 1 + (b.kind === "tree" ? 0.45 : b.kind === "mine" ? 0.8 : b.size / 2);

      return Math.abs(x - b.x) < reach && Math.abs(z - b.z) < reach;
    });

  const inPlay = (x: number, z: number): boolean => {
    const { sx, sy } = toScreen(x, z);

    return sx > 0.05 * W && sx < 0.78 * W && sy > playTop + tw && sy < playBottom - tw;
  };

  const clear: GroundPoint[] = [];
  const taken: GroundPoint[] = [];

  for (const radius of SPOT_RADII) {
    for (let k = 0; k < SPOTS_PER_RING; k++) {
      const angle = (k / SPOTS_PER_RING) * Math.PI * 2 + 0.4;
      const x = tc.x + Math.cos(angle) * radius;
      const z = tc.z + Math.sin(angle) * radius;

      if (!inPlay(x, z)) {
        continue;
      }

      (blocked(x, z) ? taken : clear).push({ x, z });
    }
  }

  // The clicked spot is the first clear one; the two refused hovers before it are the
  // taken candidates nearest to it, so all three fit in one shot.
  const green = clear[0] ?? { x: tc.x + 5, z: tc.z + 5 };
  const near = taken
    .map((s) => ({ s, d: Math.hypot(s.x - green.x, s.z - green.z) }))
    .filter((c) => c.d > SPOT_SEPARATION)
    .sort((a, b) => a.d - b.d);
  const red: GroundPoint[] = [];

  for (const c of near) {
    if (red.length < 2 && red.every((r) => Math.hypot(r.x - c.s.x, r.z - c.s.z) > SPOT_SEPARATION)) {
      red.push(c.s);
    }
  }

  while (red.length < 2) {
    red.push({ x: green.x - 3 + red.length * 6, z: green.z - 3 });
  }

  return [red[0], red[1], green];
};

export const buildLayout = (data: TurnData): Layout => {
  const W = data.frame.width;
  const H = data.frame.height;
  const tc = data.entities.find((e) => e.cls === "town_center");

  if (!tc) {
    throw new Error("The walkthrough frame has no Town Center");
  }

  const tw = data.world?.tile ?? (tc.w * W) / 4;

  const toGround = (sx: number, sy: number): GroundPoint => {
    const u = (sx - W / 2) / (tw / 2);
    const v = (sy - H / 2) / (tw / 4);

    return { x: (u + v) / 2, z: (v - u) / 2 };
  };

  const toScreen = (x: number, z: number): ScreenPoint => {
    const u = x - z;
    const v = x + z;

    return { sx: W / 2 + (u * tw) / 2, sy: H / 2 + (v * tw) / 4 };
  };

  const playTop = PLAY_TOP * H;
  const playBottom = PLAY_BOTTOM * H;
  const rand = createRandom(RANDOM_SEED);
  const frame: Frame = { data, W, H, tw, playTop, playBottom, toGround, rand };
  const placements = [...placeEntities(frame), ...placeHiddenForests(frame)];

  const corners: Layout["corners"] = [toGround(0, playTop), toGround(W, playTop), toGround(W, playBottom), toGround(0, playBottom)];
  const center = toGround(W / 2, (playTop + playBottom) / 2);
  const tcPlacement = placements.find((p) => p.entity === tc);

  if (!tcPlacement) {
    throw new Error("The Town Center has no placement");
  }

  const spots = houseSpots(frame, placements, tcPlacement, toScreen);

  return { tw, toGround, toScreen, placements, corners, center, tc: tcPlacement, rand, playTop, playBottom, W, H, spots };
};
