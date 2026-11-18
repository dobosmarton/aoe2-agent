import type * as THREE from "three";

import type { Placement } from "./layout";
import type { SpriteHandle, SpriteLayer } from "./sprites";

// The village economy. Every villager has a job, as in the game: it works at a tree,
// a gold or stone mine, or its farm until it carries a full load, walks the load to
// the nearest building that accepts it (lumber camp, mining camp, mill, or the Town
// Center, which accepts everything), drops it, and walks back. Walks follow a path
// around buildings, trees and mines. Each villager starts its loop where the detector
// saw it: at work when it stood at a resource, otherwise on its way along the path
// it stood on. The loop is shortened for the page; a real load takes 20 to 30 s.

type Point = { x: number; z: number };
type Load = "wood" | "gold" | "stone" | "food";
type Task = "chop" | "mine" | "farm";
type Gender = "m" | "f";

/** Tiles per second, a villager's speed in the game. */
const SPEED = 0.8;
/** Tiles per full walk cycle. */
const STRIDE = 1.3;
/** Frames per animation in the atlas. */
const FRAMES = 12;
/** Work animation playback, frames per second. */
const FPS = 12;
/** Seconds at the drop site. */
const DROP = 0.5;
/** Path grid cell, in tiles. */
const CELL = 0.25;
/** A villager this close to a resource (tiles) is working it. */
const AT_RESOURCE = 1.8;
/** Added to a villager's distance to a farm when it is already gathering, so it farms only when no one else is left. */
const GATHERER_PENALTY = 100;
/** Path cost per cell of crossing another villager's field. */
const FOREIGN_FIELD_COST = 60;
/** Path cost of a diagonal step (about √2). */
const DIAGONAL_STEP = 1.414;
/** Routes shorter than this (tiles) are worked standing still. */
const MIN_ROUTE = 0.05;
/** Every third villager is a woman. */
const FEMALE_EVERY = 3;

const GATHERED_LOADS: readonly Load[] = ["wood", "gold", "stone"];

const DROP_SITES: Record<Load, string[]> = {
  wood: ["lumber_camp", "town_center"],
  gold: ["mining_camp", "town_center"],
  stone: ["mining_camp", "town_center"],
  food: ["mill", "town_center"],
};

const IS_RESOURCE: Record<Load, (p: Placement) => boolean> = {
  wood: (p) => p.kind === "tree",
  gold: (p) => p.entity.cls === "gold_mine",
  stone: (p) => p.entity.cls === "stone_mine",
  food: () => false,
};

/** A villager the Track chapter can follow. */
export type TrackedVillager = {
  /** Live sprite anchor. */
  anchor: THREE.Vector3;
  /** Route length in tiles. */
  route: number;
  /** Walking at the last update. */
  walking: boolean;
};

/** Debug: per farm, the farmers assigned and every villager standing on it now; the number of villagers per load; each route's length. */
export type VillagerCheck = {
  farms: { farmers: number; standing: number }[];
  jobs: Record<string, number>;
  routes: number[];
};

export type Villagers = {
  update: (time: number) => void;
  tracked: TrackedVillager[];
  check: () => VillagerCheck;
};

/** AoE2 facing for a ground direction: the game stores 16 facings clockwise on
 * screen from facing right (0), through facing the viewer (4), left (8) and away (12). */
const facingFor = (dx: number, dz: number): number => {
  // Ground to screen: right is x - z, down is (x + z) / 2.
  const angle = Math.atan2((dx + dz) / 2, dx - dz);
  return ((Math.round(angle / (Math.PI / 8)) % 16) + 16) % 16;
};

/** The atlas keeps four facings for work animations and eight for carrying. */
const WORK_FACINGS = [2, 6, 10, 14];
const workFacing = (facing: number): number => {
  const gap = (k: number): number => Math.min(Math.abs(k - facing), 16 - Math.abs(k - facing));
  return WORK_FACINGS.reduce((best, k) => (gap(k) < gap(best) ? k : best), 2);
};
const carryFacing = (facing: number): number => (Math.round(facing / 2) * 2) % 16;

const dist = (a: Point, b: Point): number => Math.hypot(a.x - b.x, a.z - b.z);

const nearestOf = <T extends Point>(list: readonly T[], p: Point): T | null =>
  list.reduce<T | null>((best, q) => (best === null || dist(q, p) < dist(best, p) ? q : best), null);

const lengthOf = (path: readonly Point[]): number => path.reduce((sum, q, k) => (k ? sum + dist(path[k - 1], q) : 0), 0);

// ---- Path grid ---------------------------------------------------------------------------

type Heap = {
  size: () => number;
  push: (cell: number, score: number) => void;
  /** Removes and returns the lowest-scored cell. */
  pop: () => number;
};

/** Binary min-heap of cells by score. */
const createHeap = (): Heap => {
  const cells: number[] = [];
  const scores: number[] = [];
  const swap = (a: number, b: number): void => {
    [cells[a], cells[b]] = [cells[b], cells[a]];
    [scores[a], scores[b]] = [scores[b], scores[a]];
  };
  const siftUp = (start: number): void => {
    for (let i = start; i > 0; ) {
      const up = (i - 1) >> 1;
      if (scores[up] <= scores[i]) break;
      swap(i, up);
      i = up;
    }
  };
  const siftDown = (start: number): void => {
    for (let i = start; ; ) {
      const left = 2 * i + 1;
      const right = left + 1;
      let smallest = i;
      if (left < cells.length && scores[left] < scores[smallest]) smallest = left;
      if (right < cells.length && scores[right] < scores[smallest]) smallest = right;
      if (smallest === i) break;
      swap(i, smallest);
      i = smallest;
    }
  };
  return {
    size: () => cells.length,
    push: (cell: number, score: number): void => {
      cells.push(cell);
      scores.push(score);
      siftUp(cells.length - 1);
    },
    pop: (): number => {
      const top = cells[0];
      const lastCell = cells.pop();
      const lastScore = scores.pop();
      if (lastCell === undefined || lastScore === undefined) throw new Error("pop from an empty heap");
      if (cells.length) {
        cells[0] = lastCell;
        scores[0] = lastScore;
        siftDown(0);
      }
      return top;
    },
  };
};

type PathGrid = {
  /** A walk from `from` to `to` around obstacles, as few straight legs as possible. */
  path: (from: Point, to: Point, farm: number) => Point[];
  farms: Placement[];
  /** Whether a villager working `farm` (or -1) may stand at p. */
  open: (p: Point, farm: number) => boolean;
  /** The nearest point off buildings, trees, mines and fields. */
  nearestOpen: (p: Point) => Point;
};

// `farm` below is the index of the field a villager works, or -1: it may cross its own.
const createGrid = (placements: readonly Placement[], corners: readonly Point[]): PathGrid => {
  const x0 = Math.min(...corners.map((c) => c.x)) - 1;
  const z0 = Math.min(...corners.map((c) => c.z)) - 1;
  const nx = Math.ceil((Math.max(...corners.map((c) => c.x)) + 1 - x0) / CELL);
  const nz = Math.ceil((Math.max(...corners.map((c) => c.z)) + 1 - z0) / CELL);
  const blocked = new Uint8Array(nx * nz);
  const farmOf = new Int16Array(nx * nz).fill(-1);
  const farms = placements.filter((p) => p.kind === "farm");

  const mark = (p: Placement, half: number, apply: (cell: number) => void): void => {
    const i0 = Math.max(0, Math.floor((p.x - half - x0) / CELL));
    const i1 = Math.min(nx - 1, Math.floor((p.x + half - x0) / CELL));
    const k0 = Math.max(0, Math.floor((p.z - half - z0) / CELL));
    const k1 = Math.min(nz - 1, Math.floor((p.z + half - z0) / CELL));
    for (let k = k0; k <= k1; k++) for (let i = i0; i <= i1; i++) apply(k * nx + i);
  };
  const block = (cell: number): void => {
    blocked[cell] = 1;
  };
  for (const p of placements) {
    if (p.kind === "building") mark(p, p.size / 2 + 0.1, block);
    else if (p.kind === "tree") mark(p, 0.4, block);
    else if (p.kind === "mine") mark(p, 0.6, block);
  }
  farms.forEach((f, n) => mark(f, f.size / 2, (cell) => (farmOf[cell] = n)));

  const isBlocked = (cell: number): boolean => blocked[cell] === 1;
  const cellOf = (p: Point): number => {
    const i = Math.min(nx - 1, Math.max(0, Math.floor((p.x - x0) / CELL)));
    const k = Math.min(nz - 1, Math.max(0, Math.floor((p.z - z0) / CELL)));
    return k * nx + i;
  };
  const centre = (cell: number): Point => ({ x: x0 + ((cell % nx) + 0.5) * CELL, z: z0 + (Math.floor(cell / nx) + 0.5) * CELL });
  // Fields can be crossed in the game, but paths here keep off other villagers' fields
  // unless there is no other way.
  const cost = (cell: number, farm: number): number => (farmOf[cell] >= 0 && farmOf[cell] !== farm ? FOREIGN_FIELD_COST : 1);
  const open = (p: Point, farm: number): boolean => {
    const cell = cellOf(p);
    return !isBlocked(cell) && (farmOf[cell] === -1 || farmOf[cell] === farm);
  };

  /** The nearest free cell, searching outwards. */
  const free = (p: Point): number => {
    const start = cellOf(p);
    if (!isBlocked(start)) return start;
    const seen = new Set([start]);
    const queue = [start];
    for (let cell = queue.shift(); cell !== undefined; cell = queue.shift()) {
      if (!isBlocked(cell)) return cell;
      for (const n of [cell - 1, cell + 1, cell - nx, cell + nx]) {
        if (n >= 0 && n < nx * nz && !seen.has(n)) {
          seen.add(n);
          queue.push(n);
        }
      }
    }
    return start;
  };

  /** Whether a straight walk from a to b stays clear (and off other fields). */
  const clear = (a: Point, b: Point, farm: number): boolean => {
    const steps = Math.ceil(dist(b, a) / (CELL * 0.5));
    for (let s = 1; s < steps; s++) {
      const cell = cellOf({ x: a.x + ((b.x - a.x) * s) / steps, z: a.z + ((b.z - a.z) * s) / steps });
      if (isBlocked(cell) || cost(cell, farm) > 1) return false;
    }
    return true;
  };

  /** A* from start to goal over the grid, 8-connected; returns each reached cell's predecessor (-1 for none). */
  const search = (start: number, goal: number, farm: number): Int32Array => {
    const costSoFar = new Float32Array(nx * nz).fill(Infinity);
    const cameFrom = new Int32Array(nx * nz).fill(-1);
    const gx = goal % nx;
    const gz = Math.floor(goal / nx);
    // Octile distance to the goal, in cells.
    const estimate = (cell: number): number => {
      const dx = Math.abs((cell % nx) - gx);
      const dz = Math.abs(Math.floor(cell / nx) - gz);
      return Math.max(dx, dz) + 0.414 * Math.min(dx, dz);
    };
    const frontier = createHeap();
    frontier.push(start, estimate(start));
    costSoFar[start] = 0;
    while (frontier.size()) {
      const cell = frontier.pop();
      if (cell === goal) break;
      const ci = cell % nx;
      const ck = Math.floor(cell / nx);
      for (let dk = -1; dk <= 1; dk++) {
        for (let di = -1; di <= 1; di++) {
          if (di === 0 && dk === 0) continue;
          const i = ci + di;
          const k = ck + dk;
          if (i < 0 || k < 0 || i >= nx || k >= nz) continue;
          const n = k * nx + i;
          const diagonal = di !== 0 && dk !== 0;
          // No cutting corners past a blocked cell.
          if (isBlocked(n) || (diagonal && (isBlocked(ck * nx + i) || isBlocked(k * nx + ci)))) continue;
          const step = (diagonal ? DIAGONAL_STEP : 1) * cost(n, farm);
          if (costSoFar[cell] + step < costSoFar[n]) {
            costSoFar[n] = costSoFar[cell] + step;
            cameFrom[n] = cell;
            frontier.push(n, costSoFar[n] + estimate(n));
          }
        }
      }
    }
    return cameFrom;
  };

  /** Skips every corner a straight walk can cut. */
  const pullTaut = (raw: readonly Point[], farm: number): Point[] => {
    const taut = [raw[0]];
    for (let a = 0; a < raw.length - 1; ) {
      let b = raw.length - 1;
      while (b > a + 1 && !clear(raw[a], raw[b], farm)) b--;
      taut.push(raw[b]);
      a = b;
    }
    return taut;
  };

  const path = (from: Point, to: Point, farm: number): Point[] => {
    const start = free(from);
    const goal = free(to);
    const cameFrom = search(start, goal, farm);
    const cells: number[] = [];
    for (let cell = goal; cell !== -1; cell = cell === start ? -1 : cameFrom[cell]) cells.unshift(cell);
    if (cells[0] !== start) return [from, to]; // unreachable: walk straight
    return pullTaut([from, ...cells.slice(1, -1).map(centre), to], farm);
  };

  const nearestOpen = (p: Point): Point => {
    if (open(p, -1)) return p;
    for (let r = CELL; r < 6; r += CELL) {
      for (let a = 0; a < 16; a++) {
        const q = { x: p.x + Math.cos((a * Math.PI) / 8) * r, z: p.z + Math.sin((a * Math.PI) / 8) * r };
        if (open(q, -1)) return q;
      }
    }
    return p;
  };

  return { path, farms, open, nearestOpen };
};

// ---- Routes ------------------------------------------------------------------------------

type Routes = {
  resources: (load: Load) => Placement[];
  /** The drop point for a load picked up at p. */
  dropFor: (load: Load, p: Point, farm?: number) => Point;
  /** The trip from the work spot (through `via`) to the drop point it reaches soonest. */
  trip: (work: Point, via: Point | null, load: Load, farm: number) => Point[];
};

const createRoutes = (placements: readonly Placement[], grid: PathGrid): Routes => {
  const resources = (load: Load): Placement[] => placements.filter(IS_RESOURCE[load]);

  /** The drop site for a load picked up at p: the nearest building that accepts it,
   * and the clear points just outside its footprint closest to p (never on someone
   * else's field), nearest first. */
  const dropsFor = (load: Load, p: Point, farm = -1): Point[] => {
    const sites = placements.filter((q) => q.kind === "building" && DROP_SITES[load].includes(q.entity.cls));
    const site = nearestOf(sites, p);
    if (site === null) throw new Error(`No building accepts ${load}`);
    const half = site.size / 2 + 0.3;
    const ring: Point[] = [];
    for (let u = -half; u <= half; u += 0.25) {
      ring.push({ x: site.x + u, z: site.z - half }, { x: site.x + u, z: site.z + half }, { x: site.x - half, z: site.z + u }, { x: site.x + half, z: site.z + u });
    }
    const clamped = { x: Math.max(site.x - half, Math.min(site.x + half, p.x)), z: Math.max(site.z - half, Math.min(site.z + half, p.z)) };
    const clearPoints = ring.filter((q) => grid.open(q, farm)).sort((a, b) => dist(a, p) - dist(b, p));
    return clearPoints.length ? clearPoints : [clamped];
  };
  const dropFor = (load: Load, p: Point, farm = -1): Point => dropsFor(load, p, farm)[0];

  const trip = (work: Point, via: Point | null, load: Load, farm: number): Point[] => {
    const from = via ?? work;
    const head = via ? grid.path(work, via, farm) : [work];
    // A few spread-out candidates among the nearest points around the building.
    const candidates = dropsFor(load, from, farm)
      .filter((_, k) => k % 3 === 0)
      .slice(0, 5);
    const tails = candidates.map((drop) => grid.path(from, drop, farm));
    const tail = tails.reduce((best, t) => (lengthOf(t) < lengthOf(best) ? t : best));
    return [...head, ...tail.slice(1)];
  };

  return { resources, dropFor, trip };
};

// ---- Jobs --------------------------------------------------------------------------------

/** Where a villager works and, when it was seen walking, the point of its route it was seen at. */
type WorkPlan = { load: Load; work: Point; facing: number; via: Point | null };

/** Work spot to drop site, with the distance to each point. */
type Route = { points: Point[]; along: number[]; length: number };

type Job = {
  handle: SpriteHandle;
  gender: Gender;
  task: Task;
  load: Load;
  /** Where it works, facing the resource. */
  work: Point;
  facing: number;
  route: Route;
  workTime: number;
  /** Where in its loop it starts, seconds. */
  offset: number;
  farm: Placement | null;
  phase: number;
  tracked: TrackedVillager;
};

/** The resource a villager stands at, nearest first, if any. */
const resourceAt = (v: Point, routes: Routes): { load: Load; resource: Placement } | undefined =>
  GATHERED_LOADS.map((load) => ({ load, resource: nearestOf(routes.resources(load), v) }))
    .filter((o): o is { load: Load; resource: Placement } => o.resource !== null && dist(o.resource, v) < AT_RESOURCE)
    .sort((a, b) => dist(a.resource, v) - dist(b.resource, v))[0];

/** Farms first: each takes the nearest villager not already chopping or mining. */
const assignFarmers = (farms: readonly Placement[], villagers: readonly Placement[], routes: Routes): Map<Placement, Placement> => {
  const farmerOf = new Map<Placement, Placement>();
  const taken = new Set<Placement>();
  const pairs = farms
    .flatMap((farm) => villagers.map((v) => ({ farm, v, d: dist(farm, v) + (resourceAt(v, routes) ? GATHERER_PENALTY : 0) })))
    .sort((a, b) => a.d - b.d);
  for (const { farm, v } of pairs) {
    if (taken.has(farm) || farmerOf.has(v)) continue;
    taken.add(farm);
    farmerOf.set(v, farm);
  }
  return farmerOf;
};

/** A spot a third of the way from the centre to a corner, facing across the field. */
const farmWork = (farm: Placement, i: number): WorkPlan => {
  const corner = (i * 3 + 1) % 4;
  const ox = (corner & 1 ? 1 : -1) * farm.size * 0.17;
  const oz = (corner & 2 ? 1 : -1) * farm.size * 0.17;
  return { load: "food", work: { x: farm.x + ox, z: farm.z + oz }, facing: workFacing(facingFor(-ox, -oz)), via: null };
};

/** The gathering job whose trip passes closest to where a villager stood. */
const closestDetour = (v: Point, routes: Routes): { load: Load; resource: Placement } =>
  GATHERED_LOADS.flatMap((load) =>
    routes.resources(load).map((resource) => {
      const drop = routes.dropFor(load, resource);
      return { load, resource, detour: dist(resource, v) + dist(v, drop) - dist(resource, drop) };
    }),
  ).sort((a, b) => a.detour - b.detour)[0];

/**
 * At a resource, that is the job. Seen in the open, it is on its way: the job whose
 * trip passes closest to where it stood, with the route bent through it. `crowd`
 * counts the villagers already at each resource, so they spread around it.
 */
const gatherWork = (v: Placement, routes: Routes, grid: PathGrid, crowd: Map<Placement, number>): WorkPlan => {
  const atResource = resourceAt(v, routes);
  const { load, resource } = atResource ?? closestDetour(v, routes);
  const via = atResource ? null : grid.nearestOpen(v);
  // Stand at the side of the resource facing the drop site, fanned out when shared.
  const sharers = crowd.get(resource) ?? 0;
  crowd.set(resource, sharers + 1);
  const drop = routes.dropFor(load, resource);
  const toward = Math.atan2(drop.z - resource.z, drop.x - resource.x) + (sharers % 2 ? 1 : -1) * Math.ceil(sharers / 2) * 0.9;
  const reach = load === "wood" ? 0.55 : 0.8;
  const work = { x: resource.x + Math.cos(toward) * reach, z: resource.z + Math.sin(toward) * reach };
  return { load, work, facing: workFacing(facingFor(resource.x - work.x, resource.z - work.z)), via };
};

const createRoute = (points: Point[]): Route => {
  const along = [0];
  for (let k = 1; k < points.length; k++) along.push(along[k - 1] + dist(points[k - 1], points[k]));
  return { points, along, length: along[along.length - 1] };
};

/** Where in its loop a villager starts: at work, or where it was seen on its route. */
const startOffset = (route: Route, via: Point | null, i: number, workTime: number): number => {
  if (via) {
    // Half are seen carrying a load in, half heading back out.
    const seenAt = route.points.find((p) => p.x === via.x && p.z === via.z) ?? route.points[0];
    const s = route.along[route.points.indexOf(seenAt)];
    const walk = route.length / SPEED;
    return i % 2 === 0 ? workTime + s / SPEED : workTime + walk + DROP + (route.length - s) / SPEED;
  }
  return (i * 1.37) % workTime;
};

/** Puts `out` at distance d along the route and returns the leg it lies on. */
const placeAlong = (route: Route, d: number, out: THREE.Vector3): [Point, Point] => {
  let k = 1;
  while (k < route.points.length - 1 && route.along[k] < d) k++;
  const a = route.points[k - 1];
  const b = route.points[k];
  const leg = route.along[k] - route.along[k - 1] || 1;
  const t = Math.max(0, Math.min(1, (d - route.along[k - 1]) / leg));
  out.set(a.x + (b.x - a.x) * t, 0, a.z + (b.z - a.z) * t);
  return [a, b];
};

/** Position `s` along the route from the work spot; returns the facing. */
const outboundAt = (route: Route, s: number, out: THREE.Vector3): number => {
  const [a, b] = placeAlong(route, s, out);
  return facingFor(b.x - a.x, b.z - a.z);
};

/** Position `s` along the route back from the drop site; returns the facing. */
const returnAt = (route: Route, s: number, out: THREE.Vector3): number => {
  const [a, b] = placeAlong(route, route.length - s, out);
  return facingFor(a.x - b.x, a.z - b.z);
};

const updateJob = (job: Job, time: number): void => {
  const { handle, route } = job;
  const walk = route.length / SPEED;
  const loop = job.workTime + 2 * walk + DROP;
  const cycleTime = (time + job.offset) % loop;
  const working = cycleTime < job.workTime || route.length < MIN_ROUTE;
  job.tracked.walking = !working;
  if (working) {
    handle.anchor.set(job.work.x, 0, job.work.z);
    handle.name = `vil${job.gender}_${job.task}_d${job.facing}`;
    handle.frame = Math.floor(time * FPS + job.phase * 10);
  } else if (cycleTime < job.workTime + walk + DROP) {
    // Carrying the load to the drop site, then handing it over.
    const walked = Math.min(cycleTime - job.workTime, walk) * SPEED;
    const facing = outboundAt(route, walked, handle.anchor);
    handle.name = `vil${job.gender}_carry_${job.load}_d${carryFacing(facing)}`;
    handle.frame = cycleTime - job.workTime < walk ? Math.floor((walked / STRIDE) * FRAMES) % FRAMES : 0;
  } else {
    // Walking back empty-handed.
    const walked = (cycleTime - job.workTime - walk - DROP) * SPEED;
    const facing = returnAt(route, walked, handle.anchor);
    handle.name = `vil${job.gender}_walk_d${facing}`;
    handle.frame = Math.floor((walked / STRIDE) * FRAMES) % FRAMES;
  }
};

const checkJobs = (jobs: readonly Job[], farms: readonly Placement[]): VillagerCheck => ({
  farms: farms.map((f) => ({
    farmers: jobs.filter((j) => j.farm === f).length,
    standing: jobs.filter((j) => Math.abs(j.handle.anchor.x - f.x) < f.size / 2 && Math.abs(j.handle.anchor.z - f.z) < f.size / 2).length,
  })),
  jobs: jobs.reduce<Record<string, number>>((n, j) => ({ ...n, [j.load]: (n[j.load] ?? 0) + 1 }), {}),
  routes: jobs.map((j) => Number(j.route.length.toFixed(1))),
});

export const createVillagers = (placements: readonly Placement[], corners: readonly Point[], sprites: Pick<SpriteLayer, "add">): Villagers => {
  const grid = createGrid(placements, corners);
  const routes = createRoutes(placements, grid);
  const villagers = placements.filter((p) => p.kind === "villager");
  const farms = grid.farms;
  const farmerOf = assignFarmers(farms, villagers, routes);
  const crowd = new Map<Placement, number>();

  const jobs = villagers.map((v, i): Job => {
    const gender: Gender = i % FEMALE_EVERY === 0 ? "f" : "m";
    const farm = farmerOf.get(v) ?? null;
    const { load, work, facing, via } = farm ? farmWork(farm, i) : gatherWork(v, routes, grid, crowd);
    const route = createRoute(routes.trip(work, via, load, farm ? farms.indexOf(farm) : -1));
    const workTime = 5 + (i % 4) * 0.9;
    const offset = startOffset(route, via, i, workTime);
    const task: Task = farm ? "farm" : load === "wood" ? "chop" : "mine";
    const handle = sprites.add(`vil${gender}_${task}_d${facing}`, work.x, 0, work.z, i * 3);
    const tracked = { anchor: handle.anchor, route: route.length, walking: false };
    return { handle, gender, task, load, work, facing, route, workTime, offset, farm, phase: i * 0.77, tracked };
  });

  const update = (time: number): void => {
    for (const job of jobs) updateJob(job, time);
  };

  return { update, tracked: jobs.map((job) => job.tracked), check: () => checkJobs(jobs, farms) };
};
