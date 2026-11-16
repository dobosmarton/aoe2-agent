import type * as THREE from "three";

import type { Placement } from "./layout";
import type { SpriteHandle } from "./sprites";

// The village economy. Every villager has a job, as in the game: it works at a tree,
// a gold or stone mine, or its farm until it carries a full load, walks the load to
// the nearest building that accepts it (lumber camp, mining camp, mill, or the Town
// Center, which accepts everything), drops it, and walks back. Walks follow a path
// around buildings, trees and mines. Each villager starts its loop where the detector
// saw it: at work when it stood at a resource, otherwise on its way along the path
// it stood on. The loop is shortened for the page; a real load takes 20 to 30 s.

type Point = { x: number; z: number };
type Load = "wood" | "gold" | "stone" | "food";
type Sprites = { add(name: string, x: number, y: number, z: number, frame?: number): SpriteHandle };

const SPEED = 0.8; // tiles per second, a villager's speed in the game
const STRIDE = 1.3; // tiles per full walk cycle
const FRAMES = 12; // frames per animation in the atlas
const FPS = 12;
const DROP = 0.5; // seconds at the drop site
const CELL = 0.25; // path grid cell, in tiles

const DROP_SITES: Record<Load, string[]> = {
  wood: ["lumber_camp", "town_center"],
  gold: ["mining_camp", "town_center"],
  stone: ["mining_camp", "town_center"],
  food: ["mill", "town_center"],
};

/** AoE2 facing for a ground direction: the game stores 16 facings clockwise on
 * screen from facing right (0), through facing the viewer (4), left (8) and away (12). */
export function facingFor(dx: number, dz: number) {
  // Ground to screen: right is x - z, down is (x + z) / 2.
  const angle = Math.atan2((dx + dz) / 2, dx - dz);
  return ((Math.round(angle / (Math.PI / 8)) % 16) + 16) % 16;
}

/** The atlas keeps four facings for work animations and eight for carrying. */
const WORK_FACINGS = [2, 6, 10, 14];
function workFacing(f: number) {
  const gap = (k: number) => Math.min(Math.abs(k - f), 16 - Math.abs(k - f));
  return WORK_FACINGS.reduce((best, k) => (gap(k) < gap(best) ? k : best), 2);
}
const carryFacing = (f: number) => (Math.round(f / 2) * 2) % 16;

// ---- Path grid ---------------------------------------------------------------------------

/** Binary min-heap of cells by score. */
function createHeap() {
  const cells: number[] = [];
  const scores: number[] = [];
  const swap = (a: number, b: number) => {
    [cells[a], cells[b]] = [cells[b]!, cells[a]!];
    [scores[a], scores[b]] = [scores[b]!, scores[a]!];
  };
  return {
    size: () => cells.length,
    push(cell: number, score: number) {
      cells.push(cell);
      scores.push(score);
      for (let i = cells.length - 1; i > 0; ) {
        const up = (i - 1) >> 1;
        if (scores[up]! <= scores[i]!) break;
        swap(i, up);
        i = up;
      }
    },
    pop() {
      const top = cells[0]!;
      const lastCell = cells.pop()!;
      const lastScore = scores.pop()!;
      if (cells.length) {
        cells[0] = lastCell;
        scores[0] = lastScore;
        for (let i = 0; ; ) {
          const l = 2 * i + 1;
          const r = l + 1;
          let m = i;
          if (l < cells.length && scores[l]! < scores[m]!) m = l;
          if (r < cells.length && scores[r]! < scores[m]!) m = r;
          if (m === i) break;
          swap(i, m);
          i = m;
        }
      }
      return top;
    },
  };
}

function createGrid(placements: Placement[], corners: Point[]) {
  const x0 = Math.min(...corners.map((c) => c.x)) - 1;
  const z0 = Math.min(...corners.map((c) => c.z)) - 1;
  const nx = Math.ceil((Math.max(...corners.map((c) => c.x)) + 1 - x0) / CELL);
  const nz = Math.ceil((Math.max(...corners.map((c) => c.z)) + 1 - z0) / CELL);
  const blocked = new Uint8Array(nx * nz);
  const farmOf = new Int16Array(nx * nz).fill(-1);
  const farms = placements.filter((p) => p.kind === "farm");

  const mark = (p: Placement, half: number, fn: (i: number) => void) => {
    const i0 = Math.max(0, Math.floor((p.x - half - x0) / CELL));
    const i1 = Math.min(nx - 1, Math.floor((p.x + half - x0) / CELL));
    const k0 = Math.max(0, Math.floor((p.z - half - z0) / CELL));
    const k1 = Math.min(nz - 1, Math.floor((p.z + half - z0) / CELL));
    for (let k = k0; k <= k1; k++) for (let i = i0; i <= i1; i++) fn(k * nx + i);
  };
  for (const p of placements) {
    if (p.kind === "building") mark(p, p.size / 2 + 0.1, (c) => (blocked[c] = 1));
    else if (p.kind === "tree") mark(p, 0.4, (c) => (blocked[c] = 1));
    else if (p.kind === "mine") mark(p, 0.6, (c) => (blocked[c] = 1));
  }
  farms.forEach((f, n) => mark(f, f.size / 2, (c) => (farmOf[c] = n)));

  const cellOf = (p: Point) => {
    const i = Math.min(nx - 1, Math.max(0, Math.floor((p.x - x0) / CELL)));
    const k = Math.min(nz - 1, Math.max(0, Math.floor((p.z - z0) / CELL)));
    return k * nx + i;
  };
  const centre = (c: number): Point => ({ x: x0 + ((c % nx) + 0.5) * CELL, z: z0 + (Math.floor(c / nx) + 0.5) * CELL });
  // Fields can be crossed in the game, but paths here keep off other villagers' fields
  // unless there is no other way.
  const cost = (c: number, farm: number) => (farmOf[c]! >= 0 && farmOf[c] !== farm ? 60 : 1);
  /** Whether a villager working `farm` (or -1) may stand at p. */
  const open = (p: Point, farm: number) => {
    const c = cellOf(p);
    return !blocked[c] && (farmOf[c] === -1 || farmOf[c] === farm);
  };

  /** The nearest free cell, searching outwards. */
  function free(p: Point) {
    const start = cellOf(p);
    if (!blocked[start]) return start;
    const seen = new Set([start]);
    const queue = [start];
    while (queue.length) {
      const c = queue.shift()!;
      if (!blocked[c]) return c;
      for (const n of [c - 1, c + 1, c - nx, c + nx]) {
        if (n >= 0 && n < nx * nz && !seen.has(n)) {
          seen.add(n);
          queue.push(n);
        }
      }
    }
    return start;
  }

  /** Whether a straight walk from a to b stays clear (and off other fields). */
  function clear(a: Point, b: Point, farm: number) {
    const steps = Math.ceil(Math.hypot(b.x - a.x, b.z - a.z) / (CELL * 0.5));
    for (let s = 1; s < steps; s++) {
      const c = cellOf({ x: a.x + ((b.x - a.x) * s) / steps, z: a.z + ((b.z - a.z) * s) / steps });
      if (blocked[c] || cost(c, farm) > 1) return false;
    }
    return true;
  }

  /** A* over the grid, 8-connected, then pulled taut. */
  function path(from: Point, to: Point, farm: number): Point[] {
    const start = free(from);
    const goal = free(to);
    const g = new Float32Array(nx * nz).fill(Infinity);
    const came = new Int32Array(nx * nz).fill(-1);
    const gx = goal % nx;
    const gz = Math.floor(goal / nx);
    const h = (c: number) => {
      const dx = Math.abs((c % nx) - gx);
      const dz = Math.abs(Math.floor(c / nx) - gz);
      return Math.max(dx, dz) + 0.414 * Math.min(dx, dz);
    };
    const frontier = createHeap();
    frontier.push(start, h(start));
    g[start] = 0;
    while (frontier.size()) {
      const c = frontier.pop();
      if (c === goal) break;
      const ci = c % nx;
      const ck = Math.floor(c / nx);
      for (let dk = -1; dk <= 1; dk++) {
        for (let di = -1; di <= 1; di++) {
          if (!di && !dk) continue;
          const i = ci + di;
          const k = ck + dk;
          if (i < 0 || k < 0 || i >= nx || k >= nz) continue;
          const n = k * nx + i;
          // No cutting corners past a blocked cell.
          if (blocked[n] || (di && dk && (blocked[ck * nx + i] || blocked[k * nx + ci]))) continue;
          const step = (di && dk ? 1.414 : 1) * cost(n, farm);
          if (g[c]! + step < g[n]!) {
            g[n] = g[c]! + step;
            came[n] = c;
            frontier.push(n, g[n]! + h(n));
          }
        }
      }
    }
    const cells: number[] = [];
    for (let c = goal; c !== -1; c = c === start ? -1 : came[c]!) cells.unshift(c);
    if (cells[0] !== start) return [from, to]; // unreachable: walk straight
    const raw = [from, ...cells.slice(1, -1).map(centre), to];
    const taut = [raw[0]!];
    for (let a = 0; a < raw.length - 1; ) {
      let b = raw.length - 1;
      while (b > a + 1 && !clear(raw[a]!, raw[b]!, farm)) b--;
      taut.push(raw[b]!);
      a = b;
    }
    return taut;
  }

  /** The nearest point off buildings, trees, mines and fields. */
  function nearestOpen(p: Point): Point {
    if (open(p, -1)) return p;
    for (let r = CELL; r < 6; r += CELL) {
      for (let a = 0; a < 16; a++) {
        const q = { x: p.x + Math.cos((a * Math.PI) / 8) * r, z: p.z + Math.sin((a * Math.PI) / 8) * r };
        if (open(q, -1)) return q;
      }
    }
    return p;
  }

  return { path, farms, open, nearestOpen };
}

// ---- Jobs --------------------------------------------------------------------------------

type Job = {
  handle: SpriteHandle;
  gender: "m" | "f";
  task: "chop" | "mine" | "farm";
  load: Load;
  /** Where it works, facing the resource. */
  work: Point;
  facing: number;
  /** Work spot to drop site, with the distance to each point. */
  path: Point[];
  along: number[];
  length: number;
  workTime: number;
  offset: number;
  farm: Placement | null;
  phase: number;
  /** Walking at the last update. */
  moving: boolean;
};

const dist = (a: Point, b: Point) => Math.hypot(a.x - b.x, a.z - b.z);

export function createVillagers(placements: Placement[], corners: Point[], sprites: Sprites) {
  const grid = createGrid(placements, corners);
  const villagers = placements.filter((p) => p.kind === "villager");
  const farms = grid.farms;
  const resources = (load: Load) =>
    placements.filter((p) =>
      load === "wood" ? p.kind === "tree" : load === "gold" ? p.entity.cls === "gold_mine" : load === "stone" ? p.entity.cls === "stone_mine" : false,
    );
  const nearestOf = <T extends Point>(list: T[], p: Point) => list.reduce<T | null>((best, q) => (!best || dist(q, p) < dist(best, p) ? q : best), null);

  /** The drop site for a load picked up at p: the nearest building that accepts it,
   * and the clear points just outside its footprint closest to p (never on someone
   * else's field), nearest first. */
  function dropsFor(load: Load, p: Point, farm = -1): Point[] {
    const sites = placements.filter((q) => q.kind === "building" && DROP_SITES[load].includes(q.entity.cls));
    const site = nearestOf(sites, p)!;
    const half = site.size / 2 + 0.3;
    const ring: Point[] = [];
    for (let u = -half; u <= half; u += 0.25) {
      ring.push({ x: site.x + u, z: site.z - half }, { x: site.x + u, z: site.z + half }, { x: site.x - half, z: site.z + u }, { x: site.x + half, z: site.z + u });
    }
    const clamped = { x: Math.max(site.x - half, Math.min(site.x + half, p.x)), z: Math.max(site.z - half, Math.min(site.z + half, p.z)) };
    const open = ring.filter((q) => grid.open(q, farm)).sort((a, b) => dist(a, p) - dist(b, p));
    return open.length ? open : [clamped];
  }
  const dropFor = (load: Load, p: Point, farm = -1) => dropsFor(load, p, farm)[0]!;

  const lengthOf = (path: Point[]) => path.reduce((sum, q, k) => (k ? sum + dist(path[k - 1]!, q) : 0), 0);
  /** The trip from the work spot (through `via`) to the drop point it reaches soonest. */
  function trip(work: Point, via: Point | null, load: Load, farm: number) {
    const from = via ?? work;
    const head = via ? grid.path(work, via, farm) : [work];
    // A few spread-out candidates among the nearest points around the building.
    const candidates = dropsFor(load, from, farm).filter((_, k) => k % 3 === 0).slice(0, 5);
    const tails = candidates.map((drop) => grid.path(from, drop, farm));
    const tail = tails.reduce((best, t) => (lengthOf(t) < lengthOf(best) ? t : best));
    return [...head, ...tail.slice(1)];
  }

  // Farms first: each takes the nearest villager not already chopping or mining.
  const gathering = (v: Placement) =>
    (["wood", "gold", "stone"] as Load[]).some((l) => {
      const r = nearestOf(resources(l), v);
      return r !== null && dist(r, v) < 1.8;
    });
  const farmerOf = new Map<Placement, Placement>();
  const taken = new Set<Placement>();
  const pairs = farms.flatMap((f) => villagers.map((v) => ({ f, v, d: dist(f, v) + (gathering(v) ? 100 : 0) }))).sort((a, b) => a.d - b.d);
  for (const { f, v } of pairs) {
    if (taken.has(f) || farmerOf.has(v)) continue;
    taken.add(f);
    farmerOf.set(v, f);
  }

  // Villagers working one resource spread around it.
  const crowd = new Map<Placement, number>();

  const jobs: Job[] = villagers.map((v, i) => {
    const gender = i % 3 === 0 ? "f" : "m";
    const farm = farmerOf.get(v) ?? null;
    let load: Load;
    let work: Point;
    let facing: number;
    // Where along the route it was seen, when it was seen walking.
    let via: Point | null = null;

    if (farm) {
      // A spot a third of the way from the centre to a corner, facing across the field.
      const corner = (i * 3 + 1) % 4;
      const ox = (corner & 1 ? 1 : -1) * farm.size * 0.17;
      const oz = (corner & 2 ? 1 : -1) * farm.size * 0.17;
      load = "food";
      work = { x: farm.x + ox, z: farm.z + oz };
      facing = workFacing(facingFor(-ox, -oz));
    } else {
      // At a resource, that is the job. Seen in the open, it is on its way: the job
      // whose trip passes closest to where it stood, with the route bent through it.
      const atResource = (["wood", "gold", "stone"] as Load[])
        .map((l) => ({ l, r: nearestOf(resources(l), v) }))
        .filter((o): o is { l: Load; r: Placement } => o.r !== null && dist(o.r, v) < 1.8)
        .sort((a, b) => dist(a.r, v) - dist(b.r, v))[0];
      const best =
        atResource ??
        (["wood", "gold", "stone"] as Load[])
          .flatMap((l) =>
            resources(l).map((r) => {
              const drop = dropFor(l, r);
              return { l, r, detour: dist(r, v) + dist(v, drop) - dist(r, drop) };
            }),
          )
          .sort((a, b) => a.detour - b.detour)[0]!;
      if (!atResource) via = grid.nearestOpen(v);
      load = best.l;
      const r = best.r;
      // Stand at the side of the resource facing the drop site, fanned out when shared.
      const n = crowd.get(r) ?? 0;
      crowd.set(r, n + 1);
      const drop = dropFor(load, r);
      const toward = Math.atan2(drop.z - r.z, drop.x - r.x) + (n % 2 ? 1 : -1) * Math.ceil(n / 2) * 0.9;
      const reach = load === "wood" ? 0.55 : 0.8;
      work = { x: r.x + Math.cos(toward) * reach, z: r.z + Math.sin(toward) * reach };
      facing = workFacing(facingFor(r.x - work.x, r.z - work.z));
    }

    const farmIndex = farm ? farms.indexOf(farm) : -1;
    const path = trip(work, via, load, farmIndex);
    const along = [0];
    for (let k = 1; k < path.length; k++) along.push(along[k - 1]! + dist(path[k - 1]!, path[k]!));
    const length = along[along.length - 1]!;
    const workTime = 5 + (i % 4) * 0.9;
    const walk = length / SPEED;
    let offset = (i * 1.37) % workTime;
    if (via) {
      // Half are seen carrying a load in, half heading back out.
      const s = along[path.indexOf(path.find((p) => p.x === via!.x && p.z === via!.z) ?? path[0]!)]!;
      offset = i % 2 === 0 ? workTime + s / SPEED : workTime + walk + DROP + (length - s) / SPEED;
    }
    const task = farm ? "farm" : load === "wood" ? "chop" : "mine";
    const handle = sprites.add(`vil${gender}_${task}_d${facing}`, work.x, 0, work.z, i * 3);
    return { handle, gender, task, load, work, facing, path, along, length, workTime, offset, farm, phase: i * 0.77, moving: false };
  });

  /** Position and heading at a distance along a path, read forwards or backwards. */
  function pointAt(job: Job, s: number, back: boolean, out: THREE.Vector3) {
    const d = back ? job.length - s : s;
    let k = 1;
    while (k < job.path.length - 1 && job.along[k]! < d) k++;
    const a = job.path[k - 1]!;
    const b = job.path[k]!;
    const seg = job.along[k]! - job.along[k - 1]! || 1;
    const t = Math.max(0, Math.min(1, (d - job.along[k - 1]!) / seg));
    out.set(a.x + (b.x - a.x) * t, 0, a.z + (b.z - a.z) * t);
    return back ? facingFor(a.x - b.x, a.z - b.z) : facingFor(b.x - a.x, b.z - a.z);
  }

  function update(time: number) {
    for (const job of jobs) {
      const h = job.handle;
      const walk = job.length / SPEED;
      const loop = job.workTime + 2 * walk + DROP;
      const t = (time + job.offset) % loop;
      job.moving = t >= job.workTime && job.length >= 0.05;
      if (t < job.workTime || job.length < 0.05) {
        h.anchor.set(job.work.x, 0, job.work.z);
        h.name = `vil${job.gender}_${job.task}_d${job.facing}`;
        h.frame = Math.floor(time * FPS + job.phase * 10);
      } else if (t < job.workTime + walk + DROP) {
        // Carrying the load to the drop site, then handing it over.
        const s = Math.min(t - job.workTime, walk) * SPEED;
        const f = pointAt(job, s, false, h.anchor);
        h.name = `vil${job.gender}_carry_${job.load}_d${carryFacing(f)}`;
        h.frame = t - job.workTime < walk ? Math.floor((s / STRIDE) * FRAMES) % FRAMES : 0;
      } else {
        // Walking back empty-handed.
        const s = (t - job.workTime - walk - DROP) * SPEED;
        const f = pointAt(job, s, true, h.anchor);
        h.name = `vil${job.gender}_walk_d${f}`;
        h.frame = Math.floor((s / STRIDE) * FRAMES) % FRAMES;
      }
    }
  }

  /** Villagers the Track chapter can follow: live anchors, whether each is walking
   * now, and how long its route is. */
  const tracked = jobs.map((job) => ({
    anchor: job.handle.anchor,
    route: job.length,
    get walking() {
      return job.moving;
    },
  }));

  /** Debug: per farm, the farmers assigned and every villager standing on it now; and
   * the number of villagers per job. */
  const check = () => ({
    farms: farms.map((f) => ({
      farmers: jobs.filter((j) => j.farm === f).length,
      standing: jobs.filter((j) => Math.abs(j.handle.anchor.x - f.x) < f.size / 2 && Math.abs(j.handle.anchor.z - f.z) < f.size / 2).length,
    })),
    jobs: jobs.reduce<Record<string, number>>((n, j) => ({ ...n, [j.load]: (n[j.load] ?? 0) + 1 }), {}),
    routes: jobs.map((j) => +j.length.toFixed(1)),
  });

  return { update, tracked, check };
}
