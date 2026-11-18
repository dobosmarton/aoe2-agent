import * as THREE from "three";

import { CATALOG, ROLES } from "../chapters";
import type { Board } from "./board";
import { RING_X } from "./flow";
import { easeOutCubic } from "./easing";
import { childOf, el, type Label, type Labels } from "./labels";
import { fatSegments, lineMaterial, loop } from "./lines";
import type { SceneState } from "./timeline";
import type { TurnData, TurnEntity } from "./types";

// Everything written in the scene (screen-space labels, labels.ts) and the marks the
// Act chapter draws on the ground: the house's footprint and the click.

// Labelled entities: one or two of each kind, so the labels spread over the frame.
const LABEL_CLASSES = ["town_center", "mill", "house", "farm", "tree", "gold_mine", "stone_mine", "deer", "villager", "villager"];
/** Phones show the first few labels only (`is-minor` hides the rest there). */
const PHONE_LABELS = 5;
// Gaps between boxes are measured as fractions of the frame width.
/** Least gap between a labelled box and boxes of other classes. */
const CLEARANCE = 0.006;
/** Least gap between two labelled boxes. */
const LABEL_SPACING = 0.06;
const RIPPLE_SEGMENTS = 64;

type Placing = { offset: [number, number]; align: [number, number] };

const LEDGER_BESIDE: Placing = { offset: [56, -40], align: [0, 1] };
const LEDGER_ABOVE: Placing = { offset: [0, -64], align: [0.5, 1] };

type Anchors = {
  /** Scene root, for the marks that live in world coordinates. */
  root: THREE.Object3D;
  /** Cursor path in the world: a start point, then the three hovered spots. */
  cursorPath: THREE.Vector3[];
  ledgerAt: THREE.Vector3;
  green: THREE.Vector3;
};

export type Overlays = {
  update: (state: SceneState) => void;
  /** Point on the cursor path: 0 is the start, 1 to 3 the hovered spots. */
  cursorAt: (t: number, out: THREE.Vector3) => THREE.Vector3;
  /** Narrow views: the HUD chips alternate between two rows, the ledger sits above the house instead of beside it. */
  setNarrow: (on: boolean) => void;
};

type CatalogRow = {
  element: HTMLElement;
  label: Label;
  home: THREE.Vector3;
  pos: THREE.Vector3;
  row: number;
  ok: boolean;
  pick: boolean;
};

const confidence = (e: TurnEntity): string => (e.conf == null ? "" : e.conf.toFixed(2));

// Detect: class labels on a few boxes, and the count. Each class is labelled on a
// detection that stands clear of boxes of other classes: a stone pile at the edge of
// a forest shares the forest's colour, and a label there would seem to name the trees.
const createDetectLabels = (data: TurnData, board: Board, labels: Labels): ((state: SceneState) => void) => {
  const gap = (a: TurnEntity, b: TurnEntity): number => {
    const dx = Math.max(0, Math.max(a.x, b.x) - Math.min(a.x + a.w, b.x + b.w));
    const dy = Math.max(0, Math.max(a.y, b.y) - Math.min(a.y + a.h, b.y + b.h));

    return Math.hypot(dx, dy * (data.frame.height / data.frame.width));
  };

  const clearance = (e: TurnEntity): number => Math.min(1, ...data.entities.filter((o) => o.cls !== e.cls).map((o) => gap(e, o)));
  const used = new Set<string>();
  const chosen: TurnEntity[] = [];
  const entityLabels = LABEL_CLASSES.flatMap((cls, i) => {
    const candidates = data.entities.filter((e) => e.cls === cls && !used.has(e.id));
    // The most confident of those clear of other classes and of the labels already
    // placed; the clearest if none is.
    const apart = (e: TurnEntity): boolean => chosen.every((c) => gap(e, c) > LABEL_SPACING);
    const score = (e: TurnEntity): number => (clearance(e) > CLEARANCE && apart(e) ? 1 + (e.conf ?? 0) : clearance(e));
    const entity = candidates.sort((a, b) => score(b) - score(a)).at(0);

    if (!entity) {
      return [];
    }

    used.add(entity.id);
    chosen.push(entity);

    const minor = i >= PHONE_LABELS ? " is-minor" : "";
    const label = labels.add(el(`wt3d-det wt3d-cat-${entity.cat}${minor}`, `${entity.cls.replace(/_/g, " ")} <b>${confidence(entity)}</b>`), {
      at: (out) => board.group.localToWorld(board.entityTop(entity, out)),
      offset: [0, -1],
      align: [0, 1],
    });

    return [label];
  });
  const counterValue = el("wt3d-count", `<b>0</b><span>boxes · 65 classes</span>`);
  const counterNumber = childOf(counterValue, "b");
  let counted = -1;
  const counter = labels.add(counterValue, {
    at: (out) => board.group.localToWorld(out.set(-board.W / 2, board.H / 2, 0)),
    offset: [0, -10],
    align: [0, 1],
  });

  return (state) => {
    // Labels arrive with their boxes; Track hands over to the village.
    for (const label of entityLabels) {
      label.opacity = state.boxes.labels * state.boxes.alpha * (1 - state.boxes.track);
    }

    const count = board.landedCount();

    if (count !== counted) {
      counted = count;
      counterNumber.textContent = String(count);
    }

    counter.opacity = state.stack.on * (count > 0 ? 1 : 0);
  };
};

// Read the HUD: one chip per field, its value ticking up.
const createHudChips = (data: TurnData, board: Board, labels: Labels): ((state: SceneState, narrow: boolean) => void) => {
  const chips = Object.entries(board.fieldCenters).flatMap(([name, centre]) => {
    // The data keeps only the readings that were read.
    if (!Object.hasOwn(data.hud.readings, name)) {
      return [];
    }

    const reading = data.hud.readings[name];

    const element = el("wt3d-chip", `<span>${name}</span><b>0</b>`);
    const value = childOf(element, "b");
    const label = labels.add(element, {
      at: (out) => board.hud.localToWorld(out.copy(centre)),
      offset: [0, -8],
      align: [0.5, 1],
      keepInView: 4,
    });
    const target = Number(reading);

    return [{ label, value, reading: String(reading), target: Number.isFinite(target) ? target : null, shown: "" }];
  });

  // Chips lift in a row, then each value counts up.
  return (state, narrow) =>
    chips.forEach((chip, i) => {
      const t = THREE.MathUtils.clamp(state.hud.values * 1.8 - i * 0.14, 0, 1);

      chip.label.opacity = Math.min(1, t * 3) * (1 - state.sheets.lift * 2);

      const row = narrow && i % 2 === 1 ? -((chip.label.size?.[1] ?? 24) + 4) : 0;

      chip.label.shift[1] = (1 - easeOutCubic(t * 2)) * 10 + row;

      const text = chip.target == null ? (t > 0.5 ? chip.reading : "·") : String(Math.round(chip.target * easeOutCubic(t)));

      if (text !== chip.shown) {
        chip.value.textContent = text;
        chip.shown = text;
      }
    });
};

// Decide: the three roles, the strategist's goals, the catalog.
const createDecideLabels = (decision: THREE.Group, labels: Labels): ((state: SceneState) => void) => {
  const roleX = [RING_X.policy, RING_X.strategist, RING_X.executor];
  const roles = ROLES.map((role, i) =>
    labels.add(
      el(
        `wt3d-role${i === 2 ? " is-idle" : ""}`,
        `<p class="wt3d-role-name">${role.role}</p><p class="wt3d-role-model">${role.model}</p><p class="wt3d-role-when">${role.when}</p>`,
      ),
      { at: (out) => decision.localToWorld(out.set(roleX[i], 1.3, 0)), offset: [0, -6], align: [0.5, 1], keepInView: 4 },
    ),
  );
  const goals = labels.add(
    el(
      "wt3d-card wt3d-goals",
      `<p class="wt3d-card-title">Strategist goals</p><ol><li>Keep housing ahead of population</li><li>Feudal buildings for Castle Age</li></ol>`,
    ),
    { at: (out) => decision.localToWorld(out.set(RING_X.strategist + 0.4, -1.45, 0)), offset: [0, 8], align: [0, 0], keepInView: 8 },
  );
  // The catalog hangs under the policy ring as a list spaced in pixels, so its rows
  // never overlap whatever the zoom.
  const catalog = CATALOG.map((item, i): CatalogRow => {
    const element = el(`wt3d-action${item.ok ? "" : " is-out"}`, `<code>${item.id}</code><span>${item.why}</span>`);
    const home = new THREE.Vector3(RING_X.policy - 1.4, -1.9, 0);
    const pos = home.clone();
    const label = labels.add(element, { at: (out) => decision.localToWorld(out.copy(pos)), align: [0.5, 0.5] });

    return { element, label, home, pos, row: i, ok: item.ok, pick: item.id === "build_house" };
  });
  const ringSeat = new THREE.Vector3(RING_X.policy, -1.55, 0);

  return (state) => {
    const on = state.decide.on;
    const pickOf = (c: CatalogRow): number => (c.pick ? easeOutCubic(state.decide.pick) : 0);
    // The roles' names step back while the action returns to the game.
    const away = 1 - state.action.on;

    roles.forEach((role, i) => {
      role.opacity = THREE.MathUtils.clamp(on * 1.5 - i * 0.2, 0, 1) * (i === 2 ? 0.6 : 1) * away;
    });

    goals.opacity = on * THREE.MathUtils.clamp(state.decide.catalog * 1.5 - 0.3, 0, 1) * away;

    for (const [i, c] of catalog.entries()) {
      const appear = THREE.MathUtils.clamp(state.decide.catalog * 2.2 - i * 0.14, 0, 1);
      const out = c.ok ? 0 : THREE.MathUtils.clamp(state.decide.filter * 1.6 - i * 0.08, 0, 1);
      const dim = c.pick ? 1 : 1 - state.decide.pick * 0.55;
      // The picked tile hands over to the action card when Act starts.
      const handover = c.pick ? 1 - state.action.on : 1;
      const pick = pickOf(c);

      c.label.opacity = on * (c.pick ? Math.max(appear, state.decide.pick) : appear) * (1 - out * 0.6) * dim * handover;
      c.label.shift[0] = (1 - easeOutCubic(appear)) * -16 + easeOutCubic(out) * 28;
      c.label.shift[1] = c.row * ((catalog[0].label.size?.[1] ?? 26) + 4) * (1 - pick);
      c.element.classList.toggle("is-filtered", out > 0.5);

      // The pick rises into the policy ring.
      c.pos.copy(c.home).lerp(ringSeat, pick);
      c.element.classList.toggle("is-chosen", c.pick && state.decide.pick > 0.4);
    }
  };
};

// Act: the cursor, the house's footprint under it, the click.
const createActMarks = (labels: Labels, anchors: Anchors): Pick<Overlays, "update" | "cursorAt"> => {
  const cursorElement = el("wt3d-cursor", `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 3l14 8-6.5 1.5L10 19z" /></svg>`);
  const cursorPoint = new THREE.Vector3();
  const cursor = labels.add(cursorElement, { at: (out) => out.copy(cursorPoint), offset: [-4, -3], align: [0, 0] });

  const marks = new THREE.Group();

  anchors.root.add(marks);

  // The 2 x 2 tile footprint of a house, on the ground under the cursor.
  const footprintMaterial = lineMaterial({ color: new THREE.Color(2.2, 0.5, 0.45), width: 2, opacity: 0 });
  const footprint = fatSegments(
    loop([new THREE.Vector3(-1, 0.04, -1), new THREE.Vector3(1, 0.04, -1), new THREE.Vector3(1, 0.04, 1), new THREE.Vector3(-1, 0.04, 1)]),
    footprintMaterial,
  );
  const footprintFillMaterial = new THREE.MeshBasicMaterial({ color: "#ff5a4a", transparent: true, opacity: 0, depthWrite: false });
  const footprintFill = new THREE.Mesh(new THREE.PlaneGeometry(2, 2).rotateX(-Math.PI / 2).translate(0, 0.03, 0), footprintFillMaterial);

  marks.add(footprint, footprintFill);

  // The click: a ring spreading on the ground.
  const ring = Array.from({ length: RIPPLE_SEGMENTS }, (_, k) => {
    const a = (k / RIPPLE_SEGMENTS) * Math.PI * 2;

    return new THREE.Vector3(Math.cos(a), 0.05, Math.sin(a));
  });
  const rippleMaterial = lineMaterial({ color: new THREE.Color(0.8, 2.4, 1.2), width: 2.5, opacity: 0 });
  const ripple = fatSegments(loop(ring), rippleMaterial);

  ripple.position.copy(anchors.green);
  marks.add(ripple);

  const path = anchors.cursorPath;

  const cursorAt = (t: number, out: THREE.Vector3): THREE.Vector3 => {
    const along = THREE.MathUtils.clamp(t, 0, 3);
    const leg = Math.min(2, Math.floor(along));
    const f = along - leg;

    // Ease into each spot, so the cursor settles where it reads the preview.
    return out.lerpVectors(path[leg], path[leg + 1], THREE.MathUtils.smoothstep(f, 0, 1));
  };

  const update = (state: SceneState): void => {
    cursorAt(state.world.cursor, cursorPoint);
    cursor.opacity = state.act.on * Math.min(1, state.world.cursor * 4);
    cursorElement.classList.toggle("is-click", state.act.click > 0.05 && state.act.click < 0.95);

    const ghost = state.world.ghost;

    footprint.position.copy(cursorPoint).setY(0);
    footprintFill.position.copy(footprint.position);

    const ok = state.world.cursor > 2.5;

    footprintMaterial.color.setRGB(ok ? 0.5 : 2.2, ok ? 2.2 : 0.5, ok ? 0.8 : 0.45);
    footprintFillMaterial.color.set(ok ? "#5cff8a" : "#ff5a4a");
    footprintMaterial.opacity = ghost;
    footprintFillMaterial.opacity = ghost * 0.18;
    footprint.visible = footprintFill.visible = ghost > 0.01;

    const click = state.act.click;

    ripple.visible = click > 0.01 && click < 0.99;
    ripple.scale.setScalar(0.6 + 2.2 * easeOutCubic(click));
    rippleMaterial.opacity = 1 - click;
  };

  return { update, cursorAt };
};

// Verify: the ledger beside the new house, one line at a time.
const createLedger = (data: TurnData, labels: Labels, anchors: Anchors): Pick<Overlays, "update" | "setNarrow"> => {
  const wood = Number(data.hud.readings.wood);
  const [popNow, popCap] = String(data.hud.readings.population).split("/");
  const ledgerElement = el(
    "wt3d-card wt3d-ledger",
    `<p class="wt3d-card-title">Action ledger</p>
     <p class="wt3d-ledger-op"><code>build_house</code></p>
     <ol>
       <li><b>pending</b><span>wood ${wood} at the click</span></li>
       <li><b>purchased</b><span>wood ${wood - 25}, 25 spent</span></li>
       <li><b>confirmed</b><span>population ${popNow}/${popCap} → ${popNow}/${Number(popCap) + 5}</span></li>
     </ol>`,
  );
  const ledger = labels.add(ledgerElement, {
    at: (out) => out.copy(anchors.ledgerAt),
    ...LEDGER_BESIDE,
    leader: true,
    keepInView: 8,
  });
  const ledgerRows = [...ledgerElement.querySelectorAll<HTMLElement>("li")].map((row) => ({ row, state: "" }));

  const setNarrow = (on: boolean): void => {
    ledger.options.offset = on ? LEDGER_ABOVE.offset : LEDGER_BESIDE.offset;
    ledger.options.align = on ? LEDGER_ABOVE.align : LEDGER_BESIDE.align;
  };

  const update = (state: SceneState): void => {
    ledger.opacity = state.verify.on;
    ledger.shift[1] = (1 - easeOutCubic(state.verify.on)) * 10;

    ledgerRows.forEach((r, i) => {
      const rowState = state.verify.status >= i + 1.5 ? "done" : state.verify.status >= i + 0.5 ? "current" : "";

      if (rowState === r.state) {
        return;
      }

      r.state = rowState;
      r.row.classList.toggle("is-done", rowState === "done");
      r.row.classList.toggle("is-current", rowState === "current");
      r.row.style.opacity = rowState ? "1" : "0.3";
    });
  };

  return { update, setNarrow };
};

export const createOverlays = (data: TurnData, board: Board, decision: THREE.Group, labels: Labels, anchors: Anchors): Overlays => {
  const updateDetect = createDetectLabels(data, board, labels);
  const updateHud = createHudChips(data, board, labels);
  const updateDecide = createDecideLabels(decision, labels);
  const act = createActMarks(labels, anchors);
  const ledger = createLedger(data, labels, anchors);

  let narrow = false;

  const setNarrow = (on: boolean): void => {
    narrow = on;
    ledger.setNarrow(on);
  };

  const update = (state: SceneState): void => {
    updateDetect(state);
    updateHud(state, narrow);
    updateDecide(state);
    act.update(state);
    ledger.update(state);
  };

  return { update, cursorAt: act.cursorAt, setNarrow };
};
