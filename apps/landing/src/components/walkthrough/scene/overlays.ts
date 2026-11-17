import * as THREE from "three";

import { CATALOG, ROLES } from "../chapters";
import type { Board } from "./board";
import { RING_X } from "./flow";
import { el, type Labels } from "./labels";
import { fatSegments, lineMaterial, loop } from "./lines";
import type { SceneState } from "./timeline";
import type { TurnData, TurnEntity } from "./types";

// Everything written in the scene (screen-space labels, labels.ts) and the marks the
// Act chapter draws on the ground: the house's footprint and the click.

// Labelled entities: one or two of each kind, so the labels spread over the frame.
const LABEL_CLASSES = ["town_center", "mill", "house", "farm", "tree", "gold_mine", "stone_mine", "deer", "villager", "villager"];

type Anchors = {
  /** Scene root, for the marks that live in world coordinates. */
  root: THREE.Object3D;
  /** Cursor path in the world: a start point, then the three hovered spots. */
  cursorPath: THREE.Vector3[];
  ledgerAt: THREE.Vector3;
  green: THREE.Vector3;
};

const ease = (t: number) => 1 - Math.pow(1 - THREE.MathUtils.clamp(t, 0, 1), 3);

export function createOverlays(data: TurnData, board: Board, decision: THREE.Group, labels: Labels, anchors: Anchors) {
  const conf = (e: TurnEntity) => (e.conf == null ? "" : e.conf.toFixed(2));

  // ---- Detect: class labels on a few boxes, and the count --------------------------------
  // Each class is labelled on a detection that stands clear of boxes of other classes:
  // a stone pile at the edge of a forest shares the forest's colour, and a label there
  // would seem to name the trees.
  const gap = (a: TurnEntity, b: TurnEntity) => {
    const dx = Math.max(0, Math.max(a.x, b.x) - Math.min(a.x + a.w, b.x + b.w));
    const dy = Math.max(0, Math.max(a.y, b.y) - Math.min(a.y + a.h, b.y + b.h));
    return Math.hypot(dx, dy * (data.frame.height / data.frame.width));
  };
  const clearance = (e: TurnEntity) => Math.min(1, ...data.entities.filter((o) => o.cls !== e.cls).map((o) => gap(e, o)));
  const used = new Set<string>();
  const chosen: TurnEntity[] = [];
  // Phones show the first few only (`is-minor` hides the rest there).
  const entityLabels = LABEL_CLASSES.flatMap((cls, i) => {
    const candidates = data.entities.filter((e) => e.cls === cls && !used.has(e.id));
    // The most confident of those clear of other classes and of the labels already
    // placed; the clearest if none is.
    const apart = (e: TurnEntity) => chosen.every((c) => gap(e, c) > 0.06);
    const score = (e: TurnEntity) => (clearance(e) > 0.006 && apart(e) ? 1 + (e.conf ?? 0) : clearance(e));
    const entity = candidates.sort((a, b) => score(b) - score(a))[0];
    if (!entity) return [];
    used.add(entity.id);
    chosen.push(entity);
    const label = labels.add(el(`wt3d-det wt3d-cat-${entity.cat}${i >= 5 ? " is-minor" : ""}`, `${entity.cls.replace(/_/g, " ")} <b>${conf(entity)}</b>`), {
      at: (out) => board.group.localToWorld(board.entityTop(entity, out)),
      offset: [0, -1],
      align: [0, 1],
    });
    return [{ entity, label }];
  });
  const counterValue = el("wt3d-count", `<b>0</b><span>boxes · 65 classes</span>`);
  const counterNumber = counterValue.querySelector("b")!;
  let counted = -1;
  const counter = labels.add(counterValue, {
    at: (out) => board.group.localToWorld(out.set(-board.W / 2, board.H / 2, 0)),
    offset: [0, -10],
    align: [0, 1],
  });

  // ---- Read the HUD: one chip per field, its value ticking up -----------------------------
  const chips = Object.entries(board.fieldCenters).flatMap(([name, centre]) => {
    const reading = data.hud.readings[name];
    if (reading == null) return [];
    const element = el("wt3d-chip", `<span>${name}</span><b>0</b>`);
    const value = element.querySelector("b")!;
    const label = labels.add(element, {
      at: (out) => board.hud.localToWorld(out.copy(centre)),
      offset: [0, -8],
      align: [0.5, 1],
      keepInView: 4,
    });
    const target = Number(reading);
    return [{ label, value, reading: String(reading), target: Number.isFinite(target) ? target : null, shown: "" }];
  });

  // ---- Decide: the three roles, the strategist's goals, the catalog ----------------------
  const roleX = [RING_X.policy, RING_X.strategist, RING_X.executor];
  const roles = ROLES.map((role, i) =>
    labels.add(
      el(
        `wt3d-role${i === 2 ? " is-idle" : ""}`,
        `<p class="wt3d-role-name">${role.role}</p><p class="wt3d-role-model">${role.model}</p><p class="wt3d-role-when">${role.when}</p>`,
      ),
      { at: (out) => decision.localToWorld(out.set(roleX[i]!, 1.3, 0)), offset: [0, -6], align: [0.5, 1], keepInView: 4 },
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
  const catalog = CATALOG.map((item, i) => {
    const element = el(`wt3d-action${item.ok ? "" : " is-out"}`, `<code>${item.id}</code><span>${item.why}</span>`);
    const home = new THREE.Vector3(RING_X.policy - 1.4, -1.9, 0);
    const pos = home.clone();
    const label = labels.add(element, { at: (out) => decision.localToWorld(out.copy(pos)), align: [0.5, 0.5] });
    return { element, label, home, pos, row: i, ok: item.ok, pick: item.id === "build_house" };
  });
  const ringSeat = new THREE.Vector3(RING_X.policy, -1.55, 0);

  // ---- Act: the cursor, the footprint, the click; Verify: the ledger --------------------
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
  const footprintFill = new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2).rotateX(-Math.PI / 2).translate(0, 0.03, 0),
    new THREE.MeshBasicMaterial({ color: "#ff5a4a", transparent: true, opacity: 0, depthWrite: false }),
  );
  marks.add(footprint, footprintFill);
  // The click: a ring spreading on the ground.
  const ring: number[] = [];
  for (let k = 0; k < 64; k++) {
    const a = (k / 64) * Math.PI * 2;
    const b = ((k + 1) / 64) * Math.PI * 2;
    ring.push(Math.cos(a), 0.05, Math.sin(a), Math.cos(b), 0.05, Math.sin(b));
  }
  const rippleMaterial = lineMaterial({ color: new THREE.Color(0.8, 2.4, 1.2), width: 2.5, opacity: 0 });
  const ripple = fatSegments(ring, rippleMaterial);
  ripple.position.copy(anchors.green);
  marks.add(ripple);

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
    offset: [56, -40],
    align: [0, 1],
    leader: true,
    keepInView: 8,
  });
  const ledgerRows = [...ledgerElement.querySelectorAll<HTMLElement>("li")].map((row) => ({ row, state: "" }));

  const path = anchors.cursorPath;
  /** Point on the cursor path: 0 is the start, 1 to 3 the hovered spots. */
  function cursorAt(t: number, out: THREE.Vector3) {
    const c = THREE.MathUtils.clamp(t, 0, 3);
    const k = Math.min(2, Math.floor(c));
    const f = c - k;
    // Ease into each spot, so the cursor settles where it reads the preview.
    return out.lerpVectors(path[k]!, path[k + 1]!, f * f * (3 - 2 * f));
  }

  // Narrow views: the HUD chips alternate between two rows, the ledger sits above the
  // house instead of beside it.
  let narrow = false;
  function setNarrow(on: boolean) {
    narrow = on;
    ledger.options.offset = on ? [0, -64] : [56, -40];
    ledger.options.align = on ? [0.5, 1] : [0, 1];
  }

  function update(S: SceneState) {
    // Detect: labels arrive with their boxes; Track hands over to the village.
    for (const { label } of entityLabels) label.opacity = S.boxes.labels * S.boxes.alpha * (1 - S.boxes.track);
    const count = board.landedCount();
    if (count !== counted) counterNumber.textContent = String((counted = count));
    counter.opacity = S.stack.on * (count > 0 ? 1 : 0);

    // HUD: chips lift in a row, then each value counts up.
    chips.forEach((chip, i) => {
      const t = THREE.MathUtils.clamp(S.hud.values * 1.8 - i * 0.14, 0, 1);
      chip.label.opacity = Math.min(1, t * 3) * (1 - S.sheets.lift * 2);
      const row = narrow && i % 2 === 1 ? -((chip.label.size?.[1] ?? 24) + 4) : 0;
      chip.label.shift[1] = (1 - ease(t * 2)) * 10 + row;
      const text = chip.target == null ? (t > 0.5 ? chip.reading : "·") : String(Math.round(chip.target * ease(t)));
      if (text !== chip.shown) {
        chip.value.textContent = text;
        chip.shown = text;
      }
    });

    // Decide.
    const on = S.decide.on;
    const pickOf = (c: (typeof catalog)[number]) => (c.pick ? ease(S.decide.pick) : 0);
    // The roles' names step back while the action returns to the game.
    const away = 1 - S.action.on;
    roles.forEach((role, i) => (role.opacity = THREE.MathUtils.clamp(on * 1.5 - i * 0.2, 0, 1) * (i === 2 ? 0.6 : 1) * away));
    goals.opacity = on * THREE.MathUtils.clamp(S.decide.catalog * 1.5 - 0.3, 0, 1) * away;
    for (const [i, c] of catalog.entries()) {
      const appear = THREE.MathUtils.clamp(S.decide.catalog * 2.2 - i * 0.14, 0, 1);
      const out = !c.ok ? THREE.MathUtils.clamp(S.decide.filter * 1.6 - i * 0.08, 0, 1) : 0;
      const dim = c.pick ? 1 : 1 - S.decide.pick * 0.55;
      // The picked tile hands over to the action card when Act starts.
      const handover = c.pick ? 1 - S.action.on : 1;
      c.label.opacity = on * (c.pick ? Math.max(appear, S.decide.pick) : appear) * (1 - out * 0.6) * dim * handover;
      c.label.shift[0] = (1 - ease(appear)) * -16 + ease(out) * 28;
      c.label.shift[1] = c.row * ((catalog[0]!.label.size?.[1] ?? 26) + 4) * (1 - pickOf(c));
      c.element.classList.toggle("is-filtered", out > 0.5);
      // The pick rises into the policy ring.
      const pick = pickOf(c);
      c.pos.copy(c.home).lerp(ringSeat, pick);
      c.element.classList.toggle("is-chosen", c.pick && S.decide.pick > 0.4);
    }

    // Act.
    cursorAt(S.world.cursor, cursorPoint);
    cursor.opacity = S.act.on * Math.min(1, S.world.cursor * 4);
    cursorElement.classList.toggle("is-click", S.act.click > 0.05 && S.act.click < 0.95);
    const ghost = S.world.ghost;
    footprint.position.copy(cursorPoint).setY(0);
    footprintFill.position.copy(footprint.position);
    const ok = S.world.cursor > 2.5;
    footprintMaterial.color.setRGB(ok ? 0.5 : 2.2, ok ? 2.2 : 0.5, ok ? 0.8 : 0.45);
    (footprintFill.material as THREE.MeshBasicMaterial).color.set(ok ? "#5cff8a" : "#ff5a4a");
    footprintMaterial.opacity = ghost;
    (footprintFill.material as THREE.MeshBasicMaterial).opacity = ghost * 0.18;
    footprint.visible = footprintFill.visible = ghost > 0.01;
    const click = S.act.click;
    ripple.visible = click > 0.01 && click < 0.99;
    ripple.scale.setScalar(0.6 + 2.2 * ease(click));
    rippleMaterial.opacity = 1 - click;

    // Verify.
    ledger.opacity = S.verify.on;
    ledger.shift[1] = (1 - ease(S.verify.on)) * 10;
    ledgerRows.forEach((r, i) => {
      const state = S.verify.status >= i + 1.5 ? "done" : S.verify.status >= i + 0.5 ? "current" : "";
      if (state === r.state) return;
      r.state = state;
      r.row.classList.toggle("is-done", state === "done");
      r.row.classList.toggle("is-current", state === "current");
      r.row.style.opacity = state ? "1" : "0.3";
    });
  }

  return { update, cursorAt, setNarrow };
}
