import * as THREE from "three";
import { CSS3DSprite } from "three/examples/jsm/renderers/CSS3DRenderer.js";

import { CATALOG, ROLES } from "../chapters";
import type { Board } from "./board";
import { RING_X } from "./flow";
import type { SceneState } from "./timeline";
import type { TurnData, TurnEntity } from "./types";

// Text in the 3D scene is real DOM (CSS3DRenderer), so it keeps the site's type and
// stays sharp. Every element faces the camera. Styles: `.wt3d-*` in globals.css.

const PX = 0.015; // board units per CSS pixel

function el(className: string, html: string): HTMLDivElement {
  const div = document.createElement("div");
  div.className = className;
  div.innerHTML = html;
  return div;
}

function sprite(element: HTMLElement, scale = PX): CSS3DSprite {
  const s = new CSS3DSprite(element);
  s.scale.setScalar(scale);
  return s;
}

const shown = new WeakMap<HTMLElement, number>();
function fade(element: HTMLElement, opacity: number) {
  const value = Math.round(opacity * 100) / 100;
  if (shown.get(element) === value) return;
  shown.set(element, value);
  element.style.opacity = String(value);
  element.style.visibility = value <= 0 ? "hidden" : "visible";
}

// Labelled entities: one or two of each kind, so the labels spread over the board.
const LABEL_CLASSES = ["town_center", "mill", "house", "farm", "tree", "gold_mine", "stone_mine", "deer", "villager", "villager", "villager"];
// Track IDs are illustrative: no recorded run stores them.
const TRACK_IDS = [3, 12, 7, 19, 28, 34, 55, 63, 41, 90, 17];

type Anchors = {
  /** Scene root, for the elements that live in world coordinates. */
  root: THREE.Object3D;
  /** World units per mind unit. */
  mindScale: number;
  cardStart: THREE.Vector3;
  cardEnd: THREE.Vector3;
  /** Cursor path in the world: a start point, then the three hovered spots. */
  cursorPath: THREE.Vector3[];
  ledgerAt: THREE.Vector3;
};

export function createOverlays(data: TurnData, board: Board, decision: THREE.Group, anchors: Anchors) {
  const conf = (e: TurnEntity) => (e.conf == null ? "" : e.conf.toFixed(2));

  // ---- Entity labels (Detect, Track) -------------------------------------------------
  const used = new Set<string>();
  const labelled: { entity: TurnEntity; element: HTMLDivElement; sprite: CSS3DSprite; unit: boolean }[] = [];
  LABEL_CLASSES.forEach((cls, i) => {
    const entity = data.entities.find((e) => e.cls === cls && !used.has(e.id));
    if (!entity) return;
    used.add(entity.id);
    const element = el(
      `wt3d-label wt3d-cat-${entity.cat}`,
      `<span class="wt3d-label-det">${entity.cls.replace(/_/g, " ")} <b>${conf(entity)}</b></span>` +
        `<span class="wt3d-label-track">track_${TRACK_IDS[i]}</span>`,
    );
    const s = sprite(element);
    board.group.add(s);
    labelled.push({ entity, element, sprite: s, unit: entity.cat === "unit" });
  });

  // ---- HUD values (Read the HUD) -----------------------------------------------------
  const hudValues: HTMLDivElement[] = [];
  for (const [name, center] of Object.entries(board.fieldCenters)) {
    const value = data.hud.readings[name];
    if (value == null) continue;
    const element = el("wt3d-chip", `<span>${name}</span><b>${value}</b>`);
    const s = sprite(element, PX * 0.75);
    s.position.copy(center).add(new THREE.Vector3(0, 0.45, 0.15));
    board.hud.add(s);
    hudValues.push(element);
  }

  // ---- Perception card (Publish, Decide) ---------------------------------------------
  const r = data.hud.readings;
  const card = el(
    "wt3d-card wt3d-perception",
    `<p class="wt3d-card-title">[ Perception ]</p>
     <dl>
       <dt>entities</dt><dd>${data.entities.length} tracked boxes</dd>
       <dt>hud</dt><dd>W ${r.wood} · F ${r.food} · G ${r.gold} · S ${r.stone}</dd>
       <dt>population</dt><dd>${r.population} · ${r.age}</dd>
       <dt>spatial_valid</dt><dd>true</dd>
       <dt>alarm</dt><dd>false</dd>
     </dl>`,
  );
  const cardSprite = sprite(card, PX * 1.35 * anchors.mindScale);
  anchors.root.add(cardSprite);

  // ---- Decision loops (Decide) -------------------------------------------------------
  const roleX = [RING_X.policy, RING_X.strategist, RING_X.executor];
  const roles = ROLES.map((role, i) => {
    const element = el(
      `wt3d-role${i === 2 ? " is-idle" : ""}`,
      `<p class="wt3d-role-name">${role.role}</p><p class="wt3d-role-model">${role.model}</p><p class="wt3d-role-when">${role.when}</p>`,
    );
    const s = sprite(element);
    s.position.set(roleX[i]!, 1.75, 0);
    decision.add(s);
    return element;
  });
  const goals = el(
    "wt3d-card wt3d-goals",
    `<p class="wt3d-card-title">[ Strategist goals ]</p>
     <ol><li>Keep housing ahead of population</li><li>Feudal buildings for Castle Age</li></ol>`,
  );
  const goalsSprite = sprite(goals);
  goalsSprite.position.set(RING_X.strategist + 0.5, -2.2, 0);
  decision.add(goalsSprite);

  const catalog = CATALOG.map((item, i) => {
    const element = el(
      `wt3d-action${item.ok ? "" : " is-out"}${item.id === "build_house" ? " is-pick" : ""}`,
      `<code>${item.id}</code><span>${item.why}</span>`,
    );
    const s = sprite(element, PX);
    s.position.set(RING_X.policy, -1.6 - i * 0.6, 0);
    decision.add(s);
    return { element, sprite: s, home: s.position.clone(), ok: item.ok, pick: item.id === "build_house" };
  });

  // ---- Act: cursor, and Verify: ledger ------------------------------------------------
  const cursor = el(
    "wt3d-cursor",
    `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 3l14 8-6.5 1.5L10 19z" /></svg>`,
  );
  const cursorSprite = sprite(cursor, PX * 0.9);
  anchors.root.add(cursorSprite);
  const path = anchors.cursorPath;

  const wood = Number(r.wood);
  const [popNow, popCap] = String(r.population).split("/");
  const ledger = el(
    "wt3d-card wt3d-ledger",
    `<p class="wt3d-card-title">[ Action ledger ]</p>
     <p class="wt3d-ledger-op"><code>build_house</code></p>
     <ol>
       <li data-status="1"><b>pending</b><span>wood ${wood} at the click</span></li>
       <li data-status="2"><b>purchased</b><span>wood ${wood - 25}, 25 spent</span></li>
       <li data-status="3"><b>confirmed</b><span>population ${popNow}/${popCap} → ${popNow}/${Number(popCap) + 5}</span></li>
     </ol>`,
  );
  const ledgerSprite = sprite(ledger, PX * 0.75);
  ledgerSprite.position.copy(anchors.ledgerAt);
  anchors.root.add(ledgerSprite);
  const ledgerRows = [...ledger.querySelectorAll<HTMLElement>("[data-status]")];

  const tmp = new THREE.Vector3();

  function update(S: SceneState) {
    // Labels follow the tops of their volumes.
    const labelsOn = S.boxes.labels * S.boxes.alpha;
    const track = S.boxes.track > 0.5;
    for (const item of labelled) {
      board.entityTop(item.entity, item.sprite.position);
      item.sprite.position.z += 0.25;
      item.element.classList.toggle("is-track", track);
      const focus = track ? (item.unit ? 1 : 0.18) : 1;
      fade(item.element, labelsOn * focus);
    }

    hudValues.forEach((h, i) => fade(h, THREE.MathUtils.clamp(S.hud.values * 1.6 - i * 0.12, 0, 1)));

    // The card rises with the stream, then travels to the policy ring.
    const e = S.card.travel;
    const eased = e * e * (3 - 2 * e);
    cardSprite.position.lerpVectors(anchors.cardStart, anchors.cardEnd, eased);
    cardSprite.scale.setScalar(PX * anchors.mindScale * (1.35 - 0.4 * eased));
    fade(card, S.card.on);

    const on = S.decide.on;
    roles.forEach((role, i) => fade(role, on * (i === 2 ? 0.55 : 1)));
    fade(goals, on * THREE.MathUtils.clamp(S.decide.catalog * 1.5, 0, 1));
    catalog.forEach((c, i) => {
      const appear = THREE.MathUtils.clamp(S.decide.catalog * 2 - i * 0.12, 0, 1);
      const out = !c.ok ? S.decide.filter : 0;
      const dim = c.pick ? 1 : 1 - S.decide.pick * 0.45;
      fade(c.element, on * appear * (1 - out * 0.65) * dim);
      c.element.classList.toggle("is-filtered", out > 0.5);
      c.element.classList.toggle("is-chosen", c.pick && S.decide.pick > 0.5);
      tmp.copy(c.home);
      tmp.y += (1 - appear) * 0.6;
      if (c.pick) tmp.z += S.decide.pick * 0.6;
      c.sprite.position.copy(tmp);
    });

    // Cursor along its path of hovers.
    cursorAt(S.world.cursor, cursorSprite.position);
    cursorSprite.position.y += 1.2;
    fade(cursor, S.act.on);
    cursor.classList.toggle("is-click", S.act.click > 0.05 && S.act.click < 0.95);

    fade(ledger, S.verify.on);
    ledgerRows.forEach((row, i) => {
      row.classList.toggle("is-done", S.verify.status >= i + 1.5);
      row.classList.toggle("is-current", S.verify.status >= i + 0.5 && S.verify.status < i + 1.5);
      fade(row, S.verify.status >= i + 0.5 ? 1 : 0.25);
    });
  }

  /** Point on the cursor path: 0 is the start, 1 to 3 the hovered spots. */
  function cursorAt(t: number, out: THREE.Vector3) {
    const c = THREE.MathUtils.clamp(t, 0, 3);
    const k = Math.min(2, Math.floor(c));
    return out.lerpVectors(path[k]!, path[k + 1]!, c - k);
  }

  return { card: cardSprite, update, cursorAt };
}
