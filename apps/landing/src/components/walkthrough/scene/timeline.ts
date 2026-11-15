import { gsap } from "gsap";

// Every animated value of the scene lives in this one object. A paused GSAP timeline
// tweens it, and the scroll position seeks the timeline: time t means "chapter
// floor(t), fraction t % 1 of the way through it". Chapter 0 is the hero, 1 to 8 are
// the walkthrough chapters, so the timeline is 9 units long.
export function createState() {
  return {
    cam: { yaw: 0.62, pitch: 0.98, zoom: 0.94, tx: 0, ty: 0, tz: 0 },
    frame: { scan: 1, scanOn: 0, dim: 0, flash: 0, grid: 0, hudHole: 0 },
    boxes: { rise: 0.75, alpha: 0.55, fill: 0.05, labels: 0, track: 0, shimmer: 1 },
    stamp: { on: 0 },
    hud: { lift: 0, fields: 0, values: 0 },
    flow: { t: 0, on: 0 },
    card: { on: 0, travel: 0 },
    decide: { on: 0, catalog: 0, filter: 0, pick: 0 },
    act: { keys: 0, cursor: 0, click: 0, placed: 0, on: 0 },
    verify: { on: 0, status: 0 },
  };
}

export type SceneState = ReturnType<typeof createState>;

export const TIMELINE_LENGTH = 9;

// Camera targets are in board units, set by the scene once it knows the frame geometry.
export type Anchors = {
  villagers: { x: number; y: number };
  hud: { x: number; y: number };
  house: { x: number; y: number };
  decision: { x: number; y: number; z: number };
  card: { x: number; y: number; z: number };
};

export function buildTimeline(S: SceneState, A: Anchors): gsap.core.Timeline {
  const tl = gsap.timeline({ paused: true, defaults: { ease: "power2.inOut" } });
  const to = (target: object, at: number, dur: number, vars: gsap.TweenVars) => tl.to(target, { ...vars, duration: dur }, at);

  // 1 Capture: square up to the flat screen, lower the boxes, sweep a scan line.
  to(S.cam, 1, 0.3, { yaw: 0, pitch: 0, zoom: 1, tx: 0, ty: 0, tz: 0 });
  to(S.boxes, 1, 0.22, { rise: 0, alpha: 0, labels: 0 });
  to(S.stamp, 1.2, 0.12, { on: 1 });
  to(S.frame, 1.25, 0.05, { scanOn: 1, scan: 0, grid: 1 });
  to(S.frame, 1.3, 0.5, { scan: 1, ease: "none" });

  // 2 Detect: tilt to a perspective view and raise every box in a wave.
  to(S.frame, 2, 0.12, { scanOn: 0, grid: 0 });
  to(S.stamp, 2, 0.1, { on: 0 });
  to(S.cam, 2, 0.32, { yaw: 0.32, pitch: 0.92, zoom: 0.92 });
  to(S.boxes, 2.22, 0.5, { rise: 1, alpha: 1, fill: 0.1, ease: "power1.inOut" });
  to(S.boxes, 2.55, 0.15, { labels: 1 });

  // 3 Track: move in on the villagers, switch labels to track IDs.
  to(S.cam, 3, 0.32, { yaw: -0.3, pitch: 0.82, zoom: 0.72, tx: A.villagers.x, ty: A.villagers.y });
  to(S.boxes, 3.3, 0.25, { track: 1 });

  // 4 Read the HUD: lift the resource bar off the frame and read its fields.
  to(S.boxes, 4, 0.15, { track: 0, labels: 0, alpha: 0.28, fill: 0.04 });
  to(S.cam, 4, 0.32, { yaw: 0.12, pitch: 0.55, zoom: 0.62, tx: A.hud.x, ty: A.hud.y });
  to(S.hud, 4.25, 0.3, { lift: 1 });
  to(S.frame, 4.25, 0.3, { hudHole: 1 });
  to(S.hud, 4.5, 0.15, { fields: 1 });
  to(S.hud, 4.62, 0.18, { values: 1 });

  // 5 Publish: everything streams into one frozen frame above the board.
  to(S.hud, 5, 0.2, { values: 0, fields: 0, lift: 0 });
  to(S.frame, 5, 0.2, { hudHole: 0 });
  to(S.cam, 5, 0.35, { yaw: 0, pitch: 0.82, zoom: 0.86, tx: A.card.x, ty: A.card.y, tz: A.card.z * 0.4 });
  to(S.boxes, 5.05, 0.25, { alpha: 0.6, rise: 0.25 });
  to(S.flow, 5.1, 0.05, { on: 1 });
  to(S.flow, 5.12, 0.5, { t: 1, ease: "power1.inOut" });
  to(S.card, 5.45, 0.15, { on: 1 });
  to(S.frame, 5.4, 0.3, { dim: 0.55 });
  to(S.boxes, 5.45, 0.25, { alpha: 0.15 });
  to(S.flow, 5.65, 0.08, { on: 0 });

  // 6 Decide: the frame travels to the decision loops; the catalog is filtered.
  to(S.cam, 6, 0.35, { yaw: 0, pitch: 1.1, zoom: 1.05, tx: A.decision.x, ty: A.decision.y, tz: A.decision.z });
  to(S.card, 6, 0.35, { travel: 1 });
  to(S.decide, 6.1, 0.25, { on: 1 });
  to(S.decide, 6.32, 0.2, { catalog: 1 });
  to(S.decide, 6.55, 0.15, { filter: 1 });
  to(S.decide, 6.72, 0.13, { pick: 1 });

  // 7 Act: back to the board, keys, three hovers with the placement preview, a click.
  to(S.decide, 7, 0.18, { on: 0 });
  to(S.card, 7, 0.15, { on: 0 });
  to(S.frame, 7, 0.25, { dim: 0 });
  to(S.boxes, 7, 0.25, { alpha: 0.4, rise: 0.12, fill: 0.03 });
  to(S.cam, 7, 0.32, { yaw: 0.22, pitch: 0.78, zoom: 0.6, tx: A.house.x, ty: A.house.y, tz: 0 });
  to(S.act, 7.1, 0.1, { on: 1 });
  to(S.act, 7.22, 0.12, { keys: 2, ease: "steps(2)" });
  // Each hover holds long enough to read the preview.
  to(S.act, 7.3, 0.07, { cursor: 1 });
  to(S.act, 7.45, 0.07, { cursor: 2 });
  to(S.act, 7.6, 0.07, { cursor: 3 });
  to(S.act, 7.74, 0.08, { click: 1, placed: 1, ease: "power1.out" });
  to(S.act, 7.84, 0.05, { keys: 3, ease: "none" });

  // 8 Verify: the ledger settles; then the view pulls back for the next frame.
  to(S.cam, 8, 0.3, { yaw: 0.4, pitch: 0.92, zoom: 0.78 });
  to(S.verify, 8.05, 0.12, { on: 1 });
  to(S.verify, 8.15, 0.05, { status: 1, ease: "none" });
  to(S.verify, 8.3, 0.12, { status: 2, ease: "none" });
  to(S.verify, 8.5, 0.12, { status: 3, ease: "none" });
  to(S.act, 8.72, 0.12, { on: 0, cursor: 0, keys: 0 });
  to(S.verify, 8.75, 0.12, { on: 0 });
  to(S.frame, 8.8, 0.06, { flash: 0.6, ease: "power1.out" });
  to(S.frame, 8.86, 0.12, { flash: 0 });
  to(S.boxes, 8.8, 0.2, { alpha: 0.55, rise: 0.75, fill: 0.05 });
  to(S.cam, 8.75, 0.25, { yaw: 0.62, pitch: 0.98, zoom: 0.94, tx: 0, ty: 0, tz: 0 });

  tl.set({}, {}, TIMELINE_LENGTH);
  return tl;
}
