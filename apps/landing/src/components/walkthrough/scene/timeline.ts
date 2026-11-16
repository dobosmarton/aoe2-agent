import { gsap } from "gsap";

import type { CameraPose } from "./rig";

// Every animated value of the scene lives in one object. A paused GSAP timeline tweens
// it, and the scroll position seeks the timeline: time t means "chapter floor(t),
// fraction t % 1 of the way through it". Chapter 0 is the hero, 1 to 8 are the
// walkthrough chapters, so the timeline is 9 units long.
export function createState(hero: CameraPose) {
  return {
    cam: { ...hero },
    world: { dim: 0, ghost: 0, build: 0, cursor: 0 },
    pane: { on: 0, mix: 0, flash: 0 },
    slab: { on: 0, peel: 0, travel: 0 },
    mind: { gate: 0, track: 0, rail: 0, pulse: 0 },
    frame: { scan: 1, scanOn: 0, dim: 0, flash: 0, grid: 0, hudHole: 0 },
    boxes: { rise: 0, alpha: 0, fill: 0.05, labels: 0, track: 0, shimmer: 1 },
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

export type Poses = Record<
  "hero" | "game" | "capture" | "detect" | "track" | "hud" | "publish" | "decide" | "house" | "verify",
  CameraPose
>;

export function buildTimeline(S: SceneState, P: Poses): gsap.core.Timeline {
  const tl = gsap.timeline({ paused: true, defaults: { ease: "power2.inOut" } });
  const to = (target: object, at: number, dur: number, vars: gsap.TweenVars) => tl.to(target, { ...vars, duration: dur }, at);
  const cam = (at: number, dur: number, pose: keyof Poses, ease = "power2.inOut") => to(S.cam, at, dur, { ...P[pose], ease });

  // 1 Capture: rise into the game camera, the glass appears in front of the village,
  //   the shot is taken, the real frame replaces the render, the camera backs off.
  cam(1, 0.36, "game");
  to(S.pane, 1.3, 0.1, { on: 1 });
  to(S.stamp, 1.32, 0.1, { on: 1 });
  to(S.pane, 1.42, 0.03, { flash: 1, ease: "power1.out" });
  to(S.pane, 1.45, 0.1, { flash: 0 });
  to(S.pane, 1.48, 0.24, { mix: 1, ease: "none" });
  to(S.world, 1.62, 0.25, { dim: 1 });
  cam(1.7, 0.26, "capture");
  to(S.slab, 1.84, 0.02, { on: 1 });
  to(S.slab, 1.85, 0.13, { peel: 1 });

  // 2 Detect: the frame slides through the detector gate; boxes rise from it.
  to(S.stamp, 2, 0.08, { on: 0 });
  to(S.mind, 2, 0.12, { gate: 1 });
  cam(2, 0.3, "detect");
  to(S.slab, 2.1, 0.32, { travel: 1 });
  to(S.boxes, 2.42, 0.3, { rise: 1, alpha: 1, fill: 0.1, ease: "power1.inOut" });
  to(S.boxes, 2.68, 0.12, { labels: 1 });

  // 3 Track: earlier frames stack up behind; tubes thread each villager through time.
  to(S.mind, 3, 0.15, { gate: 0 });
  cam(3, 0.32, "track");
  to(S.mind, 3.22, 0.4, { track: 1, ease: "power1.inOut" });
  to(S.boxes, 3.3, 0.2, { track: 1 });

  // 4 Read the HUD: the resource bar lifts off and its fields are read.
  to(S.boxes, 4, 0.15, { track: 0, labels: 0, alpha: 0.3, fill: 0.04 });
  to(S.mind, 4, 0.15, { track: 0.15 });
  cam(4, 0.3, "hud");
  to(S.hud, 4.25, 0.3, { lift: 1 });
  to(S.frame, 4.25, 0.3, { hudHole: 1 });
  to(S.hud, 4.5, 0.15, { fields: 1 });
  to(S.hud, 4.62, 0.18, { values: 1 });

  // 5 Publish: everything streams into one Perception crystal on the FramePipe rail.
  to(S.hud, 5, 0.2, { values: 0, fields: 0, lift: 0 });
  to(S.frame, 5, 0.2, { hudHole: 0 });
  to(S.mind, 5, 0.15, { track: 0 });
  cam(5, 0.34, "publish");
  to(S.boxes, 5.05, 0.25, { alpha: 0.6, rise: 0.25 });
  to(S.flow, 5.1, 0.05, { on: 1 });
  to(S.flow, 5.12, 0.5, { t: 1, ease: "power1.inOut" });
  to(S.card, 5.45, 0.15, { on: 1 });
  to(S.mind, 5.5, 0.2, { rail: 1 });
  to(S.frame, 5.4, 0.3, { dim: 0.55 });
  to(S.boxes, 5.45, 0.25, { alpha: 0.15 });
  to(S.flow, 5.65, 0.08, { on: 0 });

  // 6 Decide: the frame rides the rail to the three loops; the catalog is filtered.
  cam(6, 0.34, "decide");
  to(S.card, 6, 0.34, { travel: 1 });
  to(S.decide, 6.1, 0.25, { on: 1 });
  to(S.decide, 6.32, 0.2, { catalog: 1 });
  to(S.decide, 6.55, 0.15, { filter: 1 });
  to(S.decide, 6.72, 0.13, { pick: 1 });

  // 7 Act: the choice flies back to the glass, the camera follows it through into the
  //   village, the house hovers over three spots and is built on the clear one.
  to(S.mind, 7, 0.2, { pulse: 1, ease: "power1.in" });
  to(S.decide, 7.04, 0.14, { on: 0 });
  to(S.card, 7.04, 0.14, { on: 0 });
  to(S.mind, 7.04, 0.14, { rail: 0 });
  cam(7.02, 0.18, "game", "power2.in");
  to(S.slab, 7.05, 0.06, { on: 0 });
  to(S.world, 7.14, 0.14, { dim: 0 });
  to(S.pane, 7.2, 0.06, { on: 0 });
  cam(7.2, 0.18, "house", "power2.out");
  to(S.act, 7.3, 0.04, { on: 1 });
  to(S.act, 7.32, 0.08, { keys: 2, ease: "steps(2)" });
  to(S.world, 7.38, 0.04, { ghost: 1 });
  to(S.world, 7.38, 0.06, { cursor: 1 });
  to(S.world, 7.5, 0.06, { cursor: 2 });
  to(S.world, 7.62, 0.06, { cursor: 3 });
  to(S.act, 7.72, 0.06, { click: 1, placed: 1, ease: "power1.out" });
  to(S.world, 7.73, 0.04, { ghost: 0 });
  to(S.world, 7.74, 0.2, { build: 5, ease: "none" });
  to(S.act, 7.84, 0.05, { keys: 3, ease: "none" });

  // 8 Verify: the ledger settles over the new house; then back to the hero shot.
  cam(8, 0.3, "verify");
  to(S.verify, 8.05, 0.12, { on: 1 });
  to(S.verify, 8.15, 0.05, { status: 1, ease: "none" });
  to(S.verify, 8.3, 0.12, { status: 2, ease: "none" });
  to(S.verify, 8.5, 0.12, { status: 3, ease: "none" });
  to(S.act, 8.72, 0.1, { on: 0, keys: 0 });
  to(S.verify, 8.74, 0.1, { on: 0 });
  cam(8.76, 0.24, "hero");

  tl.set({}, {}, TIMELINE_LENGTH);
  return tl;
}
