import { gsap } from "gsap";

import type { CameraPose } from "./rig";

// Every animated value of the scene lives in one object. A paused GSAP timeline tweens
// it, and the scroll position seeks the timeline: time t means "chapter floor(t),
// fraction t % 1 of the way through it". Chapter 0 is the hero, 1 to 8 are the
// walkthrough chapters, so the timeline is 9 units long. Each chapter enters, plays
// its action and then holds for the last stretch, so a reader can look before the next
// one starts.
export function createState(hero: CameraPose) {
  return {
    cam: { ...hero },
    world: { dim: 0, ghost: 0, build: 0, cursor: 0, track: 0 },
    pane: { on: 0, mix: 0, flash: 0, finder: 0, impact: 0 },
    slab: { on: 0, peel: 0, travel: 0 },
    stack: { on: 0, pulse: 0 },
    mind: { rail: 0, space: 0 },
    action: { on: 0, rail: 0, travel: 0, keys: 0 },
    frame: { scan: 1, scanOn: 0, dim: 0, flash: 0, grid: 0, hudHole: 0 },
    boxes: { rise: 0, alpha: 0, labels: 0, track: 0 },
    stamp: { on: 0 },
    hud: { lift: 0, fields: 0, values: 0 },
    flow: { t: 0, on: 0 },
    sheets: { lift: 0, seal: 0 },
    card: { on: 0, travel: 0 },
    decide: { on: 0, catalog: 0, filter: 0, pick: 0 },
    act: { keys: 0, cursor: 0, click: 0, placed: 0, on: 0 },
    verify: { on: 0, status: 0 },
  };
}

export type SceneState = ReturnType<typeof createState>;

export const TIMELINE_LENGTH = 9;

export type Poses = Record<
  "hero" | "game" | "capture" | "detect" | "track" | "hud" | "publish" | "decide" | "loop" | "house" | "verify",
  CameraPose
>;

// Camera moves ease in and out; things that appear decelerate into place.
const MOVE = "power3.inOut";
const ENTER = "expo.out";

export function buildTimeline(S: SceneState, P: Poses): gsap.core.Timeline {
  const tl = gsap.timeline({ paused: true, defaults: { ease: "power2.inOut" } });
  const to = (target: object, at: number, dur: number, vars: gsap.TweenVars) => tl.to(target, { ...vars, duration: dur }, at);
  const cam = (at: number, dur: number, pose: keyof Poses, ease = MOVE) => to(S.cam, at, dur, { ...P[pose], ease });

  // 1 Capture: rise into the game camera; the glass appears in front of the village,
  //   the viewfinder closes on it, the shot is taken and the real frame wipes in. The
  //   camera backs off through the glass and the frame peels off it.
  cam(1, 0.36, "game");
  to(S.pane, 1.26, 0.1, { on: 1, ease: ENTER });
  to(S.pane, 1.28, 0.14, { finder: 1, ease: ENTER });
  to(S.stamp, 1.32, 0.1, { on: 1 });
  to(S.pane, 1.43, 0.02, { flash: 1, ease: "power1.out" });
  to(S.pane, 1.45, 0.1, { flash: 0, ease: "power2.out" });
  to(S.pane, 1.47, 0.24, { mix: 1, ease: "none" });
  to(S.world, 1.62, 0.25, { dim: 1 });
  to(S.mind, 1.66, 0.25, { space: 1 });
  cam(1.68, 0.26, "capture");
  to(S.pane, 1.74, 0.08, { finder: 0 });
  to(S.slab, 1.82, 0.02, { on: 1 });
  to(S.slab, 1.83, 0.12, { peel: 1 });

  // 2 Detect: the frame glides to the station; the detector's plates unfold beside it
  //   and a pulse runs through them. Then the boxes land, outwards from the Town Center.
  to(S.stamp, 2, 0.08, { on: 0 });
  cam(2, 0.3, "detect");
  to(S.slab, 2, 0.26, { travel: 1, ease: MOVE });
  to(S.stack, 2.12, 0.2, { on: 1, ease: ENTER });
  to(S.stack, 2.26, 0.3, { pulse: 5, ease: "power1.inOut" });
  to(S.boxes, 2.5, 0.02, { alpha: 1 });
  to(S.boxes, 2.5, 0.28, { rise: 1, ease: "none" });
  to(S.boxes, 2.72, 0.1, { labels: 1, ease: ENTER });

  // 3 Track: back through the glass into the village, where each villager keeps its
  //   ID while it moves; the trail is where the tracker saw it on recent frames.
  to(S.stack, 3, 0.1, { on: 0 });
  to(S.boxes, 3, 0.08, { track: 1 });
  cam(3, 0.34, "track");
  to(S.world, 3.05, 0.2, { dim: 0 });
  to(S.mind, 3.02, 0.14, { space: 0 });
  // The glass and the frame step aside while the camera is in the village.
  to(S.pane, 3.12, 0.08, { on: 0 });
  to(S.slab, 3.12, 0.04, { on: 0 });
  to(S.world, 3.3, 0.18, { track: 1, ease: ENTER });

  // 4 Read the HUD: back to the frame; the resource bar lifts off, its fields light up
  //   and the values count in.
  to(S.world, 4, 0.12, { track: 0 });
  to(S.world, 4.02, 0.2, { dim: 1 });
  to(S.mind, 4.04, 0.2, { space: 1 });
  to(S.pane, 4.05, 0.1, { on: 1 });
  to(S.slab, 4.08, 0.04, { on: 1 });
  to(S.boxes, 4, 0.15, { labels: 0, track: 0, alpha: 0.3 });
  cam(4, 0.32, "hud");
  to(S.hud, 4.24, 0.26, { lift: 1, ease: MOVE });
  to(S.frame, 4.24, 0.26, { hudHole: 1 });
  to(S.hud, 4.44, 0.12, { fields: 1, ease: ENTER });
  to(S.hud, 4.52, 0.26, { values: 1, ease: "none" });

  // 5 Publish: the boxes, the track IDs and the readings lift off as three sheets and
  //   seal into one Perception plate; the FramePipe rail lights up.
  cam(5, 0.34, "publish");
  to(S.card, 5.06, 0.04, { on: 1 });
  to(S.boxes, 5.08, 0.2, { alpha: 0.6 });
  to(S.sheets, 5.08, 0.42, { lift: 1, ease: "none" });
  to(S.hud, 5.1, 0.2, { fields: 0, lift: 0 });
  to(S.frame, 5.1, 0.2, { hudHole: 0 });
  to(S.boxes, 5.2, 0.2, { alpha: 0.12 });
  to(S.flow, 5.1, 0.04, { on: 1 });
  to(S.flow, 5.12, 0.42, { t: 1, ease: "power1.inOut" });
  to(S.sheets, 5.42, 0.2, { seal: 1, ease: "none" });
  to(S.frame, 5.4, 0.3, { dim: 0.55 });
  to(S.mind, 5.58, 0.18, { rail: 1 });
  to(S.flow, 5.6, 0.06, { on: 0 });
  to(S.hud, 5.6, 0.05, { values: 0 });

  // 6 Decide: the plate rides the rail to the three loops; the catalog is filtered and
  //   the policy takes build_house.
  cam(6, 0.34, "decide");
  to(S.card, 6, 0.34, { travel: 1, ease: "none" });
  to(S.decide, 6.12, 0.24, { on: 1, ease: ENTER });
  to(S.decide, 6.3, 0.22, { catalog: 1, ease: "none" });
  to(S.decide, 6.54, 0.14, { filter: 1, ease: "none" });
  to(S.decide, 6.7, 0.12, { pick: 1, ease: MOVE });

  // 7 Act: the picked tile leaves the policy ring as an action card. The camera pulls
  //   back to show the whole loop at once (glass, frame, Perception card, rings) while
  //   the card rides the return rail under the pipeline to the glass; it lands as the
  //   first key presses and the camera follows it through into the village. The house
  //   hovers over three spots, its footprint red, red, green; the click; it is built.
  cam(7, 0.11, "loop");
  to(S.decide, 7, 0.06, { catalog: 0, on: 0.45 });
  to(S.action, 7, 0.04, { on: 1, ease: ENTER });
  to(S.action, 7.03, 0.06, { rail: 1, ease: "power1.inOut" });
  to(S.action, 7.05, 0.14, { travel: 1, ease: "none" });
  cam(7.14, 0.07, "game");
  to(S.pane, 7.18, 0.06, { impact: 1, ease: "power2.out" });
  to(S.action, 7.18, 0.05, { keys: 1, ease: "none" });
  to(S.act, 7.19, 0.02, { on: 1 });
  to(S.act, 7.19, 0.04, { keys: 2, ease: "steps(2)" });
  to(S.decide, 7.18, 0.04, { on: 0 });
  to(S.card, 7.18, 0.04, { on: 0 });
  to(S.mind, 7.18, 0.04, { rail: 0 });
  to(S.action, 7.19, 0.04, { rail: 0 });
  to(S.slab, 7.18, 0.04, { on: 0 });
  to(S.mind, 7.19, 0.06, { space: 0 });
  to(S.world, 7.21, 0.12, { dim: 0 });
  to(S.action, 7.245, 0.02, { keys: 0, on: 0 });
  to(S.pane, 7.26, 0.025, { on: 0 });
  cam(7.24, 0.18, "house", "power3.out");
  to(S.world, 7.42, 0.04, { ghost: 1 });
  to(S.world, 7.42, 0.08, { cursor: 1, ease: "power2.out" });
  to(S.world, 7.54, 0.08, { cursor: 2, ease: "power2.inOut" });
  to(S.world, 7.66, 0.08, { cursor: 3, ease: "power2.inOut" });
  to(S.act, 7.76, 0.08, { click: 1, ease: "power2.out" });
  to(S.act, 7.76, 0.02, { placed: 1 });
  to(S.world, 7.77, 0.04, { ghost: 0 });
  to(S.world, 7.78, 0.18, { build: 5, ease: "none" });
  to(S.act, 7.88, 0.05, { keys: 3, ease: "none" });

  // 8 Verify: the ledger settles beside the new house, one line at a time; then back
  //   to the hero shot.
  cam(8, 0.3, "verify");
  to(S.verify, 8.05, 0.12, { on: 1, ease: ENTER });
  to(S.verify, 8.15, 0.05, { status: 1, ease: "none" });
  to(S.verify, 8.3, 0.12, { status: 2, ease: "none" });
  to(S.verify, 8.5, 0.12, { status: 3, ease: "none" });
  to(S.act, 8.72, 0.1, { on: 0, keys: 0 });
  to(S.verify, 8.74, 0.1, { on: 0 });
  cam(8.76, 0.24, "hero");

  tl.set({}, {}, TIMELINE_LENGTH);
  return tl;
}
