import { gsap } from "gsap";

import type { CameraPose } from "./rig";

// Every animated value of the scene lives in one object. A paused GSAP timeline tweens
// it, and the scroll position seeks the timeline: time t means "chapter floor(t),
// fraction t % 1 of the way through it". Chapter 0 is the hero, 1 to 8 are the
// walkthrough chapters, so the timeline is 9 units long. Each chapter enters, plays
// its action and then holds for the last stretch, so a reader can look before the next
// one starts.

export type SceneState = {
  /** The camera pose the shots tween. */
  cam: CameraPose;
  /** `cursor`: 0 at the start, 1 to 3 on the hovered house spots; `build`: construction stage 0..5. */
  world: { dim: number; ghost: number; build: number; cursor: number; track: number };
  /** `mix`: the game camera's render (0) to the real frame (1). */
  pane: { on: number; mix: number; flash: number; finder: number; impact: number };
  slab: { on: number; peel: number; travel: number };
  /** `pulse`: pulses run through the detector's plates, 0..5. */
  stack: { on: number; pulse: number };
  mind: { rail: number; space: number };
  action: { on: number; rail: number; travel: number; keys: number };
  frame: { scan: number; scanOn: number; dim: number; flash: number; grid: number; hudHole: number };
  boxes: { rise: number; alpha: number; labels: number; track: number };
  stamp: { on: number };
  hud: { lift: number; fields: number; values: number };
  flow: { t: number; on: number };
  sheets: { lift: number; seal: number };
  card: { on: number; travel: number };
  decide: { on: number; catalog: number; filter: number; pick: number };
  /** `keys`: key presses sent so far, 0..3. */
  act: { keys: number; cursor: number; click: number; placed: number; on: number };
  /** `status`: ledger rows reached, 0..3. */
  verify: { on: number; status: number };
};
export const createState = (hero: CameraPose): SceneState => ({
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
});

export const TIMELINE_LENGTH = 9;

export type Poses = Record<
  "hero" | "game" | "capture" | "detect" | "track" | "hud" | "publish" | "decide" | "loop" | "house" | "verify",
  CameraPose
>;

// Camera moves ease in and out; things that appear decelerate into place.
const MOVE = "power3.inOut";
const ENTER = "expo.out";

type Tweens = {
  state: SceneState;
  /** Tweens `target` to `vars` from timeline time `at` for `duration`. */
  to: (target: object, at: number, duration: number, vars: gsap.TweenVars) => void;
  /** Moves the camera to a shot. */
  cam: (at: number, duration: number, pose: keyof Poses, ease?: string) => void;
};

// Capture: rise into the game camera; the glass appears in front of the village,
// the viewfinder closes on it, the shot is taken and the real frame wipes in. The
// camera backs off through the glass and the frame peels off it.
const captureTweens = ({ state, to, cam }: Tweens): void => {
  cam(1, 0.36, "game");
  to(state.pane, 1.26, 0.1, { on: 1, ease: ENTER });
  to(state.pane, 1.28, 0.14, { finder: 1, ease: ENTER });
  to(state.stamp, 1.32, 0.1, { on: 1 });
  to(state.pane, 1.43, 0.02, { flash: 1, ease: "power1.out" });
  to(state.pane, 1.45, 0.1, { flash: 0, ease: "power2.out" });
  to(state.pane, 1.47, 0.24, { mix: 1, ease: "none" });
  to(state.world, 1.62, 0.25, { dim: 1 });
  to(state.mind, 1.66, 0.25, { space: 1 });
  cam(1.68, 0.26, "capture");
  to(state.pane, 1.74, 0.08, { finder: 0 });
  to(state.slab, 1.82, 0.02, { on: 1 });
  to(state.slab, 1.83, 0.12, { peel: 1 });
};

// Detect: the frame glides to the station; the detector's plates unfold beside it
// and a pulse runs through them. Then the boxes land, outwards from the Town Center.
const detectTweens = ({ state, to, cam }: Tweens): void => {
  to(state.stamp, 2, 0.08, { on: 0 });
  cam(2, 0.3, "detect");
  to(state.slab, 2, 0.26, { travel: 1, ease: MOVE });
  to(state.stack, 2.12, 0.2, { on: 1, ease: ENTER });
  to(state.stack, 2.26, 0.3, { pulse: 5, ease: "power1.inOut" });
  to(state.boxes, 2.5, 0.02, { alpha: 1 });
  to(state.boxes, 2.5, 0.28, { rise: 1, ease: "none" });
  to(state.boxes, 2.72, 0.1, { labels: 1, ease: ENTER });
};

// Track: back through the glass into the village, where each villager keeps its
// ID while it moves; the trail is where the tracker saw it on recent frames.
const trackTweens = ({ state, to, cam }: Tweens): void => {
  to(state.stack, 3, 0.1, { on: 0 });
  to(state.boxes, 3, 0.08, { track: 1 });
  cam(3, 0.34, "track");
  to(state.world, 3.05, 0.2, { dim: 0 });
  to(state.mind, 3.02, 0.14, { space: 0 });
  // The glass and the frame step aside while the camera is in the village.
  to(state.pane, 3.12, 0.08, { on: 0 });
  to(state.slab, 3.12, 0.04, { on: 0 });
  to(state.world, 3.3, 0.18, { track: 1, ease: ENTER });
};

// Read the HUD: back to the frame; the resource bar lifts off, its fields light up
// and the values count in.
const readHudTweens = ({ state, to, cam }: Tweens): void => {
  to(state.world, 4, 0.12, { track: 0 });
  to(state.world, 4.02, 0.2, { dim: 1 });
  to(state.mind, 4.04, 0.2, { space: 1 });
  to(state.pane, 4.05, 0.1, { on: 1 });
  to(state.slab, 4.08, 0.04, { on: 1 });
  to(state.boxes, 4, 0.15, { labels: 0, track: 0, alpha: 0.3 });
  cam(4, 0.32, "hud");
  to(state.hud, 4.24, 0.26, { lift: 1, ease: MOVE });
  to(state.frame, 4.24, 0.26, { hudHole: 1 });
  to(state.hud, 4.44, 0.12, { fields: 1, ease: ENTER });
  to(state.hud, 4.52, 0.26, { values: 1, ease: "none" });
};

// Publish: the boxes, the track IDs and the readings lift off as three sheets and
// seal into one Perception plate; the FramePipe rail lights up.
const publishTweens = ({ state, to, cam }: Tweens): void => {
  cam(5, 0.34, "publish");
  to(state.card, 5.06, 0.04, { on: 1 });
  to(state.boxes, 5.08, 0.2, { alpha: 0.6 });
  to(state.sheets, 5.08, 0.42, { lift: 1, ease: "none" });
  to(state.hud, 5.1, 0.2, { fields: 0, lift: 0 });
  to(state.frame, 5.1, 0.2, { hudHole: 0 });
  to(state.boxes, 5.2, 0.2, { alpha: 0.12 });
  to(state.flow, 5.1, 0.04, { on: 1 });
  to(state.flow, 5.12, 0.42, { t: 1, ease: "power1.inOut" });
  to(state.sheets, 5.42, 0.2, { seal: 1, ease: "none" });
  to(state.frame, 5.4, 0.3, { dim: 0.55 });
  to(state.mind, 5.58, 0.18, { rail: 1 });
  to(state.flow, 5.6, 0.06, { on: 0 });
  to(state.hud, 5.6, 0.05, { values: 0 });
};

// Decide: the plate rides the rail to the three loops; the catalog is filtered and
// the policy takes build_house.
const decideTweens = ({ state, to, cam }: Tweens): void => {
  cam(6, 0.34, "decide");
  to(state.card, 6, 0.34, { travel: 1, ease: "none" });
  to(state.decide, 6.12, 0.24, { on: 1, ease: ENTER });
  to(state.decide, 6.3, 0.22, { catalog: 1, ease: "none" });
  to(state.decide, 6.54, 0.14, { filter: 1, ease: "none" });
  to(state.decide, 6.7, 0.12, { pick: 1, ease: MOVE });
};

// Act: the picked tile leaves the policy ring as an action card. The camera pulls
// back to show the whole loop at once (glass, frame, Perception card, rings) while
// the card rides the return rail under the pipeline to the glass; it lands as the
// first key presses and the camera follows it through into the village. The house
// hovers over three spots, its footprint red, red, green; the click; it is built.
const actTweens = ({ state, to, cam }: Tweens): void => {
  cam(7, 0.11, "loop");
  to(state.decide, 7, 0.06, { catalog: 0, on: 0.45 });
  to(state.action, 7, 0.04, { on: 1, ease: ENTER });
  to(state.action, 7.03, 0.06, { rail: 1, ease: "power1.inOut" });
  to(state.action, 7.05, 0.14, { travel: 1, ease: "none" });
  cam(7.14, 0.07, "game");
  to(state.pane, 7.18, 0.06, { impact: 1, ease: "power2.out" });
  to(state.action, 7.18, 0.05, { keys: 1, ease: "none" });
  to(state.act, 7.19, 0.02, { on: 1 });
  to(state.act, 7.19, 0.04, { keys: 2, ease: "steps(2)" });
  to(state.decide, 7.18, 0.04, { on: 0 });
  to(state.card, 7.18, 0.04, { on: 0 });
  to(state.mind, 7.18, 0.04, { rail: 0 });
  to(state.action, 7.19, 0.04, { rail: 0 });
  to(state.slab, 7.18, 0.04, { on: 0 });
  to(state.mind, 7.19, 0.06, { space: 0 });
  to(state.world, 7.21, 0.12, { dim: 0 });
  to(state.action, 7.245, 0.02, { keys: 0, on: 0 });
  to(state.pane, 7.26, 0.025, { on: 0 });
  cam(7.24, 0.18, "house", "power3.out");
  to(state.world, 7.42, 0.04, { ghost: 1 });
  to(state.world, 7.42, 0.08, { cursor: 1, ease: "power2.out" });
  to(state.world, 7.54, 0.08, { cursor: 2, ease: "power2.inOut" });
  to(state.world, 7.66, 0.08, { cursor: 3, ease: "power2.inOut" });
  to(state.act, 7.76, 0.08, { click: 1, ease: "power2.out" });
  to(state.act, 7.76, 0.02, { placed: 1 });
  to(state.world, 7.77, 0.04, { ghost: 0 });
  to(state.world, 7.78, 0.18, { build: 5, ease: "none" });
  to(state.act, 7.88, 0.05, { keys: 3, ease: "none" });
};

// Verify: the ledger settles beside the new house, one line at a time; then back
// to the hero shot.
const verifyTweens = ({ state, to, cam }: Tweens): void => {
  cam(8, 0.3, "verify");
  to(state.verify, 8.05, 0.12, { on: 1, ease: ENTER });
  to(state.verify, 8.15, 0.05, { status: 1, ease: "none" });
  to(state.verify, 8.3, 0.12, { status: 2, ease: "none" });
  to(state.verify, 8.5, 0.12, { status: 3, ease: "none" });
  to(state.act, 8.72, 0.1, { on: 0, keys: 0 });
  to(state.verify, 8.74, 0.1, { on: 0 });
  cam(8.76, 0.24, "hero");
};

export const buildTimeline = (state: SceneState, poses: Poses): gsap.core.Timeline => {
  const timeline = gsap.timeline({ paused: true, defaults: { ease: "power2.inOut" } });
  const to: Tweens["to"] = (target, at, duration, vars) => {
    timeline.to(target, { ...vars, duration }, at);
  };
  const cam: Tweens["cam"] = (at, duration, pose, ease = MOVE) => to(state.cam, at, duration, { ...poses[pose], ease });
  const tweens = { state, to, cam };
  captureTweens(tweens);
  detectTweens(tweens);
  trackTweens(tweens);
  readHudTweens(tweens);
  publishTweens(tweens);
  decideTweens(tweens);
  actTweens(tweens);
  verifyTweens(tweens);
  timeline.set({}, {}, TIMELINE_LENGTH);
  return timeline;
};
