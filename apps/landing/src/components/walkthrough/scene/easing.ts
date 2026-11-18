import * as THREE from "three";

// Easing curves shared by the scene. Each takes progress 0..1 (clamped) and returns
// eased progress.

const clamp01 = (t: number): number => THREE.MathUtils.clamp(t, 0, 1);

/** Starts slow, speeds up, settles: camera moves and things that travel. */
export const easeInOutCubic = (t: number): number => {
  const x = clamp01(t);

  return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2;
};

/** Decelerates into place: things that appear. */
export const easeOutCubic = (t: number): number => 1 - Math.pow(1 - clamp01(t), 3);

/** Overshoots by about 10% and settles back: things that land. */
export const easeOutBack = (t: number): number => {
  const x = clamp01(t) - 1;

  return 1 + 2.7 * x * x * x + 1.7 * x * x;
};

/** 0 before `inStart`, rising to 1 by `inEnd`, held, then falling to 0 from `outStart` to `outEnd`. */
export const fadeWindow = (t: number, inStart: number, inEnd: number, outStart: number, outEnd: number): number =>
  THREE.MathUtils.smoothstep(t, inStart, inEnd) * (1 - THREE.MathUtils.smoothstep(t, outStart, outEnd));
