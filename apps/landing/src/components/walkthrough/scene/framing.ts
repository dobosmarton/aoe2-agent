// Framing rules the page script shares with the scene. The page loads this before
// three.js, so it imports nothing.

/** Part of the view, in CSS pixels from its top-left, that no caption covers. */
export type SafeArea = { left: number; top: number; right: number; bottom: number };

// A pose's `size` is the half-height of a subject this much wider than tall; the
// camera fits that box into the safe area.
const SUBJECT_ASPECT = 1.3;

// The hero's text fades over this stretch of the scroll (timeline units), and the
// scene's framing hands over from the hero's safe area to the chapters' with it.
export const HERO_FADE_START = 0.62;
export const HERO_FADE_END = 0.92;

/** How much a view `height` CSS px tall widens so the subject box fits `area`. */
export const subjectFit = (height: number, area: SafeArea): number => {
  const width = Math.max(1, area.right - area.left);
  const safeHeight = Math.max(1, area.bottom - area.top);

  return Math.max(height / safeHeight, (SUBJECT_ASPECT * height) / width);
};
