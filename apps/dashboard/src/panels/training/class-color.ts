/**
 * Golden-angle hue rotation: consecutive class ids land far apart on the wheel,
 * so neighbouring classes stay distinguishable without a hand-picked palette.
 *
 * Shared by the annotation overlay and the legend, which must agree on colour.
 */
export const classColor = (classId: number, alpha = 1): string =>
  `hsl(${String((classId * 137.508) % 360)} 90% 60% / ${String(alpha)})`;
