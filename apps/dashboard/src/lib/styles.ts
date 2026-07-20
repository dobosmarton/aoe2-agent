/**
 * Tailwind class tokens for typographic roles used across several panels.
 *
 * A shared constant rather than a component: the call sites pass these to an
 * existing element's `className` (usually shadcn's CardTitle), so wrapping them
 * in a component would add a node just to carry a string.
 */

/** Small uppercase heading above a panel section. */
export const SECTION_TITLE =
  "text-muted-foreground text-xs font-semibold uppercase tracking-wide";
