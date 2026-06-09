// Shared constants for the operator fork form. Non-component module so the
// panel, its field components, and the mutation summary can share them without
// import cycles.

import type { Age, MutationPatch } from "@/lib/api";

/** Age options for the fork age override, in chronological order. */
export const AGES: readonly Age[] = [
  "Dark Age",
  "Feudal Age",
  "Castle Age",
  "Imperial Age",
] as const;

/** Mutation patch fields, in display order. */
export const MUTATION_FIELDS: ReadonlyArray<keyof MutationPatch> = [
  "food",
  "wood",
  "gold",
  "stone",
  "population",
  "pop_cap",
  "age",
] as const;

/** Theme color token per resource field (non-resource fields have no swatch). */
export const RESOURCE_COLORS: Partial<Record<keyof MutationPatch, string>> = {
  food: "var(--food)",
  wood: "var(--wood)",
  gold: "var(--gold)",
  stone: "var(--stone)",
};
