import { defineCollection, z } from "astro:content";
import { glob } from "astro/loaders";

// Source markdown lives in `docs/` at the repo root — two levels up from
// this file (apps/landing/src/content.config.ts → ../../docs). We use Astro's
// `glob` loader so the content layer is the single source of truth for which
// docs exist; everything else (sidebar, prev/next, search index, sitemap)
// derives from this collection.
//
// Schema is intentionally loose: most docs in this repo have no frontmatter,
// and the few that do (Obsidian-style notes under `docs/explorations/`) use
// a variety of keys. Optional() everywhere lets us ingest them all without
// rejecting any.
const docs = defineCollection({
  loader: glob({
    pattern: "**/*.md",
    base: "../../docs",
  }),
  schema: z.object({
    title: z.string().optional(),
    description: z.string().optional(),
    part: z.number().int().min(1).max(8).optional(),
    chapter: z.number().int().optional(),
    type: z
      .enum(["adr", "runbook", "design", "exploration", "tutorial", "deployment"])
      .optional(),
    status: z.string().optional(),
    phase: z.string().optional(),
    capability: z.string().optional(),
    related: z.array(z.string()).optional(),
  }),
});

export const collections = { docs };
