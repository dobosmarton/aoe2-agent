import { defineCollection, z } from "astro:content";
import { glob } from "astro/loaders";

// Source markdown lives in `docs/` at the repo root, two levels up from
// this file (apps/landing/src/content.config.ts → ../../docs). We use Astro's
// `glob` loader so the content layer is the single source of truth for which
// docs exist; everything else (sidebar, prev/next, search index, sitemap)
// derives from this collection.
//
// The docs carry no frontmatter: a page's title comes from its first H1.
// `title` and `description` stay as optional overrides because the doc page
// route reads them; any other frontmatter key is ignored.
const docs = defineCollection({
  loader: glob({
    pattern: "**/*.md",
    base: "../../docs",
  }),
  schema: z.object({
    title: z.string().optional(),
    description: z.string().optional(),
  }),
});

export const collections = { docs };
