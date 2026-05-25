import { defineConfig } from "astro/config";
import react from "@astrojs/react";
import mdx from "@astrojs/mdx";
import sitemap from "@astrojs/sitemap";
import tailwindcss from "@tailwindcss/vite";
import rehypeMermaid from "rehype-mermaid";
import rehypeSlug from "rehype-slug";
import rehypeAutolinkHeadings from "rehype-autolink-headings";
import remarkGfm from "remark-gfm";
import { remarkRewriteMdLinks } from "./src/lib/remark-rewrite-md-links.ts";

// Update SITE to your production URL before deploy.
const SITE = process.env.SITE_URL ?? "https://aoe2-llm-arena.example.com";

export default defineConfig({
  site: SITE,
  output: "static",
  integrations: [react(), mdx(), sitemap()],
  vite: {
    plugins: [tailwindcss()],
    resolve: {
      alias: {
        "@": new URL("./src", import.meta.url).pathname,
      },
    },
  },
  markdown: {
    // Shiki runs before our rehype plugins. We exclude `mermaid` so the
    // <pre><code class="language-mermaid"> blocks survive into the rehype
    // stage where rehype-mermaid can replace them with inline SVG.
    syntaxHighlight: {
      type: "shiki",
      excludeLangs: ["mermaid"],
    },
    shikiConfig: {
      themes: { light: "github-light", dark: "github-dark" },
      wrap: true,
    },
    remarkPlugins: [remarkGfm, remarkRewriteMdLinks],
    rehypePlugins: [
      // Build-time mermaid -> SVG via headless Chromium (Playwright).
      // strategy "inline-svg" embeds the SVG directly in the page; ships zero JS.
      // errorFallback keeps the block as a code-styled element if rendering
      // fails (e.g. Chromium not installed during `astro dev`).
      [
        rehypeMermaid,
        {
          strategy: "inline-svg",
          dark: true,
          errorFallback: (node, error, file) => {
            file.message(`mermaid render failed: ${String(error)}`);
            return {
              type: "element",
              tagName: "pre",
              properties: {
                className: ["mermaid-error"],
                "data-language": "mermaid",
              },
              children: [
                {
                  type: "element",
                  tagName: "code",
                  properties: {},
                  children: node.children,
                },
              ],
            };
          },
        },
      ],
      rehypeSlug,
      [
        rehypeAutolinkHeadings,
        {
          behavior: "append",
          properties: { className: ["heading-anchor"], ariaLabel: "Permalink" },
        },
      ],
    ],
  },
});
