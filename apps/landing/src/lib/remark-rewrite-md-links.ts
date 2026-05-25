import path from "node:path";
import type { Root } from "mdast";
import type { Plugin } from "unified";
import type { VFile } from "vfile";
import { visit } from "unist-util-visit";

/**
 * Rewrite relative `.md` link targets to their resolved `/docs/<slug>` URL
 * form.
 *
 * Source markdown in `docs/` uses GitHub-style relative links like
 * `./04-provider-pattern.md` or `../part1-architecture/02-game-loop-pipeline.md`.
 * Those work on GitHub (which transforms them at render time) but break in
 * the built site because the routes are `/docs/<part>/<chapter>` with no
 * `.md` extension.
 *
 * This plugin walks markdown link nodes, resolves relative `.md` targets
 * against the source file's directory, then rewrites them to a `/docs/...`
 * URL stripped of the `.md` extension. Anchors are preserved.
 *
 * Non-rewrites:
 * - http(s):// and mailto: (external)
 * - links starting with `/` (already absolute)
 * - pure anchors like `#section`
 * - links to non-`.md` files (images, code, etc.)
 * - links whose resolved target falls outside the `docs/` tree
 */
export const remarkRewriteMdLinks: Plugin<[], Root> = () => {
  return (tree, file: VFile) => {
    const sourcePath = file.history?.[0] ?? file.path;
    if (!sourcePath || typeof sourcePath !== "string") return;
    const sourceDir = path.dirname(sourcePath);

    visit(tree, "link", (node) => {
      const url = node.url;
      if (typeof url !== "string") return;
      if (/^(https?:|mailto:|tel:|#|\/\/)/.test(url)) return;
      if (url.startsWith("/")) return;
      if (!/\.md($|#)/i.test(url)) return;

      const hashIdx = url.indexOf("#");
      const targetPath = hashIdx >= 0 ? url.slice(0, hashIdx) : url;
      const anchor = hashIdx >= 0 ? url.slice(hashIdx) : "";

      const resolved = path.resolve(sourceDir, targetPath);

      // Locate the `docs/` segment in the resolved path. The actual docs
      // directory might be `/Users/.../agent/docs/...`, so we look for the
      // last occurrence of `/docs/` and treat everything after as the slug.
      const docsMarker = `${path.sep}docs${path.sep}`;
      const docsIdx = resolved.lastIndexOf(docsMarker);
      if (docsIdx < 0) return;

      const relative = resolved.slice(docsIdx + docsMarker.length);
      const slug = relative.replace(/\\/g, "/").replace(/\.md$/i, "");

      node.url = `/docs/${slug}${anchor}`;
    });
  };
};
