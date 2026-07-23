import path from "node:path";

import tailwindcss from "@tailwindcss/vite";
import { tanstackRouter } from "@tanstack/router-plugin/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Backend routing:
//   - Default (no env var): the dev proxy below forwards /runs, /events,
//     /forks, /health to FastAPI on :8000. Same relative URLs work in the
//     prod build, where the SPA is served from the API's origin.
//   - Cross-origin: set VITE_API_BASE_URL=http://host:port to bypass the
//     proxy entirely (e.g. backend on a VM, frontend dev locally). The
//     backend must allow the SPA origin via ARENA_WEB_CORS_ORIGINS.
export default defineConfig({
  plugins: [
    // Must precede react(): the router plugin rewrites route modules before
    // the React refresh transform sees them. It generates src/routeTree.gen.ts
    // from src/routes/**, which is committed because `bun run build` runs
    // `tsc -b` before vite — a missing tree would fail the build on a fresh
    // clone, before the plugin ever had a chance to write it.
    tanstackRouter({ target: "react", autoCodeSplitting: true }),
    // React Compiler memoises components and hooks automatically, which is why
    // the panels below carry almost no useMemo/useCallback by hand. Where it
    // cannot compile a component it bails out silently at build time — the
    // react-hooks lint rules surface those bail-outs so they stay visible.
    react({ babel: { plugins: ["babel-plugin-react-compiler"] } }),
    tailwindcss(),
  ],
  resolve: {
    alias: { "@": path.resolve(__dirname, "src") },
  },
  server: {
    port: 5173,
    proxy: {
      // `/runs` is both an API path AND a client route (/runs, /runs/$runId).
      // The proxy is matched before the SPA fallback, so without this a browser
      // navigation to /runs/<id> would render the raw JSON run list. Document
      // requests ask for text/html; fetch()/XHR never do — so that header is
      // what separates "show me the app" from "give me the data".
      "/runs": {
        target: "http://localhost:8000",
        bypass: (req) =>
          req.headers.accept?.includes("text/html") === true ? "/index.html" : undefined,
      },
      "/events": "http://localhost:8000",
      "/forks": "http://localhost:8000",
      "/health": "http://localhost:8000",
      // Training tracker API (apps/training-api on :8100).
      "/classes": "http://localhost:8100",
      "/images": "http://localhost:8100",
      "/annotations": "http://localhost:8100",
      "/datasets": "http://localhost:8100",
      "/stats": "http://localhost:8100",
      "/thumbs": "http://localhost:8100",
      "/raw": "http://localhost:8100",
    },
  },
});
