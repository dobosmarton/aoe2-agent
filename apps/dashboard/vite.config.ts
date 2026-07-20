import path from "node:path";

import tailwindcss from "@tailwindcss/vite";
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
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { "@": path.resolve(__dirname, "src") },
  },
  server: {
    port: 5173,
    proxy: {
      "/runs": "http://localhost:8000",
      "/events": "http://localhost:8000",
      "/forks": "http://localhost:8000",
      "/health": "http://localhost:8000",
      // Training tracker API (apps/training-api on :8100).
      "/classes": "http://localhost:8100",
      "/images": "http://localhost:8100",
      "/datasets": "http://localhost:8100",
      "/stats": "http://localhost:8100",
      "/thumbs": "http://localhost:8100",
      "/raw": "http://localhost:8100",
    },
  },
});
