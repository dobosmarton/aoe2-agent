import path from "node:path";

import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Dev-server proxy forwards backend calls to FastAPI (Phase 7.1) on :8000.
// In production the SPA is built to dist/ and served from the same origin as
// the API, so the same relative URLs (/runs, /events, /health) just work.
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
      "/health": "http://localhost:8000",
    },
  },
});
