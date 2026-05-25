/**
 * Capture the four arena UI panel screenshots that the landing page's
 * "See it in action" section embeds. Runs Playwright against a live
 * dashboard + api pair, drives the UI through a real run from the DuckDB
 * fixtures under logs/arena/, and writes PNGs to:
 *
 *   apps/landing/public/screenshots/arena/{world,trace,diff,operator}.png
 *
 * Usage (from repo root):
 *
 *   # Auto-start the servers, capture, tear down (recommended):
 *   bun --filter aoe2-llm-arena-web capture:screenshots
 *
 *   # If you already have api (:8000) and dashboard (:5173) running, the
 *   # script reuses them and skips startup.
 *
 * Requires Playwright Chromium (installed once via `bunx playwright install
 * chromium` from apps/landing/).
 */
import { chromium, type Browser, type Page } from "playwright";
import { spawn, type ChildProcess } from "node:child_process";
import { mkdir } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(SCRIPT_DIR, "../../..");
const SCREENSHOTS_DIR = resolve(REPO_ROOT, "apps/landing/public/screenshots/arena");
const API_BASE = "http://localhost:8000";
const DASHBOARD_BASE = "http://localhost:5173";
const VIEWPORT = { width: 1600, height: 900 } as const;
const PANELS = ["world", "trace", "diff", "operator"] as const;

async function probe(url: string): Promise<boolean> {
  try {
    const res = await fetch(url, { signal: AbortSignal.timeout(1500) });
    return res.ok || res.status === 404; // 404 = server up, route absent (fine)
  } catch {
    return false;
  }
}

async function waitForUp(url: string, label: string, timeoutMs = 30_000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await probe(url)) {
      console.log(`  ✓ ${label} up at ${url}`);
      return;
    }
    await new Promise((r) => setTimeout(r, 500));
  }
  throw new Error(`Timed out waiting for ${label} at ${url}`);
}

function spawnDetached(cmd: string, args: string[], env: Record<string, string>): ChildProcess {
  return spawn(cmd, args, {
    cwd: REPO_ROOT,
    env: { ...process.env, ...env },
    stdio: "ignore",
    detached: false,
  });
}

async function main(): Promise<void> {
  await mkdir(SCREENSHOTS_DIR, { recursive: true });

  // 1. Probe — reuse already-running servers if present.
  const apiUp = await probe(`${API_BASE}/health`);
  const dashboardUp = await probe(DASHBOARD_BASE);
  const spawned: ChildProcess[] = [];

  if (!apiUp) {
    console.log("→ Starting FastAPI api on :8000 …");
    spawned.push(
      spawnDetached("uv", ["run", "--package", "arena-web", "aoe2-arena-web", "--port", "8000"], {
        ARENA_LOGS_ROOT: resolve(REPO_ROOT, "logs/arena"),
      }),
    );
    await waitForUp(`${API_BASE}/health`, "api");
  } else {
    console.log("→ Reusing api already running on :8000");
  }

  if (!dashboardUp) {
    console.log("→ Starting Vite dashboard on :5173 …");
    spawned.push(spawnDetached("bun", ["--filter", "arena-web-ui", "dev"], {}));
    await waitForUp(DASHBOARD_BASE, "dashboard");
  } else {
    console.log("→ Reusing dashboard already running on :5173");
  }

  // 2. Drive the UI with Playwright.
  console.log("→ Launching Chromium …");
  const browser: Browser = await chromium.launch({ headless: true });
  try {
    const page: Page = await browser.newPage({ viewport: VIEWPORT });
    await page.goto(DASHBOARD_BASE, { waitUntil: "networkidle" });

    // 3. Pick the first run from the sidebar. RunList renders each run as a
    //    clickable shadcn Card (a div, not a button) inside <ol><li>. The
    //    first <li> on the page is the first run.
    console.log("→ Selecting first run from sidebar …");
    const firstRun = page.locator("ol > li").first();
    await firstRun.waitFor({ state: "visible", timeout: 15_000 });
    await firstRun.click();

    // Wait for events to start streaming — the status badge text contains
    // "events" once useEvents subscribes. Cold DuckDB fixtures load fast, so
    // the status often goes straight from "Connecting…" to "Complete".
    await page
      .getByText(/\d+ events/)
      .first()
      .waitFor({ state: "visible", timeout: 15_000 });
    await page.waitForTimeout(1500); // let a few turns paint

    // 4. Screenshot each panel. Tabs are Radix TabsTrigger; their visible
    //    text matches the panel name (capitalized), which is the most robust
    //    selector since Radix doesn't expose `value` as a DOM attribute.
    for (const panel of PANELS) {
      console.log(`→ Capturing ${panel} panel …`);
      const tabName = panel.charAt(0).toUpperCase() + panel.slice(1);
      await page.getByRole("tab", { name: tabName, exact: true }).click();
      await page.waitForTimeout(750); // tab content transition + initial paint

      const out = resolve(SCREENSHOTS_DIR, `${panel}.png`);
      await page.screenshot({ path: out, fullPage: false });
      console.log(`  ✓ wrote ${out}`);
    }
  } finally {
    await browser.close();

    // 5. Tear down servers we started.
    if (spawned.length > 0) {
      console.log(`→ Stopping ${spawned.length} spawned server(s) …`);
      for (const proc of spawned) {
        proc.kill("SIGTERM");
      }
    }
  }

  console.log("✓ Done. Screenshots saved under apps/landing/public/screenshots/arena/.");
}

main().catch((err: unknown) => {
  console.error("✗ Capture failed:", err);
  process.exit(1);
});
