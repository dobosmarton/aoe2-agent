/**
 * Render the home page walkthrough at each chapter and save the frames.
 *
 *   bun run capture:walkthrough            # stills for static mode and the poster
 *   bun run capture:walkthrough -- --review # views at desktop and phone size
 *   bun run capture:walkthrough -- --review --static # the reduced-motion fallback, full page
 *
 * Stills go to public/walkthrough/stills/<chapter>.jpg (plus hero.jpg, the poster, and
 * public/og-image.png). Review shots go to the directory in WT_REVIEW_DIR (default
 * tmp/walkthrough-review). Needs the site served at WT_BASE (default
 * http://localhost:4321, e.g. `bun run preview`) and Playwright Chromium.
 */
import { chromium, type Page } from "playwright";
import { mkdir } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const LANDING = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const BASE = process.env.WT_BASE ?? "http://localhost:4321";
const REVIEW = process.argv.includes("--review");
const STATIC = process.argv.includes("--static");
const REVIEW_DIR = resolve(process.env.WT_REVIEW_DIR ?? resolve(LANDING, "../../tmp/walkthrough-review"));
const STILLS_DIR = resolve(LANDING, "public/walkthrough/stills");
// Metal on macOS renders the scene as a browser would; elsewhere fall back to SwiftShader.
const GPU_ARGS =
  process.platform === "darwin"
    ? ["--use-angle=metal", "--enable-gpu", "--ignore-gpu-blocklist"]
    : ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"];
const CHAPTERS = ["hero", "capture", "detect", "track", "hud", "publish", "decide", "act", "verify"];

// Scrolls so the scene sits at timeline time `p`, then waits until it has caught up.
async function seek(page: Page, p: number) {
  await page.evaluate((p) => {
    const articles = [...document.querySelectorAll<HTMLElement>("[data-wt-chapter]")];
    const i = Math.min(Math.floor(p), articles.length - 1);
    const rect = articles[i]!.getBoundingClientRect();
    const top = rect.top + window.scrollY;
    window.scrollTo({ top: top + (p - i) * rect.height - window.innerHeight / 2, behavior: "instant" });
  }, p);
  await page.waitForFunction(
    (p) => {
      const stage = document.querySelector("[data-wt-stage]") as (HTMLElement & { wtProgress?: number }) | null;
      return stage?.wtProgress != null && Math.abs(stage.wtProgress - p) < 0.003;
    },
    p,
    { timeout: 30_000, polling: 100 },
  );
  await page.waitForTimeout(400);
}

async function open(page: Page) {
  await page.goto(BASE, { waitUntil: "networkidle" });
  await page.waitForSelector("#walkthrough[data-ready]", { timeout: 60_000 });
}

async function stills() {
  await mkdir(STILLS_DIR, { recursive: true });
  const browser = await chromium.launch({ args: GPU_ARGS });
  const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
  await open(page);
  // Stills show the scene alone, centred: no text, progress, shade or header.
  await page.addStyleTag({
    content: `.wt-chapters > *, .wt-chapters .wt-card { visibility: hidden !important; }
      .wt-shade, .wt-progress, [data-site-header] { display: none !important; }`,
  });
  await page.evaluate(() => (document.querySelector("[data-wt-stage]") as HTMLElement & { wtDebug: { setCentred(on: boolean): void } }).wtDebug.setCentred(true));
  const stage = page.locator("[data-wt-stage]");
  for (const [i, id] of CHAPTERS.entries()) {
    await seek(page, i === 0 ? 0.5 : i + 0.88);
    await stage.screenshot({ path: resolve(STILLS_DIR, `${id}.jpg`), type: "jpeg", quality: 80 });
    if (i === 0) {
      await page.setViewportSize({ width: 1200, height: 630 });
      await seek(page, 0.5);
      await stage.screenshot({ path: resolve(LANDING, "public/og-image.png") });
      await page.setViewportSize({ width: 1600, height: 900 });
    }
    console.log(`  ✓ ${id}`);
  }
  await browser.close();
}

async function review() {
  await mkdir(REVIEW_DIR, { recursive: true });
  const browser = await chromium.launch({ args: GPU_ARGS });
  for (const [name, viewport] of [
    ["desktop", { width: 1440, height: 900 }],
    ["phone", { width: 390, height: 844 }],
  ] as const) {
    const page = await browser.newPage({ viewport, deviceScaleFactor: 1, reducedMotion: STATIC ? "reduce" : "no-preference" });
    const errors: string[] = [];
    page.on("console", (m) => (m.type() === "error" || m.type() === "warning") && errors.push(m.text()));
    page.on("pageerror", (e) => errors.push(String(e)));
    if (STATIC) {
      await page.goto(BASE, { waitUntil: "networkidle" });
      const mode = await page.locator("#walkthrough").getAttribute("data-mode");
      await page.evaluate(() => document.querySelectorAll("img[loading=lazy]").forEach((i) => i.setAttribute("loading", "eager")));
      await page.waitForTimeout(1500);
      const loaded = await page.evaluate(() =>
        [...document.querySelectorAll<HTMLImageElement>(".wt-still")].filter((i) => i.naturalWidth > 0).length,
      );
      for (const k of [0, 1, 2, 3]) {
        const clip = { x: 0, y: k * viewport.height * 1.2, width: viewport.width, height: viewport.height };
        await page.screenshot({ path: resolve(REVIEW_DIR, `${name}-static-${k}.jpg`), type: "jpeg", quality: 60, fullPage: true, clip });
      }
      console.log(`${name}: mode=${mode}, stills loaded ${loaded}, ${errors.length} console errors/warnings`);
      await page.close();
      continue;
    }
    await open(page);
    const points = process.env.WT_POINTS?.split(",").map(Number) ?? CHAPTERS.map((_, i) => (i === 0 ? 0.5 : i + 0.85));
    for (const p of points) {
      await seek(page, p);
      await page.screenshot({ path: resolve(REVIEW_DIR, `${name}-${p.toFixed(2)}.jpg`), type: "jpeg", quality: 70 });
    }
    console.log(`${name}: ${errors.length} console errors/warnings`);
    for (const e of errors.slice(0, 10)) console.log("   ", e.slice(0, 200));
    await page.close();
  }
  await browser.close();
}

await (REVIEW ? review() : stills());
