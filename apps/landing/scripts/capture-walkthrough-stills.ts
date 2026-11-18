/**
 * Render the home page walkthrough at each chapter and save the frames.
 *
 *   bun run capture:walkthrough            # stills for static mode and the poster
 *   bun run capture:walkthrough -- --review # views at desktop, tablet and phone sizes
 *   bun run capture:walkthrough -- --review --static # the reduced-motion fallback, full page
 *
 * Stills go to public/walkthrough/stills/<chapter>.jpg (plus hero.jpg, the poster, and
 * public/og-image.jpg). Review shots go to the directory in WT_REVIEW_DIR (default
 * tmp/walkthrough-review). Needs the site served at WT_BASE (default
 * http://localhost:4321, e.g. `bun run preview`) and Playwright Chromium.
 */
import { chromium, type Browser, type Page } from "playwright";
import { mkdir } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import type { DebugStage } from "../src/components/walkthrough/scene";

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

const STILL_VIEWPORT = { width: 1600, height: 900 };
const OG_IMAGE_VIEWPORT = { width: 1200, height: 630 };

// Timeline points: the hero is taken halfway through its chapter, every other chapter
// near its end, once its build has finished.
const HERO_PROGRESS = 0.5;
const STILL_CHAPTER_OFFSET = 0.88;
const REVIEW_CHAPTER_OFFSET = 0.85;

// How close the scene's progress must come to the target before a frame is taken, and
// how long it then gets to settle.
const PROGRESS_TOLERANCE = 0.003;
const SETTLE_MS = 400;
const SEEK_TIMEOUT_MS = 30_000;
const READY_TIMEOUT_MS = 60_000;

// Static review: time for the lazy stills to load, and how many viewport-high slices of
// the page to save (each one 1.2 viewports below the last).
const STATIC_IMAGE_WAIT_MS = 1500;
const STATIC_SLICES = 4;
const STATIC_SLICE_STRIDE = 1.2;
const MAX_LOGGED_ERRORS = 10;

// The debug hooks the walkthrough scene sets on its stage element.
type Viewport = { width: number; height: number };

type ReviewSize = { name: string; viewport: Viewport; mobile: boolean };

const chapterPoint = (index: number, offset: number): number => (index === 0 ? HERO_PROGRESS : index + offset);

// Scrolls so the scene sits at timeline time `progress`, then waits until it has caught up.
// The callbacks run in the page, so they get their values as arguments.
const seek = async (page: Page, progress: number): Promise<void> => {
  await page.evaluate((target) => {
    const articles = [...document.querySelectorAll<HTMLElement>("[data-wt-chapter]")];
    const index = Math.min(Math.floor(target), articles.length - 1);
    const article = articles[index];
    if (!article) throw new Error("The walkthrough has no chapters to scroll to");
    const rect = article.getBoundingClientRect();
    const top = rect.top + window.scrollY;
    window.scrollTo({ top: top + (target - index) * rect.height - window.innerHeight / 2, behavior: "instant" });
  }, progress);
  await page.waitForFunction(
    ({ target, tolerance }) => {
      const stage = document.querySelector<DebugStage>("[data-wt-stage]");
      return stage?.wtProgress != null && Math.abs(stage.wtProgress - target) < tolerance;
    },
    { target: progress, tolerance: PROGRESS_TOLERANCE },
    { timeout: SEEK_TIMEOUT_MS, polling: 100 },
  );
  await page.waitForTimeout(SETTLE_MS);
};

const open = async (page: Page): Promise<void> => {
  await page.goto(BASE, { waitUntil: "networkidle" });
  await page.waitForSelector("#walkthrough[data-ready]", { timeout: READY_TIMEOUT_MS });
};

// Stills show the scene alone, centred: no text, progress, shade or header.
const isolateScene = async (page: Page): Promise<void> => {
  await page.addStyleTag({
    content: `.wt-chapters > *, .wt-chapters .wt-card { visibility: hidden !important; }
      .wt-shade, .wt-progress, [data-site-header] { display: none !important; }`,
  });
  await page.evaluate(() => {
    const stage = document.querySelector<DebugStage>("[data-wt-stage]");
    if (!stage?.wtDebug) throw new Error("The walkthrough stage has no debug hooks");
    stage.wtDebug.setCentred(true);
  });
};

const captureOgImage = async (page: Page): Promise<void> => {
  await page.setViewportSize(OG_IMAGE_VIEWPORT);
  await seek(page, HERO_PROGRESS);
  await page.locator("[data-wt-stage]").screenshot({ path: resolve(LANDING, "public/og-image.jpg"), type: "jpeg", quality: 85 });
  await page.setViewportSize(STILL_VIEWPORT);
};

const stills = async (): Promise<void> => {
  await mkdir(STILLS_DIR, { recursive: true });
  const browser = await chromium.launch({ args: GPU_ARGS });
  const page = await browser.newPage({ viewport: STILL_VIEWPORT });
  await open(page);
  await isolateScene(page);
  const stage = page.locator("[data-wt-stage]");
  for (const [index, id] of CHAPTERS.entries()) {
    await seek(page, chapterPoint(index, STILL_CHAPTER_OFFSET));
    await stage.screenshot({ path: resolve(STILLS_DIR, `${id}.jpg`), type: "jpeg", quality: 80 });
    if (index === 0) await captureOgImage(page);
    console.log(`  ✓ ${id}`);
  }
  await browser.close();
};

// Review sizes. Phones and tablets are emulated as such (mobile viewport, touch), so
// anything that widens the page shows up as it would on the device.
const REVIEW_SIZES: readonly ReviewSize[] = [
  { name: "desktop", viewport: { width: 1440, height: 900 }, mobile: false },
  { name: "tablet", viewport: { width: 768, height: 1024 }, mobile: true },
  { name: "phone", viewport: { width: 390, height: 844 }, mobile: true },
  { name: "phone-small", viewport: { width: 360, height: 640 }, mobile: true },
  { name: "phone-landscape", viewport: { width: 844, height: 390 }, mobile: true },
];

const openReviewPage = async (browser: Browser, { viewport, mobile }: ReviewSize, errors: string[]): Promise<Page> => {
  const page = await browser.newPage({
    viewport,
    deviceScaleFactor: mobile ? 2 : 1,
    isMobile: mobile,
    hasTouch: mobile,
    reducedMotion: STATIC ? "reduce" : "no-preference",
  });
  page.on("console", (message) => {
    if (message.type() === "error" || message.type() === "warning") errors.push(message.text());
  });
  page.on("pageerror", (error) => errors.push(String(error)));
  return page;
};

// Returns a line describing the overflow when the page is wider than the viewport.
const findOverflow = async (page: Page, name: string, viewport: Viewport): Promise<string | undefined> => {
  const width = await page.evaluate(() => document.documentElement.scrollWidth);
  return width > viewport.width ? `${name}: page is ${width}px wide in a ${viewport.width}px viewport` : undefined;
};

const reviewStatic = async (page: Page, { name, viewport }: ReviewSize, errors: string[]): Promise<string | undefined> => {
  await page.goto(BASE, { waitUntil: "networkidle" });
  const mode = await page.locator("#walkthrough").getAttribute("data-mode");
  await page.evaluate(() =>
    document.querySelectorAll("img[loading=lazy]").forEach((image) => image.setAttribute("loading", "eager")),
  );
  await page.waitForTimeout(STATIC_IMAGE_WAIT_MS);
  const loaded = await page.evaluate(
    () => [...document.querySelectorAll<HTMLImageElement>(".wt-still")].filter((image) => image.naturalWidth > 0).length,
  );
  for (let slice = 0; slice < STATIC_SLICES; slice++) {
    const clip = { x: 0, y: slice * viewport.height * STATIC_SLICE_STRIDE, width: viewport.width, height: viewport.height };
    await page.screenshot({ path: resolve(REVIEW_DIR, `${name}-static-${slice}.jpg`), type: "jpeg", quality: 60, fullPage: true, clip });
  }
  console.log(`${name}: mode=${mode}, stills loaded ${loaded}, ${errors.length} console errors/warnings`);
  return findOverflow(page, name, viewport);
};

const reviewPoints = (): number[] =>
  process.env.WT_POINTS?.split(",").map(Number) ?? CHAPTERS.map((_, index) => chapterPoint(index, REVIEW_CHAPTER_OFFSET));

const reviewScroll = async (page: Page, { name, viewport }: ReviewSize, errors: string[]): Promise<string | undefined> => {
  await open(page);
  for (const point of reviewPoints()) {
    await seek(page, point);
    await page.screenshot({ path: resolve(REVIEW_DIR, `${name}-${point.toFixed(2)}.jpg`), type: "jpeg", quality: 70 });
  }
  const overflow = await findOverflow(page, name, viewport);
  console.log(`${name}: ${errors.length} console errors/warnings`);
  for (const error of errors.slice(0, MAX_LOGGED_ERRORS)) console.log("   ", error.slice(0, 200));
  return overflow;
};

const review = async (): Promise<void> => {
  await mkdir(REVIEW_DIR, { recursive: true });
  const browser = await chromium.launch({ args: GPU_ARGS });
  const overflows: string[] = [];
  for (const size of REVIEW_SIZES) {
    const errors: string[] = [];
    const page = await openReviewPage(browser, size, errors);
    const overflow = STATIC ? await reviewStatic(page, size, errors) : await reviewScroll(page, size, errors);
    if (overflow) overflows.push(overflow);
    await page.close();
  }
  await browser.close();
  if (overflows.length > 0) {
    for (const overflow of overflows) console.error(`  ✗ ${overflow}`);
    process.exit(1);
  }
};

await (REVIEW ? review() : stills());
