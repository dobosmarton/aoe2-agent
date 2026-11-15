import * as THREE from "three";
import { CSS3DRenderer } from "three/examples/jsm/renderers/CSS3DRenderer.js";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";

import turn from "../turn-data.json";
import { createBoard } from "./board";
import { createFlow, createRings } from "./flow";
import { createOverlays } from "./overlays";
import { buildTimeline, createState, TIMELINE_LENGTH, type Anchors } from "./timeline";
import type { TurnData } from "./types";

export type SceneHandle = {
  /** Scroll position in timeline units: chapter index plus the fraction read. */
  setProgress(p: number): void;
  dispose(): void;
};

type MountOptions = {
  /** Holds the WebGL canvas and the CSS3D layer; sized by CSS. */
  view: HTMLElement;
  /** Receives `--wt-*` custom properties and state classes for 2D overlays. */
  stage: HTMLElement;
  /** Base URL of the textures in public/walkthrough. */
  assets: string;
};

const FOV = 32;

function loadTexture(url: string): Promise<THREE.Texture> {
  return new Promise((resolve, reject) => new THREE.TextureLoader().load(url, resolve, undefined, reject));
}

export async function mountScene({ view, stage, assets }: MountOptions): Promise<SceneHandle> {
  const data = turn as TurnData;
  const small = window.matchMedia("(max-width: 767px)").matches;
  const [frameTexture, hudTexture] = await Promise.all([
    loadTexture(`${assets}/frame-${small ? 1024 : 2048}.webp`),
    loadTexture(`${assets}/hud.webp`),
  ]);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.setClearColor(0x000000, 0);
  for (const texture of [frameTexture, hudTexture]) {
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.anisotropy = renderer.capabilities.getMaxAnisotropy();
  }
  renderer.domElement.className = "wt-canvas";
  view.append(renderer.domElement);

  const cssRenderer = new CSS3DRenderer();
  cssRenderer.domElement.className = "wt-css3d";
  view.append(cssRenderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(FOV, 1, 0.1, 400);

  const board = createBoard(data, frameTexture, hudTexture);
  scene.add(board.group);

  // The decision loops float above the far edge of the board, turned to face the
  // camera's pose in the Decide chapter (pitch 1.1).
  const decisionAnchor = new THREE.Vector3(0.6, board.H * 0.58, 4.4);
  const decision = new THREE.Group();
  decision.position.copy(decisionAnchor);
  decision.rotation.x = 1.1;
  scene.add(decision);
  const rings = createRings();
  decision.add(rings.group);

  const cardStart = new THREE.Vector3(0, -0.2, 3.4);
  const flow = createFlow(data, board, cardStart);
  scene.add(flow.object);
  const overlays = createOverlays(data, board, decision, cardStart);
  scene.add(overlays.card);

  const villagers = data.entities.filter((e) => e.cls === "villager");
  const median = (xs: number[]) => xs.sort((a, b) => a - b)[Math.floor(xs.length / 2)] ?? 0.5;
  const villagerCenter = board.toBoard(
    median(villagers.map((e) => e.x + e.w / 2)),
    median(villagers.map((e) => e.y + e.h / 2)),
  );
  const hudField = board.fieldCenters.population ?? new THREE.Vector3();
  const anchors: Anchors = {
    villagers: { x: villagerCenter.x, y: villagerCenter.y },
    hud: { x: board.hud.position.x + hudField.x * 0.6, y: board.hud.position.y - 0.9 },
    house: { x: board.spots[2]!.x * 0.6, y: board.spots[2]!.y * 0.6 + 0.6 },
    decision: { x: decisionAnchor.x - 1.9, y: decisionAnchor.y - 0.6, z: decisionAnchor.z - 0.6 },
    card: { x: cardStart.x, y: cardStart.y, z: cardStart.z },
  };

  const S = createState();
  const timeline = buildTimeline(S, anchors);

  // Bloom only on larger screens; phones render straight to the canvas.
  const composer = small ? null : new EffectComposer(renderer);
  let bloom: UnrealBloomPass | null = null;
  if (composer) {
    composer.addPass(new RenderPass(scene, camera));
    bloom = new UnrealBloomPass(new THREE.Vector2(256, 256), 0.85, 0.55, 0.92);
    composer.addPass(bloom);
    composer.addPass(new OutputPass());
  }

  // ---- Layout --------------------------------------------------------------------------
  let width = 1;
  let height = 1;
  let fitDistance = 30;
  let pointScale = 1;
  const resize = () => {
    width = Math.max(1, view.clientWidth);
    height = Math.max(1, view.clientHeight);
    const dpr = Math.min(window.devicePixelRatio, small ? 1.5 : 2);
    renderer.setPixelRatio(dpr);
    renderer.setSize(width, height, false);
    cssRenderer.setSize(width, height);
    composer?.setPixelRatio(dpr);
    composer?.setSize(width, height);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    // Fit the flat frame to a share of the visible stage: the view is wider than the
    // stage by the CSS shift, so measure against the stage itself.
    const stageWidth = stage.clientWidth;
    const stageHeight = stage.clientHeight;
    const wantWidth = stageWidth * (stageWidth < 768 ? 0.94 : stageWidth < 1200 ? 0.62 : 0.56);
    const wantHeight = stageHeight * (stageWidth < 768 ? 0.42 : 0.62);
    const tanHalf = Math.tan(THREE.MathUtils.degToRad(FOV / 2));
    const byWidth = (board.W * width) / (wantWidth * 2 * tanHalf * camera.aspect);
    const byHeight = (board.H * height) / (wantHeight * 2 * tanHalf);
    fitDistance = Math.max(byWidth, byHeight);
    pointScale = ((height * dpr) / 2 / tanHalf) * 0.012;
  };
  resize();
  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(view);
  resizeObserver.observe(stage);

  // ---- Pointer parallax in the hero ----------------------------------------------------
  const pointer = { x: 0, y: 0, sx: 0, sy: 0 };
  const onPointer = (event: PointerEvent) => {
    pointer.x = (event.clientX / window.innerWidth) * 2 - 1;
    pointer.y = (event.clientY / window.innerHeight) * 2 - 1;
  };
  window.addEventListener("pointermove", onPointer, { passive: true });

  // ---- Loop ----------------------------------------------------------------------------
  let target = 0;
  let current = -1;
  let running = false;
  let visible = true;
  let frameId = 0;
  const timer = new THREE.Timer();
  const lookAt = new THREE.Vector3();
  const up = new THREE.Vector3();

  const render = () => {
    timer.update();
    const time = timer.getElapsed();
    current = current < 0 ? target : current + (target - current) * 0.1;
    if (Math.abs(target - current) < 0.0005) current = target;
    timeline.seek(Math.min(TIMELINE_LENGTH, Math.max(0, current)), false);
    // Read by scripts/capture-walkthrough-stills.ts to know when the scene has settled.
    (stage as HTMLElement & { wtProgress?: number }).wtProgress = current;

    // The hero drifts and follows the pointer; the effect fades out by chapter 1.
    const hero = THREE.MathUtils.clamp(1 - current, 0, 1);
    pointer.sx += (pointer.x - pointer.sx) * 0.05;
    pointer.sy += (pointer.y - pointer.sy) * 0.05;
    const yaw = S.cam.yaw + hero * (Math.sin(time * 0.17) * 0.07 + pointer.sx * 0.1);
    const pitch = S.cam.pitch + hero * (Math.sin(time * 0.23) * 0.03 + pointer.sy * 0.05);

    const d = fitDistance * S.cam.zoom;
    lookAt.set(S.cam.tx, S.cam.ty, S.cam.tz);
    camera.position.set(
      lookAt.x + d * Math.sin(yaw) * Math.sin(pitch),
      lookAt.y - d * Math.cos(yaw) * Math.sin(pitch),
      lookAt.z + d * Math.cos(pitch),
    );
    up.set(-Math.sin(yaw) * Math.cos(pitch), Math.cos(yaw) * Math.cos(pitch), Math.sin(pitch));
    camera.up.copy(up);
    camera.lookAt(lookAt);

    board.update(S, time);
    flow.update(S, pointScale);
    rings.update(S, time);
    overlays.update(S);

    stage.style.setProperty("--wt-stamp", String(S.stamp.on));
    stage.style.setProperty("--wt-input", String(S.act.on));
    // Input strip: q, q, hover, click, h.
    const step =
      S.act.keys >= 2.5 ? 5 : S.act.click > 0.05 ? 4 : S.act.cursor > 0.05 ? 3 : Math.round(S.act.keys);
    stage.dataset.step = String(step);

    if (composer) composer.render();
    else renderer.render(scene, camera);
    cssRenderer.render(scene, camera);
  };

  const tick = () => {
    if (!running) return;
    render();
    frameId = requestAnimationFrame(tick);
  };
  const start = () => {
    if (running || !visible || document.hidden) return;
    running = true;
    frameId = requestAnimationFrame(tick);
  };
  const stop = () => {
    running = false;
    cancelAnimationFrame(frameId);
  };

  const intersection = new IntersectionObserver(([entry]) => {
    visible = entry?.isIntersecting ?? true;
    if (visible) start();
    else stop();
  });
  intersection.observe(stage);
  const onVisibility = () => (document.hidden ? stop() : start());
  document.addEventListener("visibilitychange", onVisibility);

  render();
  start();

  return {
    setProgress(p) {
      target = p;
    },
    dispose() {
      stop();
      intersection.disconnect();
      resizeObserver.disconnect();
      document.removeEventListener("visibilitychange", onVisibility);
      window.removeEventListener("pointermove", onPointer);
      timeline.kill();
      timer.dispose();
      composer?.dispose();
      renderer.dispose();
      renderer.domElement.remove();
      cssRenderer.domElement.remove();
    },
  };
}
