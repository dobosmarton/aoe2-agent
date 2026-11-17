import * as THREE from "three";

import turn from "../turn-data.json";
import { createBoard } from "./board";
import { createFlow, createRings } from "./flow";
import { buildLayout } from "./layout";
import { createLabels } from "./labels";
import { setLineResolution } from "./lines";
import { createMind, DOCK, MIND } from "./mind";
import { createOverlays } from "./overlays";
import { createPost } from "./post";
import { applyPose, type CameraPose } from "./rig";
import { createScreen } from "./screen";
import { createTracking } from "./track";
import type { SpriteMeta } from "./sprites";
import { buildTimeline, createState, TIMELINE_LENGTH, type Poses } from "./timeline";
import type { TurnData } from "./types";
import { createWorld } from "./world";

/** Part of the view, in CSS pixels from its top-left, that no caption covers. */
export type SafeArea = { left: number; top: number; right: number; bottom: number };

export type SceneHandle = {
  /** Scroll position in timeline units: chapter index plus the fraction read. */
  setProgress(p: number): void;
  /** Where to frame each shot's subject. */
  setSafe(area: SafeArea): void;
  dispose(): void;
};

type MountOptions = {
  /** Holds the WebGL canvas and the label layer; sized by CSS. */
  view: HTMLElement;
  /** Receives `--wt-*` custom properties and state attributes for 2D overlays. */
  stage: HTMLElement;
  /** Base URL of public/walkthrough. */
  assets: string;
};

type DebugStage = HTMLElement & {
  wtProgress?: number;
  wtDebug?: {
    setPose(pose: Partial<CameraPose> | null): void;
    setCentred(on: boolean): void;
    info(): unknown;
    /** A pose given in the mind's local frame, as the poses are written. */
    mindPose(local: number[], dAz: number, dEl: number, size: number): void;
    /** Screen positions (CSS px) of points in the mind's local frame. */
    projectMind(points: number[][]): number[][];
  };
};

// The game camera: 45 degrees round, 30 degrees up, so tiles draw as 2:1 diamonds.
const GAME_AZ = 45;
const GAME_EL = 30;
const GAME_FOV = 8;
// Distance from the village centre to the glass pane, along the game camera's axis.
const PANE_DISTANCE = 45;
// A pose's `size` is the half-height of a subject this much wider than tall; the
// camera fits that box into the safe area.
const SUBJECT_ASPECT = 1.3;

// The space around the map and the mind: deep blue-black with a soft glow behind the
// subject, darker towards the corners. Grain in post (post.ts) keeps it from banding.
function createBackdrop(): THREE.Texture {
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 288;
  const g = canvas.getContext("2d")!;
  g.fillStyle = "#03060e";
  g.fillRect(0, 0, 512, 288);
  const glow = g.createRadialGradient(300, 120, 10, 300, 130, 320);
  glow.addColorStop(0, "rgba(38, 70, 140, 0.55)");
  glow.addColorStop(0.45, "rgba(22, 40, 92, 0.32)");
  glow.addColorStop(1, "rgba(3, 6, 14, 0)");
  g.fillStyle = glow;
  g.fillRect(0, 0, 512, 288);
  const low = g.createRadialGradient(90, 300, 10, 90, 300, 260);
  low.addColorStop(0, "rgba(70, 30, 140, 0.16)");
  low.addColorStop(1, "rgba(3, 6, 14, 0)");
  g.fillStyle = low;
  g.fillRect(0, 0, 512, 288);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function loadTexture(url: string): Promise<THREE.Texture> {
  return new Promise((resolve, reject) => new THREE.TextureLoader().load(url, resolve, undefined, reject));
}

async function loadJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  return (await response.json()) as T;
}

export async function mountScene({ view, stage, assets }: MountOptions): Promise<SceneHandle> {
  const data = turn as TurnData;
  // Canvas text in the scene uses the site's mono face; wait for it.
  await document.fonts?.ready;
  const mono = getComputedStyle(stage).getPropertyValue("--font-mono").trim() || "ui-monospace, monospace";
  const small = window.matchMedia("(max-width: 767px)").matches;
  const art = `${assets}/aoe2`;
  const [meta, atlas, grass, patchy, dirt, forest, farm, visibility, frameTexture, hudTexture] = await Promise.all([
    loadJson<SpriteMeta>(`${art}/sprites.json`),
    loadTexture(`${art}/sprites.webp`),
    loadTexture(`${art}/grass.webp`),
    loadTexture(`${art}/grass_patchy.webp`),
    loadTexture(`${art}/dirt.webp`),
    loadTexture(`${art}/forest.webp`),
    loadTexture(`${art}/farm.webp`),
    loadTexture(`${art}/visibility.png`),
    loadTexture(`${assets}/frame-${small ? 1024 : 2048}.webp`),
    loadTexture(`${assets}/hud.webp`),
  ]);

  const renderer = new THREE.WebGLRenderer({ powerPreference: "high-performance", antialias: false, stencil: false, depth: false });
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  // Count draw calls over the whole frame (shadow, scene and post passes), not per pass.
  renderer.info.autoReset = false;
  for (const texture of [frameTexture, hudTexture]) {
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.anisotropy = renderer.capabilities.getMaxAnisotropy();
  }
  renderer.domElement.className = "wt-canvas";
  view.append(renderer.domElement);
  const labels = createLabels(view);

  const scene = new THREE.Scene();
  scene.background = createBackdrop();
  const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 500);

  // ---- World ---------------------------------------------------------------------------
  const layout = buildLayout(data);
  // The world is drawn unlit, as the game draws it: sprites carry their own shading
  // and shadows. The lights only shape the mind's few lit materials.
  const world = createWorld(layout, { atlas, grass, patchy, dirt, forest, farm, visibility }, meta);
  scene.add(world.group);
  const rim = new THREE.DirectionalLight("#8fb4ff", 1.2);
  rim.position.set(-20, 30, 20);
  scene.add(new THREE.HemisphereLight("#e2eeff", "#1a2030", 1.2), rim);

  // ---- The game camera and the glass pane ------------------------------------------------
  // Half-height of the recorded frame on the ground plane's screen, in world units.
  const frameAspect = data.frame.width / data.frame.height;
  const gameSize = data.frame.height / 2 / (layout.tw / Math.SQRT2);
  const az = THREE.MathUtils.degToRad(GAME_AZ);
  const el = THREE.MathUtils.degToRad(GAME_EL);
  const axis = new THREE.Vector3(Math.cos(el) * Math.cos(az), Math.sin(el), Math.cos(el) * Math.sin(az));
  const right = new THREE.Vector3(Math.SQRT1_2, 0, -Math.SQRT1_2);
  const up = new THREE.Vector3().crossVectors(axis, right);

  const gameCamera = new THREE.OrthographicCamera(-gameSize * frameAspect, gameSize * frameAspect, gameSize, -gameSize, 1, 400);
  gameCamera.position.copy(axis).multiplyScalar(150);
  gameCamera.lookAt(0, 0, 0);
  const gameTarget = new THREE.WebGLRenderTarget(small ? 768 : 1280, Math.round((small ? 768 : 1280) / frameAspect), {
    colorSpace: THREE.SRGBColorSpace,
    samples: 4,
  });

  // The pane sits on the game camera's axis, sized to fill that camera's view there,
  // so the village behind lines up with the picture on the glass.
  const screen = createScreen(frameTexture, gameTarget.texture, frameAspect);
  const mindScale = (gameSize - PANE_DISTANCE * Math.tan(THREE.MathUtils.degToRad(GAME_FOV / 2))) / (screen.height / 2);
  const mindRoot = new THREE.Group();
  mindRoot.position.copy(axis).multiplyScalar(PANE_DISTANCE);
  mindRoot.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(right, up, axis));
  mindRoot.scale.setScalar(mindScale);
  scene.add(mindRoot);
  mindRoot.add(screen.group);

  // ---- The mind --------------------------------------------------------------------------
  const board = createBoard(data, frameTexture, hudTexture);
  mindRoot.add(board.group);
  const mind = createMind(data, board, frameTexture, labels, mono);
  mindRoot.add(mind.group);
  const flow = createFlow(data, board, MIND.plate.clone().sub(MIND.slab));
  board.group.add(flow.object);
  const decision = new THREE.Group();
  decision.position.copy(MIND.decision);
  mindRoot.add(decision);
  const rings = createRings();
  decision.add(rings.group);

  mindRoot.updateMatrixWorld(true);
  const toWorld = (v: THREE.Vector3) => mindRoot.localToWorld(v.clone());

  const green = layout.spots[2]!;
  const spotsCentre = layout.spots.reduce((a, s) => ({ x: a.x + s.x / 3, z: a.z + s.z / 3 }), { x: 0, z: 0 });
  const spotsReach = Math.max(...layout.spots.map((s) => Math.hypot(s.x - spotsCentre.x, s.z - spotsCentre.z)));
  const spotPoint = (s: { x: number; z: number }) => new THREE.Vector3(s.x, 0, s.z);
  const overlays = createOverlays(data, board, decision, labels, {
    root: scene,
    cursorPath: [spotPoint({ x: green.x + 3, z: green.z + 3 }), ...layout.spots.map(spotPoint)],
    ledgerAt: new THREE.Vector3(green.x, 1.6, green.z),
    green: spotPoint(green),
  });

  // ---- Tracking in the village: the walking villagers nearest the Town Center ----------
  // Villagers with the longest walks near the Town Center show the trails best; the
  // nearest others fill the group.
  const near = (t: { anchor: THREE.Vector3 }) => Math.hypot(t.anchor.x - layout.tc.x, t.anchor.z - layout.tc.z);
  const byDistance = [...world.tracked].sort((a, b) => near(a) - near(b));
  const walkers = byDistance.filter((t) => t.route > 2 && near(t) < 11).slice(0, 6);
  const followed = [...walkers, ...byDistance.filter((t) => !walkers.includes(t))].slice(0, 9);
  const tracking = createTracking(followed, [41, 7, 63, 12, 28, 55, 3, 90, 34], labels);
  const trackCentre = followed.reduce((c, t) => ({ x: c.x + t.anchor.x / followed.length, z: c.z + t.anchor.z / followed.length }), { x: 0, z: 0 });
  const spread = Math.max(...followed.map((t) => Math.hypot(t.anchor.x - trackCentre.x, t.anchor.z - trackCentre.z)));
  const trackSize = THREE.MathUtils.clamp(spread * 0.8, 5.5, 9);

  // ---- Camera poses ----------------------------------------------------------------------
  const inMind = (local: THREE.Vector3, dAz: number, dEl: number, size: number, fov = 30): CameraPose => {
    const t = toWorld(local);
    return { tx: t.x, ty: t.y, tz: t.z, az: GAME_AZ + dAz, el: GAME_EL + dEl, size: size * mindScale, fov };
  };
  const hudLocal = board.hud.position.clone().add(MIND.slab);
  const poses: Poses = {
    // World shots stay close to the game's own angle, where its sprites are drawn for.
    hero: { tx: layout.tc.x * 0.5 + 1.5, ty: 0, tz: layout.tc.z * 0.5 - 1.5, az: 45, el: 30, size: 9.5, fov: 28 },
    // A little wider than the frame, so the glass shows with the village around it.
    game: { tx: 0, ty: 0, tz: 0, az: GAME_AZ, el: GAME_EL, size: gameSize * 1.75, fov: GAME_FOV },
    capture: inMind(new THREE.Vector3(3, 0, 1), 20, -4, 8.5),
    // The frame and the detector's plates beside it, seen from the front left.
    detect: inMind(MIND.slab.clone().add(new THREE.Vector3(3.5, 2.4, -2)), 22, -5, 9.2),
    track: { tx: trackCentre.x, ty: 0.3, tz: trackCentre.z, az: 50, el: 33, size: trackSize, fov: 26 },
    hud: inMind(new THREE.Vector3(hudLocal.x + 1.2, hudLocal.y - 1.4, hudLocal.z + 2), 10, 6, 5.2),
    // The frame and the plate the sheets seal into.
    publish: inMind(MIND.slab.clone().lerp(MIND.plate, 0.55).add(new THREE.Vector3(0, 0.5, 1)), 16, -5, 8.5),
    decide: inMind(MIND.decision.clone().add(new THREE.Vector3(-1.2, -1.5, 0)), 4, 0, 7),
    // The whole loop at once: the glass, the frame, the Perception card and the rings,
    // with the return rail under them.
    loop: inMind(new THREE.Vector3(28, -1.8, 3), 8, 6, 27),
    house: { tx: spotsCentre.x, ty: 0.6, tz: spotsCentre.z, az: 48, el: 31, size: Math.max(6, spotsReach * 1.3), fov: 26 },
    verify: { tx: green.x, ty: 1, tz: green.z, az: 52, el: 32, size: 6.5, fov: 26 },
  };
  const S = createState(poses.hero);
  scene.add(tracking.group);
  const timeline = buildTimeline(S, poses);

  const post = createPost(renderer, scene, camera, small);

  // ---- Layout ----------------------------------------------------------------------------
  // Each shot's subject is framed in the safe area: the camera's principal point moves
  // to the safe area's centre (a view offset), and `fit` scales the view so the subject
  // box fits inside it.
  let width = 1;
  let height = 1;
  let fit = 1;
  let pointScale = 1;
  let safe: SafeArea | null = null;
  // Full sharpness on desktop; phones and slow machines render at 1.5x.
  let maxDpr = small ? 1.5 : 2;
  let centred = false;
  const frameSafe = () => {
    const s = !centred && safe ? safe : { left: 0, top: 0, right: width, bottom: height };
    const sw = Math.max(1, s.right - s.left);
    const sh = Math.max(1, s.bottom - s.top);
    fit = Math.max(height / sh, (SUBJECT_ASPECT * height) / sw);
    camera.setViewOffset(width, height, width / 2 - (s.left + s.right) / 2, height / 2 - (s.top + s.bottom) / 2, width, height);
  };
  const resize = () => {
    width = Math.max(1, view.clientWidth);
    height = Math.max(1, view.clientHeight);
    const dpr = Math.min(window.devicePixelRatio, maxDpr);
    renderer.setPixelRatio(dpr);
    renderer.setSize(width, height, false);
    setLineResolution(width, height);
    post.setSize(width, height);
    frameSafe();
    pointScale = ((height * dpr) / 2 / Math.tan(THREE.MathUtils.degToRad(15))) * 0.012 * mindScale;
  };
  resize();
  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(view);

  const pointer = { x: 0, y: 0, sx: 0, sy: 0 };
  const onPointer = (event: PointerEvent) => {
    pointer.x = (event.clientX / window.innerWidth) * 2 - 1;
    pointer.y = (event.clientY / window.innerHeight) * 2 - 1;
  };
  window.addEventListener("pointermove", onPointer, { passive: true });

  // ---- Loop ------------------------------------------------------------------------------
  let target = 0;
  let current = -1;
  let running = false;
  let visible = true;
  let frameId = 0;
  let debugPose: Partial<CameraPose> | null = null;
  const timer = new THREE.Timer();
  let slowFrames = 0;
  const pose: CameraPose = { ...poses.hero };
  const ghostAt = new THREE.Vector3();

  const render = () => {
    renderer.info.reset();
    timer.update();
    const time = timer.getElapsed();
    const delta = timer.getDelta();
    // Fall back to 1.5x if frames keep running long (a 2x frame costs about 1.8x).
    if (maxDpr > 1.5) {
      slowFrames = delta > 0.022 ? slowFrames + 1 : Math.max(0, slowFrames - 1);
      if (slowFrames > 90) {
        maxDpr = 1.5;
        resize();
      }
    }
    // Lenis already smooths the wheel; this light damping only removes steps from
    // keyboard and scrollbar jumps. Frame-rate independent.
    current = current < 0 ? target : THREE.MathUtils.damp(current, target, 9, Math.min(delta, 0.1));
    if (Math.abs(target - current) < 0.0005) current = target;
    timeline.seek(Math.min(TIMELINE_LENGTH, Math.max(0, current)), false);
    (stage as DebugStage).wtProgress = current;

    // On load the map opens out of the dark and the camera settles (expo out, 1.8 s).
    const intro = 1 - Math.pow(2, -10 * Math.min(1, time / 1.8));
    // The hero drifts and follows the pointer; the effect fades out by chapter 1.
    const hero = THREE.MathUtils.clamp(1 - current, 0, 1) + THREE.MathUtils.clamp(current - 8.8, 0, 1);
    pointer.sx += (pointer.x - pointer.sx) * 0.05;
    pointer.sy += (pointer.y - pointer.sy) * 0.05;
    Object.assign(pose, S.cam, debugPose ?? {});
    if (!debugPose) {
      pose.az += hero * (Math.sin(time * 0.08) * 3 + pointer.sx * 3);
      pose.el += hero * pointer.sy * 1.5;
      // Every other shot breathes a little and leans towards the pointer, so a held
      // beat is never frozen.
      const held = 1 - hero;
      pose.az += held * (Math.sin(time * 0.13) * 0.7 + pointer.sx * (small ? 0 : 1.1));
      pose.el += held * (Math.sin(time * 0.1 + 1.3) * 0.35 + pointer.sy * (small ? 0 : 0.5));
      pose.size *= 1 + hero * Math.sin(time * 0.11) * 0.03;
      pose.size *= 1 + 0.14 * (1 - intro);
      pose.el += 5 * (1 - intro);
    }
    applyPose(camera, pose, width / height, fit);

    const dim = S.world.dim;
    overlays.cursorAt(S.world.cursor, ghostAt);
    world.update(time, camera, {
      light: 1 - 0.78 * dim,
      open: intro,
      ghost: S.world.ghost,
      ghostX: ghostAt.x,
      ghostZ: ghostAt.z,
      ghostOk: S.world.cursor > 2.5,
      build: S.world.build,
    });
    tracking.update(S.world.track, time);
    screen.update(S.pane, time);
    board.update(S, time);
    mind.update(S, time, pointScale / mindScale);
    flow.update(S, pointScale);
    rings.update(S, time);
    overlays.update(S);
    mindRoot.visible = S.pane.on > 0.002 || S.slab.on > 0.01 || S.card.on > 0.01 || S.decide.on > 0.01 || S.action.on > 0.001 || S.mind.space > 0.01;

    stage.style.setProperty("--wt-stamp", String(S.stamp.on));
    stage.style.setProperty("--wt-input", String(S.act.on));
    const step = S.act.keys >= 2.5 ? 5 : S.act.click > 0.05 ? 4 : S.world.cursor > 0.05 ? 3 : Math.round(S.act.keys);
    stage.dataset.step = String(step);

    // The glass shows the game camera's render until the real frame has replaced it.
    if (S.pane.on > 0.002 && S.pane.mix < 0.999) {
      const mindWasVisible = mindRoot.visible;
      mindRoot.visible = false;
      renderer.setRenderTarget(gameTarget);
      renderer.render(scene, gameCamera);
      renderer.setRenderTarget(null);
      mindRoot.visible = mindWasVisible;
    }

    // The miniature look is for wide world shots; close-ups keep more in focus.
    const wide = 0.35 + 0.65 * THREE.MathUtils.smoothstep(pose.size, 5, 10);
    post.setMiniature((1 - dim) * THREE.MathUtils.smoothstep(pose.fov, 12, 24) * wide);
    post.setBloom(0.55 + 0.75 * dim);
    post.render(delta);
    labels.update(camera, width, height);
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

  (stage as DebugStage).wtDebug = {
    setPose(p) {
      debugPose = p;
    },
    setCentred(on) {
      centred = on;
      frameSafe();
    },
    mindPose(local, dAz, dEl, size) {
      debugPose = inMind(new THREE.Vector3(local[0], local[1], local[2]), dAz, dEl, size);
    },
    projectMind(points) {
      return points.map((p) => {
        const v = toWorld(new THREE.Vector3(p[0], p[1], p[2])).project(camera);
        return [Math.round((v.x * 0.5 + 0.5) * width), Math.round((-v.y * 0.5 + 0.5) * height)];
      });
    },
    info() {
      return { calls: renderer.info.render.calls, triangles: renderer.info.render.triangles, mindScale, gameSize, poses, followed: followed.map((t) => [t.anchor.x, t.anchor.z, t.walking]), villagers: world.villagerCheck() };
    },
  };

  render();
  start();

  return {
    setProgress(p) {
      target = p;
    },
    setSafe(area) {
      safe = area;
      frameSafe();
    },
    dispose() {
      stop();
      intersection.disconnect();
      resizeObserver.disconnect();
      document.removeEventListener("visibilitychange", onVisibility);
      window.removeEventListener("pointermove", onPointer);
      timeline.kill();
      timer.dispose();
      gameTarget.dispose();
      post.dispose();
      renderer.dispose();
      renderer.domElement.remove();
      labels.dispose();
    },
  };
}
