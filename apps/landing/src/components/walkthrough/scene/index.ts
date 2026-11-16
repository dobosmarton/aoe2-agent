import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { MeshoptDecoder } from "three/examples/jsm/libs/meshopt_decoder.module.js";
import { CSS3DRenderer } from "three/examples/jsm/renderers/CSS3DRenderer.js";

import turn from "../turn-data.json";
import { createBoard } from "./board";
import { createFlow, createRings } from "./flow";
import { buildLayout } from "./layout";
import { createMind, MIND } from "./mind";
import { createOverlays } from "./overlays";
import { createPost } from "./post";
import { applyPose, type CameraPose } from "./rig";
import { createScreen } from "./screen";
import { buildTimeline, createState, TIMELINE_LENGTH, type Poses } from "./timeline";
import type { TurnData } from "./types";
import { createWorld } from "./world";

export type SceneHandle = {
  /** Scroll position in timeline units: chapter index plus the fraction read. */
  setProgress(p: number): void;
  dispose(): void;
};

type MountOptions = {
  /** Holds the WebGL canvas and the CSS3D layer; sized by CSS. */
  view: HTMLElement;
  /** Receives `--wt-*` custom properties and state attributes for 2D overlays. */
  stage: HTMLElement;
  /** Base URL of public/walkthrough. */
  assets: string;
};

type DebugStage = HTMLElement & {
  wtProgress?: number;
  wtDebug?: { setPose(pose: Partial<CameraPose> | null): void; info(): unknown };
};

// The game camera: 45 degrees round, 30 degrees up, so tiles draw as 2:1 diamonds.
const GAME_AZ = 45;
const GAME_EL = 30;
const GAME_FOV = 8;
// Distance from the village centre to the glass pane, along the game camera's axis.
const PANE_DISTANCE = 45;

function loadTexture(url: string): Promise<THREE.Texture> {
  return new Promise((resolve, reject) => new THREE.TextureLoader().load(url, resolve, undefined, reject));
}

async function loadModels(url: string) {
  const loader = new GLTFLoader();
  loader.setMeshoptDecoder(MeshoptDecoder);
  const gltf = await loader.loadAsync(url);
  const models = new Map<string, THREE.Object3D>();
  for (const child of gltf.scene.children) models.set(child.name, child);
  return models;
}

export async function mountScene({ view, stage, assets }: MountOptions): Promise<SceneHandle> {
  const data = turn as TurnData;
  const small = window.matchMedia("(max-width: 767px)").matches;
  const [models, frameTexture, hudTexture] = await Promise.all([
    loadModels(`${assets}/village.glb`),
    loadTexture(`${assets}/frame-${small ? 1024 : 2048}.webp`),
    loadTexture(`${assets}/hud.webp`),
  ]);

  const renderer = new THREE.WebGLRenderer({ powerPreference: "high-performance", antialias: false, stencil: false, depth: false });
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFShadowMap;
  // Count draw calls over the whole frame (shadow, scene and post passes), not per pass.
  renderer.info.autoReset = false;
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
  scene.background = new THREE.Color("#0a1020");
  const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 500);

  // ---- World ---------------------------------------------------------------------------
  const layout = buildLayout(data);
  const world = await createWorld(layout, models);
  scene.add(world.group);

  const hemi = new THREE.HemisphereLight("#e2eeff", "#5a4630", 1.5);
  const sun = new THREE.DirectionalLight("#ffe4b8", 3.6);
  sun.position.set(18, 26, -10);
  sun.castShadow = true;
  sun.shadow.mapSize.set(small ? 1024 : 2048, small ? 1024 : 2048);
  Object.assign(sun.shadow.camera, { left: -34, right: 34, top: 34, bottom: -34, near: 1, far: 90 });
  sun.shadow.camera.updateProjectionMatrix();
  sun.shadow.bias = -0.0004;
  sun.shadow.normalBias = 0.03;
  sun.shadow.radius = 3;
  const rim = new THREE.DirectionalLight("#8fb4ff", 0.6);
  rim.position.set(-20, 12, 20);
  scene.add(hemi, sun, sun.target, rim);

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
  const mind = createMind(data, board, frameTexture);
  mindRoot.add(mind.group);
  const flow = createFlow(data, board, MIND.crystal.clone().sub(MIND.slab));
  board.group.add(flow.object);
  const decision = new THREE.Group();
  decision.position.copy(MIND.decision);
  mindRoot.add(decision);
  const rings = createRings();
  decision.add(rings.group);
  const mindLight = new THREE.PointLight("#6aa2ff", 60, 0, 1.4);
  mindLight.position.copy(MIND.crystal).add(new THREE.Vector3(0, 6, 8));
  mindRoot.add(mindLight);

  mindRoot.updateMatrixWorld(true);
  const toWorld = (v: THREE.Vector3) => mindRoot.localToWorld(v.clone());

  const green = layout.spots[2]!;
  const spotsCentre = layout.spots.reduce((a, s) => ({ x: a.x + s.x / 3, z: a.z + s.z / 3 }), { x: 0, z: 0 });
  const spotsReach = Math.max(...layout.spots.map((s) => Math.hypot(s.x - spotsCentre.x, s.z - spotsCentre.z)));
  const spotPoint = (s: { x: number; z: number }) => new THREE.Vector3(s.x, 0, s.z);
  const overlays = createOverlays(data, board, decision, {
    root: scene,
    mindScale,
    cardStart: toWorld(MIND.crystal.clone().add(new THREE.Vector3(0, 3.2, 1))),
    cardEnd: toWorld(MIND.decision.clone().add(new THREE.Vector3(-6.3, 0, 0))),
    cursorPath: [spotPoint({ x: green.x + 3, z: green.z + 3 }), ...layout.spots.map(spotPoint)],
    ledgerAt: new THREE.Vector3(green.x, 3.4, green.z),
  });

  // ---- Camera poses ----------------------------------------------------------------------
  const inMind = (local: THREE.Vector3, dAz: number, dEl: number, size: number, fov = 30): CameraPose => {
    const t = toWorld(local);
    return { tx: t.x, ty: t.y, tz: t.z, az: GAME_AZ + dAz, el: GAME_EL + dEl, size: size * mindScale, fov };
  };
  const hudLocal = board.hud.position.clone().add(MIND.slab);
  const poses: Poses = {
    hero: { tx: layout.tc.x * 0.4, ty: 0.6, tz: layout.tc.z * 0.4, az: 70, el: 26, size: 11, fov: 30 },
    // A little wider than the frame, so the glass shows with the village around it.
    game: { tx: 0, ty: 0, tz: 0, az: GAME_AZ, el: GAME_EL, size: gameSize * 1.75, fov: GAME_FOV },
    capture: inMind(new THREE.Vector3(3, 0, 1), 20, -4, 8.5),
    detect: inMind(new THREE.Vector3(22, 0, 2), 28, -8, 9),
    track: inMind(new THREE.Vector3(25, 0.5, -2), 62, -10, 8.2),
    hud: inMind(new THREE.Vector3(hudLocal.x + 4.5, hudLocal.y - 1.2, hudLocal.z + 2), 10, 6, 4.8),
    publish: inMind(new THREE.Vector3(37, 0, 3), 22, -6, 9.5),
    decide: inMind(MIND.decision.clone().add(new THREE.Vector3(-0.9, -0.8, 0)), 0, 0, 7.6),
    house: { tx: spotsCentre.x, ty: 0.8, tz: spotsCentre.z, az: 75, el: 36, size: Math.max(5.5, spotsReach * 1.25), fov: 30 },
    verify: { tx: green.x, ty: 1.4, tz: green.z, az: 105, el: 28, size: 7, fov: 30 },
  };
  const S = createState(poses.hero);
  const timeline = buildTimeline(S, poses);

  const post = createPost(renderer, scene, camera, small);

  // ---- Layout ----------------------------------------------------------------------------
  let width = 1;
  let height = 1;
  let fit = 1;
  let pointScale = 1;
  const resize = () => {
    width = Math.max(1, view.clientWidth);
    height = Math.max(1, view.clientHeight);
    // Ambient occlusion, shadows and three post passes run per pixel: 1.5 is the cap.
    const dpr = Math.min(window.devicePixelRatio, 1.5);
    renderer.setPixelRatio(dpr);
    renderer.setSize(width, height, false);
    cssRenderer.setSize(width, height);
    post.setSize(width, height);
    // Narrow (portrait) views need a taller framing to keep the same width in view.
    fit = Math.max(1, 1.6 / (width / height));
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
  const pose: CameraPose = { ...poses.hero };
  const ghostAt = new THREE.Vector3();

  const render = () => {
    renderer.info.reset();
    timer.update();
    const time = timer.getElapsed();
    const delta = timer.getDelta();
    current = current < 0 ? target : current + (target - current) * 0.09;
    if (Math.abs(target - current) < 0.0005) current = target;
    timeline.seek(Math.min(TIMELINE_LENGTH, Math.max(0, current)), false);
    (stage as DebugStage).wtProgress = current;

    // The hero drifts and follows the pointer; the effect fades out by chapter 1.
    const hero = THREE.MathUtils.clamp(1 - current, 0, 1) + THREE.MathUtils.clamp(current - 8.8, 0, 1);
    pointer.sx += (pointer.x - pointer.sx) * 0.05;
    pointer.sy += (pointer.y - pointer.sy) * 0.05;
    Object.assign(pose, S.cam, debugPose ?? {});
    if (!debugPose) {
      pose.az += hero * (Math.sin(time * 0.08) * 6 + pointer.sx * 4);
      pose.el += hero * pointer.sy * 2;
    }
    applyPose(camera, pose, width / height, fit);

    const dim = S.world.dim;
    sun.intensity = 3.6 * (1 - 0.85 * dim);
    hemi.intensity = 1.5 * (1 - 0.75 * dim);
    sun.castShadow = dim < 0.95;

    overlays.cursorAt(S.world.cursor, ghostAt);
    world.update(time, {
      ghost: S.world.ghost,
      ghostX: ghostAt.x,
      ghostZ: ghostAt.z,
      ghostOk: S.world.cursor > 2.5,
      build: S.world.build,
    });
    screen.update(S.pane, time);
    board.update(S, time);
    mind.update(S, time);
    flow.update(S, pointScale);
    rings.update(S, time);
    overlays.update(S);
    mindRoot.visible = S.pane.on > 0.002 || S.slab.on > 0.01 || S.card.on > 0.01 || S.decide.on > 0.01 || S.mind.pulse > 0.001;

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
    const wide = 0.4 + 0.6 * THREE.MathUtils.smoothstep(pose.size, 6, 11);
    post.setMiniature((1 - dim) * THREE.MathUtils.smoothstep(pose.fov, 12, 24) * wide);
    post.setBloom(0.55 + 0.75 * dim);
    post.render(delta);
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

  (stage as DebugStage).wtDebug = {
    setPose(p) {
      debugPose = p;
    },
    info() {
      return { calls: renderer.info.render.calls, triangles: renderer.info.render.triangles, mindScale, gameSize };
    },
  };

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
      gameTarget.dispose();
      post.dispose();
      renderer.dispose();
      renderer.domElement.remove();
      cssRenderer.domElement.remove();
    },
  };
}
