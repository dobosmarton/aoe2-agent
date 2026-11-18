import * as THREE from "three";

import turn from "../turn-data.json";
import { type Board, createBoard } from "./board";
import { createCanvas } from "./card";
import { fadeWindow } from "./easing";
import { createFlow, createRings, type Flow, type Rings } from "./flow";
import { HERO_FADE_END, HERO_FADE_START, type SafeArea, subjectFit } from "./framing";
import { buildLayout, type GroundPoint, type Layout } from "./layout";
import { createLabels, type Labels } from "./labels";
import { setLineResolution } from "./lines";
import { createMind, MIND, type Mind } from "./mind";
import { createOverlays } from "./overlays";
import { createPost, type Post } from "./post";
import { applyPose, type CameraPose } from "./rig";
import { createScreen, type Screen } from "./screen";
import type { SpriteMeta } from "./sprites";
import { buildTimeline, createState, TIMELINE_LENGTH, type Poses, type SceneState } from "./timeline";
import { createTracking } from "./track";
import type { TurnData } from "./types";
import type { TrackedVillager, VillagerCheck } from "./villagers";
import { createWorld, type World, type WorldTextures } from "./world";

export type SceneHandle = {
  /** Scroll position in timeline units: chapter index plus the fraction read. */
  setProgress: (progress: number) => void;
  /**
   * Where to frame each shot's subject, and the hero's (defaults to the same).
   * `stacked`: the caption sits under the scene (phones and tablets held upright).
   */
  setSafe: (area: SafeArea, layout?: { hero?: SafeArea; stacked?: boolean }) => void;
  dispose: () => void;
};

type MountOptions = {
  /** Holds the WebGL canvas and the label layer; sized by CSS. */
  view: HTMLElement;
  /** Receives `--wt-*` custom properties and state attributes for 2D overlays. */
  stage: HTMLElement;
  /** Base URL of public/walkthrough. */
  assets: string;
  /** Where the reader already is (timeline units); a reload mid-page starts there. */
  progress: number;
};

type DebugInfo = {
  frames: number;
  narrow: boolean;
  dpr: number;
  calls: number;
  triangles: number;
  mindScale: number;
  gameSize: number;
  poses: Poses;
  /** Ground position and walking state of each followed villager. */
  followed: [number, number, boolean][];
  villagers: VillagerCheck;
};

type DebugHooks = {
  setPose: (pose: Partial<CameraPose> | null) => void;
  setCentred: (on: boolean) => void;
  info: () => DebugInfo;
  /** A pose given in the mind's local frame, as the poses are written. */
  mindPose: (local: number[], dAz: number, dEl: number, size: number) => void;
  /** Screen positions (CSS px) of points in the mind's local frame. */
  projectMind: (points: number[][]) => number[][];
  /** Mean ms per frame over `frames` frames drawn back to back, GPU included. */
  bench: (frames: number) => number;
};

/** Hooks for the stills script and the browser console. */
export type DebugStage = HTMLElement & { wtProgress?: number; wtDebug?: DebugHooks };

// The game camera: 45 degrees round, 30 degrees up, so tiles draw as 2:1 diamonds.
const GAME_AZ = 45;
const GAME_EL = 30;
const GAME_FOV = 8;
// Distance from the village centre to the glass pane, along the game camera's axis.
const PANE_DISTANCE = 45;
// Most pixels a frame renders (device pixels), about 2x of a 1440 x 900 window.
const PIXEL_BUDGET = 5.2e6;
// Pixel ratio phones always use, and that other screens fall back to when frames run long.
const REDUCED_DPR = 1.5;
// Frames longer than this (seconds), back to back, mean the GPU cannot keep up at 2x.
const SLOW_FRAME = 0.026;
const SLOW_FRAMES_BEFORE_FALLBACK = 60;
// Frame pacing (see createFrameLoop).
const ACTIVE_INTERVAL = 1000 / 60;
const IDLE_INTERVAL = 1000 / 30;
const REST_AFTER = 60_000;
const POINTER_ACTIVE = 1500;
// Scroll positions closer than this to the target count as settled.
const SETTLED = 0.0005;
// Seconds the map takes to open out of the dark on load.
const INTRO = 1.8;
// The hero drifts and follows the pointer up to chapter 1, and again from here on, as
// the last chapter hands back to the hero shot.
const HERO_RETURN = 8.8;
// The stacked layout frames the narrow shots, and so does a safe area narrower than
// this many CSS px (phones on their side).
const NARROW_WIDTH = 544;
// Half-height (mind units) of the shot that rides along with the action on phones.
const FOLLOW_SIZE = 5.5;
const SMALL_SCREEN = "(max-width: 767px)";
/** Illustrative IDs of the followed villagers; the Track chapter's copy names track_41. */
const TRACK_IDS = [41, 7, 63, 12, 28, 55, 3, 90, 34];
/** Followed villagers that walk; the nearest others fill the group. */
const MAX_WALKERS = 6;
const INPUTS = ["keydown", "wheel", "touchstart", "pointerdown"] as const;
const BACKDROP_SIZE = { w: 512, h: 288 };

type Art = { meta: SpriteMeta; world: WorldTextures; frame: THREE.Texture; hud: THREE.Texture };

/** The game camera, the view it renders onto the glass, and its axes. */
type GameView = {
  camera: THREE.OrthographicCamera;
  target: THREE.WebGLRenderTarget;
  /** Half-height of the recorded frame on the ground plane's screen, in world units. */
  size: number;
  /** The recorded frame's width over its height. */
  aspect: number;
  /** Towards the camera, and the screen's right and up. */
  axis: THREE.Vector3;
  right: THREE.Vector3;
  up: THREE.Vector3;
};

type MindPoser = (local: THREE.Vector3, dAz: number, dEl: number, size: number, fov?: number) => CameraPose;

type ViewSize = {
  /** CSS px. */
  width: number;
  height: number;
  /** Scale of the mind's point sprites, from the view's height in device pixels. */
  pointScale: number;
};

type Pointer = {
  /** Pointer position, -1..1 over the window, and its smoothed follower. */
  x: number;
  y: number;
  sx: number;
  sy: number;
  /** When a mouse last moved (ms); a touch is a scroll and does not lean the camera. */
  lastMouse: number;
};

type Render = (delta: number, active: boolean) => void;

// The space around the map and the mind: deep blue-black with a soft glow behind the
// subject, darker towards the corners. Grain in post (post.ts) keeps it from banding.
const createBackdrop = (): THREE.Texture => {
  const { w, h } = BACKDROP_SIZE;
  const [canvas, g] = createCanvas(w, h);

  g.fillStyle = "#03060e";
  g.fillRect(0, 0, w, h);

  const glow = g.createRadialGradient(300, 120, 10, 300, 130, 320);

  glow.addColorStop(0, "rgba(38, 70, 140, 0.55)");
  glow.addColorStop(0.45, "rgba(22, 40, 92, 0.32)");
  glow.addColorStop(1, "rgba(3, 6, 14, 0)");
  g.fillStyle = glow;
  g.fillRect(0, 0, w, h);

  const low = g.createRadialGradient(90, 300, 10, 90, 300, 260);

  low.addColorStop(0, "rgba(70, 30, 140, 0.16)");
  low.addColorStop(1, "rgba(3, 6, 14, 0)");
  g.fillStyle = low;
  g.fillRect(0, 0, w, h);

  const texture = new THREE.CanvasTexture(canvas);

  texture.colorSpace = THREE.SRGBColorSpace;

  return texture;
};

const loadTexture = (url: string): Promise<THREE.Texture> =>
  new Promise((resolve, reject) => new THREE.TextureLoader().load(url, resolve, undefined, reject));

const isSpriteMeta = (value: unknown): value is SpriteMeta =>
  typeof value === "object" && value !== null && "size" in value && "sprites" in value;

const loadSpriteMeta = async (url: string): Promise<SpriteMeta> => {
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`${url}: ${response.status}`);
  }

  const meta: unknown = await response.json();

  if (!isSpriteMeta(meta)) {
    throw new Error(`${url}: not a sprite sheet description`);
  }

  return meta;
};

const loadArt = async (assets: string, small: boolean): Promise<Art> => {
  const art = `${assets}/aoe2`;
  const [meta, atlas, grass, patchy, dirt, forest, farm, visibility, frame, hud] = await Promise.all([
    loadSpriteMeta(`${art}/sprites.json`),
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

  return { meta, world: { atlas, grass, patchy, dirt, forest, farm, visibility }, frame, hud };
};

/** The renderer, with the textures shown on the glass set up for it. */
const createRenderer = (screenTextures: readonly THREE.Texture[]): THREE.WebGLRenderer => {
  // The default power preference lets dual-GPU laptops stay on the integrated GPU.
  const renderer = new THREE.WebGLRenderer({ antialias: false, stencil: false, depth: false });

  renderer.outputColorSpace = THREE.SRGBColorSpace;
  // Count draw calls over the whole frame (shadow, scene and post passes), not per pass.
  renderer.info.autoReset = false;

  for (const texture of screenTextures) {
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.anisotropy = renderer.capabilities.getMaxAnisotropy();
  }

  renderer.domElement.className = "wt-canvas";

  return renderer;
};

// The world is drawn unlit, as the game draws it: sprites carry their own shading
// and shadows. The lights only shape the mind's few lit materials.
const addLights = (scene: THREE.Scene): void => {
  const rim = new THREE.DirectionalLight("#8fb4ff", 1.2);

  rim.position.set(-20, 30, 20);
  scene.add(new THREE.HemisphereLight("#e2eeff", "#1a2030", 1.2), rim);
};

const createGameView = (frame: TurnData["frame"], tileWidth: number, small: boolean): GameView => {
  const aspect = frame.width / frame.height;
  const size = frame.height / 2 / (tileWidth / Math.SQRT2);
  const az = THREE.MathUtils.degToRad(GAME_AZ);
  const el = THREE.MathUtils.degToRad(GAME_EL);
  const axis = new THREE.Vector3(Math.cos(el) * Math.cos(az), Math.sin(el), Math.cos(el) * Math.sin(az));
  const right = new THREE.Vector3(Math.SQRT1_2, 0, -Math.SQRT1_2);
  const up = new THREE.Vector3().crossVectors(axis, right);

  const camera = new THREE.OrthographicCamera(-size * aspect, size * aspect, size, -size, 1, 400);

  camera.position.copy(axis).multiplyScalar(150);
  camera.lookAt(0, 0, 0);

  const targetWidth = small ? 768 : 1280;
  const target = new THREE.WebGLRenderTarget(targetWidth, Math.round(targetWidth / aspect), {
    colorSpace: THREE.SRGBColorSpace,
    samples: 4,
  });

  return { camera, target, size, aspect, axis, right, up };
};

// The pane sits on the game camera's axis, sized to fill that camera's view there,
// so the village behind lines up with the picture on the glass.
const createMindRoot = (game: GameView, paneHeight: number): { root: THREE.Group; scale: number } => {
  const scale = (game.size - PANE_DISTANCE * Math.tan(THREE.MathUtils.degToRad(GAME_FOV / 2))) / (paneHeight / 2);
  const root = new THREE.Group();

  root.position.copy(game.axis).multiplyScalar(PANE_DISTANCE);
  root.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(game.right, game.up, game.axis));
  root.scale.setScalar(scale);

  return { root, scale };
};

type SceneParts = {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  layout: Layout;
  world: World;
  game: GameView;
  screen: Screen;
  /** Holds the glass and everything behind it, in the mind's local frame. */
  mindRoot: THREE.Group;
  /** World units per mind unit. */
  mindScale: number;
  board: Board;
  mind: Mind;
  flow: Flow;
  /** Holds the three decision loops. */
  decision: THREE.Group;
  rings: Rings;
};

/** The village, the glass in front of it and the mind behind the glass. */
const assembleScene = (data: TurnData, art: Art, labels: Labels, mono: string, small: boolean): SceneParts => {
  const scene = new THREE.Scene();

  scene.background = createBackdrop();

  const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 500);

  const layout = buildLayout(data);
  const world = createWorld(layout, art.world, art.meta);

  scene.add(world.group);
  addLights(scene);

  const game = createGameView(data.frame, layout.tw, small);
  const screen = createScreen(art.frame, game.target.texture, game.aspect);
  const { root: mindRoot, scale: mindScale } = createMindRoot(game, screen.height);

  scene.add(mindRoot);
  mindRoot.add(screen.group);

  const board = createBoard(data, art.frame, art.hud);

  mindRoot.add(board.group);

  const mind = createMind(data, board, art.frame, labels, mono);

  mindRoot.add(mind.group);

  const flow = createFlow(data, board, MIND.plate.clone().sub(MIND.slab));

  board.group.add(flow.object);

  const decision = new THREE.Group();

  decision.position.copy(MIND.decision);
  mindRoot.add(decision);

  const rings = createRings();

  decision.add(rings.group);

  mindRoot.updateMatrixWorld(true);

  return { scene, camera, layout, world, game, screen, mindRoot, mindScale, board, mind, flow, decision, rings };
};

// Villagers with the longest walks near the Town Center show the trails best; the
// nearest others fill the group.
const pickFollowed = (tracked: readonly TrackedVillager[], tc: GroundPoint): TrackedVillager[] => {
  const near = (t: TrackedVillager): number => Math.hypot(t.anchor.x - tc.x, t.anchor.z - tc.z);
  const byDistance = [...tracked].sort((a, b) => near(a) - near(b));
  const walkers = byDistance.filter((t) => t.route > 2 && near(t) < 11).slice(0, MAX_WALKERS);

  return [...walkers, ...byDistance.filter((t) => !walkers.includes(t))].slice(0, TRACK_IDS.length);
};

/** Centre of a group of ground points, and how far the farthest one is from it. */
const spread = (points: readonly GroundPoint[]): { centre: GroundPoint; reach: number } => {
  const centre = points.reduce((c, p) => ({ x: c.x + p.x / points.length, z: c.z + p.z / points.length }), { x: 0, z: 0 });
  const reach = Math.max(...points.map((p) => Math.hypot(p.x - centre.x, p.z - centre.z)));

  return { centre, reach };
};

const createMindPoser = (toWorld: (local: THREE.Vector3) => THREE.Vector3, mindScale: number): MindPoser => {
  return (local, dAz, dEl, size, fov = 30) => {
    const t = toWorld(local);

    return { tx: t.x, ty: t.y, tz: t.z, az: GAME_AZ + dAz, el: GAME_EL + dEl, size: size * mindScale, fov };
  };
};

type ShotSubjects = {
  layout: Layout;
  game: GameView;
  board: Board;
  inMind: MindPoser;
  /** The followed villagers. */
  track: { centre: GroundPoint; size: number };
  /** The three house spots. */
  spots: { centre: GroundPoint; reach: number; green: GroundPoint };
};

const createWidePoses = ({ layout, game, board, inMind, track, spots }: ShotSubjects): Poses => {
  const hudLocal = board.hud.position.clone().add(MIND.slab);

  return {
    // World shots stay close to the game's own angle, where its sprites are drawn for.
    hero: { tx: layout.tc.x * 0.5 + 1.5, ty: 0, tz: layout.tc.z * 0.5 - 1.5, az: 45, el: 30, size: 9.5, fov: 28 },
    // A little wider than the frame, so the glass shows with the village around it.
    game: { tx: 0, ty: 0, tz: 0, az: GAME_AZ, el: GAME_EL, size: game.size * 1.75, fov: GAME_FOV },
    capture: inMind(new THREE.Vector3(3, 0, 1), 20, -4, 8.5),
    // The frame and the detector's plates beside it, seen from the front left.
    detect: inMind(MIND.slab.clone().add(new THREE.Vector3(3.5, 2.4, -2)), 22, -5, 9.2),
    track: { tx: track.centre.x, ty: 0.3, tz: track.centre.z, az: 50, el: 33, size: track.size, fov: 26 },
    hud: inMind(new THREE.Vector3(hudLocal.x + 1.2, hudLocal.y - 1.4, hudLocal.z + 2), 10, 6, 5.2),
    // The frame and the plate the sheets seal into.
    publish: inMind(MIND.slab.clone().lerp(MIND.plate, 0.55).add(new THREE.Vector3(0, 0.5, 1)), 16, -5, 8.5),
    decide: inMind(MIND.decision.clone().add(new THREE.Vector3(-1.2, -1.5, 0)), 4, 0, 7),
    // The whole loop at once: the glass, the frame, the Perception card and the rings,
    // with the return rail under them.
    loop: inMind(new THREE.Vector3(28, -1.8, 3), 8, 6, 27),
    house: { tx: spots.centre.x, ty: 0.6, tz: spots.centre.z, az: 48, el: 31, size: Math.max(6, spots.reach * 1.3), fov: 26 },
    verify: { tx: spots.green.x, ty: 1, tz: spots.green.z, az: 52, el: 32, size: 6.5, fov: 26 },
  };
};

// Narrow views (the stacked layout, phones on their side) fit a subject by its width:
// these shots are re-centred and widened so nothing runs off the sides.
const createNarrowPoses = (wide: Poses, board: Board, inMind: MindPoser): Poses => {
  const hudLocal = board.hud.position.clone().add(MIND.slab);

  return {
    ...wide,
    capture: inMind(new THREE.Vector3(2.2, 0, 1), 20, -4, 9.6),
    hud: inMind(new THREE.Vector3(hudLocal.x + 2.4, hudLocal.y - 1.6, hudLocal.z + 2), 10, 6, 7),
    publish: inMind(new THREE.Vector3(27, 1.4, 5), 24, -5, 10.2),
    decide: inMind(MIND.decision.clone().add(new THREE.Vector3(0, -1.4, 0)), 12, 2, 7.2),
  };
};

type Scrubber = {
  /** Sets the scroll position the scene settles towards (timeline units). */
  setTarget: (progress: number) => void;
  /** Damps the position towards the target and seeks the timeline there; returns the position. */
  advance: (delta: number) => number;
  /** True until the position has settled on the target. */
  isSettling: () => boolean;
  /** Rebuilds the timeline for other camera poses. */
  rebuild: (poses: Poses) => void;
  dispose: () => void;
};

const createScrubber = (state: SceneState, poses: Poses, initial: number): Scrubber => {
  let timeline = buildTimeline(state, poses);
  let target = initial;
  let current = -1;
  let seeked = -1;

  const advance = (delta: number): number => {
    // Lenis already smooths the wheel; this light damping only removes steps from
    // keyboard and scrollbar jumps. Frame-rate independent.
    current = current < 0 ? target : THREE.MathUtils.damp(current, target, 9, delta);

    if (Math.abs(target - current) < SETTLED) {
      current = target;
    }

    if (current !== seeked) {
      timeline.seek(Math.min(TIMELINE_LENGTH, Math.max(0, current)), false);
      seeked = current;
    }

    return current;
  };

  // GSAP records each tween's start when it first plays; rewind the old timeline to put
  // every value back where it started before the new one is built from them.
  const rebuild = (next: Poses): void => {
    timeline.seek(0, false);
    timeline.kill();
    timeline = buildTimeline(state, next);
    seeked = -1;
  };

  return {
    setTarget: (progress) => {
      target = progress;
    },
    advance,
    isSettling: () => current < 0 || Math.abs(target - current) > SETTLED,
    rebuild,
    dispose: () => {
      timeline.kill();
    },
  };
};

type Framing = {
  /** Points the camera at the safe area's centre for this view size (CSS px). */
  apply: (width: number, height: number) => void;
  setSafe: (area: SafeArea, hero: SafeArea) => void;
  /** 1 while the hero's own area frames the shot, 0 once the chapters' area does. */
  setHeroWeight: (weight: number) => void;
  /** Debug: frame on the whole view, ignoring the safe areas. */
  setCentred: (on: boolean) => void;
  /** How much the view widens so the subject box fits the safe area. */
  fit: () => number;
};

// Each shot's subject is framed in the safe area: the camera's principal point moves
// to the safe area's centre (a view offset), and `fit` scales the view so the subject
// box fits inside it. The hero frames its subject in its own area (above its text on
// phones) and hands over to the chapters' area as its text fades.
const createFraming = (camera: THREE.PerspectiveCamera): Framing => {
  let fit = 1;
  let safe: SafeArea | null = null;
  let heroSafe: SafeArea | null = null;
  let heroWeight = 1;
  let centred = false;
  let framed = "";

  const apply = (width: number, height: number): void => {
    const whole = { left: 0, top: 0, right: width, bottom: height };
    const a = !centred && safe ? safe : whole;
    const b = !centred && heroSafe ? heroSafe : a;
    const w = heroWeight;
    const s = {
      left: a.left + (b.left - a.left) * w,
      top: a.top + (b.top - a.top) * w,
      right: a.right + (b.right - a.right) * w,
      bottom: a.bottom + (b.bottom - a.bottom) * w,
    };
    const key = `${width} ${height} ${s.left.toFixed(1)} ${s.top.toFixed(1)} ${s.right.toFixed(1)} ${s.bottom.toFixed(1)}`;

    if (key === framed) {
      return;
    }

    framed = key;
    fit = subjectFit(height, s);
    camera.setViewOffset(width, height, width / 2 - (s.left + s.right) / 2, height / 2 - (s.top + s.bottom) / 2, width, height);
  };

  return {
    apply,
    setSafe: (area, hero) => {
      safe = area;
      heroSafe = hero;
    },
    setHeroWeight: (weight) => {
      heroWeight = weight;
    },
    setCentred: (on) => {
      centred = on;
    },
    fit: () => fit,
  };
};

type Viewport = {
  size: Readonly<ViewSize>;
  /** Sizes the renderer, the post chain and the framing to the view. */
  resize: () => void;
  /** Falls back to the reduced pixel ratio if back-to-back frames keep running long. */
  noteFrameTime: (delta: number) => void;
};

type ViewportParts = {
  view: HTMLElement;
  renderer: THREE.WebGLRenderer;
  post: Post;
  framing: Framing;
  small: boolean;
  mindScale: number;
};

// Full sharpness on desktop; phones and slow machines render at 1.5x. Large windows
// render at most PIXEL_BUDGET pixels: a frame's cost follows its pixel count.
const createViewport = ({ view, renderer, post, framing, small, mindScale }: ViewportParts): Viewport => {
  const size: ViewSize = { width: 1, height: 1, pointScale: 1 };
  let maxDpr = small ? REDUCED_DPR : 2;
  let slowFrames = 0;

  const resize = (): void => {
    size.width = Math.max(1, view.clientWidth);
    size.height = Math.max(1, view.clientHeight);

    const { width, height } = size;
    const budget = Math.max(1, Math.sqrt(PIXEL_BUDGET / (width * height)));
    const dpr = Math.min(window.devicePixelRatio, maxDpr, budget);

    renderer.setPixelRatio(dpr);
    renderer.setSize(width, height, false);
    setLineResolution(width, height);
    post.setSize(width, height);
    framing.apply(width, height);
    size.pointScale = ((height * dpr) / 2 / Math.tan(THREE.MathUtils.degToRad(15))) * 0.012 * mindScale;
  };

  // A 2x frame costs about 1.8x as much as a 1.5x one.
  const noteFrameTime = (delta: number): void => {
    if (maxDpr <= REDUCED_DPR) {
      return;
    }

    slowFrames = delta > SLOW_FRAME ? slowFrames + 1 : Math.max(0, slowFrames - 1);

    if (slowFrames > SLOW_FRAMES_BEFORE_FALLBACK) {
      maxDpr = REDUCED_DPR;
      resize();
    }
  };

  return { size, resize, noteFrameTime };
};

/** Tracks the pointer and wakes the loop on any input; returns the clean-up. */
const listenToInput = (pointer: Pointer, wake: () => void): (() => void) => {
  const onPointer = (event: PointerEvent): void => {
    // Only a mouse leans the camera; a touch is a scroll.
    if (event.pointerType === "mouse") {
      pointer.x = (event.clientX / window.innerWidth) * 2 - 1;
      pointer.y = (event.clientY / window.innerHeight) * 2 - 1;
      pointer.lastMouse = performance.now();
    }

    wake();
  };

  window.addEventListener("pointermove", onPointer, { passive: true });

  for (const type of INPUTS) {
    window.addEventListener(type, wake, { passive: true });
  }

  return () => {
    window.removeEventListener("pointermove", onPointer);

    for (const type of INPUTS) {
      window.removeEventListener(type, wake);
    }
  };
};

type FrameLoop = {
  start: () => void;
  /** Draws soon: input arrived or something changed. */
  wake: () => void;
  dispose: () => void;
};

// The scene draws only as often as it has to. Up to 60 fps while something moves on
// purpose: the scroll position settling, the mouse leaning the camera, the intro.
// 30 fps while it only idles (villagers, clouds, drift), woken by a timer rather than
// every display refresh. Nothing after a minute without input, off screen or in a
// hidden tab; any input wakes it. Scene time only advances while it draws, so the
// villagers carry on where they stopped.
const createFrameLoop = (stage: HTMLElement, isActive: (now: number) => boolean, render: Render): FrameLoop => {
  let running = false;
  let visible = true;
  let rafId = 0;
  let timeoutId = 0;
  let lastFrame = -1;
  let lastInput = performance.now();

  // `frame` and `schedule` call each other; `frame` only runs once `schedule` exists.
  const frame = (now: number): void => {
    rafId = 0;

    if (!running) {
      return;
    }

    const active = isActive(now);

    // A few ms of slack, so a 60 fps target on a 120 Hz display draws every second refresh.
    if (lastFrame < 0 || now - lastFrame >= (active ? ACTIVE_INTERVAL : IDLE_INTERVAL) - 4) {
      const delta = lastFrame < 0 ? 1 / 60 : Math.min(0.1, (now - lastFrame) / 1000);

      lastFrame = now;
      render(delta, active);
    }

    schedule();
  };

  const schedule = (): void => {
    if (!running || rafId || timeoutId) {
      return;
    }

    const now = performance.now();

    if (isActive(now)) {
      rafId = requestAnimationFrame(frame);

      return;
    }

    // At rest: wait for input (wake).
    if (now - lastInput > REST_AFTER) {
      return;
    }

    timeoutId = window.setTimeout(
      () => {
        timeoutId = 0;

        if (running) {
          rafId = requestAnimationFrame(frame);
        }
      },
      Math.max(0, lastFrame + IDLE_INTERVAL - now - 6),
    );
  };

  const wake = (): void => {
    lastInput = performance.now();

    // Active input should not wait for an idle timer.
    if (timeoutId && isActive(lastInput)) {
      clearTimeout(timeoutId);
      timeoutId = 0;
    }

    schedule();
  };

  const start = (): void => {
    if (running || !visible || document.hidden) {
      return;
    }

    running = true;
    lastFrame = -1;
    lastInput = performance.now();
    schedule();
  };

  const stop = (): void => {
    running = false;
    cancelAnimationFrame(rafId);
    clearTimeout(timeoutId);
    rafId = timeoutId = 0;
  };

  // Entries queue in order: the last one is the stage's current state.
  const intersection = new IntersectionObserver((entries) => {
    const entry: IntersectionObserverEntry | undefined = entries.at(-1);

    visible = entry?.isIntersecting ?? true;

    if (visible) {
      start();
    } else {
      stop();
    }
  });

  intersection.observe(stage);

  const onVisibility = (): void => {
    if (document.hidden) {
      stop();
    } else {
      start();
    }
  };

  document.addEventListener("visibilitychange", onVisibility);

  const dispose = (): void => {
    stop();
    intersection.disconnect();
    document.removeEventListener("visibilitychange", onVisibility);
  };

  return { start, wake, dispose };
};

// Narrow screens: the whole loop is too small to read, so the camera rides along with
// the action card on its way back to the glass instead.
const createActionFollow = (
  actionAt: (travel: number, out: THREE.Vector3) => THREE.Vector3,
  mindRoot: THREE.Object3D,
  mindScale: number,
  start: CameraPose,
): ((pose: CameraPose, action: SceneState["action"]) => void) => {
  const cardWorld = new THREE.Vector3();
  const follow: CameraPose = { ...start };

  return (pose, action) => {
    if (action.on < 0.001) {
      return;
    }

    const weight = action.on * fadeWindow(action.travel, 0, 0.15, 0.8, 1);

    if (weight < 0.001) {
      return;
    }

    mindRoot.localToWorld(actionAt(action.travel, cardWorld));
    follow.tx = cardWorld.x;
    follow.ty = cardWorld.y;
    follow.tz = cardWorld.z;
    follow.az = GAME_AZ + 14;
    follow.el = GAME_EL + 4;
    follow.size = FOLLOW_SIZE * mindScale;
    follow.fov = 30;

    for (const k of ["tx", "ty", "tz", "az", "el", "size", "fov"] as const) {
      pose[k] += (follow[k] - pose[k]) * weight;
    }
  };
};

type CameraDirector = {
  /** The pose the camera took this frame. */
  pose: Readonly<CameraPose>;
  /** Places the camera for this frame: the timeline's shot with its drift, pointer lean and intro, or the debug pose. */
  place: (time: number, delta: number, progress: number, intro: number) => void;
  setDebugPose: (pose: Partial<CameraPose> | null) => void;
  hasDebugPose: () => boolean;
};

type DirectorParts = {
  camera: THREE.PerspectiveCamera;
  state: SceneState;
  pointer: Pointer;
  framing: Framing;
  size: Readonly<ViewSize>;
  small: boolean;
  start: CameraPose;
  /** Bends the pose towards a subject that moves (the action card on narrow screens). */
  follow: (pose: CameraPose) => void;
};

const createCameraDirector = ({ camera, state, pointer, framing, size, small, start, follow }: DirectorParts): CameraDirector => {
  const pose: CameraPose = { ...start };
  let debugPose: Partial<CameraPose> | null = null;

  // The hero drifts and follows the pointer; every other shot breathes a little and
  // leans towards the pointer, so a held beat is never frozen. On load the camera
  // settles in from a little wider and higher.
  const animate = (time: number, hero: number, intro: number): void => {
    pose.az += hero * (Math.sin(time * 0.08) * 3 + pointer.sx * 3);
    pose.el += hero * pointer.sy * 1.5;

    const held = 1 - hero;

    pose.az += held * (Math.sin(time * 0.13) * 0.7 + pointer.sx * (small ? 0 : 1.1));
    pose.el += held * (Math.sin(time * 0.1 + 1.3) * 0.35 + pointer.sy * (small ? 0 : 0.5));
    pose.size *= 1 + hero * Math.sin(time * 0.11) * 0.03;
    pose.size *= 1 + 0.14 * (1 - intro);
    follow(pose);
    pose.el += 5 * (1 - intro);
  };

  const place = (time: number, delta: number, progress: number, intro: number): void => {
    // The hero's drift and pointer lean fade out by chapter 1.
    const hero = THREE.MathUtils.clamp(1 - progress, 0, 1) + THREE.MathUtils.clamp(progress - HERO_RETURN, 0, 1);

    pointer.sx = THREE.MathUtils.damp(pointer.sx, pointer.x, 3, delta);
    pointer.sy = THREE.MathUtils.damp(pointer.sy, pointer.y, 3, delta);

    Object.assign(pose, state.cam, debugPose ?? {});

    if (!debugPose) {
      animate(time, hero, intro);
    }

    applyPose(camera, pose, size.width / size.height, framing.fit());
  };

  return {
    pose,
    place,
    setDebugPose: (next) => {
      debugPose = next;
    },
    hasDebugPose: () => debugPose !== null,
  };
};

// The 2D overlays read these; write them only when they change.
const createStageWriter = (stage: HTMLElement): ((state: SceneState) => void) => {
  const written = { stamp: "", input: "", step: "" };

  return (state) => {
    const stamp = state.stamp.on.toFixed(3);

    if (stamp !== written.stamp) {
      written.stamp = stamp;
      stage.style.setProperty("--wt-stamp", stamp);
    }

    const input = state.act.on.toFixed(3);

    if (input !== written.input) {
      written.input = input;
      stage.style.setProperty("--wt-input", input);
    }

    const { keys, click } = state.act;
    const step = String(keys >= 2.5 ? 5 : click > 0.05 ? 4 : state.world.cursor > 0.05 ? 3 : Math.round(keys));

    if (step !== written.step) {
      written.step = step;
      stage.dataset.step = step;
    }
  };
};

// The glass shows the game camera's render until the real frame has replaced it.
// That camera never moves, so 30 updates a second keep the villagers in it smooth.
const createGlassRenderer = (
  renderer: THREE.WebGLRenderer,
  scene: THREE.Scene,
  game: GameView,
  mindRoot: THREE.Object3D,
): ((time: number, pane: SceneState["pane"]) => void) => {
  let liveAt = Number.NEGATIVE_INFINITY;

  return (time, pane) => {
    const due = pane.on > 0.002 && pane.mix < 0.999 && time - liveAt >= 1 / 31;

    if (due) {
      liveAt = time;

      const mindWasVisible = mindRoot.visible;

      mindRoot.visible = false;
      renderer.setRenderTarget(game.target);
      renderer.render(scene, game.camera);
      renderer.setRenderTarget(null);
      mindRoot.visible = mindWasVisible;
    }
  };
};

const setPostEffects = (post: Post, pose: CameraPose, state: SceneState, mindVisible: boolean): void => {
  const dim = state.world.dim;
  // The miniature look is for wide world shots; close-ups keep more in focus.
  const wide = 0.35 + 0.65 * THREE.MathUtils.smoothstep(pose.size, 5, 10);

  post.setMiniature((1 - dim) * THREE.MathUtils.smoothstep(pose.fov, 12, 24) * wide);

  // Only the mind and the marks drawn on the world (track boxes, the footprint, the
  // click) are bright enough to bloom; the world on its own never is.
  const click = state.act.click;
  const bright = mindVisible || state.world.track > 0.005 || state.world.ghost > 0.005 || (click > 0.005 && click < 0.995);

  post.setBloom(bright ? 0.55 + 0.75 * dim : 0);
};

const isMindVisible = (state: SceneState): boolean =>
  state.pane.on > 0.002 ||
  state.slab.on > 0.01 ||
  state.card.on > 0.01 ||
  state.decide.on > 0.01 ||
  state.action.on > 0.001 ||
  state.mind.space > 0.01;

type DebugParts = {
  renderer: THREE.WebGLRenderer;
  camera: THREE.Camera;
  size: Readonly<ViewSize>;
  inMind: MindPoser;
  toWorld: (local: THREE.Vector3) => THREE.Vector3;
  render: Render;
  setPose: DebugHooks["setPose"];
  setCentred: DebugHooks["setCentred"];
  info: DebugHooks["info"];
};

const createDebugHooks = ({ renderer, camera, size, inMind, toWorld, render, setPose, setCentred, info }: DebugParts): DebugHooks => ({
  setPose,
  setCentred,
  mindPose: (local, dAz, dEl, poseSize) => setPose(inMind(new THREE.Vector3(local[0], local[1], local[2]), dAz, dEl, poseSize)),
  projectMind: (points) =>
    points.map((p) => {
      const v = toWorld(new THREE.Vector3(p[0], p[1], p[2])).project(camera);

      return [Math.round((v.x * 0.5 + 0.5) * size.width), Math.round((-v.y * 0.5 + 0.5) * size.height)];
    }),
  bench: (frames) => {
    const gl = renderer.getContext();
    const pixel = new Uint8Array(4);
    const sync = (): void => gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixel);

    render(1 / 60, true);
    sync();

    const t0 = performance.now();

    for (let i = 0; i < frames; i++) {
      render(1 / 60, true);
    }

    sync();

    return (performance.now() - t0) / frames;
  },
  info,
});

export const mountScene = async ({ view, stage, assets, progress }: MountOptions): Promise<SceneHandle> => {
  const data: TurnData = turn;

  // Canvas text in the scene uses the site's mono face; wait for it.
  await document.fonts.ready;

  const mono = getComputedStyle(stage).getPropertyValue("--font-mono").trim() || "ui-monospace, monospace";
  const small = window.matchMedia(SMALL_SCREEN).matches;
  const art = await loadArt(assets, small);

  const renderer = createRenderer([art.frame, art.hud]);

  view.append(renderer.domElement);

  const labels = createLabels(view);

  const { scene, camera, layout, world, game, screen, mindRoot, mindScale, board, mind, flow, decision, rings } = assembleScene(
    data,
    art,
    labels,
    mono,
    small,
  );

  const toWorld = (local: THREE.Vector3): THREE.Vector3 => mindRoot.localToWorld(local.clone());

  const [, , green] = layout.spots;
  const spots = spread(layout.spots);
  const groundPoint = (p: GroundPoint): THREE.Vector3 => new THREE.Vector3(p.x, 0, p.z);
  const overlays = createOverlays(data, board, decision, labels, {
    root: scene,
    cursorPath: [groundPoint({ x: green.x + 3, z: green.z + 3 }), ...layout.spots.map(groundPoint)],
    ledgerAt: new THREE.Vector3(green.x, 1.6, green.z),
    green: groundPoint(green),
  });

  const followed = pickFollowed(world.tracked, layout.tc);
  const tracking = createTracking(followed, TRACK_IDS, labels);
  const trackSpread = spread(followed.map((t) => t.anchor));
  const trackSize = THREE.MathUtils.clamp(trackSpread.reach * 0.8, 5.5, 9);

  const inMind = createMindPoser(toWorld, mindScale);
  const widePoses = createWidePoses({
    layout,
    game,
    board,
    inMind,
    track: { centre: trackSpread.centre, size: trackSize },
    spots: { ...spots, green },
  });
  const narrowPoses = createNarrowPoses(widePoses, board, inMind);

  let narrow = false;
  let poses = widePoses;
  const state = createState(poses.hero);

  scene.add(tracking.group);

  const scrubber = createScrubber(state, poses, progress);

  const setNarrow = (on: boolean): void => {
    if (on === narrow) {
      return;
    }

    narrow = on;
    poses = on ? narrowPoses : widePoses;
    overlays.setNarrow(on);
    scrubber.rebuild(poses);
  };

  const post = createPost(renderer, scene, camera, small);

  const framing = createFraming(camera);
  const viewport = createViewport({ view, renderer, post, framing, small, mindScale });
  const { size } = viewport;

  viewport.resize();

  const debugStage: DebugStage = stage;
  const pointer: Pointer = { x: 0, y: 0, sx: 0, sy: 0, lastMouse: Number.NEGATIVE_INFINITY };
  let clock = 0;
  let frames = 0;
  const ghostAt = new THREE.Vector3();

  const followAction = createActionFollow(mind.actionAt, mindRoot, mindScale, widePoses.loop);
  const director = createCameraDirector({
    camera,
    state,
    pointer,
    framing,
    size,
    small,
    start: poses.hero,
    follow: (pose) => {
      if (narrow) {
        followAction(pose, state.action);
      }
    },
  });
  const writeStage = createStageWriter(stage);
  const renderGlass = createGlassRenderer(renderer, scene, game, mindRoot);

  const isActive = (now: number): boolean =>
    scrubber.isSettling() || clock < INTRO + 0.2 || now - pointer.lastMouse < POINTER_ACTIVE || director.hasDebugPose();

  const updateParts = (time: number, intro: number): void => {
    const dim = state.world.dim;

    overlays.cursorAt(state.world.cursor, ghostAt);

    world.update(time, camera, {
      light: 1 - 0.78 * dim,
      open: intro,
      ghost: state.world.ghost,
      ghostX: ghostAt.x,
      ghostZ: ghostAt.z,
      ghostOk: state.world.cursor > 2.5,
      build: state.world.build,
    });

    tracking.update(state.world.track, time);
    screen.update(state.pane, time);
    board.update(state);
    mind.update(state, time, size.pointScale / mindScale);
    flow.update(state, size.pointScale);
    rings.update(state, time);
    overlays.update(state);
    mindRoot.visible = isMindVisible(state);
  };

  const render: Render = (delta, active) => {
    frames++;
    renderer.info.reset();
    clock += delta;

    const time = clock;

    if (active) {
      viewport.noteFrameTime(delta);
    }

    const progress = scrubber.advance(delta);

    debugStage.wtProgress = progress;
    framing.setHeroWeight(1 - THREE.MathUtils.smoothstep(progress, HERO_FADE_START, HERO_FADE_END));
    framing.apply(size.width, size.height);

    // On load the map opens out of the dark and the camera settles (expo out, 1.8 s).
    const intro = 1 - Math.pow(2, -10 * Math.min(1, time / INTRO));

    director.place(time, delta, progress, intro);
    updateParts(time, intro);
    writeStage(state);
    renderGlass(time, state.pane);
    setPostEffects(post, director.pose, state, mindRoot.visible);
    post.render(delta);
    labels.update(camera, size.width, size.height);
  };

  const loop = createFrameLoop(stage, isActive, render);
  const resizeObserver = new ResizeObserver(() => {
    viewport.resize();
    loop.wake();
  });

  resizeObserver.observe(view);

  const stopListening = listenToInput(pointer, loop.wake);

  debugStage.wtDebug = createDebugHooks({
    renderer,
    camera,
    size,
    inMind,
    toWorld,
    render,
    setPose: (pose) => {
      director.setDebugPose(pose);
      loop.wake();
    },
    setCentred: (on) => {
      framing.setCentred(on);
      framing.apply(size.width, size.height);
      loop.wake();
    },
    info: () => ({
      frames,
      narrow,
      dpr: renderer.getPixelRatio(),
      calls: renderer.info.render.calls,
      triangles: renderer.info.render.triangles,
      mindScale,
      gameSize: game.size,
      poses,
      followed: followed.map((t) => [t.anchor.x, t.anchor.z, t.walking]),
      villagers: world.villagerCheck(),
    }),
  });

  render(1 / 60, true);
  loop.start();

  return {
    setProgress: (progress) => {
      scrubber.setTarget(progress);
      loop.wake();
    },
    setSafe: (area, areas) => {
      framing.setSafe(area, areas?.hero ?? area);
      setNarrow(Boolean(areas?.stacked) || area.right - area.left < NARROW_WIDTH);
      labels.setInsetTop(area.top);
      framing.apply(size.width, size.height);
      loop.wake();
    },
    dispose: () => {
      loop.dispose();
      resizeObserver.disconnect();
      stopListening();
      scrubber.dispose();
      game.target.dispose();
      post.dispose();
      renderer.dispose();
      renderer.domElement.remove();
      labels.dispose();
    },
  };
};
