import * as React from "react";
import { Pause, Play } from "lucide-react";

/**
 * Animated hero figure: one agent turn played end-to-end.
 *
 * Loops through capture → detect → reason → act using a real Mac+VMware
 * screenshot of a Dark Age AoE2 game, with the actual detector's predicted
 * bounding boxes from the labeling pipeline. Strategist and executor copy
 * is hand-written but uses the exact string shapes that the live agent
 * emits (entity_utils.py:58, claude_tools.py).
 *
 * All animation derives from a single elapsedMs counter, which makes
 * step-jumping, pausing, and reduced-motion fallback trivial.
 */

// Frame dimensions of /screenshots/agent/turn-demo.jpg (resized from the
// real 3024x1964 Mac capture; aspect ratio preserved).
const IMG_W = 1440;
const IMG_H = 935;

// Five hex strings mirroring apps/agent/src/overlay.py:51-61. Duplicated by
// design — the landing workspace can't import Python and these change once
// a year at most.
const CATEGORY_COLOR = {
  resource: "#228B22",
  animal: "#FFA500",
  building: "#4169E1",
  defense: "#9400D3",
  unit: "#DC143C",
} as const;
type Category = keyof typeof CATEGORY_COLOR;

interface Entity {
  id: string;
  class: string;
  category: Category;
  cx: number; // normalized 0..1
  cy: number;
  w: number;
  h: number;
  conf: number; // 0..1
}

// The detector's complete output on this frame — 6 entries (4 forest
// clusters + 2 villagers). Coordinates verbatim from
// packages/detection/.../prelabeled_new/labels/Screenshot 2026-01-19 at 19.05.35.txt.
// Confidences are illustrative; the prelabel format doesn't store them.
// The detector legitimately missed the small Dark Age building sprites
// on this frame — that's the actual model output, not a curated subset.
const ENTITIES: Entity[] = [
  { id: "tree_0", class: "tree", category: "resource", cx: 0.1824, cy: 0.247, w: 0.165, h: 0.2271, conf: 0.94 },
  { id: "tree_1", class: "tree", category: "resource", cx: 0.69, cy: 0.2074, w: 0.1739, h: 0.236, conf: 0.93 },
  { id: "tree_2", class: "tree", category: "resource", cx: 0.24, cy: 0.6501, w: 0.1407, h: 0.1902, conf: 0.91 },
  { id: "tree_3", class: "tree", category: "resource", cx: 0.4532, cy: 0.2753, w: 0.0509, h: 0.0909, conf: 0.78 },
  { id: "villager_0", class: "villager", category: "unit", cx: 0.4826, cy: 0.3303, w: 0.0224, h: 0.0337, conf: 0.83 },
  { id: "villager_1", class: "villager", category: "unit", cx: 0.2144, cy: 0.3146, w: 0.0265, h: 0.0417, conf: 0.86 },
];

// The "send villager to wood" narrative: villager_0 (idle, center)
// → tree_0 (nearest unexplored forest).
const ACTION_SOURCE_ID = "villager_0";
const ACTION_TARGET_ID = "tree_0";

interface Step {
  title: string;
  caption: string;
}

// Per prompts/strategist.md:1-9, the strategist reads the *screenshot* (for
// resource bar / pop / age text that YOLO can't detect) and emits goals;
// the executor is text-only and uses YOLO entity ids to resolve targets
// (providers/claude.py:233). Captions reflect that data flow.
const STEPS: Step[] = [
  { title: "1. Capture screen", caption: "Frame grabbed from the AoE2 client" },
  { title: "2. Detect entities", caption: "YOLOv8 finds units, buildings, resources" },
  { title: "3. Strategist reasons", caption: "Sonnet reads the screenshot, sets goals" },
  { title: "4. Executor acts", caption: "Haiku turns goals into tool calls on YOLO ids" },
];

const STEP_DURATIONS_MS = [1500, 2500, 2500, 2500] as const;
const CYCLE_MS = STEP_DURATIONS_MS.reduce((a, b) => a + b, 0);
const STEP_STARTS_MS = STEP_DURATIONS_MS.reduce<number[]>((acc, d, i) => {
  acc.push(i === 0 ? 0 : acc[i - 1] + STEP_DURATIONS_MS[i - 1]);
  return acc;
}, []);

const BOX_REVEAL_STAGGER_MS = 160;
const TYPE_MS_PER_CHAR = 10;

// Entity summary block — uses entity_utils.py:58's exact format.
const ENTITY_SUMMARY_LINES = ENTITIES.map(
  (e) =>
    `${e.id}: ${e.class} at (${Math.round(e.cx * IMG_W)},${Math.round(e.cy * IMG_H)}) [${Math.round(e.conf * 100)}%]`,
);

// Strategist-shaped output: reasoning + resource readings (read from the
// screenshot's UI bar, not from YOLO) + prioritized goals. Mirrors the
// StrategistResponse model in providers/strategist.py:43-48.
const REASONING_TEXT =
  "Dark Age opener — resource bar reads F=200 W=100 G=0 S=200, pop 3/5. Wood is the bottleneck. Goals: gather wood→200 (P9), queue villagers→10 (P8), advance to Feudal Age (P4).";

// Real tool names from apps/agent/src/providers/claude_tools.py.
const TOOL_CALLS = [
  `click target="villager_0"`,
  `right_click target="tree_0"`,
];

function stepFromElapsed(elapsedMs: number): { step: number; stepElapsed: number } {
  const t = elapsedMs % CYCLE_MS;
  for (let i = STEP_DURATIONS_MS.length - 1; i >= 0; i--) {
    if (t >= STEP_STARTS_MS[i]) {
      return { step: i, stepElapsed: t - STEP_STARTS_MS[i] };
    }
  }
  return { step: 0, stepElapsed: t };
}

function typedSlice(text: string, ms: number): string {
  if (ms <= 0) return "";
  return text.slice(0, Math.min(text.length, Math.floor(ms / TYPE_MS_PER_CHAR)));
}

export default function GameplayLoopDemo(): React.ReactElement {
  // The whole component runs off this single counter.
  const [elapsedMs, setElapsedMs] = React.useState(0);
  const [paused, setPaused] = React.useState(false);
  const [reducedMotion, setReducedMotion] = React.useState(false);

  React.useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    setReducedMotion(mq.matches);
    const onChange = (e: MediaQueryListEvent) => setReducedMotion(e.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);

  React.useEffect(() => {
    if (paused || reducedMotion) return;
    const tickMs = 60;
    const id = window.setInterval(() => setElapsedMs((e) => e + tickMs), tickMs);
    return () => window.clearInterval(id);
  }, [paused, reducedMotion]);

  // When reduced motion is on we freeze on the final composite (everything
  // visible, no typing in progress). Otherwise derive from elapsedMs.
  const { step, stepElapsed } = reducedMotion
    ? { step: 3, stepElapsed: STEP_DURATIONS_MS[3] }
    : stepFromElapsed(elapsedMs);

  // Box visibility: at step 0 nothing; at step 1 staggered reveal; later always visible.
  const isBoxVisible = (index: number) => {
    if (step === 0) return false;
    if (step >= 2) return true;
    return stepElapsed >= (index + 1) * BOX_REVEAL_STAGGER_MS;
  };

  // Side-panel typing: each section types in during its own step. In later
  // steps the section is already complete.
  const entitiesTyped =
    step <= 0
      ? ""
      : step === 1
        ? typedSlice(ENTITY_SUMMARY_LINES.join("\n"), stepElapsed)
        : ENTITY_SUMMARY_LINES.join("\n");
  const reasoningTyped =
    step < 2 ? "" : step === 2 ? typedSlice(REASONING_TEXT, stepElapsed) : REASONING_TEXT;
  const toolCallsFull = TOOL_CALLS.join("\n");
  const toolCallsTyped = step < 3 ? "" : typedSlice(toolCallsFull, stepElapsed);

  const source = ENTITIES.find((e) => e.id === ACTION_SOURCE_ID)!;
  const target = ENTITIES.find((e) => e.id === ACTION_TARGET_ID)!;

  const jumpToStep = (s: number) => {
    setElapsedMs(STEP_STARTS_MS[s]);
    setPaused(false);
  };

  return (
    <div
      className="rounded-xl border bg-card p-4"
      role="figure"
      aria-label="Animated demonstration of one agent turn: capture, detect, reason, act"
    >
      <div className="grid gap-4 lg:grid-cols-[1.6fr_1fr]">
        {/* Image + SVG overlay */}
        <figure className="m-0 flex flex-col">
          <div
            className="relative w-full overflow-hidden rounded-md border bg-muted"
            style={{ aspectRatio: `${IMG_W} / ${IMG_H}` }}
          >
            <img
              src="/screenshots/agent/turn-demo.jpg"
              alt="Real screenshot of Age of Empires II Definitive Edition running in VMware Fusion on macOS. Dark Age, with two villagers and a few forest patches on a partially-explored map."
              className="absolute inset-0 h-full w-full object-cover"
              loading="eager"
              decoding="async"
            />
            <svg
              viewBox={`0 0 ${IMG_W} ${IMG_H}`}
              preserveAspectRatio="none"
              className="absolute inset-0 h-full w-full"
              aria-hidden="true"
            >
              {ENTITIES.map((e, i) => {
                const x = (e.cx - e.w / 2) * IMG_W;
                const y = (e.cy - e.h / 2) * IMG_H;
                const w = e.w * IMG_W;
                const h = e.h * IMG_H;
                const visible = isBoxVisible(i);
                const isActionSource = step === 3 && e.id === ACTION_SOURCE_ID;
                const isActionTarget = step === 3 && e.id === ACTION_TARGET_ID;
                const color = CATEGORY_COLOR[e.category];
                return (
                  <g
                    key={e.id}
                    style={{
                      opacity: visible ? 1 : 0,
                      transition: "opacity 220ms ease-out",
                    }}
                  >
                    <title>{`${e.id} · ${e.class} · ${Math.round(e.conf * 100)}% confidence`}</title>
                    <rect
                      x={x}
                      y={y}
                      width={w}
                      height={h}
                      fill="none"
                      stroke={color}
                      strokeWidth={isActionSource || isActionTarget ? 6 : 3}
                      style={
                        isActionSource && !reducedMotion
                          ? { animation: "gld-pulse 900ms ease-in-out infinite" }
                          : undefined
                      }
                    />
                    <rect
                      x={x}
                      y={y - 22}
                      width={Math.min(w + 80, e.id.length * 11 + 16)}
                      height={20}
                      fill={color}
                      opacity={0.92}
                    />
                    <text
                      x={x + 6}
                      y={y - 7}
                      fill="white"
                      fontSize="13"
                      fontFamily="ui-monospace, SFMono-Regular, Menlo, monospace"
                      fontWeight="600"
                    >
                      {`${e.id} ${Math.round(e.conf * 100)}%`}
                    </text>
                  </g>
                );
              })}
              {/* Action vector: villager_3 → tree_0 */}
              {step === 3 && (
                <g>
                  <defs>
                    <marker
                      id="gld-arrow"
                      viewBox="0 0 10 10"
                      refX="9"
                      refY="5"
                      markerWidth="8"
                      markerHeight="8"
                      orient="auto-start-reverse"
                    >
                      <path d="M 0 0 L 10 5 L 0 10 z" fill={CATEGORY_COLOR.unit} />
                    </marker>
                  </defs>
                  <line
                    x1={source.cx * IMG_W}
                    y1={source.cy * IMG_H}
                    x2={target.cx * IMG_W}
                    y2={target.cy * IMG_H}
                    stroke={CATEGORY_COLOR.unit}
                    strokeWidth={5}
                    strokeDasharray="14 10"
                    markerEnd="url(#gld-arrow)"
                    opacity={0.95}
                  />
                </g>
              )}
            </svg>
            <style>{`
              @keyframes gld-pulse {
                0%, 100% { stroke-opacity: 1; stroke-width: 6; }
                50%      { stroke-opacity: 0.5; stroke-width: 10; }
              }
            `}</style>
          </div>
          <figcaption className="mt-3 flex items-baseline gap-3">
            <span className="font-semibold text-foreground">{STEPS[step].title}</span>
            <span className="text-sm text-muted-foreground">{STEPS[step].caption}</span>
          </figcaption>
        </figure>

        {/* Narration panel */}
        <aside className="flex flex-col gap-3 text-xs">
          <Section label="Entity detections (YOLOv8)" active={step === 1} done={step > 1}>
            <pre className="whitespace-pre-wrap font-mono text-[11px] leading-snug text-foreground">
              {entitiesTyped || " "}
              {step === 1 && !reducedMotion && <Caret />}
            </pre>
          </Section>
          <Section label="Strategist (Sonnet · reads screenshot)" active={step === 2} done={step > 2} dimmed={step < 2}>
            <p className="leading-snug text-foreground">
              {reasoningTyped || " "}
              {step === 2 && !reducedMotion && <Caret />}
            </p>
          </Section>
          <Section label="Executor (Haiku · text only, uses YOLO ids)" active={step === 3} dimmed={step < 3}>
            <pre className="whitespace-pre-wrap font-mono text-[11px] leading-snug text-foreground">
              {toolCallsTyped || " "}
              {step === 3 && !reducedMotion && <Caret />}
            </pre>
          </Section>
        </aside>
      </div>

      {/* Controls */}
      <div className="mt-4 flex items-center gap-3">
        <button
          type="button"
          onClick={() => setPaused((p) => !p)}
          disabled={reducedMotion}
          aria-label={paused ? "Play" : "Pause"}
          className="inline-flex h-8 w-8 items-center justify-center rounded-md border bg-background text-muted-foreground hover:text-foreground disabled:opacity-40"
        >
          {paused ? <Play className="h-4 w-4" /> : <Pause className="h-4 w-4" />}
        </button>
        <div className="flex items-center gap-1.5">
          {STEPS.map((s, i) => (
            <button
              key={s.title}
              type="button"
              onClick={() => jumpToStep(i)}
              aria-label={`Jump to ${s.title}`}
              aria-current={step === i ? "step" : undefined}
              className={`h-2 w-6 rounded-full transition-colors ${
                step === i ? "bg-foreground" : "bg-border hover:bg-muted-foreground"
              }`}
            />
          ))}
        </div>
        <p className="ml-auto text-xs text-muted-foreground">
          Real screenshot · real detector output (6 entities on this frame) · strategist/executor copy is illustrative.
        </p>
      </div>
    </div>
  );
}

function Section({
  label,
  active,
  done,
  dimmed,
  children,
}: {
  label: string;
  active?: boolean;
  done?: boolean;
  dimmed?: boolean;
  children: React.ReactNode;
}): React.ReactElement {
  return (
    <div
      className={`rounded-md border p-3 transition-colors ${
        active ? "border-foreground bg-background" : "border-border bg-background/40"
      } ${dimmed ? "opacity-50" : ""}`}
    >
      <div className="mb-1.5 flex items-center justify-between">
        <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
          {label}
        </span>
        {done && <span className="text-[10px] text-muted-foreground">✓</span>}
      </div>
      {children}
    </div>
  );
}

function Caret(): React.ReactElement {
  return (
    <span
      aria-hidden="true"
      className="ml-0.5 inline-block w-[1px] -translate-y-px"
      style={{
        height: "1em",
        backgroundColor: "currentColor",
        animation: "gld-blink 1s steps(2) infinite",
        verticalAlign: "middle",
      }}
    >
      <style>{`@keyframes gld-blink { 50% { opacity: 0; } }`}</style>
    </span>
  );
}
