/**
 * Single source of truth for the sequential reading path through docs/.
 *
 * Hand-curated rather than derived from frontmatter because the docs have
 * no frontmatter. Sections and pages follow the order of docs/index.md, and
 * titles and blurbs are taken from its tables.
 *
 * If `docs/index.md` changes, update this file. A section id is its
 * directory under docs/, a page slug is the filename without `.md`, and a
 * full slug is `<section>/<page>`.
 */

export type SectionId = "agent" | "detection" | "arena" | "web" | "autoresearch";

export interface Page {
  /** Filename under the section directory, without `.md`. */
  slug: string;
  /** Display title for the sidebar and prev/next links. */
  title: string;
  /** One-line description, shown on hover. */
  blurb: string;
}

export interface Section {
  /** Directory name under docs/. */
  id: SectionId;
  /** Full title. */
  title: string;
  pages: Page[];
}

export const SECTIONS: Section[] = [
  {
    id: "agent",
    title: "Gameplay agent",
    pages: [
      {
        slug: "overview",
        title: "Overview",
        blurb:
          "What the agent is, the three loops, deployment topologies, repo layout, configuration.",
      },
      {
        slug: "perception",
        title: "Perception",
        blurb:
          "Window capture on Windows and macOS, the capture, detection and HUD lanes, OCR, ownership, how a Perception frame is versioned and published.",
      },
      {
        slug: "tracking",
        title: "Tracking",
        blurb:
          "Stable track IDs and smoothed labels, the screentrack Rust tracker, camera-motion resets, the tracker benchmark.",
      },
      {
        slug: "decision-loops",
        title: "Decision loops",
        blurb:
          "TypeSafe policy, strategist and executor on the perceive, act and deliberate clocks; the feasible catalog, revalidation, fallback, goals, alarms and recovery.",
      },
      {
        slug: "actions-and-execution",
        title: "Actions and execution",
        blurb:
          "The action model, executor tools, spend checks, target resolution, input on Windows and macOS, the action ledger, the offline scenario runner.",
      },
      {
        slug: "llm-providers",
        title: "LLM providers",
        blurb:
          "ChatWire adapters, wire selection, model defaults, the TypeSafe client, cost accounting, the text completer.",
      },
      {
        slug: "prompts-and-context",
        title: "Prompts and context",
        blurb: "Prompt files, what each role reads per turn, cross-game memories.",
      },
      {
        slug: "game-knowledge",
        title: "Game knowledge",
        blurb:
          "The packages/data reference database and where the agent's runtime game facts actually live.",
      },
    ],
  },
  {
    id: "detection",
    title: "Detection",
    pages: [
      {
        slug: "detector",
        title: "Entity detector",
        blurb:
          "The inference library, the served YOLO26n model, pre and postprocessing, thresholds, ownership, the 65-class taxonomy.",
      },
      {
        slug: "serving-and-evaluation",
        title: "Serving and evaluation",
        blurb:
          "The detection server, the one-inference-at-a-time rule, model export, the model registry, release evaluation.",
      },
      {
        slug: "training",
        title: "Training",
        blurb:
          "Real, synthetic and composite training data, train_yolo.py settings, local and cloud runs, promotion.",
      },
      {
        slug: "labeling",
        title: "Labeling",
        blurb:
          "Capture sessions, prelabels, X-AnyLabeling review, label aliases, the independent test set.",
      },
      {
        slug: "sprite-extraction",
        title: "Sprite extraction",
        blurb: "Decoding .sld sprites into PNGs for the synthetic generators.",
      },
    ],
  },
  {
    id: "arena",
    title: "Synthetic arena",
    pages: [
      {
        slug: "overview",
        title: "Overview",
        blurb:
          "The aoe2-arena CLI, profiles, how a variant is invoked, the infra stack.",
      },
      {
        slug: "world-sim",
        title: "World sim",
        blurb: "The AoE2-lite economy model and what the arena model sees.",
      },
      {
        slug: "ranking",
        title: "Ranking",
        blurb:
          "Round-robin play over scenarios, Bradley-Terry ratings, bootstrap intervals.",
      },
      {
        slug: "event-broker",
        title: "Event broker",
        blurb:
          "The EventBroker protocol, in-process and Redis Streams backends, backpressure.",
      },
      {
        slug: "event-log-and-replay",
        title: "Event log and replay",
        blurb: "Event types, the DuckDB persister and reader, the fork primitive.",
      },
    ],
  },
  {
    id: "web",
    title: "Arena web",
    pages: [
      {
        slug: "architecture",
        title: "Architecture",
        blurb:
          "FastAPI endpoints, live versus cold event streams, forks, the reaper, the dashboard routes.",
      },
      {
        slug: "run-viewer",
        title: "Run viewer",
        blurb:
          "The run page: timeline, World, Trace, Diff and Operator panels, forking from the UI.",
      },
      {
        slug: "running-locally",
        title: "Running locally",
        blurb:
          "just recipes, the Vite proxy, environment variables, deployment modes.",
      },
    ],
  },
  {
    id: "autoresearch",
    title: "Autoresearch",
    pages: [
      {
        slug: "overview",
        title: "Overview",
        blurb:
          "Playing and scoring real games, the experiment ledger and merge gate, the prompt tournament.",
      },
      {
        slug: "mutation-and-memory",
        title: "Mutation and memory",
        blurb: "PromptMutator and MemoryChain.",
      },
    ],
  },
];

export interface FlatPage {
  fullSlug: string;
  section: Section;
  page: Page;
}

export const FLAT_PAGES: FlatPage[] = SECTIONS.flatMap((section) =>
  section.pages.map((page) => ({
    fullSlug: `${section.id}/${page.slug}`,
    section,
    page,
  })),
);

const BY_SLUG = new Map(FLAT_PAGES.map((p) => [p.fullSlug, p]));

export function getPage(fullSlug: string): FlatPage | undefined {
  return BY_SLUG.get(fullSlug);
}

export function prevPage(fullSlug: string): FlatPage | undefined {
  const i = FLAT_PAGES.findIndex((p) => p.fullSlug === fullSlug);
  return i > 0 ? FLAT_PAGES[i - 1] : undefined;
}

export function nextPage(fullSlug: string): FlatPage | undefined {
  const i = FLAT_PAGES.findIndex((p) => p.fullSlug === fullSlug);
  return i >= 0 && i < FLAT_PAGES.length - 1 ? FLAT_PAGES[i + 1] : undefined;
}

export function sectionFor(fullSlug: string): Section | undefined {
  return getPage(fullSlug)?.section;
}

/** True when a doc id belongs to one of the sections above. */
export function isSectionPage(fullSlug: string): boolean {
  const dir = fullSlug.split("/")[0];
  return SECTIONS.some((s) => s.id === dir);
}

/** First page of the reading path, used by the "Start reading" links. */
export const FIRST_PAGE = FLAT_PAGES[0]!;
