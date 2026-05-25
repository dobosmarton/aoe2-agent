/**
 * Single source of truth for the sequential tutorial path through docs/.
 *
 * Hand-curated rather than derived from frontmatter because most docs in
 * the corpus don't have frontmatter. The ordering matches docs/index.md
 * (Parts 1–8, chapters 01–23), which is the README the rest of the team
 * is already using as a navigation anchor.
 *
 * If `docs/index.md` changes, update this file. Chapter slugs are the
 * filename without the `.md` extension; full slugs are `<part>/<chapter>`.
 */

export type PartId =
  | "part1-architecture"
  | "part2-llm-integration"
  | "part3-entity-detection"
  | "part4-game-knowledge"
  | "part5-operations"
  | "part6-evaluation-arena"
  | "part7-arena-web"
  | "part8-autoresearch";

export interface Chapter {
  /** Slug under the part directory, without `.md`. */
  slug: string;
  /** Display title for the sidebar / TOC. */
  title: string;
  /** One-line description for hover and the landing page. */
  blurb: string;
}

export interface Part {
  id: PartId;
  /** Roman numeral or short label shown in the timeline. */
  label: string;
  /** Full title. */
  title: string;
  /** One-line description for the landing page. */
  blurb: string;
  chapters: Chapter[];
}

export const PARTS: Part[] = [
  {
    id: "part1-architecture",
    label: "I",
    title: "Real-game architecture",
    blurb:
      "Two-tier design, graceful degradation, the capture → detect → think → act loop.",
    chapters: [
      {
        slug: "01-system-overview",
        title: "System Overview",
        blurb: "Two-tier design, graceful degradation, async architecture.",
      },
      {
        slug: "02-game-loop-pipeline",
        title: "Game Loop Pipeline",
        blurb:
          "Capture-detect-alarm-strategist-execute-verify cycle.",
      },
      {
        slug: "03-action-model-and-execution",
        title: "Action Model & Execution",
        blurb: "Pydantic action types, target resolution.",
      },
    ],
  },
  {
    id: "part2-llm-integration",
    label: "II",
    title: "LLM integration",
    blurb: "Strategist + executor provider pattern, prompts, context injection.",
    chapters: [
      {
        slug: "04-provider-pattern",
        title: "Provider Pattern",
        blurb:
          "Abstract base, Claude executor (text-only), strategist (vision).",
      },
      {
        slug: "05-prompt-engineering",
        title: "Prompt Engineering",
        blurb: "Executor + strategist prompt design.",
      },
      {
        slug: "06-context-injection",
        title: "Context Injection",
        blurb: "Memory, goals, resources, dynamic game knowledge.",
      },
    ],
  },
  {
    id: "part3-entity-detection",
    label: "III",
    title: "Entity detection",
    blurb: "YOLO architecture, training pipeline, labeling workflow.",
    chapters: [
      {
        slug: "07-detector-architecture",
        title: "Detector Architecture",
        blurb: "EntityDetector, PyTorch/ONNX/Mock backends, 60-class taxonomy.",
      },
      {
        slug: "08-training-pipeline",
        title: "Training Pipeline",
        blurb: "Synthetic data, augmentations, YOLO11n training.",
      },
      {
        slug: "09-labeling-and-active-learning",
        title: "Labeling & Active Learning",
        blurb: "CVAT workflow, COCO/YOLO conversion, class remapping.",
      },
    ],
  },
  {
    id: "part4-game-knowledge",
    label: "IV",
    title: "Game knowledge",
    blurb: "SQLite knowledge DB, sprite extraction.",
    chapters: [
      {
        slug: "10-knowledge-database",
        title: "Knowledge Database",
        blurb: "SQLite schema, data sources, dynamic queries.",
      },
      {
        slug: "11-sprite-extraction",
        title: "Sprite Extraction",
        blurb: "SLD format, DXT1 decompression, player color recoloring.",
      },
    ],
  },
  {
    id: "part5-operations",
    label: "V",
    title: "Operations",
    blurb: "Cloud training, class schema evolution.",
    chapters: [
      {
        slug: "12-cloud-training",
        title: "Cloud Training",
        blurb: "Lambda Labs workflow, dataset packaging, cost analysis.",
      },
      {
        slug: "13-class-schema-evolution",
        title: "Class Schema Evolution",
        blurb: "Schema history, unified 60-class taxonomy, legacy mapping.",
      },
    ],
  },
  {
    id: "part6-evaluation-arena",
    label: "VI",
    title: "Evaluation arena",
    blurb:
      "race / smoke / rank CLI, broker, DuckDB log, Bradley-Terry ranking, synthetic world.",
    chapters: [
      {
        slug: "14-arena-overview",
        title: "Arena Overview",
        blurb: "race / smoke / rank — when to use which.",
      },
      {
        slug: "15-event-broker",
        title: "Event Broker",
        blurb: "In-process vs Redis, backpressure, /metrics.",
      },
      {
        slug: "16-duckdb-persister-and-replay",
        title: "DuckDB Persister & Replay",
        blurb: "Event log schema, cold-path reader, fork primitive.",
      },
      {
        slug: "17-ranking-pipeline",
        title: "Ranking Pipeline",
        blurb: "Bradley-Terry MLE, scenarios, bootstrap CIs.",
      },
      {
        slug: "18-synthetic-world-sim",
        title: "Synthetic World Sim",
        blurb: "AoE2-lite economy + perception projection.",
      },
    ],
  },
  {
    id: "part7-arena-web",
    label: "VII",
    title: "Arena web",
    blurb: "FastAPI + SSE backend, fork/diff UI, local dev.",
    chapters: [
      {
        slug: "19-web-architecture",
        title: "Web Architecture",
        blurb: "FastAPI lifespan, /events dispatch, reaper, /forks flow.",
      },
      {
        slug: "20-fork-and-diff-ui",
        title: "Fork and Diff UI",
        blurb: "Timeline scrubber, World/Trace/Diff/Operator tabs.",
      },
      {
        slug: "21-running-the-ui-locally",
        title: "Running the UI Locally",
        blurb: "Dev proxy, VITE_API_BASE_URL, deployment modes.",
      },
    ],
  },
  {
    id: "part8-autoresearch",
    label: "VIII",
    title: "Autoresearch",
    blurb: "Mutate → run → score → accept/revert prompt evolution loop.",
    chapters: [
      {
        slug: "22-autoresearch-overview",
        title: "Autoresearch Overview",
        blurb: "Mutate → run → score → accept/revert loop.",
      },
      {
        slug: "23-prompt-mutation-and-memory",
        title: "Prompt Mutation & Memory",
        blurb: "Mutator constraints, protected sections, memory chain.",
      },
    ],
  },
];

export interface FlatChapter {
  fullSlug: string;
  part: Part;
  chapter: Chapter;
  index: number;
}

export const FLAT_CHAPTERS: FlatChapter[] = PARTS.flatMap((part, partIdx) =>
  part.chapters.map((chapter, chapterIdx) => ({
    fullSlug: `${part.id}/${chapter.slug}`,
    part,
    chapter,
    index: partIdx * 100 + chapterIdx,
  })),
);

const BY_SLUG = new Map(FLAT_CHAPTERS.map((c) => [c.fullSlug, c]));

export function getChapter(fullSlug: string): FlatChapter | undefined {
  return BY_SLUG.get(fullSlug);
}

export function prevChapter(fullSlug: string): FlatChapter | undefined {
  const i = FLAT_CHAPTERS.findIndex((c) => c.fullSlug === fullSlug);
  return i > 0 ? FLAT_CHAPTERS[i - 1] : undefined;
}

export function nextChapter(fullSlug: string): FlatChapter | undefined {
  const i = FLAT_CHAPTERS.findIndex((c) => c.fullSlug === fullSlug);
  return i >= 0 && i < FLAT_CHAPTERS.length - 1
    ? FLAT_CHAPTERS[i + 1]
    : undefined;
}

export function partFor(fullSlug: string): Part | undefined {
  return getChapter(fullSlug)?.part;
}

/** First chapter of the tutorial — used as the default "Start tutorial" CTA. */
export const FIRST_CHAPTER = FLAT_CHAPTERS[0]!;
