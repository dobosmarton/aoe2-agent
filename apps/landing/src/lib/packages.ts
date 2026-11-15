/**
 * Static manifest of the nine Python members of the uv workspace (apps/* and
 * packages/*, listed in the root pyproject.toml). Each entry powers a card in
 * <PackageMap />. Keep in sync with each member's pyproject.toml and the
 * repository layout table in docs/agent/overview.md.
 */

import type { SectionId } from "./taxonomy";

export interface PackageInfo {
  /** Workspace package name. */
  id: string;
  title: string;
  blurb: string;
  /** Relative path inside the repo. */
  path: string;
  /** Doc page that covers this package, as `<section>/<page>`. */
  doc: `${SectionId}/${string}` | null;
  /** Workspace packages this one depends on (used to draw arrows). */
  dependsOn: string[];
  /** Group the package is shown under. */
  tier: "shared" | "real-game" | "detection" | "arena" | "ops" | "reference";
}

export const PACKAGES: PackageInfo[] = [
  {
    id: "core",
    title: "core",
    blurb: "Shared types such as Event, WorldState and DetectedEntity. No I/O.",
    path: "packages/core",
    doc: null,
    dependsOn: [],
    tier: "shared",
  },
  {
    id: "data",
    title: "data",
    blurb:
      "AoE2 reference data: a SQLite database of unit and building stats and JSON exports. No app imports it at runtime.",
    path: "packages/data",
    doc: "agent/game-knowledge",
    dependsOn: [],
    tier: "reference",
  },
  {
    id: "detection",
    title: "detection",
    blurb:
      "YOLO inference, ownership, training, labeling and sprite extraction. 65 classes; the served model is aoe2_yolo_v12_nms736.",
    path: "packages/detection",
    doc: "detection/detector",
    dependsOn: ["core"],
    tier: "detection",
  },
  {
    id: "detection-server",
    title: "detection-server",
    blurb:
      "FastAPI detection server (aoe2-server) with /health, /detect and /detect/sahi. Runs on the macOS host so inference can use CoreML.",
    path: "apps/detection-server",
    doc: "detection/serving-and-evaluation",
    dependsOn: ["core", "detection"],
    tier: "detection",
  },
  {
    id: "gameplay-agent",
    title: "gameplay-agent",
    blurb:
      "The agent (aoe2-agent): perceive, act and deliberate loops, the TypeSafe policy, the strategist and executor LLMs, input and prompts.",
    path: "apps/agent",
    doc: "agent/overview",
    dependsOn: ["core", "detection", "evaluation"],
    tier: "real-game",
  },
  {
    id: "evaluation",
    title: "evaluation",
    blurb:
      "Event broker (in-process or Redis Streams), DuckDB persister and the synthetic world sim.",
    path: "packages/evaluation",
    doc: "arena/event-broker",
    dependsOn: ["core"],
    tier: "arena",
  },
  {
    id: "arena",
    title: "arena",
    blurb:
      "Synthetic arena CLI (aoe2-arena): race, smoke and rank, with Bradley-Terry ranking.",
    path: "apps/arena",
    doc: "arena/overview",
    dependsOn: ["core", "evaluation", "gameplay-agent"],
    tier: "arena",
  },
  {
    id: "arena-web",
    title: "arena-web",
    blurb:
      "FastAPI and SSE backend (aoe2-arena-web) for live and replayed runs and forks. The dashboard in apps/dashboard calls it.",
    path: "apps/api",
    doc: "web/architecture",
    dependsOn: ["core", "evaluation", "gameplay-agent", "arena"],
    tier: "arena",
  },
  {
    id: "autoresearch",
    title: "autoresearch",
    blurb:
      "Plays and scores real games, keeps the experiment ledger and merge gate, and runs prompt tournaments.",
    path: "apps/autoresearch",
    doc: "autoresearch/overview",
    dependsOn: ["core", "gameplay-agent"],
    tier: "ops",
  },
];
