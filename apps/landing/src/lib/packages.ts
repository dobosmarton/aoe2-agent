/**
 * Static manifest of the 9-package uv workspace under packages/. Each entry
 * powers a card in <PackageMap />. Keep in sync with docs/index.md and the
 * actual contents of packages/*.
 */

import type { PartId } from "./taxonomy";

export interface PackageInfo {
  id: string;
  title: string;
  blurb: string;
  /** Relative path inside the repo. */
  path: string;
  /** Which Part of the tutorial this package is documented under. */
  part: PartId | null;
  /** IDs of packages this one depends on (used to draw arrows). */
  dependsOn: string[];
  /** Tier the package belongs to. */
  tier: "shared" | "real-game" | "detection" | "arena" | "ops";
}

export const PACKAGES: PackageInfo[] = [
  {
    id: "core",
    title: "core",
    blurb: "Shared event / payload / world-state types. The protocol everyone agrees on.",
    path: "packages/core",
    part: null,
    dependsOn: [],
    tier: "shared",
  },
  {
    id: "data",
    title: "data",
    blurb: "SQLite game-knowledge database: tech trees, building stats, unit properties.",
    path: "packages/data",
    part: "part4-game-knowledge",
    dependsOn: [],
    tier: "shared",
  },
  {
    id: "detection",
    title: "detection",
    blurb:
      "YOLO inference, model training, labeling UI, ownership classification (60 classes, 92.2% mAP50).",
    path: "packages/detection",
    part: "part3-entity-detection",
    dependsOn: ["core"],
    tier: "detection",
  },
  {
    id: "detection-server",
    title: "detection-server",
    blurb:
      "macOS CoreML/ONNX inference endpoint (~15ms Neural Engine vs 1.2s CPU).",
    path: "packages/detection-server",
    part: "part3-entity-detection",
    dependsOn: ["detection"],
    tier: "detection",
  },
  {
    id: "gameplay-agent",
    title: "gameplay-agent",
    blurb:
      "Real-game loop, goal manager, alarm system, Sonnet strategist + Haiku executor.",
    path: "packages/gameplay-agent",
    part: "part1-architecture",
    dependsOn: ["core", "data", "detection"],
    tier: "real-game",
  },
  {
    id: "evaluation",
    title: "evaluation",
    blurb:
      "Event broker (in-process / Redis), DuckDB persister, deterministic world_sim projection.",
    path: "packages/evaluation",
    part: "part6-evaluation-arena",
    dependsOn: ["core"],
    tier: "arena",
  },
  {
    id: "arena",
    title: "arena",
    blurb:
      "Synthetic CLI (race/smoke/rank), Bradley-Terry ranking, multi-run orchestration.",
    path: "packages/arena",
    part: "part6-evaluation-arena",
    dependsOn: ["core", "evaluation", "gameplay-agent"],
    tier: "arena",
  },
  {
    id: "arena-web",
    title: "arena-web",
    blurb:
      "FastAPI + SSE backend for live tailing, DuckDB queries, fork replay. Powers the internal UI.",
    path: "packages/arena-web",
    part: "part7-arena-web",
    dependsOn: ["core", "evaluation", "arena"],
    tier: "arena",
  },
  {
    id: "autoresearch",
    title: "autoresearch",
    blurb:
      "Automated prompt-optimization loop: mutator, game_runner, memory-chain evolution.",
    path: "packages/autoresearch",
    part: "part8-autoresearch",
    dependsOn: ["gameplay-agent"],
    tier: "ops",
  },
];
