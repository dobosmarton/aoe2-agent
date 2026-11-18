// Copy for the home page walkthrough: one real turn from a recorded game, in eight
// chapters. Facts follow docs/agent/*.md; numbers about the frame come from
// turn-data.json (scripts/build_walkthrough_data.py). Bodies are trusted HTML written
// here, rendered with set:html so inline <code> works.

import turn from "./turn-data.json";

export type Chapter = {
  /** Anchor id, also used by the rail. */
  id: string;
  /** Short name for the rail. */
  name: string;
  title: string;
  body: string;
  /** Docs page that covers the chapter, as `<section>/<page>` plus an optional hash. */
  doc: { slug: string; hash?: string; label: string };
  /** Scroll length of the chapter, in viewport heights. */
  length: number;
};

export type Role = { role: string; model: string; when: string; output: string };

export type CatalogEntry = {
  id: string;
  /** Whether this frame allows the action. */
  ok: boolean;
  /** The reading that allows or rules it out. */
  why: string;
};

const classCounts: Record<string, number> = turn.classes;
const count = (cls: string): number => classCounts[cls] ?? 0;
const hud = turn.hud.readings;

export const TURN_FACTS = {
  boxes: turn.entities.length,
  villagers: count("villager"),
  trees: count("tree"),
  farms: count("farm"),
  houses: count("house"),
  hud,
};

/** The input the picked action becomes, in the order the actor sends it. */
export const ACTION_INPUT: readonly string[] = ["q", "q", "hover", "click", "h"];

export const CHAPTERS: Chapter[] = [
  {
    id: "capture",
    name: "Capture",
    title: "Grab the window",
    body: `The perceive loop captures the game window every 0.1 s: an <code>mss</code> region grab in the Windows VM, a ScreenCaptureKit stream on macOS. Each capture gets an increasing <code>capture_id</code>, and detection, OCR and camera analysis read the same bytes.`,
    doc: { slug: "agent/perception", label: "Perception" },
    length: 1.4,
  },
  {
    id: "detect",
    name: "Detect",
    title: "Find every entity",
    body: `A YOLO26n model trained on 65 classes boxes the units, buildings, animals and resources. On this frame it returns ${TURN_FACTS.boxes} boxes: ${TURN_FACTS.villagers} villagers, ${TURN_FACTS.trees} trees, ${TURN_FACTS.farms} farms, ${TURN_FACTS.houses} houses and the rest.`,
    doc: { slug: "detection/detector", label: "Entity detector" },
    length: 1.75,
  },
  {
    id: "track",
    name: "Track",
    title: "Keep identities across frames",
    body: `The screentrack tracker matches each box to a track from earlier frames, so <code>track_41</code> names the same villager two frames later. It smooths the class label by vote and drops every track when the camera jumps. Military units and gates also get an owner from the player colour.`,
    doc: { slug: "agent/tracking", label: "Tracking" },
    length: 1.4,
  },
  {
    id: "hud",
    name: "Read the HUD",
    title: "Read the resource bar",
    body: `Template matching reads the resource bar on the machine itself; no image goes to a model. This frame: ${hud.wood} wood, ${hud.food} food, ${hud.gold} gold, ${hud.stone} stone, population ${hud.population}, ${hud.age}.`,
    doc: { slug: "agent/perception", hash: "hud-reading", label: "HUD reading" },
    length: 1.4,
  },
  {
    id: "publish",
    name: "Publish",
    title: "Publish one frozen frame",
    body: `Boxes, track IDs and HUD readings become one immutable <code>Perception</code>, stamped with capture IDs and the ledger's revisions. The <code>FramePipe</code> hands it to the act and deliberate loops. From here on, every role reads text.`,
    doc: { slug: "agent/perception", hash: "the-published-frame", label: "The published frame" },
    length: 1.4,
  },
  {
    id: "decide",
    name: "Decide",
    title: "Pick one action",
    body: `The act loop filters the action catalog against the frame. Population is at its cap and no villager is idle, so <code>queue_villager</code> and the <code>assign_*</code> actions drop out. The TypeSafe policy picks <code>build_house</code>. The strategist runs on its own 25 s clock and only sets goals; the executor LLM wakes for alarms, handoffs and recovery.`,
    doc: { slug: "agent/decision-loops", label: "Decision loops" },
    length: 2.2,
  },
  {
    id: "act",
    name: "Act",
    title: "Place the house",
    body: `Under the input lock the actor checks the choice against the newest frame, selects a builder and presses <kbd>q</kbd> <kbd>q</kbd>. With the house on the cursor it hovers candidate spots and reads the game's own placement preview, which turns red where the ground is taken. It clicks the first clear spot and presses <kbd>h</kbd>.`,
    doc: { slug: "agent/actions-and-execution", hash: "building-placement", label: "Building placement" },
    length: 2.5,
  },
  {
    id: "verify",
    name: "Verify",
    title: "Check the result",
    body: `The action ledger records the order as pending. When 25 wood leaves the HUD it is purchased; when the population cap rises by 5 it is confirmed. A new frame is already on its way, and the loop starts again.`,
    doc: { slug: "agent/actions-and-execution", hash: "action-ledger-and-reconciliation", label: "Action ledger" },
    length: 1.8,
  },
];

/** Roles shown in the Decide chapter (docs/agent/overview.md, "Roles"). */
export const ROLES: Role[] = [
  { role: "TypeSafe policy", model: "jev-1.13.0", when: "Each frame, at most every 0.5 s", output: "One catalog action" },
  { role: "Strategist", model: "gpt-5.6-terra", when: "Every 25 s, at once on an age change or alarm", output: "Goals and allocation" },
  { role: "Executor", model: "gpt-5.6-luna", when: "Alarm, tactical handoff or recovery", output: "Guarded tool calls" },
];

/** Catalog entries shown in the Decide chapter. Feasibility follows this frame's HUD. */
export const CATALOG: CatalogEntry[] = [
  { id: "build_house", ok: true, why: "wood 25" },
  { id: "queue_villager", ok: false, why: `population ${hud.population}` },
  { id: "assign_wood", ok: false, why: "no idle villager" },
  { id: "build_farm", ok: true, why: "wood 60" },
  { id: "build_barracks", ok: true, why: "wood 175" },
  { id: "advance_to_castle", ok: false, why: "needs two Feudal buildings" },
  { id: "tactical_handoff", ok: false, why: "no army in view" },
  { id: "wait", ok: true, why: "always" },
];
