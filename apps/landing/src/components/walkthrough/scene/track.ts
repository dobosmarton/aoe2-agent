import * as THREE from "three";
import { LineSegmentsGeometry } from "three/examples/jsm/lines/LineSegmentsGeometry.js";

import { createBillboardBoxes } from "./boxes";
import { el, type Label, type Labels } from "./labels";
import { fatSegments, lineMaterial } from "./lines";

// Tracking, shown in the village: each tracked villager gets a box facing the camera,
// as the tracker draws it on the screen, with two fainter boxes where it stood on
// earlier frames, a trail of its recent positions and its track ID. The trail and the
// earlier boxes are what tracking adds over detection: the same ID across frames. IDs
// and trails are illustrative; no recorded run stores track histories.

const TRACK_COLOR = new THREE.Color(0.5, 0.95, 2.1);
/** Remembered positions per villager, newest first. */
const SAMPLES = 16;
/** Seconds between remembered positions, like frames arriving. */
const SAMPLE_EVERY = 0.2;
const BOX_W = 0.62;
const BOX_H = 1.0;
/** Earlier boxes as [samples ago, opacity], oldest first so the current box draws on top. */
const EARLIER_BOXES = [
  [8, 0.22],
  [4, 0.4],
] as const;
/** Small screens show the first few IDs only, so the crowd by the Town Center stays legible. */
const PHONE_TAGS = 5;
/** Trail points when there is nothing to draw: one segment under the ground. */
const NO_TRAIL = [0, -10, 0, 0, -10, 0];

export type Tracked = { anchor: THREE.Vector3; walking: boolean };

export type Tracking = {
  group: THREE.Group;
  /** `on` fades the boxes, trails and IDs in; `time` in seconds paces the remembered positions. */
  update: (on: number, time: number) => void;
};

type TrackItem = {
  tracked: Tracked;
  history: THREE.Vector3[];
  geometry: LineSegmentsGeometry;
  label: Label;
  /** Time the last position was remembered. */
  last: number;
};

const writeTrail = (item: TrackItem): void => {
  const points: number[] = [];
  for (let k = 0; k + 1 < item.history.length; k++) {
    const a = item.history[k];
    const b = item.history[k + 1];
    if (a.distanceToSquared(b) < 1e-6) continue;
    points.push(a.x, 0.03, a.z, b.x, 0.03, b.z);
  }
  item.geometry.setPositions(points.length === 0 ? NO_TRAIL : points);
};

export const createTracking = (tracked: readonly Tracked[], ids: readonly number[], labels: Labels): Tracking => {
  const group = new THREE.Group();
  group.name = "tracking";
  const boxes = createBillboardBoxes(tracked.length * 3, { line: 1.25, bracket: 2.6, fill: 0.05 });
  group.add(boxes.mesh);
  const trailMaterial = lineMaterial({ color: TRACK_COLOR, width: 2.25, opacity: 0 });

  const items = tracked.map((villager, i): TrackItem => {
    const history = Array.from({ length: SAMPLES }, () => villager.anchor.clone());
    const geometry = new LineSegmentsGeometry();
    const trail = fatSegments([0, 0, 0, 0, 0, 0], trailMaterial);
    trail.geometry.dispose();
    trail.geometry = geometry;
    group.add(trail);
    const label = labels.add(el(`wt3d-tag${i >= PHONE_TAGS ? " is-minor" : ""}`, `track_${ids[i % ids.length]}`), {
      at: (out) => out.copy(villager.anchor).setY(villager.anchor.y + BOX_H + 0.05),
      offset: [0, -4],
      align: [0.5, 1],
    });
    return { tracked: villager, history, geometry, label, last: -1 };
  });

  const centre = new THREE.Vector3();
  const faint = TRACK_COLOR.clone();

  const update = (on: number, time: number): void => {
    group.visible = on > 0.005;
    trailMaterial.opacity = on * 0.85;
    boxes.begin();
    for (const item of items) {
      const { anchor, walking } = item.tracked;
      if (time - item.last > SAMPLE_EVERY) {
        item.last = time;
        item.history.pop();
        item.history.unshift(anchor.clone());
        if (group.visible) writeTrail(item);
      }
      item.label.opacity = on;
      if (!group.visible) continue;
      for (const [samplesAgo, alpha] of EARLIER_BOXES) {
        const p = item.history[samplesAgo];
        if (!walking || p.distanceToSquared(anchor) < 0.01) continue;
        centre.set(p.x, BOX_H / 2, p.z);
        boxes.set(centre, BOX_W, BOX_H, faint, on * alpha);
      }
      centre.copy(anchor).setY(BOX_H / 2);
      boxes.set(centre, BOX_W, BOX_H, TRACK_COLOR, on);
    }
    boxes.end();
  };

  return { group, update };
};
