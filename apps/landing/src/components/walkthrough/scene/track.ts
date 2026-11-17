import * as THREE from "three";
import { LineSegmentsGeometry } from "three/examples/jsm/lines/LineSegmentsGeometry.js";

import { createBillboardBoxes } from "./boxes";
import { el, type Labels } from "./labels";
import { fatSegments, lineMaterial } from "./lines";

// Tracking, shown in the village: each tracked villager gets a box facing the camera,
// as the tracker draws it on the screen, with two fainter boxes where it stood on
// earlier frames, a trail of its recent positions and its track ID. The trail and the
// earlier boxes are what tracking adds over detection: the same ID across frames. IDs
// and trails are illustrative; no recorded run stores track histories.

const TRACK_COLOR = new THREE.Color(0.5, 0.95, 2.1);
const SAMPLES = 16;
const SAMPLE_EVERY = 0.2; // seconds between remembered positions, like frames arriving
const BOX_W = 0.62;
const BOX_H = 1.0;

export type Tracked = { anchor: THREE.Vector3; walking: boolean };

export function createTracking(tracked: Tracked[], ids: number[], labels: Labels) {
  const group = new THREE.Group();
  group.name = "tracking";
  const boxes = createBillboardBoxes(tracked.length * 3, { line: 1.25, bracket: 2.6, fill: 0.05 });
  group.add(boxes.mesh);
  const trailMaterial = lineMaterial({ color: TRACK_COLOR, width: 2.25, opacity: 0 });

  const items = tracked.map((t, i) => {
    const history = Array.from({ length: SAMPLES }, () => t.anchor.clone());
    const geometry = new LineSegmentsGeometry();
    const trail = fatSegments([0, 0, 0, 0, 0, 0], trailMaterial);
    trail.geometry.dispose();
    trail.geometry = geometry;
    group.add(trail);
    const label = labels.add(el("wt3d-tag", `track_${ids[i % ids.length]}`), {
      at: (out) => out.copy(t.anchor).add(new THREE.Vector3(0, BOX_H + 0.05, 0)),
      offset: [0, -4],
      align: [0.5, 1],
    });
    return { t, history, trail, geometry, label, last: -1 };
  });

  const centre = new THREE.Vector3();
  const faint = TRACK_COLOR.clone();

  function writeTrail(item: (typeof items)[number]) {
    const points: number[] = [];
    for (let k = 0; k + 1 < item.history.length; k++) {
      const a = item.history[k]!;
      const b = item.history[k + 1]!;
      if (a.distanceToSquared(b) < 1e-6) continue;
      points.push(a.x, 0.03, a.z, b.x, 0.03, b.z);
    }
    if (points.length === 0) points.push(0, -10, 0, 0, -10, 0);
    item.geometry.setPositions(points);
  }

  function update(on: number, time: number) {
    group.visible = on > 0.005;
    trailMaterial.opacity = on * 0.85;
    boxes.begin();
    for (const item of items) {
      // Remember a position every SAMPLE_EVERY seconds, as frames would arrive.
      if (time - item.last > SAMPLE_EVERY) {
        item.last = time;
        item.history.pop();
        item.history.unshift(item.t.anchor.clone());
        if (group.visible) writeTrail(item);
      }
      item.label.opacity = on;
      if (!group.visible) continue;
      // Earlier frames first, so the current box draws on top.
      for (const [k, alpha] of [[8, 0.22], [4, 0.4]] as const) {
        const p = item.history[k]!;
        if (!item.t.walking || p.distanceToSquared(item.t.anchor) < 0.01) continue;
        centre.set(p.x, BOX_H / 2, p.z);
        boxes.set(centre, BOX_W, BOX_H, faint, on * alpha);
      }
      centre.copy(item.t.anchor).setY(BOX_H / 2);
      boxes.set(centre, BOX_W, BOX_H, TRACK_COLOR, on);
    }
    boxes.end();
  }

  return { group, update };
}
