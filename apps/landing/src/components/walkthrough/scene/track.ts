import * as THREE from "three";
import { CSS3DSprite } from "three/examples/jsm/renderers/CSS3DRenderer.js";

// Tracking, shown in the village: each tracked villager gets a box that follows it,
// its track ID, and a trail of where the tracker saw it on recent frames. The trail
// is what tracking adds over detection: the same ID across frames. IDs and trails are
// illustrative; no recorded run stores track histories.

const TRACK_COLOR = new THREE.Color(0.55, 1.05, 2.2);
const SAMPLES = 14;
const SAMPLE_EVERY = 0.22; // seconds between remembered positions, like frames arriving

export type Tracked = { anchor: THREE.Vector3; walking: boolean };

// The 12 edges of a box from (-0.5, 0, -0.5) to (0.5, 1, 0.5).
const BOX_EDGES = (() => {
  const c = [
    [-0.5, 0, -0.5], [0.5, 0, -0.5], [0.5, 0, 0.5], [-0.5, 0, 0.5],
    [-0.5, 1, -0.5], [0.5, 1, -0.5], [0.5, 1, 0.5], [-0.5, 1, 0.5],
  ];
  const pairs = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
  return pairs.flatMap(([a, b]) => [...c[a!]!, ...c[b!]!]);
})();

export function createTracking(tracked: Tracked[], ids: number[]) {
  const group = new THREE.Group();
  group.name = "tracking";
  const boxMaterial = new THREE.LineBasicMaterial({ color: TRACK_COLOR, transparent: true, opacity: 0, toneMapped: false });
  const boxGeometry = new THREE.BufferGeometry();
  boxGeometry.setAttribute("position", new THREE.Float32BufferAttribute(BOX_EDGES, 3));
  const dotMaterial = new THREE.PointsMaterial({ color: TRACK_COLOR, size: 6, sizeAttenuation: false, transparent: true, opacity: 0, toneMapped: false });

  const items = tracked.map((t, i) => {
    const box = new THREE.LineSegments(boxGeometry, boxMaterial);
    box.scale.set(0.5, 0.9, 0.5);

    // Trail: a line through the remembered positions, fading towards the oldest.
    const trailPositions = new Float32Array(SAMPLES * 3);
    const trailColors = new Float32Array(SAMPLES * 3);
    const trailGeometry = new THREE.BufferGeometry();
    trailGeometry.setAttribute("position", new THREE.BufferAttribute(trailPositions, 3));
    trailGeometry.setAttribute("color", new THREE.BufferAttribute(trailColors, 3));
    const trail = new THREE.Line(
      trailGeometry,
      new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0, toneMapped: false }),
    );
    const dots = new THREE.Points(trailGeometry, dotMaterial);
    trail.frustumCulled = dots.frustumCulled = false;

    const label = document.createElement("div");
    label.className = "wt3d-label is-track";
    label.innerHTML = `<span class="wt3d-label-track">track_${ids[i % ids.length]}</span>`;
    const tag = new CSS3DSprite(label);
    tag.scale.setScalar(0.017);

    group.add(box, trail, dots, tag);
    const history = Array.from({ length: SAMPLES }, () => t.anchor.clone());
    return { t, box, trail, trailGeometry, trailPositions, trailColors, tag, label, history, last: 0 };
  });

  function update(on: number, time: number) {
    group.visible = on > 0.005;
    if (!group.visible) return;
    boxMaterial.opacity = on;
    dotMaterial.opacity = on * 0.9;
    for (const item of items) {
      // Remember a position every SAMPLE_EVERY seconds, as frames would arrive.
      if (time - item.last > SAMPLE_EVERY) {
        item.last = time;
        item.history.pop();
        item.history.unshift(item.t.anchor.clone());
      }
      item.box.position.copy(item.t.anchor);
      item.history.forEach((p, k) => {
        item.trailPositions.set([p.x, 0.04, p.z], k * 3);
        const fade = 1 - k / SAMPLES;
        item.trailColors.set([TRACK_COLOR.r * fade, TRACK_COLOR.g * fade, TRACK_COLOR.b * fade], k * 3);
      });
      item.trailGeometry.attributes.position!.needsUpdate = true;
      item.trailGeometry.attributes.color!.needsUpdate = true;
      (item.trail.material as THREE.LineBasicMaterial).opacity = on * (item.t.walking ? 1 : 0.4);
      item.tag.position.set(item.t.anchor.x, 1.25, item.t.anchor.z);
      item.label.style.opacity = String(on);
      item.label.style.visibility = on > 0.01 ? "visible" : "hidden";
    }
  }

  return { group, update };
}
