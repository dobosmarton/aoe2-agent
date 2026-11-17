import * as THREE from "three";

import { el, type Labels } from "./labels";
import { fatSegments, lineMaterial } from "./lines";
import { ACCENT, canvas, INK, MUTED, roundRect, sheet, texture } from "./publish";

// Act: the action the policy picked travels back to the game. It leaves the policy ring
// as a small glass card (the same family as the Perception card that came in) and
// rides a return rail under the pipeline to the glass, where it becomes what the agent
// really sends: key presses, then the mouse.

const CARD_W = 6.4;

function drawCard(mono: string) {
  const W = 960;
  const H = 520;
  const [c, g] = canvas(W, H);
  g.fillStyle = "rgba(5, 10, 24, 0.95)";
  roundRect(g, 4, 4, W - 8, H - 8, 30);
  g.fill();
  g.strokeStyle = "rgba(124, 196, 255, 0.95)";
  g.lineWidth = 4;
  g.stroke();
  g.textBaseline = "alphabetic";
  g.font = `600 34px ${mono}`;
  g.fillStyle = ACCENT;
  g.fillText("ACTION", 52, 86);
  g.font = `400 30px ${mono}`;
  g.fillStyle = MUTED;
  g.textAlign = "right";
  g.fillText("policy → actor", W - 52, 86);
  g.textAlign = "left";
  g.font = `700 92px ${mono}`;
  g.fillStyle = INK;
  g.fillText("build_house", 52, 210);
  g.font = `400 36px ${mono}`;
  g.fillStyle = MUTED;
  g.fillText("house · 25 wood · first clear spot", 52, 272);
  // The input it becomes.
  let x = 52;
  for (const key of ["q", "q", "hover", "click", "h"]) {
    g.font = `600 34px ${mono}`;
    const w = Math.max(70, g.measureText(key).width + 40);
    g.fillStyle = "rgba(124, 196, 255, 0.12)";
    roundRect(g, x, 352, w, 76, 12);
    g.fill();
    g.strokeStyle = "rgba(124, 196, 255, 0.7)";
    g.lineWidth = 2.5;
    g.stroke();
    g.fillStyle = INK;
    g.textAlign = "center";
    g.fillText(key, x + w / 2, 402);
    g.textAlign = "left";
    x += w + 18;
  }
  return { map: texture(c), aspect: W / H };
}

export function createAction(curve: THREE.Curve<THREE.Vector3>, labels: Labels, mono: string, toWorld: (local: THREE.Vector3, out: THREE.Vector3) => void) {
  const group = new THREE.Group();

  // The return rail, in the pipeline's blue; its dashes flow towards the glass.
  const SEGMENTS = 90;
  const samples = curve.getSpacedPoints(SEGMENTS);
  const points: number[] = [];
  for (let i = 0; i < SEGMENTS; i++) points.push(...samples[i]!.toArray(), ...samples[i + 1]!.toArray());
  const railMaterial = lineMaterial({ color: new THREE.Color(0.45, 0.95, 2.2), width: 2, opacity: 0, dashed: true, dashSize: 0.5, gapSize: 0.32 });
  const rail = fatSegments(points, railMaterial);
  group.add(rail);

  const { map, aspect } = drawCard(mono);
  const card = sheet(map, CARD_W, CARD_W / aspect);
  card.mesh.renderOrder = 9;
  group.add(card.mesh);

  // Where the rail meets the glass, the first two key presses land, with what each one
  // does in the game: with a villager selected, q opens the economic build menu and the
  // second q picks the house.
  const landing = samples[SEGMENTS]!.clone();
  const keysElement = el(
    "wt3d-keyseq",
    `<span class="wt3d-keyseq-step"><kbd>q</kbd>build menu</span><span class="wt3d-keyseq-arrow">→</span><span class="wt3d-keyseq-step"><kbd>q</kbd>house</span>`,
  );
  const steps = [...keysElement.querySelectorAll<HTMLElement>(".wt3d-keyseq-step")];
  const keys = labels.add(keysElement, { at: (out) => toWorld(landing, out), align: [0.5, 0.5] });
  const railLabel = labels.add(el("wt3d-rail", "← input"), {
    at: (out) => toWorld(curve.getPointAt(0.42, out), out),
    offset: [0, 12],
    align: [0.5, 0],
  });

  const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

  /** `on` shows the card, `rail` draws the rail in, `travel` carries the card, `keys` lands the presses. */
  function update(s: { on: number; rail: number; travel: number; keys: number }, time: number) {
    group.visible = s.on > 0.002 || s.rail > 0.002;
    rail.geometry.instanceCount = Math.max(1, Math.floor(SEGMENTS * THREE.MathUtils.clamp(s.rail, 0, 1)));
    railMaterial.opacity = Math.min(1, s.rail * 3) * 0.85;
    railMaterial.dashOffset = -time * 1.2;
    railLabel.opacity = THREE.MathUtils.smoothstep(s.rail, 0.5, 1) * (1 - THREE.MathUtils.smoothstep(s.travel, 0.7, 0.95));

    const t = ease(THREE.MathUtils.clamp(s.travel, 0, 1));
    curve.getPointAt(Math.min(0.999, t), card.mesh.position);
    // It leaves the ring at the tile's size, grows a little so it reads in the wide
    // shot, and presses flat onto the glass over the last stretch.
    const arrive = THREE.MathUtils.smoothstep(t, 0.9, 1);
    card.mesh.scale.setScalar((1 + 0.6 * THREE.MathUtils.smoothstep(t, 0, 0.25)) * (1 - 0.45 * arrive));
    card.uniforms.uOpacity.value = s.on * (1 - arrive);
    card.uniforms.uGlow.value = 1 + 0.5 * (1 - THREE.MathUtils.smoothstep(t, 0, 0.15));
    card.mesh.visible = s.on > 0.002 && arrive < 0.999;

    // The chip appears, then each key goes down in turn.
    keys.opacity = Math.min(1, s.keys * 3);
    keys.shift[1] = (1 - Math.min(1, s.keys * 2)) * 8;
    steps.forEach((step, i) => step.classList.toggle("is-down", s.keys > 0.45 + i * 0.35));
  }

  return { group, update };
}
