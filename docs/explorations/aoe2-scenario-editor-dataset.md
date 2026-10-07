# Using the AoE II: DE scenario editor for detector screenshots

## Recommendation

Use the **Scenario Editor as a targeted capture generator**, not as the sole dataset or an automatic source of bounding-box labels. Generate a normal random-map base, add underrepresented objects and later-age scenes, run **Test Scenario**, and capture the game at the same resolution, zoom, UI settings, and graphics settings as the agent. Keep normal-match screenshots as the final evaluation set. This is a proposed workflow, not a measured improvement yet.

AoE II: DE's editor supports generated maps and copying the generated seed; its units/buildings and terrain catalog is updated with game content; it also supports map copy, player architecture sets, per-player view, triggers, and in-editor testing. These are sufficient to construct varied gameplay scenes without playing every match from Dark Age to Imperial. [Generated map seed](https://www.ageofempires.com/news/aoe2de-update-37650/), [units, buildings, terrain and Generate Map](https://www.ageofempires.com/news/age-of-empires-ii-definitive-edition-update-61321/), [architecture, view and Test Scenario](https://www.ageofempires.com/news/aoe2de-update-42848/).

## Options for image collection

| Source | Best use | Main limitation |
| --- | --- | --- |
| **Editor-created, then tested in-game** | Quickly stage missing sheep, knights, farms, construction, ownership colors, architecture, clutter, and occlusions at multiple map seeds and camera positions. | Hand arrangement can create unnatural co-occurrences and spacing. Editor tile positions are not automatically trustworthy pixel bounding boxes. |
| **Recorded normal matches** | Mine authentic later-age compositions, motion, combat, damage, and occlusion without replaying the whole match manually. | Searching and labeling remains manual; replay/spectator has a distinct UI, and old recordings can be build-sensitive. Official notes distinguish the spectator/replay UI and warned that some recordings only play correctly on the same build. [Replay UI](https://www.ageofempires.com/news/aoe2de-update-34055/), [build caveat](https://www.ageofempires.com/news/age-of-empires-ii-definitive-edition-update-73855/). |
| **Agent/normal live matches** | Validate performance in exactly the camera, HUD, UI, and distribution the agent will see. | Slow to reach later ages or rare classes; poor source for deliberately balanced labels. |

The best mix is editor-tested screenshots for **targeted training additions**, normal-match/replay screenshots for natural training diversity, and a held-out set of *live* agent-view screenshots for the published metric. Do not use editor frames, near-duplicate frames, or the same scenario seed on both sides of a train/test split. The split and domain-gap guidance are dataset-design recommendations, not claims made by the game documentation.

## Small capture protocol

1. Build a few reusable scenario templates from different generated map seeds. Record the seed, map, game build, civ/architecture, ownership colors, graphics/UI settings, and scenario ID. The editor exposes generated-map seeds and architecture selection. [Seed](https://www.ageofempires.com/news/aoe2de-update-37650/), [architecture](https://www.ageofempires.com/news/aoe2de-update-42848/).
2. For each weak class, stage plausible rather than isolated scenes: sheep near an opening TC; knights in moving/mixed cavalry groups; farms around TCs or mills; buildings at varied construction states; and examples partially obscured by trees, buildings, or other units. Vary terrain, elevation, orientation, team color, density, and camera position. Triggers and camera/view controls exist if repetitive scene setup becomes worthwhile. [Scenario Editor controls and triggers](https://www.ageofempires.com/news/aoe2de-update-42848/).
3. Capture **Test Scenario gameplay**, not the editor canvas, as the training frame. Test a sample screenshot against the agent's real-game screenshot first: an editor or spectator overlay may change the image distribution. The official notes explicitly distinguish in-editor scenario testing and spectator/replay UI. [Test Scenario](https://www.ageofempires.com/news/age-of-empires-ii-definitive-edition-update-107882/), [spectator/replay UI](https://www.ageofempires.com/news/aoe2de-update-34055/).
4. Label visible pixel boxes in the captured image and review them manually. Preserve difficult negatives and unlabeled-object audits; do not assume known scenario object coordinates map to exact screen boxes after isometric projection, animation, camera movement, or occlusion.
5. Retrain only after checking a batch of captured frames for visual realism and annotation quality. Evaluate on a fresh, untouched live-match holdout; compare per-class precision/recall and error examples, not only aggregate F1.

## Caveats

- This is **real game rendering of staged scenes**, not synthetic compositing, but it still has a scenario-to-live distribution gap. Deliberately staged rows of units are useful for a class sanity check, not a substitute for realistic training or evaluation.
- Capturing the editor viewport is less representative than Test Scenario gameplay; capturing a replay's spectator UI may likewise differ from the agent's player UI. Whether a particular test scenario displays the identical HUD must be verified on the Windows VM.
- The editor is not a pixel-label exporter. Even if scenario object IDs and map positions are known, accurate detection boxes require image-level inspection.
- Stored replays may fail across game builds; the official update specifically warned about same-build replay compatibility. [AoE II: DE Update 73855](https://www.ageofempires.com/news/age-of-empires-ii-definitive-edition-update-73855/).
