# Improving AoE2 detector training with synthetic data

Research date: 2026-10-04. Status: research and proposed experiments; no new training results.

## Conclusion

The premise is sound: exact game assets make AoE2 a particularly promising case for synthetic training. A gameplay screenshot is itself a rendering of a constrained set of assets. The main challenge is reproducing their appearance **and the distribution of scenes the detector sees**: scale, animation, terrain, overlapping objects, construction states, UI, and correct labels. This is an application inference, not a measured result for our model.

Pursue two complementary sources: a corrected, calibrated sprite compositor for cheap labeled volume, and procedurally staged scenarios captured through the actual game for rendering fidelity. Keep ordinary gameplay captures as the independent measurement of whether either source improves useful detection. This could substantially reduce manual labeling; current evidence does not establish that real gameplay labels can be eliminated.

## What the local audit actually found

Inspected the current extractor, generator, terrain builder, trainer, `tmp/sprites_v6`, and the existing `training_data_v9_slim` images and labels. These findings separate verified defects from proposed improvements. No model was retrained for this research.

### 1. Some extracted sprites are incomplete

The SLD decoder reads the previous-frame reuse flag but ignores it, creating a transparent output buffer and leaving skipped blocks transparent. SLD skipped blocks can instead refer to pixels from the preceding frame. The current decoder also skips shadow, damage and player-color layers. [Local decoder](../../packages/detection/src/extraction/sld_extractor.py), [SLD format and reuse semantics](https://github.com/SFTtech/openage/blob/master/doc/media/sld-files.md), [reference implementation](https://github.com/SFTtech/openage/blob/master/openage/convert/value_object/read/media/sld.pyx)

A temporary diagnostic decoded frames sequentially and copied the reused blocks at aligned canvas coordinates. Its uncorrected pixels matched the existing PNG assets byte-for-byte. For `sheep_00`, frames 4, 8 and 12 all use the reuse flag; the current PNGs omit respectively **30.7%, 17.0%, and 10.4%** of the opaque pixels in the reconstructed main layer. The visible result is missing legs and pieces of the body. Selected villager frames also lose pixels; knight frame 8 loses 3.3%. This verifies an extraction defect, not a hypothesis that synthetic imagery is inherently unsuitable. The diagnostic does not validate the entire SLD format or replace production decoder tests.

Temporary diagnostic artifacts from this audit: `/private/tmp/aoe2_sld_delta_diagnostic.py`, `/private/tmp/aoe2_sld_delta_diagnostic.json`, and `/private/tmp/aoe2_sld_delta_diagnostic.png`. Production work should turn these cases into permanent decoder fixtures and tests.

### 2. Source selection introduces incorrect classes and misses important states

The current asset selection relies on broad filename patterns and a capped number of matching files. Verified examples in the current PNG library:

- `ram_02_f0.png` is byte-identical to the decoded Karambit Warrior source: `*ram*idle*` also matches `kaRAMbitwarrior`.
- `relic_00_f0.png` and `relic_01_f0.png` are relic carts, not the ordinary ground relic.
- Some selected fishing/unique-ship sources are destruction animations; the exclusion filter is applied only to buildings.
- The Dark Age TC source exists locally, but TC patterns only request age 2–4 sources.
- Sheep have four selected frames from one idle source. Selecting frames 0/4/8/12 does not provide four independent headings: the inspected sheep frames face the same direction. Villager work/carry animations are not requested by the idle-only patterns.
- The dataset has no synthetic farm labels. Farm coverage currently comes from real images.

Use explicit audited source identities, animation semantics and a source manifest instead of relying on substring matches. Retain heading, frame, player-color mask, ground anchor, layer offsets, architecture and multipart-building information. [Extraction patterns and selection](../../packages/detection/src/extraction/extract_sprites.py)

The existing runbook names `tmp/sprites_v6`, and the current v9 synthetic label counts agree across the local copies. However, there is no per-image asset manifest proving which source PNG/version produced every historical training instance. The PNG defects are verified in the current library; do not present their frequency in the trained model's historical corpus as measured. [Runbook](../runbooks/retrain-detection-v6.md)

### 3. The generated objects are mostly much larger than the labeled gameplay objects

Measured every label in the existing validation directory: 600 synthetic images and 32 real screenshots. The table gives median bounding-box dimensions at a **1280-pixel long-edge input scale**, using aspect-preserving resizing. Both groups already have width 1280, so this does not enlarge either group. Padding for letterboxing does not change object pixel dimensions. These are distributions of the current labels, not a claim that every deployment uses this exact zoom.

| Class | Synthetic median width × height | Real median width × height | Synthetic / real support |
| --- | --- | --- | --- |
| Sheep | 39 × 27 px | 14.6 × 12.1 px | 565 / 6 |
| Town center | 390 × 271 px | 165.2 × 118.4 px | 599 / 18 |
| House | 195 × 162 px | 73.4 × 65.2 px | 1,124 / 110 |
| Villager | 18 × 41 px | 11 × 17 px | 3,021 / 218 |
| Knight line | 65 × 59 px | 26.7 × 26.4 px | 848 / 29 |
| Farm | Absent | 120.8 × 78.6 px | 0 / 80 |

This is a large training-distribution mismatch. The small real sheep/knight samples cannot define the final production distribution; calibrate additional native VM frames before freezing ranges. The current trainer applies additional scaling/mosaic, so these pre-augmentation statistics do not establish the exact effective training distribution or quantify their causal contribution to model error. They do establish that the base generated scenes are not already scale-matched. [Dataset](../../packages/detection/src/training_data_v9_slim/dataset.yaml), [labels](../../packages/detection/src/training_data_v9_slim/val/labels), [trainer](../../packages/detection/src/training/train_yolo.py)

### 4. The compositor is a collage, with label-handling defects

Representative existing outputs are [`img_00129.jpg`](../../packages/detection/src/training_data_v9_slim/val/images/img_00129.jpg) and [`img_00317.jpg`](../../packages/detection/src/training_data_v9_slim/val/images/img_00317.jpg). They contain mixed architectures, independently sized units/buildings, sparse resource arrangements, and crude HUD shapes. Random placement alone is not proven harmful; the evidence below is more specific:

- Rendering sorts by category only: resources, then buildings, then units. Every unit is painted over buildings and trees, independent of its ground position. Same-category occlusion is not accounted for in the subsequent label filter.
- Occlusion is estimated by summing intersecting **rectangles**, not visible alpha masks or a union. Transparent pixels count as opaque; overlapping occluders can be counted more than once. Labels can be removed while recognizable object pixels remain in the image.
- Fog rectangles with opacity 120/255 are treated as label-removing occluders, even though the object remains substantially visible. This can create contradictory training targets.
- Each object gets an independent scale; a separate tiny-unit band approximates “distance” even though a coherent game camera should govern scale. Sheep are not included in that tiny-unit band.
- DDS backgrounds are cropped, resized, blended and sometimes heavily blurred. The builder does not reconstruct game terrain transitions, shorelines or elevation. Those are fidelity gaps to test, not proof that every simple background is harmful.
- Optional real-screenshot backgrounds retain existing objects without importing their annotations. This is a code-path risk; the inspected runbook does **not** enable that option, so it is not established as a defect of this historical corpus.

The generator emits box text and images but no per-instance provenance/masks. A corrected version should retain an internal scene/instance manifest and use final visibility to derive labels under one documented box convention. [Generator](../../packages/detection/src/training/generate_training_data.py), [terrain builder](../../packages/detection/src/training/build_terrain_backgrounds.py)

### 5. Training augmentation needs its own controlled check

The trainer applies ±10° in-plane rotation, broad color/scale variation, mosaic and MixUp. In-plane image rotation is not the same as a unit changing its facing in an isometric game. These augmentations may still regularize a detector, so disable or reduce them in a controlled comparison rather than attributing poor results to them without a test. Generator and trainer augmentations must be recorded together. [Training settings](../../packages/detection/src/training/train_yolo.py)

### Recommended priority from these findings

1. **Repair asset decoding and class integrity first.** Add delta-frame regression fixtures, correct masks/layers/anchors, audited identity mapping and visual contact-sheet checks. Include Dark Age TCs, farms and relevant unit states. Do not increase dataset volume before this gate.
2. **Produce a small calibrated compositor dataset.** Match actual input-scale distributions and native capture treatment; use scene-wide zoom, correct depth/visibility, and clean labeled terrain. Start with food economy, base infrastructure, and cavalry scenes. Preserve randomness and hard negatives without forcing every class into every scene.
3. **Use generated game scenarios to calibrate the difficult parts.** Test farms, multipart buildings, forests, construction, player colors and overlap in the actual renderer. This can avoid implementing a complete game renderer while giving the compositor an accurate reference.
4. **Compare at equal training budget, then scale.** Keep architecture and real training frames fixed; compare current data against repaired data and then added scene variation. Assess held-out gameplay precision/recall/F1 and action-critical classes. Treat the current 32 repeatedly inspected screenshots as development data, not a new independent test.

The initial recommendation is therefore to improve synthetic-data correctness and coverage before funding a broad new manual-labeling campaign. A focused set of native captures remains important for calibration and honest evaluation. No numeric model-quality improvement is claimed until retraining and independent evaluation demonstrate it.

## What primary research establishes

| Finding | Evidence and scope | Implication for AoE2 |
| --- | --- | --- |
| Compositing can improve detection without reproducing an entire renderer. | Dwibedi, Misra and Hebert, *Cut, Paste and Learn* (ICCV 2017), train instance detectors using segmented objects placed on backgrounds. Their experiments show useful transfer and complementary benefit when synthetic and real images are combined. Edge artifacts and blending are examined explicitly. These are benchmark findings, not guaranteed AoE2 gains. [Paper](https://arxiv.org/pdf/1708.01642) | Reusing sprite pixels is a credible approach. Audit sprite boundaries and source/background compatibility before increasing image count. |
| Context is worth testing, but there is no universal requirement for elaborate placement models. | Dvornik et al. (ECCV 2018) find contextual placement beneficial and random placement harmful in their VOC experiments. Ghiasi et al. (CVPR 2021) find simple random Copy-Paste improves strong COCO/LVIS baselines without context modeling. The tasks and training setups differ. [Context paper](https://openaccess.thecvf.com/content_ECCV_2018/html/NIKITA_DVORNIK_Modeling_Visual_Context_ECCV_2018_paper.html), [Copy-Paste paper](https://arxiv.org/html/2012.07177v2) | Compare plausible economy/army scenes with unrestricted placements; do not build a complicated learned scene generator before this comparison. |
| Occlusion changes the annotation, not just the pixels. | Ghiasi et al. remove fully hidden instances and update masks and boxes for partial occlusions. They also find elaborate blending unnecessary in their setup. [Method, section 3](https://arxiv.org/html/2012.07177v2#S3) | Preserve instance masks during composition. Label the resulting scene consistently with the real-frame annotation policy, rather than retaining every original sprite rectangle. |
| Variation can matter as much as apparent realism. | Tremblay et al. (CVPR Workshops 2018) compare randomized synthetic data with a more realistic virtual dataset for car detection. Results vary by detector; synthetic pretraining followed by real-data fine-tuning can outperform real-only training. Their ablations show different rendering variations have different value. [Paper](https://arxiv.org/html/1804.06516v1) | Randomization is an experiment to calibrate, not a reason to apply every possible distortion. Matching the game's narrow visual domain may be more valuable than arbitrary textures or camera angles. |

These papers establish that synthetic data can be useful; none establishes an optimum synthetic/real ratio or expected precision/recall/F1 gain for this detector. Human visual plausibility is a useful audit but is not an acceptance metric.

## Recommended image-generation contract

The following are engineering recommendations to test against our gameplay distribution.

1. **Render at calibrated gameplay scale.** Measure representative sheep, villagers, cavalry, trees, and buildings in native VM captures. Use one coherent camera/zoom transform for a scene and preserve relative object sizes. Verify the complete native-capture → resize/letterbox → model-input transformation. A detailed sheep at the wrong effective pixel size does not reproduce the inference problem.
2. **Use complete asset variants.** Include valid headings and animation frames; villager working, carrying, walking and idle states; player colors; relevant architecture sets; building construction and damage states. Dead units, foundations and completed buildings need deliberate class rules. Existing sprite access should not be confused with possession of editable original 3D models.
3. **Build plausible terrain and arrangements.** Sample resource patches, forests with edges, farms around mills/TCs, sheep near workers, mixed cavalry, paths, elevation and terrain transitions. Use object ground anchors and a consistent depth order. Include a modest random-placement component as an ablation, rather than assuming all scene constraints help.
4. **Treat every visible target as labeled.** If a gameplay image supplies the background, existing units, trees and buildings must already be annotated or excluded from usable background regions. Otherwise the generator silently teaches the detector that genuine objects are background. Use audited empty terrain, labeled source frames, or a fully generated scene.
5. **Track masks and occlusion internally.** Retain per-instance alpha masks and IDs after scaling, clipping and composition. Remove fully hidden objects. Define a minimum visible size/fraction for annotation review. Decide whether the real dataset uses visible or inferred full-object boxes and apply that same convention everywhere; changing conventions can change IoU scores without improving recognition. Shadows and selection indicators should not accidentally enlarge the semantic object box.
6. **Match captured game effects.** Cover fog edges, partial offscreen objects, shadows, selection outlines, health bars, construction overlays, and the HUD where they occur in the actual input. Apply measured resize/compression effects after composition. Keep a clean PNG master; avoid excessive blur, arbitrary rotation or unrealistic color changes by default.
7. **Include mistakes the model must reject.** Sample terrain clutter, decorative objects, corpses, selection/UI graphics, and visually similar classes that caused real false positives. Include sparse and empty scenes as well as dense scenes. Oversampling sheep or knights should not make their presence in every frame an artificial certainty.

This requires an offline compositor with masks, even if the final model remains a box detector. Ultralytics' current `copy_paste` augmentation requires polygon annotations for segmentation/OBB tasks; enabling that flag on a box-only detection dataset does not provide the proposed sprite pipeline. Retain masks internally and export ordinary YOLO box labels after composition. [Ultralytics augmentation documentation](https://docs.ultralytics.com/guides/yolo-data-augmentation/#copy-paste-augmentations)

## Actual-game rendering as a second synthetic source

The practical middle option is **synthetic scenario layouts, rendered by AoE2 itself**. The game supplies its actual terrain treatment, unit animation, shadows, depth behavior and UI, so we need not independently reproduce all of them. Scenario distributions still need to resemble gameplay.

AoE2's official notes document generated-map seeds, player architecture/view options, and scenario testing. Capture Test Scenario gameplay with the agent's actual graphics, resolution, zoom and UI configuration. [Generated seeds](https://www.ageofempires.com/news/aoe2de-update-37650/), [editor controls](https://www.ageofempires.com/news/aoe2de-update-42848/)

For automation, `AoE2ScenarioParser` is a community library with source and documentation for editing `.aoe2scenario` files. Its unit API can add units and buildings with player, world position, rotation and animation frame. The repository currently advertises scenario-format support through 1.59, but compatibility with our installed game must be checked using a saved scenario. This allows generating many controlled arrangements without clicking every object into place. It is **not** a headless renderer or a pixel-annotation exporter. [Repository](https://github.com/KSneijders/AoE2ScenarioParser), [unit editing API](https://github.com/KSneijders/AoE2ScenarioParser/blob/master/docs/cheatsheets/units.md)

Known world coordinates and unit IDs can help annotation, but cannot be treated as exact screen boxes. A future label experiment could compare a paused scene with and without a given object at a fixed camera and use the changed pixels as a candidate mask. Animation, shadows, visibility, terrain changes and other units can contaminate that difference. First validate isolated static objects and manually check the masks; no verified automatic labeling capability is claimed here.

## Experiments before scaling generation

Freeze class definitions, label policy, runtime preprocessing, model architecture/initialization, training update budget and evaluation code. Correct objectively invalid labels before treating a generator as a meaningful baseline. Do not compare a much larger new dataset trained for the same number of epochs without also reporting the additional optimizer updates.

| Experiment | Controlled comparison | Question answered |
| --- | --- | --- |
| Baseline and label correction | Current generator versus corrected masks, clipping, class mapping and box policy | How much is annotation error rather than insufficient volume? |
| Scale and asset coverage | Corrected baseline versus measured scale plus additional headings/states/colors | Is the appearance distribution missing deployment cases? |
| Context and occlusion | Same object/image counts; random placement versus scene templates; with/without realistic overlaps | Do scene structure and difficult visibility improve the real task? |
| Source of pixels | Calibrated composition versus game-rendered staged scenes at similar labeling/capture budgets | Is recreating the renderer the remaining bottleneck? |
| Training mixture | Synthetic-only; real-only; equal-budget mixed training; synthetic pretraining then real fine-tuning | How much real calibration data is still useful? |
| Sampling ratio | For mixed training, try real-image sampling probabilities of 25%, 50%, and 75% | Which ratio works without allowing abundant synthetic files to dominate by accident? |

The ratios are proposed trial points, not literature-derived optima. Use a small first pass; repeat the strongest comparisons with multiple training seeds before a major dataset build. Prefer sampling controls to physically duplicating real images. Keep synthetic and real metrics separate.

Evaluate on ordinary gameplay sessions held apart from all training backgrounds, scenarios, near-duplicate frames and generator calibration. Keep the existing repeatedly inspected validation data for development and create a fresh final test set. Group splits by match/session and scenario family/seed. A random split of adjacent screenshots substantially overstates diversity.

Report precision, recall and F1 at the same selected confidence/IoU settings; also report AP across IoUs, per-class support, small/occluded-object slices and false positives per frame. Select thresholds on validation only. Include image examples of changed errors and uncertainty across session groups. Improvements concentrated in common trees should not conceal regressions on sheep, villagers, TCs or knights. A synthetic score alone cannot establish improvement on gameplay.

## Proposed decision gate

First inspect a small paired gallery: generated versus gameplay sheep/workers, farms/TCs, and mixed cavalry at the same effective input scale, with label overlays. Fix recognizable rendering/label defects. Then run the controlled generator comparison with the same architecture and training budget. Scale only the variants that improve held-out gameplay detection and do not regress the action-critical classes. The purpose of scenario captures is both to supply difficult examples and to calibrate the cheaper compositor.
