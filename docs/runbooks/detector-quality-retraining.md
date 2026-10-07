# From the current AoE2 detector to a qualified replacement

This guide is the work sequence for improving detection with game assets, synthetic scenes, and scale robust training. It describes **work to do**, not a new measured model result. The current deployed checkpoint remains the reference until a candidate passes the final gates.

The target is reliable detection of objects the agent must act on across normal gameplay views. Measure precision, recall, F1, missed action targets, false positives per frame, and latency. Overall F1 alone is insufficient: the agent needs sheep, villagers, food sources, farms, town centers, houses, and later cavalry even when common trees dominate the image count.

The [synthetic data audit](../explorations/aoe2-synthetic-data-quality.md) establishes broken SLD delta frames, wrong source matches, missing states and farms, inconsistent occlusion labels, and large object-size differences. The [scale study](../explorations/aoe2-detector-scale-robustness.md) shows that a 10% diagnostic shrink changed current real-validation F1 only from 0.688 to 0.682; severe shrinking hurts much more. The objective is a supported **range** of object sizes, sampled automatically, rather than a single exact game zoom.

## 1. Define the result and freeze the current detector

Record the current model bytes, 60-class schema, thresholds, preprocessing, NMS, and evaluator as the reference. The published development result is precision **0.753**, recall **0.633**, F1 **0.688** on 32 real frames at IoU 0.50. Those frames share capture sessions with training, so this is a development number, not the release benchmark. [Current evaluation](../../packages/detection/registry/aoe2-entity-detector/EVALUATION.md)

Write down the primary comparison before training: use the same detector path and IoU rule for the reference and every candidate; report both precision and recall with F1. Report each action-critical class, its ground-truth count, false positives per frame, and p50/p95 inference and requested-refresh time. State the zoom and capture-size range actually tested. Select confidence thresholds on development data only.

The existing development result can be reproduced from the repository root with the local weights present:

```bash
uv run --no-sync python -m detection_server.evaluate \
  --model packages/detection/src/inference/models/aoe2_yolo_v9.onnx \
  --data packages/detection/src/training_data_v9_slim \
  --output tmp/evaluations/current-development.json
```

**Exit check:** a frozen, reproducible baseline report and metric definition. No new model is called better based solely on synthetic validation or the existing 32 frames.

The [frozen baseline record](detector-quality-baseline.md) completes this step:
it includes exact artifact/source hashes, the reproduced development score,
action-critical class counts, a local CPU latency diagnostic, and predeclared
comparison rules. Independent session results and Windows refresh latency
remain pending later steps.

## 2. Capture separate gameplay development and final-test sessions

Capture native agent-view screenshots from several distinct Windows game sessions: Dark Age openings with sheep/berries/TCs, established economies with farms and buildings, and later mixed armies. Cover normal and extreme zoom settings the agent may encounter, more than one map terrain, ownership colors, empty areas, and visually confusing objects. Save capture resolution, zoom, UI/game settings, match/session ID, time, and frame source. Avoid adjacent near-duplicate frames.

Split **by match/session**, before annotation, into development and a final test set. Keep the final-test sessions out of training backgrounds, sprite calibration, threshold selection, scene-template design, and scenario families. The existing January split stays in development. Start with a manageable pilot of independent sessions and add frames until every action-critical class being compared has enough labeled examples for an interpretable estimate; six sheep boxes are not enough. Retain normal gameplay as the final test source even if staged scenarios help training.

Extend the current real-frame evaluator to consume the session manifest and each frozen split explicitly; its existing `real_*` validation-directory discovery alone does not enforce a session holdout. Run the unchanged reference detector on the new test only at the final comparison, using the same frozen scoring code as the candidate.

Prelabel frames using the current detector, then manually correct **all** visible objects under one box/occlusion convention. Check a sample of completed frames for missed labels and wrong classes. Incomplete ground truth can make both training and evaluation misleading. CVAT is already part of the project's labeling workflow; staged Test Scenario captures can fill specific training gaps but their world coordinates are not pixel boxes. [Scenario capture options](../explorations/aoe2-scenario-editor-dataset.md)

**Exit check:** versioned image/label manifests; session-disjoint development and sealed final test; class counts, zoom coverage, and annotation QA report. This step needs Windows capture access and human review of labels; tooling and prelabels can be automated.

The [session capture and labeling runbook](detector-session-capture.md) now
provides lossless game-window capture, session/split manifests, current-model
development prelabels, strict corrected-label import, audit and gallery tools,
and a session-aware evaluator. The capture and human-review exit gate remains
pending until new Windows matches are recorded.

## 3. Probe inference with the current model before retraining

On development frames, compare the live single full-frame pass with a small number of overlapping crops taken from the **original** captured pixels. Feed each crop through the same 1280×1280 letterbox contract and map detections back to screen coordinates before deduplication. Test small-object recall, large-building precision, duplicate rate, coordinate error, and end-to-end latency. Include objects outside the chosen crop region so a narrow crop selector cannot appear to solve the whole screen.

The shipped ONNX declares one static `[1, 3, 1280, 1280]` input. The older 640-pixel tiled/batched path is incompatible with that artifact; it must be revised rather than enabled as-is. A fixed model tensor does not constrain the Windows game resolution. [Runtime path](../../apps/agent/src/detection_phase.py), [server preprocessing](../../apps/detection-server/src/app.py), [SAHI study](https://arxiv.org/abs/2202.06934)

**Exit check:** measured full-frame versus full-frame-plus-crop tradeoff and a choice of inference protocol for subsequent training. No production switch is needed until accuracy and latency have been compared.

## 4. Repair the sprite library and labels

Fix SLD previous-frame block reuse and add decoder fixtures for sheep, villagers, and cavalry. Preserve layer offsets and ground hotspots. Extract shadows and player-color information with an explicit composition policy. Replace broad filename substring matches with an audited source manifest: source ID/path, class, age/architecture, animation action, heading, frame, and asset hash. Fail ambiguous or missing mappings rather than silently using another class.

Cover Dark Age town centers; sheep headings; villagers idle, walking, gathering, and carrying; cavalry headings/actions; relevant construction and damage states. Farms need their terrain-based graphics represented or sufficient game-rendered training captures. Define how foundations, destroyed buildings, corpses, and unsupported states are labeled before generating them. Verify a contact sheet per class and compare reconstructed sprites with game captures.

**Exit check:** repeatable extraction; tests for delta-frame reuse and source identity; no known Karambit-as-ram/relic-cart-as-relic cases; complete coverage table. Rebuilding images from the current sprite directory before this gate would reproduce the defects. [Extractor](../../packages/detection/src/extraction/sld_extractor.py), [source selection](../../packages/detection/src/extraction/extract_sprites.py), [SLD format](https://github.com/SFTtech/openage/blob/master/doc/media/sld-files.md)

## 5. Replace the collage generator with a traceable scene renderer

Build a small set of plausible scene templates first: opening TC plus sheep/berries/workers; forest and lumber camp; farms around drop-off buildings; housing/construction; mixed cavalry and army. Retain some random layouts and hard negatives for diversity. Apply one scene camera/zoom transform so relative sizes remain coherent. Place sprites by ground anchor and draw them in depth order. Keep source alpha masks through overlap, clipping, fog, and HUD occlusion; derive final boxes from the documented annotation convention. A partly visible target must not become an unlabeled positive-looking object.

Use clean game terrain or fully annotated source frames. Never paste onto a gameplay background whose existing units/buildings have no labels. Record, per image, the random seed, source sprites and hashes, background, scene template, zoom, object states, final masks/boxes, and augmentation settings. This makes bad examples traceable. Use game-rendered Test Scenario frames for complex farms, building parts, shadows, and overlap when the compositor cannot yet reproduce them; review their pixel boxes.

**Exit check:** a 200–500-image pilot, overlaid label gallery, machine checks for invalid/unlabeled objects and class coverage, and a manifest from which every image can be regenerated. Do not scale to thousands of frames until the pilot passes. [Current generator](../../packages/detection/src/training/generate_training_data.py)

## 6. Make scale coverage automatic

From **development** gameplay captures, measure box widths/heights after the exact full-frame and crop preprocessing. For each critical class, note typical and difficult small/large cases. Sample scene-wide zoom continuously across the supported gameplay interval, with a modest margin and explicit small-object coverage. Check the **post-augmentation** distribution against development captures; native sprite dimensions alone are the wrong measure. Preserve real relative building/unit sizes within a scene.

The current trainer already sets affine `scale=0.5`; it does not explicitly enable Ultralytics `multi_scale` training. Pilot the corrected generator first, then test a modest nonzero `multi_scale` setting as a separate experiment. Inspect augmented batches: scene zoom, affine scaling, mosaic, and training-size variation compound. More augmentation may erase tiny details or create unnatural scenes. Multi-scale training can still be exported to a fixed 1280 deployment tensor. [Pinned trainer semantics](https://raw.githubusercontent.com/ultralytics/ultralytics/v8.4.54/ultralytics/models/yolo/detect/train.py)

**Exit check:** histograms by class for model-input object size, including small/large tails; verified label overlays at several actual game zooms. No per-run manual adjustment of game resolution or zoom is required.

## 7. Train controlled candidates

Keep the 60-class mapping, model family, initialization, unique real-training examples, preprocessing, optimizer-update budget, and development evaluation fixed for the first comparisons. Record seeds, package/model versions, dataset hashes, augmentation arguments, and checkpoint/export hashes. When dataset size changes, matching epochs alone does not match the number of optimizer updates.

Run a small ablation ladder:

1. Reproduce a reference training run using the old dataset and the chosen fixed budget.
2. Change only corrected asset decoding, source mapping, and label handling.
3. Add coherent scenes and sampled scale coverage at a comparable object/image budget.
4. Compare synthetic-only, mixed batches, and synthetic pretraining followed by real fine-tuning. Keep the unique real images the same; control sampling instead of assuming 10 disk duplicates are optimal.
5. Separately test modest `multi_scale` and any stronger augmentation. Train the most promising settings with multiple seeds.

Use the existing YOLO26n architecture first. Track whole-frame and critical-class development metrics, precision-recall curves, false positives, and latency. If a candidate helps rare classes but harms essential economy objects, keep the tradeoff visible rather than collapsing it into one overall number.

**Exit check:** an experiment table that attributes gains to a specific change and identifies the candidate to take to the final test. GPU training and labeling review are the main external resources; code, generation, audits, evaluation, and experiment bookkeeping can be automated.

## 8. Choose the inference and model package

Re-evaluate the best trained checkpoint with the full-frame and compatible crop protocols from step 3. Crops must come from native capture pixels. Merge in screenshot coordinates; check overlapping detections and target-click alignment. Keep the input size required by the exported model explicit and verify ONNX Runtime versus the Mac inference provider. Measure the **whole detection request**, not only model-forward time.

If corrected data plus crops still miss small classes, test a P2/finer-feature candidate or a larger standard model under the same latency budget. These require new training and export checks. Select per-class thresholds on development sessions only, then freeze the entire inference package: weights, preprocessing, crop policy, merge/NMS rules, class list, and thresholds. [YOLO26 variants](https://docs.ultralytics.com/models/yolo26)

**Exit check:** one frozen candidate with reproducible coordinates, provider parity checked, and p95 latency within the agent's target-refresh budget (the prior plan calls for at most two seconds). Record any speed/accuracy compromise.

## 9. Run the sealed test and qualify in the game

Evaluate the frozen candidate and the frozen reference **once on the same held-out sessions**. Report overall and per-class precision, recall, F1, IoU rule, support counts, false positives per frame, performance by actual zoom/model-pixel-size band, and latency. Use session-aware uncertainty intervals where sample size permits. If a result is weak or ambiguous, return to development data and reserve a *new* final test before making a new generalization claim.

Then run controlled Windows game checks across opening and later-age views. Inspect model boxes, their conversion to click coordinates, post-camera-jump target reacquisition, and whether sheep/berries/TC/farm/cavalry detections support the intended action. Save failures by source image and object size. A high offline F1 is not enough when the remaining errors break food gathering or production.

**Exit check:** a candidate improves the predeclared gameplay detection measures over the reference, has no material regression on action-critical objects, meets the refresh budget, and supports successful in-game target interactions. If a class still has too few independent examples, report it as unqualified rather than declaring it solved.

## 10. Release and continue the error loop

Publish the selected checkpoint as a new immutable version of `model.onnx` in the model registry, with source revision, training manifest, model/schema hashes, class mapping, inference contract, test protocol/results, and known limitations. Pin the agent/server to that version and verify both ends load the expected checksum. Keep the previous version available for rollback.

After deployment, save false negatives, wrong classes, duplicate boxes, bad click targets, zoom, and pixel size from real runs. Add them to development/training after review. Keep the final test set sealed; create a new one for the next release. [Registry contract](../../packages/detection/registry/aoe2-entity-detector/README.md)

**Exit check:** reproducible model release and a short, actionable list of remaining error clusters.

## Division of work

| Can be done in the repository | Needs Windows/game or human input |
| --- | --- |
| Fix extractor and generator; make manifests and quality checks; implement compatible crop inference; prepare prelabels; run analysis and training code; evaluate; package candidate model | Capture representative native game frames at actual zooms; verify difficult class labels and box policy; inspect game-rendered assets and interactions; provide access to the chosen GPU/training host if local compute is insufficient |

The most useful immediate session is a varied Windows capture with a few Dark Age openings, farms, and cavalry scenes at normal and edge zoom settings. It supplies calibration and independent test material without requiring anyone to build a large labeled dataset before code work begins.
