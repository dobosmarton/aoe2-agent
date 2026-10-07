# Detector labeling and evaluation recommendation

## Decision

Yes, label more **real game frames**, but first reserve a small, independently
captured test set. Then label a separate, targeted training batch. The current
0.753 precision / 0.633 recall / 0.688 F1 is an operating-point measurement on
32 validation images, not proof of performance in a new game session. Those
images share January 18–19 capture sessions with the training split, and the
confidence thresholds may have been tuned on them. [Local evaluation protocol](../../packages/detection/registry/aoe2-entity-detector/EVALUATION.md)

The aggregate score also hides important differences. In this validation set,
the detector found 3/6 sheep, 4/6 berry bushes, 33/80 farms, and 165/218
villagers. Six sheep or berry labels are too few to characterize reliability in
new games; the farm misses are a clearer source of concern. These are
observations from the [per-class results](../../packages/detection/registry/aoe2-entity-detector/evaluation.json),
not predictions of live agent success. Detection quality alone cannot establish
that selection, clicking, OCR, or action confirmation works.

The real training split contains 187 unique screenshots, duplicated tenfold
for training. Counting only the unique label files gives 70 sheep boxes, 50
berry-bush boxes, and 533 farm boxes. More distinct sheep and berry scenes are
plausibly useful; farm recall needs an error audit because its problem is not
simply a lack of labeled instances. These counts come from the local
`training_data_v9_slim/train/labels` files, excluding `__dup*` copies.

## Recommended sequence

1. **Freeze the baseline.** Keep the present ONNX bytes, preprocessing,
   thresholds, class schema, and scorer fixed while constructing the test set.
   Ultralytics distinguishes a labeled, held-out test split from predictions on
   new unlabeled images: only the former supports quantitative metrics.
   [Model testing](https://docs.ultralytics.com/guides/model-testing)
2. **Create an untouched test set from new game sessions.** Sample at the
   actual capture resolution and interface profile, across Dark Age openings,
   food-resource views, later economies, Arabia and Highland, camera positions,
   clutter, and occasional empty/negative views. Keep whole sessions outside
   training and threshold tuning; do not split adjacent frames from one game
   across train and test. The session-level rule is our conservative inference
   from the documented risk of leakage and the need for previously unseen,
   deployment-representative test images—not a special Ultralytics requirement.
   Record session IDs and selection rules so the split can be audited.
   [Model testing](https://docs.ultralytics.com/guides/model-testing),
   [CV project steps](https://docs.ultralytics.com/guides/steps-of-a-cv-project)
3. **Audit annotation quality before increasing quantity.** Define precisely
   what counts as each class and box, including occluded livestock, overlapping
   villagers, foundations, and UI-covered objects. Review missed and false
   detections visually; fully label all in-scope visible objects in selected
   images rather than labeling only a desired class. Inconsistent or missing
   boxes can themselves depress measured precision or train the wrong behavior.
   [Ultralytics data-collection guide](https://docs.ultralytics.com/guides/data-collection-and-annotation),
   [YOLO training tips](https://docs.ultralytics.com/yolov5/tutorials/tips-for-best-training-results)
4. **Build a separate training batch from model failures and diversity.** Start
   with distinct, nonadjacent frames where the model misses or confuses sheep,
   berries, villagers, TC, farm, or another class the agent actually needs to
   act on. Include lookalike hard negatives and varied zoom, terrain, ownership
   colors, and occlusion. Do not spend the first labeling budget uniformly over
   all 60 classes: unsupported naval classes, for example, do not improve the
   opening economy. Use uncertainty/error triage *and* scene diversity rather
   than collecting many near-identical frames. This prioritization is an
   application-specific proposal, supported in principle by Ultralytics'
   active-labeling guidance and object-detection active-learning work; its
   benefit for this model must be measured.
   [Ultralytics annotation strategies](https://docs.ultralytics.com/guides/data-collection-and-annotation),
   [Yang et al., CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Plug_and_Play_Active_Learning_for_Object_Detection_CVPR_2024_paper.html)
5. **Retrain once and compare on the frozen test set.** Report overall and
   per-class precision/recall/F1 with ground-truth counts, false positives,
   misses, and representative failure images; include inference latency and
   the exact model/manifest hashes. Keep the old 32-image result labeled as
   in-distribution validation, and avoid comparing scores produced by different
   thresholds or scoring rules. If the new test set is repeatedly used to
   select labels, thresholds, or models, promote it to development data and
   create another untouched holdout before making a generalization claim.
   [Model testing](https://docs.ultralytics.com/guides/model-testing),
   [Validation mode](https://docs.ultralytics.com/modes/val)

No fixed number of extra frames guarantees a useful model. Start with a
manageable test-set pilot from several new sessions, inspect class support and
failure diversity, and expand the set where uncertainty remains high. Labeling
many neighboring frames or oversampling the existing January sessions would
increase image counts without resolving the present generalization question.

## Source boundaries

- The numerical results and split history above are from this repository's
  evaluation files, not from external papers.
- The session-level separation, class-priority order, and staged workflow are
  recommendations inferred for this game agent. The cited external sources
  support held-out testing, leakage checks, accurate complete labels, and
  uncertainty/diversity-based sample selection; they do not validate this
  detector or prescribe a universal sample count.
