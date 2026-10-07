# Making the AoE2 detector robust to object size

Research date: 2026-10-04. This is a research and experiment proposal, not a claim that a new detector has been trained or qualified.

## Conclusion

Exact matching between synthetic and gameplay object sizes is not the objective. The objective is a detector with measured accuracy throughout a supported range of projected object sizes. Scale augmentation, multi-scale feature extraction, and detail-preserving inference are established approaches to that problem. They provide robustness, not perfect invariance at arbitrary sizes. The SNIP paper specifically studies why scale variation remains difficult for CNN detectors. [SNIP](https://arxiv.org/abs/1711.08189), [Feature Pyramid Networks](https://arxiv.org/abs/1612.03144).

For this project, prioritize correct synthetic assets/labels and continuous scene-level scale coverage, then test full-frame versus crop-based inference on the current model. Architecture changes should follow those controlled measurements. This prioritization is a project recommendation, not a result established by the cited papers.

## Local implementation and a completed diagnostic

The current gameplay route calls `detect_fast`, so fresh detections normally use one full-image pass, not the existing `detect_fast_multi` or tiled path. Local initialization explicitly disables SAHI. Aspect-preserving letterboxing is already shared by the local/server ONNX routes; replacing it with a stretched square would be a regression. [Gameplay route](../../apps/agent/src/detection_phase.py), [letterboxing](../../packages/detection/src/inference/preprocess.py)

The inspected `aoe2_yolo_v9.onnx` declares **`[1, 3, 1280, 1280]`**, and its metadata records YOLO26n, Ultralytics 8.4.54 and `dynamic=False`. That is a tensor-shape requirement, **not a requirement that the game run at 1280 resolution or that every sheep match one training size**. The comments in `config.py` conflate training resolution with the deployment shape contract and should be clarified during implementation. A fixed-shape model can process different capture resolutions and crop sizes through correct preprocessing. [Configuration](../../apps/agent/src/config.py)

The older tiled code needs revision before it can be enabled for this artifact: the server defaults to 640-square tiles and stacks all ONNX tiles into a batch, conflicting with both the artifact's 1280-square shape and batch size 1. A compatible crop path would preprocess every crop to 1280, perform supported batch-1 calls, and merge in original screen coordinates. A larger inference size would require a suitable export and provider qualification; changing only the config value is insufficient. [Server tiling](../../apps/detection-server/src/app.py), [local tiling](../../packages/detection/src/inference/sahi.py)

The remote two-pass helper already uses the configured 1280 input for both the full view and center crop, but it is not the live gameplay route. Center-only detail does not solve coverage at the edges. The local helper also carries a legacy crop-to-640 assumption: on a dynamic/PyTorch model, halving both crop width and input width would provide approximately the same object pixel size as the full pass, not twice the detail. The static ONNX loader overrides that requested size. This is another reason to use one explicit preprocessing and crop contract across providers. [Remote helper](../../packages/detection/src/inference/remote_detector.py), [local helper](../../packages/detection/src/inference/detector.py)

### Does a modest size change actually collapse the current model?

A read-only diagnostic on 2026-10-04 used all **32 existing real validation images / 1,212 labels**, the shipped model, CPU ONNX, and the existing server thresholds, NMS and IoU 0.50 matching. For each scale, it resized the original frame with bicubic interpolation, centered it on a same-size gray `(114,114,114)` canvas, and transformed every ground-truth box by the exact rounded resize and offset. No objects were cropped out. It then ran the production single-pass preprocessing and scorer. The unmodified baseline exactly reproduces the recorded micro metrics. [Evaluation implementation](../../apps/detection-server/src/evaluate.py)

| Linear object scale | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| 100% / unmodified | 0.753 | 0.633 | 0.688 |
| 90% | 0.775 | 0.610 | 0.682 |
| 75% | 0.771 | 0.569 | 0.655 |
| 50% | 0.724 | 0.433 | 0.542 |

At 90%, villager F1 changes from 0.738 to 0.733. At 50%, it drops to 0.513; sheep recall drops from 3/6 to 0/6, although six sheep are far too few for a stable class estimate. Reduced recall is the main aggregate failure at severe shrinking.

This does **not** establish robustness at all zooms or resolutions. The test also changes padding, image context and resampling; it does not reveal new terrain like zooming out in-game, and it starts from previously resized JPEGs rather than native captures. Its variants are correlated observations of the same development scenes, not 128 independent test frames. It does directly contradict the stronger claim that a 10% size difference necessarily makes this model collapse. Temporary reproducer: `/private/tmp/aoe2_scale_probe.py`.

The preceding synthetic audit found roughly 2–3× differences in median linear size across several classes, not a slight mismatch. The recommendation is to cover an appropriate distribution automatically, not to force the user to reproduce one exact scale. [Synthetic audit](aoe2-synthetic-data-quality.md)

## Three different meanings of size

- **Game camera zoom:** changes the size of rendered units/buildings and how much world fits in the viewport.
- **Captured image dimensions:** may change the viewport extent, the rendered scale, or both, depending on game/display settings. Resizing an existing screenshot changes neither the original detail nor the game camera; it only resamples pixels.
- **Detector input dimensions:** preprocessing changes how many pixels each captured object occupies when the network receives it. Letterboxing preserves aspect ratio; stretching does not.

For aspect-preserving preprocessing, an object's model-input width is its capture width multiplied by `min(input_width / capture_width, input_height / capture_height)`. Padding does not enlarge the object. Thus a hypothetical 30-pixel-wide object in a 1920-wide frame becomes 20 pixels wide when the whole frame is reduced to 1280. A 960-wide crop from the original capture, resized to 1280, makes that object 40 pixels wide. The crop uses more of the original image detail than the reduced full-frame path; it does not create newly observed detail. These are geometric consequences, not model benchmark results. [Ultralytics LetterBox implementation](https://github.com/ultralytics/ultralytics/blob/v8.4.54/ultralytics/data/augment.py).

The absolute number of original pixels still matters. Enlarging a nearly featureless tiny sprite cannot reveal details the capture never contained. There is no universal minimum width at which every AoE2 class becomes identifiable; that needs measurement by class and visual condition. Slicing is intended to avoid losing useful high-resolution information when preparing network inputs. [SAHI paper](https://arxiv.org/abs/2202.06934).

## Training practices

### Cover a distribution rather than one calibrated size

Generate each synthetic scene with a shared camera scale, retaining meaningful building/unit size relationships. Sample the supported zoom interval and a modest margin around it; include deliberate coverage of small, medium, and large examples of important classes. Record object dimensions after the actual inference preprocessing, not only dimensions in the source PNG. This is a proposed AoE2-specific sampling design.

Scene scale need not be manually tuned per image. A one-time calibration from several game zoom settings can define automated rendering ranges. Supplemental copy-paste augmentation can vary object scales more freely; physically coherent scenes need not be the only training examples. Research found both large-scale jittering and copy-paste useful, but its Mask R-CNN/COCO results are not an expected gain for this YOLO/AoE2 system. The paper's very broad 0.1–2.0 jitter range is evidence that broad variation can be useful, not a setting to copy blindly for tiny sprites. [Simple Copy-Paste, §3–4](https://arxiv.org/html/2012.07177v2).

### Separate image zoom augmentation from multi-scale training

The repository lock pins Ultralytics 8.4.54. Its source supports two separate controls:

- `scale=0.5` samples image zoom between 0.5 and 1.5, with corresponding crop/padding and label transforms. An explicit `(min, max)` interval is also supported. The repository trainer already sets `scale=0.5`; scale augmentation is not currently absent. [Pinned RandomPerspective implementation](https://raw.githubusercontent.com/ultralytics/ultralytics/v8.4.54/ultralytics/data/augment.py).
- `multi_scale` is a **floating-point fraction**, defaulting to `0.0` in this version. A positive value samples the network batch dimensions around `imgsz`, rounded to model stride. For example, `imgsz=1280, multi_scale=0.25` targets approximately 960–1600. It varies training computation/memory as well as effective object sizes. Do not use advice for older boolean-only versions without checking the pinned API. [Pinned trainer](https://raw.githubusercontent.com/ultralytics/ultralytics/v8.4.54/ultralytics/models/yolo/detect/train.py), [pinned defaults](https://raw.githubusercontent.com/ultralytics/ultralytics/v8.4.54/ultralytics/cfg/default.yaml).

Broader augmentation should be tested incrementally: the composed effects of synthetic scene zoom, mosaic, affine scale and batch resizing can make many objects unrecognizably small. Inspect the actual augmented training batches and per-class pixel-size distribution. More aggressive settings are not automatically better. This is an experimental safeguard rather than a claimed optimal recipe.

Multi-scale training does not require dynamic-shape deployment: one trained detector can still be exported for a selected fixed input shape. Conversely, enabling dynamic ONNX dimensions only changes the accepted shapes; it does not train scale robustness. [Ultralytics export documentation](https://docs.ultralytics.com/modes/export/).

## Inference practices

### Preserve detail with overlapping crops when needed

SAHI combines overlapping image patches, per-patch detection and coordinate-aware merging. Its authors also train on both full images and resized patches: full images retain large-object context while patches improve small-object exposure. Smaller patches can cut large objects apart, increase duplicate detections, and cost additional forward passes. The paper demonstrates these tradeoffs, not a guarantee that slicing increases our F1. [SAHI method and experiments](https://arxiv.org/html/2202.06934v1).

For an interactive agent, a reasonable candidate is a full-frame pass plus a bounded high-detail pass around the intended interaction region, with occasional broader tiled refreshes if benchmarks justify them. This is a project-specific latency design. Evaluate missing objects outside the selected region: a detector-driven crop selector cannot recover every object the first pass never noticed.

Crop first from the original capture, then resize the crop to the detector's supported input. Cropping after the full screenshot has already been reduced cannot recover pixels lost by that reduction. Keep exact crop/letterbox transforms for converting detections back to screen input coordinates. Never silently send a 640-square tensor to a static 1280-square ONNX model.

### Test a finer feature pyramid only after simpler changes

Standard YOLO26 uses detection levels P3/8, P4/16 and P5/32. The current upstream P2 variant adds a finer P2/4 level. Finer sampling is a plausible benefit for small sprites, but it adds compute and does not restore missing source detail. A stride of 8 is not a hard minimum detectable object width. [Standard YOLO26 architecture](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/26/yolo26.yaml), [P2 architecture](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/26/yolo26-p2.yaml).

Current official documentation lists P2 as an architecture-only YAML, with no released P2-specific pretrained checkpoints. Its presence in our pinned 8.4.54 package was not verified; adoption may require an explicit version decision. Benchmark a trained P2 candidate or a slightly larger standard detector against the corrected-data baseline, including export compatibility and CPU p95 latency. [YOLO26 architecture variants](https://docs.ultralytics.com/models/yolo26).

## Proposed scale-robustness evaluation

These are proposed project gates, not completed experiments:

1. Keep a session-separated real holdout. Capture some equivalent scenes at several actual game zoom settings; do not rely exclusively on resizing the same screenshot.
2. Add a deterministic resampling stress test to isolate sensitivity to preprocessing. Treat its transformed images as repeated measurements of the same scenes, not independent new test examples.
3. Report precision, recall and F1 by class and model-input pixel-size bucket, alongside aggregate metrics. Also report performance by capture resolution, actual zoom and occlusion, where metadata exists. Thresholds are chosen on development data and frozen for final comparison.
4. Compare the existing model's full-frame inference with detail-preserving crops first. Record p50/p95 end-to-end latency, forward-pass count, duplicate rate and coordinate error as well as detection accuracy.
5. Compare retraining with corrected synthetic data, then added scale coverage, then optional multi-scale training. Hold model family, real examples and training budget constant for the initial ablations. If stronger augmentation has not converged at that budget, report it rather than concluding the technique cannot help.
6. Only then test a finer-head/larger detector, and compare on the same speed–accuracy budget.

The intended acceptance condition is stable useful accuracy throughout the declared gameplay operating range, including modest zoom/resolution variation. It is not identical accuracy at every possible scale, nor a requirement to reproduce one exact training size during play.
