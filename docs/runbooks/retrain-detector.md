# Retrain the detector

Takes reviewed real frames to a served detector: label, build the dataset, train, export, score on the independent test set, register, deploy. Run every command from the `agent/` repo root on the Mac unless a step says otherwise. Background for each step is in [labeling](../detection/labeling.md) and [training](../detection/training.md).

`<date>` and `vN` stand for the dataset date and the next model version. Every output directory must not exist yet; the tools refuse to overwrite.

## 1. Get frames

Use frames already in `packages/detection/src/real_screenshots/`, or capture a session on the machine running the game:

```bash
uv run --no-sync python -m gameplay_agent.capture_dataset --session-id dev_<date>_match01 \
  --split development --source live_game --map <map> --civilization <civ> \
  --game-version "<build>" --graphics-preset <preset> --ui-scale <scale> \
  --game-stage dark_age --zoom default --count 20 --interval 15
```

Copy whole session directories to the Mac without re-encoding. Put the chosen frames in a batch folder `real_screenshots/labeling_batch_<date>_<name>/images/` and write its `manifest.csv` with the columns `image`, `source_path`, `source_session`, `split` (`development`), `width`, `height`, `sha256`. Do not pick frames from a session in the independent test batch (`labeling_batch_20261029_independent_test/manifest.csv`).

## 2. Prelabel and review

```bash
uv run --no-sync python -m detection_server.session_tools prelabel-batch \
  --manifest <batch>/manifest.csv \
  --model packages/detection/src/inference/models/aoe2_yolo_v12_nms736.onnx \
  --output <batch>/prelabels.coco.json
uv run --no-sync python -c "from detection.labeling.class_mapping import get_annotation_labels; print('\n'.join(get_annotation_labels()))" > <batch>/classes.txt
```

Import the prelabels into X-AnyLabeling, then review every image:

```bash
xanylabeling --filename <batch>/images --labels <batch>/classes.txt \
  --work-dir <batch> --no-auto-update-check --nodata
```

Fix and add boxes per the convention in [labeling](../detection/labeling.md#review-in-x-anylabeling), and mark each image checked only after reviewing all of it. Then export:

```bash
xanylabeling convert --task xlabel2coco --mode detect \
  --images <batch>/images --labels <batch>/images \
  --output <batch>/export_<date> --classes <batch>/classes.txt
```

## 3. Build the real-only dataset

This dataset is the background source for the composites. `prepare_training` takes one `--images-dir`, and the exports together must cover exactly the frames in the manifests passed. Copy the images and sidecars of every reviewed batch into one directory (`<reviewed>`), then pass every export and every manifest:

```bash
R=packages/detection/src/real_screenshots
uv run --no-sync python -m detection.labeling.prepare_training \
  --cvat-export $R/labeling_batch_20261011_xanylabeling_review/export_relabel_20261030 \
  --cvat-export <batch>/export_<date> \
  --images-dir <reviewed> \
  --manifest $R/labeling_batch_20261011_development/manifest.csv \
  --manifest $R/labeling_batch_20261011_scenario_addendum/manifest.csv \
  --manifest <batch>/manifest.csv \
  --historical-real packages/detection/src/training_data_v9_slim \
  --oversample-real 4 \
  --output packages/detection/src/training_data_real_<date> --dry-run
```

Check the summary (real train and validation counts, classes without real boxes), then rerun without `--dry-run`.

## 4. Generate composites and the synthetic source

```bash
uv run python -m detection.training.generate_composites \
  --dataset packages/detection/src/training_data_real_<date> \
  --sprites tmp/sprites_v8 \
  --output packages/detection/src/training_data_composite_<date> \
  --train 1000 --val 100
```

Open a sample of `train/images/` with its labels before going on. Then make `packages/detection/src/training_data_synth_<date>/` with a copy of `dataset.yaml` and `train` and `val` directories whose `images/` and `labels/` symlink both `training_data_v9_synth` (prefix `render_`) and the composite set (prefix `comp_`). Image and label stems must match.

## 5. Build the training dataset

Repeat the step 3 command with the same exports, images and manifests, plus `--synthetic packages/detection/src/training_data_synth_<date>` and `--output packages/detection/src/training_data_<date>`. Check the dry run, then rerun without `--dry-run`. The real train/validation split comes out the same because the session split uses the same seed.

## 6. Train

On a machine with an NVIDIA GPU and the repo:

```bash
just train yolo26n.pt --data packages/detection/src/training_data_<date>/dataset.yaml \
  --name aoe2_yolo_vN --cache ram
```

On a rented GPU, tar the dataset, upload it with `packages/detection/src/training/train_yolo.py`, and run the same flags with `python train_yolo.py`. The RunPod scripts in `tmp/r1_optimizer_2x2/` automate pod creation, upload, polling, download and pod deletion; copy them and edit the run list and dataset hashes first (see [training](../detection/training.md#where-training-runs)). Keep the defaults (MuSGD, lr 0.01, 150 epochs, no early stop) unless the change is the experiment.

Copy the final checkpoint to `packages/detection/src/inference/models/aoe2_yolo_vN.pt`. Use `last.pt`, not `best.pt`.

## 7. Export

```bash
M=packages/detection/src/inference/models
cp $M/aoe2_yolo_vN.pt $M/aoe2_yolo_vN_nms736.pt
just export-onnx $M/aoe2_yolo_vN_nms736.pt --imgsz 736 1280 --no-end2end
```

## 8. Score on the independent test set

Rebuild the test set with the candidate dataset's manifest added; the build fails if any test session was used for training:

```bash
uv run --no-sync python -m detection.labeling.build_test_set \
  --batch packages/detection/src/real_screenshots/labeling_batch_20261029_independent_test \
  --training-manifest packages/detection/src/training_data_reviewed_20261018/dataset_manifest.json \
  --training-manifest packages/detection/src/training_data_<date>/dataset_manifest.json \
  --output tmp/test_data_independent_<date>
```

Score the served model and the candidate on the same set:

```bash
for m in aoe2_yolo_v12_nms736 aoe2_yolo_vN_nms736; do
  uv run --no-sync python -m detection_server.evaluate --split test \
    --model packages/detection/src/inference/models/$m.onnx \
    --data tmp/test_data_independent_<date> \
    --output tmp/evaluations/$m-independent-<date>.json
done
```

Compare micro precision, recall and F1, per-class recall for villager, sheep, deer, boar, berry bush, town center and house, and the `fine_labels` ibex recall. Promote only a candidate that wins on the same frames. A frame that informed training or threshold choice cannot stay in the test set; if one did, label a fresh test batch.

## 9. Register

Update `packages/detection/registry/aoe2-entity-detector/` (`README.md`, `EVALUATION.md`, `evaluation.json` from step 8, `inference-contract.json` with the model, checkpoint and class-schema SHA-256) and publish `model.onnx`, `model.pt` and `classes.yaml` as a tagged version. Contents and rules are in [serving and evaluation](../detection/serving-and-evaluation.md).

## 10. Deploy

On the Mac, restart the detection server on the exported file and check the reported model and class count:

```bash
just server --model packages/detection/src/inference/models/aoe2_yolo_vN_nms736.onnx
curl http://localhost:8420/health
```

Point the agent at the same version: set `AOE2_DETECTION_MODEL=aoe2_yolo_vN_nms736`, or change the `detection_model` default in `apps/agent/src/config.py`. Keep the previous `.onnx` for rollback.
