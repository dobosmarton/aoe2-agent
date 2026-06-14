# Runbook: Retrain the detection model (v6 / YOLO26n) with cvat.ai + Lambda

End-to-end checklist to produce `aoe2_yolo_v6.pt` / `aoe2_yolo_v6.onnx` — the YOLO26n (NMS-free) model the refactored detector now resolves. Annotation is done on **cvat.ai** (hosted), training on **Lambda Cloud** (A100).

> **Why this is needed:** `get_detector()` ([detector.py:835](../../packages/detection/src/inference/detector.py)) resolves `aoe2_yolo_v6.onnx → aoe2_yolo_v6.pt → mock`. Neither v6 file exists yet, so the agent currently runs in **mock detection**. This runbook produces them.

All commands run from the repo root (`agent/`) inside the uv env — prefix with `uv run`, or activate the venv and drop the prefix. Module path is `detection.*`.

---

## The loop at a glance

```
                    ┌─────────────────────────────────────────────┐
                    │                                             │
 generate_training_data.py        prelabel.py (model | open-vocab)│
 (synthetic + rebalancing)        active_learning.py / hard_negatives.py
          │                                  │  (what to label next)
          ▼                                  ▼
   synthetic data ───►  prepare_training.py  ◄─── cvat.ai (annotate → COCO 1.0)
                              │
                              ▼
                        training_data/ (merged + dataset.yaml)
                              │
                              ▼
                        Lambda A100 (train YOLO26n → aoe2_yolo_v6)
                              │
                              ▼
                 models/aoe2_yolo_v6.{pt,onnx}  ──► deploy to Windows VM
```

## Prerequisites

- **Accounts:** [app.cvat.ai](https://app.cvat.ai) and [Lambda Cloud](https://cloud.lambdalabs.com).
- **SSH key** for Lambda saved locally (e.g. `~/.ssh/lambda-aoe2-training.pem`, `chmod 600`).
- **Raw screenshots** in `packages/detection/src/real_screenshots/raw/` (220 captured; 58 labeled so far).
- **Sprites extracted** to `tmp/sprites/` (only if regenerating synthetic data — see Phase A).
- For the optional DINO-X open-vocab backend: `export DINOX_API_KEY=...` and install the extra: `uv pip install -e 'packages/detection[autolabel]'`. The default `yoloe` backend needs no key.

---

## Phase A — Generate synthetic data (local)

The v6 refactor added dataset-level rebalancing (oversample rare/confusable classes, render distant ~20px units) directly into the generator, so you just regenerate.

```bash
# Run from the repo root. Pass --game-dir / --output / --sprites as ABSOLUTE paths
# (see trap below) — under `uv run` the module's cwd is packages/detection/src, so
# relative paths resolve to the wrong place.

# 1. (Re)extract sprites from the local game graphics into a fresh library.
#    Produces all 59 classes incl. the 6 newly-added (galley, fire_galley,
#    siege_tower, krepost, goose, fish). Source SLDs: game_graphics/ (6,817 files,
#    already exported from the VM). Verified run: 59/59 classes, 775 sprites, 0 failed.
uv run python -m detection.extraction.extract_sprites \
    --game-dir "$PWD/game_graphics" \
    --output "$PWD/tmp/sprites_v6" \
    --multi-frame              # 4 rotation frames per unit; add --player-colors to recolor units

# 2. Build real-terrain backgrounds from the game's ground textures.
#    Source: game_terrain/ — the DDS tiles from resources/_common/terrain/textures
#    (exported from the VM; Pillow reads DXT1 directly, no extra deps). Gives scenes
#    real game ground (grass/dirt/desert/snow...) instead of flat procedural color.
#    Defaults soften the ground to match in-game rendering and keep a realism→soft
#    spectrum so objects stay recognizable: capped zoom (--zoom-max 1.6), bimodal
#    blur (--soft-fraction 0.35 heavily muted, rest mild --blur 1.6), muted contrast/
#    saturation, busy cobblestone/rock down-weighted. Excludes void (g_bla), water
#    (g_wt*), near-black, and DE placeholder terrains (o_* "PLACEHOLDER" grid) → 63 tiles.
uv run python -m detection.training.build_terrain_backgrounds \
    --terrain-dir "$PWD/game_terrain" \
    --output "$PWD/tmp/terrain_backgrounds" \
    --count 200                # scale up for the full run; all softening defaults are baked in

# 3. Generate synthetic data from the v6 library on the real-terrain backgrounds.
uv run python -m detection.training.generate_training_data \
    --num-images 3000 \
    --sprites "$PWD/tmp/sprites_v6" \
    --backgrounds "$PWD/tmp/terrain_backgrounds" \
    --output training_data_v6 \
    --train-split 0.8
```

> **Path trap — verify before spending a GPU hour.** Under `uv run`, both the extractor and the generator resolve relative paths against the **package dir** (`packages/detection/src/`), not your cwd. So the extractor's default `--game-dir game_graphics` errors with *"Directory not found: …/packages/detection/src/game_graphics"*, and the generator's default `--sprites tmp/sprites` silently points at a non-existent dir (the repo-root `tmp/sprites` it *looks* like it means is a **stale 140-sprite set** with pre-unification `war_wagon`/`longbowman`/`mangudai`, missing every `unique_*` and `wall`). **Pass `--game-dir`, `--output`, and `--sprites` as absolute paths** (`"$PWD/…"` from the repo root) — pathlib returns an absolute path unchanged, sidestepping the broken join. The fresh, complete library is **`tmp/sprites_v6`** (775 sprites, 59/60 classes); your prior `tmp/sprites_v5` (718, 53 classes) is left untouched as a fallback. `--output training_data_v6` (relative) lands in `packages/detection/src/training_data_v6/` — a **fresh** dir, because the generator doesn't clean its output (writing into the old `training_data/` would mix stale v5 images with new ones). Phase C must then read this dir via an explicit `--synthetic` (its default still points at the old `training_data/`).

> **Synthetic coverage — 6 of 7 gaps now closed.** Adding a class requires entries in **two** separate configs: `extract_sprites.py`'s `SPRITE_CONFIG` (produces the sprite PNGs) **and** `generate_training_data.py`'s `_BASE_SPRITE_CONFIGS` (places them into composited images, with `z_order`/scale/count). Both were updated for **`galley`, `fire_galley`, `siege_tower`, `krepost`, `goose`, `fish`** and verified end-to-end — a 300-image preview produced ~170–280 instances each (`krepost` is sparse at ~4/300 since it's a single rare building; raise its `count_range` or add an `oversample_weight` if detection is weak). **`farm` is the one remaining gap and is intentional** — the farm field is a terrain texture, not an SLD sprite (see the `# Farm buildings are terrain textures ... skipping` note in `extract_sprites.py`), so it can only be learned from **real CVAT annotations** (Phase B). Make sure your annotation batches cover `farm`.

This writes `train/` and `val/` with `classes.yaml` IDs. Scale `--num-images` up for the real run.

---

## Phase B — Annotate real screenshots on cvat.ai

### B1. Decide *what* to label (active learning)

Don't label blindly — surface the images/cases the current model is weakest on:

```bash
# Score all raw screenshots by model uncertainty, copy a CVAT-ready batch + pre-labels
uv run python -m detection.labeling.active_learning prepare \
    --batch-size 20 --conf 0.25

# OR target the specific cavalry-line confusions (camel vs cav-archer vs battle-elephant)
uv run python -m detection.labeling.hard_negatives --max-conf 0.5
```

### B2. Pre-label to bootstrap the annotations

Pre-labels give annotators boxes to *correct* instead of drawing from scratch. Use the current model (SAHI for high-res), or an open-vocab backend when no good model exists yet:

```bash
# Model-based, with SAHI tiling for retina screenshots
uv run python -m detection.labeling.prelabel --sahi --conf 0.15

# OR open-vocabulary bootstrap (no trained model needed)
uv run python -m detection.labeling.prelabel --open-vocab yoloe    # local
uv run python -m detection.labeling.prelabel --open-vocab dinox    # hosted (needs DINOX_API_KEY)
```

Output (YOLO `.txt` + `classes.txt`) lands in `labeling/output/prelabeled/`.

### B3. Create the cvat.ai project + 60 labels

1. On [app.cvat.ai](https://app.cvat.ai): **Projects → +** → name it e.g. `aoe2-detection`.
2. Open the project → **Raw** label editor and paste the 60-class label JSON. Generate it from the single source of truth:

   ```bash
   uv run python -c "import json; from detection.labeling.class_mapping import get_classes_for_cvat; print(json.dumps([{'name': n, 'attributes': []} for n in get_classes_for_cvat()]))" > cvat_labels.json
   ```

   Paste the contents of `cvat_labels.json` into the Raw editor and **Done**. (Label *order* doesn't matter — `prepare_training.py` maps by name, not by ID.)

### B4. Create a task and upload images + pre-labels

1. Inside the project: **+ Create a new task**, attach the screenshots you selected in B1.
2. Open the task → **Actions → Upload annotations** → format **YOLO 1.1**, and upload your pre-label `.txt` set (zipped with `classes.txt`). This seeds the boxes from B2.

### B5. Annotate / correct

Fix the pre-labeled boxes and add anything missed. Polygons are fine for precise outlines — they're preserved by the COCO export in the next step.

### B6. Export as COCO 1.0

> **Why COCO, not YOLO:** cvat.ai's **YOLO 1.1 export silently drops polygon annotations** — only rectangles survive. Export **COCO 1.0** instead; `prepare_training.py` computes bboxes from the polygon vertices.

**Task → Actions → Export annotations → COCO 1.0** (include images if convenient). Download and unzip, e.g. to `~/cvat_exports/batch1/` (expects `annotations/instances_default.json`).

---

## Phase C — Merge real + synthetic into the training set (local)

```bash
uv run python -m detection.labeling.prepare_training \
    --cvat-export ~/cvat_exports/batch1 \
    --synthetic "$PWD/packages/detection/src/training_data_v6" \
    --val-split 0.15
# add more --cvat-export <dir> flags to merge several export batches
```

The explicit `--synthetic` points at your Phase A v6 output (its default still targets the old `training_data/`); `--output` writes the merged set to `packages/detection/src/training_data_v2/`. The step converts COCO→YOLO (by class *name*), 85/15 splits the real images (`seed=42`), copies synthetic + `real_`-prefixed real images together, and writes `dataset.yaml` + `merge_summary.json`. **Check `merge_summary.json`** for per-class counts before spending money on a GPU.

---

## Phase D — Train on Lambda Cloud

### D1. Launch the instance
Lambda dashboard → launch **1× A100 (40 GB SXM4)**, Lambda Stack 22.04, attach your SSH key. Note the instance IP.

### D2. Package + upload the dataset

```bash
# the merged set from Phase C lives at packages/detection/src/training_data_v2/
tar -czf training_data_v2.tar.gz -C packages/detection/src training_data_v2
scp -i ~/.ssh/lambda-aoe2-training.pem training_data_v2.tar.gz ubuntu@<IP>:/home/ubuntu/
ssh -i ~/.ssh/lambda-aoe2-training.pem ubuntu@<IP> 'tar -xzf training_data_v2.tar.gz'
```

### D3. Fix the dataset.yaml path (known gotcha)

> ultralytics resolves relative `path:` from its **own install dir**, not your cwd. On Lambda the `path:` must be **absolute**, or training silently finds zero images.

```bash
ssh -i ~/.ssh/lambda-aoe2-training.pem ubuntu@<IP>
# on the instance, edit training_data_v2/dataset.yaml so the top reads:
#   path: /home/ubuntu/training_data_v2
#   train: train/images
#   val: val/images
```

### D4. Set up the environment

```bash
python3 -m venv ~/yolo_env && source ~/yolo_env/bin/activate
pip install --upgrade pip
pip install 'numpy<2' ultralytics     # numpy 2.x breaks PyTorch C-extensions
```

### D5. Train (YOLO26n → aoe2_yolo_v6)

Mirrors `train_yolo.py`'s isometric hyperparameters. The `cls=` line is optional — raise it to push the model to separate the confusable cavalry lines:

```bash
python -c "
from ultralytics import YOLO
model = YOLO('yolo26n.pt')                 # NMS-free base; ships STAL head for small objects
model.train(
    data='/home/ubuntu/training_data_v2/dataset.yaml',
    epochs=150, imgsz=640, batch=16, device=0, workers=8, patience=20,
    project='runs', name='aoe2_yolo_v6', exist_ok=True,
    # isometric-tuned augmentation
    flipud=0.0, fliplr=0.5, degrees=10, translate=0.1, scale=0.5,
    mosaic=1.0, mixup=0.1, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    # cls=1.0,   # optional: raise cls-gain for cavalry-line separation
)
"
```

~50–60 min on an A100. Watch mAP50 / mAP50-95 per epoch.

### D6. Export ONNX (NMS-free, dynamic batch)

```bash
python -c "from ultralytics import YOLO; YOLO('runs/aoe2_yolo_v6/weights/best.pt').export(format='onnx', imgsz=640, simplify=True, dynamic=True)"
```

### D7. Download both artifacts → local models dir

```bash
M=packages/detection/src/inference/models
scp -i ~/.ssh/lambda-aoe2-training.pem ubuntu@<IP>:/home/ubuntu/runs/aoe2_yolo_v6/weights/best.pt   $M/aoe2_yolo_v6.pt
scp -i ~/.ssh/lambda-aoe2-training.pem ubuntu@<IP>:/home/ubuntu/runs/aoe2_yolo_v6/weights/best.onnx $M/aoe2_yolo_v6.onnx
```

### D8. Terminate the instance

> Lambda bills by the hour. **Terminate from the dashboard immediately** after the scp completes.

---

## Phase E — Wire up, verify, deploy

1. **Check the model in.** `.gitignore` ignores all `models/*.{pt,onnx}` except a v5 allowlist (lines ~79–80). Flip it to v6 so the snapshot is committable:
   ```
   !packages/detection/src/inference/models/aoe2_yolo_v6.pt
   !packages/detection/src/inference/models/aoe2_yolo_v6.onnx
   ```
2. **Smoke test locally** — the detector should now load ONNX instead of falling back to mock:
   ```bash
   uv run python -m detection.testing.test_real_detection --model detection/inference/models/aoe2_yolo_v6.pt
   # optional: confirm the ONNX (num_boxes, 6) layout on the real export
   uv run python -m detection.training.spike_yolo26_onnx --model detection/inference/models/aoe2_yolo_v6.onnx --imgsz 1280
   ```
3. **Deploy the ONNX to the Windows VM** per [windows-vm-agent-bringup.md](./windows-vm-agent-bringup.md) (already points at `aoe2_yolo_v6.onnx`).
4. **Commit** the new model snapshot + any label/data tooling changes (no `Co-Authored-By: Claude` trailer, per project convention).

---

## Iterate (active-learning loop)

After a training run, feed the model's mistakes back in: re-run `active_learning prepare` / `hard_negatives` (Phase B1) on the new model, label that batch on cvat.ai, re-merge (Phase C), retrain (Phase D). Each loop targets the weakest classes — currently the cavalry lines and long-tail unique units.

## Cost / time

| Phase | Time | Cost |
|-------|------|------|
| Synthetic gen + merge (local) | minutes | $0 |
| Annotation (cvat.ai) | hours (manual) | free tier |
| Lambda A100 training | ~50–60 min | ~$1.30 |
| **Per training cycle** | **~1 h GPU** | **~$1.30** |

## Troubleshooting

- **Agent still detects nothing / "Using mock detection":** the v6 files aren't where `get_detector()` looks — confirm `models/aoe2_yolo_v6.onnx` (or `.pt`) exists; ONNX is preferred.
- **Training finds 0 images on Lambda:** `dataset.yaml` `path:` isn't absolute (Phase D3).
- **PyTorch import/C-extension errors on Lambda:** you skipped `pip install 'numpy<2'` (Phase D4).
- **Lost polygon labels after export:** you exported YOLO 1.1 instead of COCO 1.0 (Phase B6).

## Related

- [Chapter 9: Labeling and Active Learning](../part3-entity-detection/09-labeling-and-active-learning.md)
- [Chapter 12: Cloud Training](../part5-operations/12-cloud-training.md)
- [Chapter 8: Training Pipeline](../part3-entity-detection/08-training-pipeline.md) — rebalancing, loss-gain knobs
- `packages/detection/src/docs/TRAINING_GUIDE.md` — package-level training reference
