# Runbook: Retrain the detection model (v6 / YOLO26n) with cvat.ai + cloud GPU

End-to-end checklist to produce `aoe2_yolo_v6.pt` / `aoe2_yolo_v6.onnx` — the YOLO26n (NMS-free) model the refactored detector now resolves. Annotation is done on **cvat.ai** (hosted); training on a cloud GPU — the **tested path is RunPod** (RTX 4090, ~$0.7/hr). Lambda Cloud (A100) steps are kept as an alternative, but Lambda rejects prepaid/most debit cards (incl. Revolut), so RunPod is the working default.

> **Why this is needed:** `get_detector()` ([detector.py:835](../../packages/detection/src/inference/detector.py)) resolves `aoe2_yolo_v6.onnx → aoe2_yolo_v6.pt → mock`. Without the v6 files the agent runs in **mock detection**. This runbook produces them.

> **Latest run (water-scene + real data).** Synthetic-only v6 hit 82.7% mAP50 overall but **`fish` collapsed to 0.146** — fish/naval sprites were composited on *land* (a scene that never occurs in-game). The fix: a **water-scene mode** (Phase A) that composites fish/naval only on real water textures and removes them from land scenes, plus merging **187 real CVAT screenshots**. Result: **`fish` 0.146 → 0.545 mAP50** (recall 0.079 → 0.512), overall **0.827 → 0.834** (held). The diagnosis generalizes — *if a class only exists in a specific scene, the synthetic generator must place it only in that scene.*

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
              cloud GPU: RunPod 4090 (train YOLO26n → aoe2_yolo_v6)
                              │
                              ▼
                 models/aoe2_yolo_v6.{pt,onnx}  ──► deploy to Windows VM
```

## Prerequisites

- **Accounts:** [app.cvat.ai](https://app.cvat.ai) and a GPU host — [RunPod](https://runpod.io) (tested) or [Lambda Cloud](https://cloud.lambdalabs.com).
- **SSH key** saved locally — e.g. `~/.ssh/runpod-aoe2` (`chmod 600`); you'll paste its `.pub` into the pod (Phase D1). Lambda alt: `~/.ssh/lambda-aoe2-training.pem`.
- **Raw screenshots** in `packages/detection/src/real_screenshots/raw/` (220 captured; 220 labeled, 187 merged into the v7 train split after dedup).
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

# 2b. Build WATER-only backgrounds for naval/fish scenes (the fish fix).
#     Same softening defaults, but --only-water loads ONLY the g_wt*/g_wtr DDS tiles
#     that step 2 excludes. ~60 is plenty (water scenes are a fraction of the set).
uv run python -m detection.training.build_terrain_backgrounds \
    --only-water \
    --terrain-dir "$PWD/game_terrain" \
    --output "$PWD/tmp/water_backgrounds" \
    --count 60

# 3. Generate synthetic data on the real-terrain backgrounds, WITH water-scene mode.
#    --water-backgrounds + --water-fraction enable per-image scene typing: with prob
#    --water-fraction the image is a WATER scene (water bg, only fishing_ship/unique_ship/
#    fish/galley/fire_galley placed); otherwise a LAND scene (land bg, those 5 classes
#    excluded). Net effect: fish/naval ONLY on water, land units ONLY on land — no more
#    fish-on-grass. Omit the two water flags to fall back to legacy un-gated behaviour.
uv run python -m detection.training.generate_training_data \
    --num-images 3000 \
    --sprites "$PWD/tmp/sprites_v6" \
    --backgrounds "$PWD/tmp/terrain_backgrounds" \
    --water-backgrounds "$PWD/tmp/water_backgrounds" \
    --water-fraction 0.15 \
    --output training_data_v6 \
    --train-split 0.8
```

> **Scene-separation invariant — verify before training.** The 5 water classes (`fishing_ship`, `unique_ship`, `fish`, `galley`, `fire_galley`) must appear **only** on water backgrounds and **never** on land. `dock`=15 stays a **land** class (it sits on the shoreline, learned from real data). Generate a small preview and assert no land image carries a water-class label — a single regression here reintroduces the fish-on-grass failure that capped fish at 0.146.

> **Path trap — verify before spending a GPU hour.** Under `uv run`, both the extractor and the generator resolve relative paths against the **package dir** (`packages/detection/src/`), not your cwd. So the extractor's default `--game-dir game_graphics` errors with *"Directory not found: …/packages/detection/src/game_graphics"*, and the generator's default `--sprites tmp/sprites` silently points at a non-existent dir (the repo-root `tmp/sprites` it *looks* like it means is a **stale 140-sprite set** with pre-unification `war_wagon`/`longbowman`/`mangudai`, missing every `unique_*` and `wall`). **Pass `--game-dir`, `--output`, and `--sprites` as absolute paths** (`"$PWD/…"` from the repo root) — pathlib returns an absolute path unchanged, sidestepping the broken join. The fresh, complete library is **`tmp/sprites_v6`** (775 sprites, 59/60 classes); your prior `tmp/sprites_v5` (718, 53 classes) is left untouched as a fallback. `--output training_data_v6` (relative) lands in `packages/detection/src/training_data_v6/` — a **fresh** dir, because the generator doesn't clean its output (writing into the old `training_data/` would mix stale v5 images with new ones). Phase C must then read this dir via an explicit `--synthetic` (its default still points at the old `training_data/`).

> **Synthetic coverage — 6 of 7 gaps now closed.** Adding a class requires entries in **two** separate configs: `extract_sprites.py`'s `SPRITE_CONFIG` (produces the sprite PNGs) **and** `generate_training_data.py`'s `_BASE_SPRITE_CONFIGS` (places them into composited images, with `z_order`/scale/count). Both were updated for **`galley`, `fire_galley`, `siege_tower`, `krepost`, `goose`, `fish`** and verified end-to-end — a 300-image preview produced ~170–280 instances each (`krepost` is sparse at ~4/300 since it's a single rare building; raise its `count_range` or add an `oversample_weight` if detection is weak). **`farm` is the one remaining gap and is intentional** — the farm field is a terrain texture, not an SLD sprite (see the `# Farm buildings are terrain textures ... skipping` note in `extract_sprites.py`), so it can only be learned from **real CVAT annotations** (Phase B). Make sure your annotation batches cover `farm`.

> **Real-screenshot UI overlays (on by default).** The generator layers in-game UI on top of placed entities — selection ellipses + health bars on units, garrison badges on buildings, and a large bottom command-panel HUD — so the model isn't brittle to artifacts that only appear in *real* screenshots (the synthetic sprites are otherwise pristine). Overlays are purely visual: they never move a label, and the HUD panel is registered as an occluder so entities buried beneath it get their labels dropped. They ride the same `enable_enhanced_augmentations` switch, so `--no-enhanced-aug` turns them off with the rest. Overlay colours reuse the extractor's 8-colour `PLAYER_COLORS`, so they match `--player-colors` sprite recolouring.

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
    --cvat-export ~/cvat_exports/batch2 \
    --cvat-export ~/cvat_exports/full220 \
    --synthetic "$PWD/packages/detection/src/training_data_v6" \
    --output training_data_v8 \
    --oversample-real 10 \
    --val-split 0.15
```

> **`--oversample-real N` — counter synthetic dominance.** With ~2400 synthetic vs ~187 real train images, the real signal is <8% of the loss, which is a prime suspect for the ~0% military recall. `--oversample-real N` duplicates each real **train** pair N× on disk (the validation split is never duplicated, so metrics stay honest). Pick N to roughly balance the two pools — e.g. `10` lifts 187 real → 1870, on par with the synthetic 2400. This is the local, no-GPU half of the two-stage sim-to-real plan; the GPU half is fine-tuning `best.pt` on this oversampled set. The per-class counts in `merge_summary.json` stay **unique** (oversample-independent) on purpose, so the scarcity signal still reads true; the `oversample_real` factor is recorded alongside them.

`--cvat-export` is repeatable — pass each export batch. Each export must be restructured as `<dir>/annotations/instances_default.json` (copy a loose `instances_default.json` into an `annotations/` subdir). **Order matters: list the batch whose labels should win first** — dedup is by image *stem*, first-wins, so the fish-rich `batch2` goes ahead of the broader `full220` so its fish/galley labels survive overlaps. The explicit `--synthetic` points at your Phase A v6 output (its default still targets the old `training_data/`); `--output training_data_v7` writes to `packages/detection/src/training_data_v7/`. The step converts COCO→YOLO (by class *name*), 85/15 splits the real images (`seed=42`), copies synthetic + `real_`-prefixed real images together (here: 2400 synthetic + 187 real = 2587 train, 633 val, 60 classes), and writes `dataset.yaml` (`path: .`) + `merge_summary.json`. **Check `merge_summary.json`** for per-class counts before spending money on a GPU — confirm fish/galley are present from *both* synthetic-water and real.

---

## Phase D — Train on a cloud GPU (RunPod — tested path)

An RTX 4090 (24 GB) trains this set in ~30 min for 150 epochs (~12 s/epoch). All remote commands assume `root@<IP> -p <PORT>`.

### D1. Provision the pod + fix SSH (the recurring gotcha)

RunPod dashboard → deploy a **PyTorch** pod on an **RTX 4090**. Then, **before connecting over SSH**:

> **RunPod wipes `~/.ssh/authorized_keys` on every pod (re)start, and key injection on the PyTorch template is unreliable.** Add your public key manually via the in-browser **Connect → Start Web Terminal**, and re-do it after any restart:
> ```bash
> mkdir -p ~/.ssh && echo "<paste your id_ed25519.pub / runpod-aoe2.pub>" >> ~/.ssh/authorized_keys \
>   && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys
> ```
> Connect with the **private key whose `.pub` you pasted** (a wrong key fails silently as `Permission denied (publickey)`; diagnose with `ssh -v …` — it prints the offered key's fingerprint). `Permission denied` means the pod is *up* but missing your key (re-add it); `Connection refused`/timeout means the pod is stopped.

```bash
ssh -i ~/.ssh/runpod-aoe2 -p <PORT> root@<IP>
```

### D2. Package + upload the dataset

> **Use plain `tar` (no `-z`) for image datasets.** The set is ~95% JPG/PNG, which is *already compressed* — gzip shaves ~2–3 % off the wire while burning minutes of CPU, and a backgrounded `tar -czf` can hit its timeout mid-archive and leave a **truncated tarball** (this bit us: 1413/2587 images). Plain `tar` is an I/O-bound byte copy (~10 s) and the upload size is essentially identical.

```bash
# from the repo root — merged set lives at packages/detection/src/training_data_v7/
tar -cf tmp/runpod/training_data_v7.tar -C packages/detection/src training_data_v7
# sanity: tar -tf tmp/runpod/training_data_v7.tar | grep -c train/images   # expect 2587
scp -i ~/.ssh/runpod-aoe2 -P <PORT> tmp/runpod/training_data_v7.tar root@<IP>:/root/
ssh -i ~/.ssh/runpod-aoe2 -p <PORT> root@<IP> 'cd /root && tar -xf training_data_v7.tar'
# verify integrity: byte counts must match
wc -c < tmp/runpod/training_data_v7.tar
ssh -i ~/.ssh/runpod-aoe2 -p <PORT> root@<IP> 'wc -c < /root/training_data_v7.tar'
```

`tar` may print harmless `Ignoring unknown extended header keyword 'LIBARCHIVE.xattr.com.apple.provenance'` lines on Linux — macOS xattrs that don't map onto the target FS. Safe to ignore.

### D3. Set up the environment

> On the PyTorch template, Python lives at **`/opt/conda`** and is **not on the non-login SSH PATH** — prefix every remote command with `export PATH=/opt/conda/bin:$PATH`. ultralytics + CUDA-enabled torch are preinstalled. Minimal images are missing **`libGL.so.1`**, so the OpenCV import inside ultralytics fails until you install it.

```bash
ssh -i ~/.ssh/runpod-aoe2 -p <PORT> root@<IP>
export PATH=/opt/conda/bin:$PATH
apt-get update && apt-get install -y libgl1 libglib2.0-0     # fixes "libGL.so.1: cannot open shared object file"
python -c "import torch, ultralytics, cv2; print(ultralytics.__version__, torch.cuda.is_available())"
```

### D4. Fix the dataset.yaml path (known gotcha)

> ultralytics resolves relative `path:` from its **own install dir**, not your cwd. The merged `dataset.yaml` ships `path: .`, which finds zero images. Make it absolute.

```bash
sed -i 's|^path:.*|path: /root/training_data_v7|' /root/training_data_v7/dataset.yaml
```

### D5. Train (YOLO26n → aoe2_yolo_v6)

Mirrors `train_yolo.py`'s isometric hyperparameters (batch 32 fits the 4090's 24 GB; bump `cls=` to push cavalry-line separation). Run under **`nohup`** so it survives an SSH drop, and tail the log:

```bash
cat > /root/train_v7.py << 'PYEOF'
from ultralytics import YOLO
YOLO('yolo26n.pt').train(                  # NMS-free base; STAL head for small objects
    data='/root/training_data_v7/dataset.yaml',
    epochs=150, imgsz=640, batch=32, device=0, workers=8, patience=20,
    project='/root/runs', name='aoe2_yolo_v6', exist_ok=True,
    flipud=0.0, fliplr=0.5, degrees=10, translate=0.1, scale=0.5,   # isometric-tuned aug
    mosaic=1.0, mixup=0.1, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
)
PYEOF
export PATH=/opt/conda/bin:$PATH
nohup python /root/train_v7.py > /root/train.log 2>&1 &
tail -f /root/train.log     # Ctrl-C to detach; training continues
```

The full **per-class** table (the `fish` row you care about) prints only at the **final** validation, not per-epoch — `results.csv` holds aggregate mAP only. To poll for completion, match the *specific* invocation, **not** the bare script name: `pgrep -f "python /root/train_v7.py"` — a loop that greps `train_v7.py` matches its own command line and never exits.

### D6. Export ONNX (NMS-free)

```bash
export PATH=/opt/conda/bin:$PATH
python -c "from ultralytics import YOLO; YOLO('/root/runs/aoe2_yolo_v6/weights/best.pt').export(format='onnx', opset=12, simplify=True)"
# confirm the layout the detector expects: output0 = [batch, 300, 6]
python -c "import onnx; m=onnx.load('/root/runs/aoe2_yolo_v6/weights/best.onnx'); print([[d.dim_value for d in o.type.tensor_type.shape.dim] for o in m.graph.output])"
```

### D7. Download both artifacts → local models dir

```bash
M=packages/detection/src/inference/models
scp -i ~/.ssh/runpod-aoe2 -P <PORT> root@<IP>:/root/runs/aoe2_yolo_v6/weights/best.pt   $M/aoe2_yolo_v6.pt
scp -i ~/.ssh/runpod-aoe2 -P <PORT> root@<IP>:/root/runs/aoe2_yolo_v6/weights/best.onnx $M/aoe2_yolo_v6.onnx
# verify against the inference runtime (output0 must be [1, 300, 6])
cd packages/detection && uv run python -c "import onnxruntime as ort; s=ort.InferenceSession('src/inference/models/aoe2_yolo_v6.onnx'); print(s.get_outputs()[0].shape)"
```

### D8. Terminate the pod

> RunPod bills by the hour. **Stop/terminate the pod from the dashboard immediately** after the scp completes — a 4090 is ~$0.7/hr.

### Alternative: Lambda Cloud (A100)

Same flow with `ubuntu@<IP>`, key `~/.ssh/lambda-aoe2-training.pem`, and home `/home/ubuntu`. Lambda needs an explicit venv: `python3 -m venv ~/yolo_env && source ~/yolo_env/bin/activate && pip install 'numpy<2' ultralytics` (numpy 2.x breaks PyTorch C-extensions). **Caveat:** Lambda rejects prepaid/most debit cards (incl. Revolut) — it needs a major credit card.

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
3. **Measure real performance — the metric of record.** `test_real_detection` only *counts* boxes; `evaluate_real.py` scores **per-class precision/recall/F1** by IoU-matching against the ground-truth labels in a `training_data_vN/val/` split, and reports **real images separately from synthetic** (a blended mAP over a ~95%-synthetic val set hides real-world performance). Run it after every retrain:
   ```bash
   # real-only, single-pass at the model's TRAINING resolution (the realistic number)
   uv run --project packages/detection python -m detection.testing.evaluate_real \
       --model detection/inference/models/aoe2_yolo_v6.onnx \
       --data detection/training_data_v8 --mode detect_fast --imgsz 640
   # also score synthetic and recommend per-class thresholds in one pass:
   #   ... --split both --conf-sweep
   ```
   > **Inference resolution must match training resolution.** v6 was trained at `imgsz=640`, and its best real F1 is a **single pass at `--imgsz 640`** — *not* SAHI. SAHI tiles the image and runs each tile at a different effective scale than the model trained on; that scale mismatch *lowers* real F1 here. This is why the agent is pinned to `detection_imgsz=640` with `adaptive_sahi=False`. Only re-enable SAHI after retraining at a resolution whose tiles match the SAHI tile scale (the P3.3 decision).
4. **Tune per-class thresholds (optional, post-eval).** `--conf-sweep` writes the best-F1 confidence per class into `recommended_thresholds` in `eval_real_summary.json`. Promote them into `thresholds.py` — the single source the detector and server read — with the sync tool (overlays recommendations on the current values; print-only by default, `--write` to apply):
   ```bash
   uv run python -m detection.inference.sync_thresholds \
       packages/detection/src/training_data_v8/eval_real_summary.json --write
   ```
   The tuned block is a plain literal between `# BEGIN/END GENERATED CLASS_THRESHOLDS` markers, so the change lands as a reviewable git diff.
5. **Deploy the ONNX to the Windows VM** per [windows-vm-agent-bringup.md](./windows-vm-agent-bringup.md) (already points at `aoe2_yolo_v6.onnx`).
6. **Commit** the new model snapshot + any label/data tooling changes (no `Co-Authored-By: Claude` trailer, per project convention).

---

## Iterate (active-learning loop)

After a training run, feed the model's mistakes back in: re-run `active_learning prepare` / `hard_negatives` (Phase B1) on the new model, label that batch on cvat.ai, re-merge (Phase C), retrain (Phase D). Each loop targets the weakest classes — currently the cavalry lines and long-tail unique units.

## Cost / time

| Phase | Time | Cost |
|-------|------|------|
| Synthetic gen + merge (local) | minutes | $0 |
| Annotation (cvat.ai) | hours (manual) | free tier |
| RunPod RTX 4090 training (150 epochs) | ~30 min + upload | ~$0.40 |
| *(alt) Lambda A100 training* | *~50–60 min* | *~$1.30* |
| **Per training cycle** | **<1 h GPU** | **~$0.40 (RunPod)** |

## Troubleshooting

- **Agent still detects nothing / "Using mock detection":** the v6 files aren't where `get_detector()` looks — confirm `models/aoe2_yolo_v6.onnx` (or `.pt`) exists; ONNX is preferred.
- **`fish` (or any naval class) mAP collapses:** the synthetic generator placed it on land. Re-check the scene-separation invariant (Phase A) and that you passed `--water-backgrounds` + `--water-fraction`.
- **`Permission denied (publickey)` on RunPod after it was working:** the pod restarted and wiped `authorized_keys` — re-add your key via the Web Terminal (Phase D1). `ssh -v` shows the offered key's fingerprint if you suspect a key mismatch.
- **`libGL.so.1: cannot open shared object file`:** `apt-get install -y libgl1 libglib2.0-0` on the pod (Phase D3).
- **`python: command not found` over SSH:** PATH missing conda — `export PATH=/opt/conda/bin:$PATH` (Phase D3).
- **Truncated/short tarball on the pod:** you gzipped a media dataset and it timed out mid-archive — use plain `tar -cf` and verify byte counts match (Phase D2).
- **Training finds 0 images:** `dataset.yaml` `path:` isn't absolute (Phase D4).
- **PyTorch import/C-extension errors (Lambda venv):** you skipped `pip install 'numpy<2'`.
- **Lost polygon labels after export:** you exported YOLO 1.1 instead of COCO 1.0 (Phase B6).

## Related

- [Chapter 9: Labeling and Active Learning](../part3-entity-detection/09-labeling-and-active-learning.md)
- [Chapter 12: Cloud Training](../part5-operations/12-cloud-training.md)
- [Chapter 8: Training Pipeline](../part3-entity-detection/08-training-pipeline.md) — rebalancing, loss-gain knobs
- `packages/detection/src/docs/TRAINING_GUIDE.md` — package-level training reference
