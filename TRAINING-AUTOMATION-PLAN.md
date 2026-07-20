# Training Workflow Automation — Implementation Plan

> Extend the dashboard to internalize the AoE2 detection retraining loop:
> in-app annotation editor (replaces CVAT) + MinIO-brokered, one-click RunPod
> training (replaces the manual SSH checklist). Reuses the existing
> `packages/detection` modules — only the orchestration + editor UI are new.

**Status:** planned · **Owner:** —  · **Created:** 2026-07-19

---

## 0. Decisions (locked)

| Question | Decision |
|---|---|
| Annotation UI | **Full in-dashboard editor** (bbox + polygon), replaces hosted CVAT |
| RunPod automation | **Custom Docker image + MinIO** artifact bus; no SSH |
| Backend home | **New `apps/training-api`** (FastAPI, :8100); `apps/api` untouched |
| First milestone | **Dataset + annotation tracker** (visibility first) |

Reuse-don't-rebuild targets (import, do not fork):
`labeling/prepare_training.py`, `scripts/pack_dataset_for_runpod.py`,
`training/{train_yolo.py,export_onnx.py}`, `testing/evaluate_real.py`,
`inference/detector.py`, `training/config/classes.yaml`
(+ `class_mapping.load_classes_yaml`).

Strict-typing house rules (root `pyproject.toml`): ruff `ANN` + basedpyright
`reportAny=error`/`reportExplicitAny=error`. All new Python fully annotated,
no bare `Any`; all new TS `strict`.

---

## Phase 0 — Scaffolding (prereq)

**Goal:** a runnable empty `apps/training-api` wired into the workspace + a "Training"
tab shell in the dashboard.

### Tasks
- [ ] `apps/training-api/pyproject.toml` — package `training-api`, CLI `aoe2-training-api`.
      Copy `apps/api/pyproject.toml`; deps: `detection`, `fastapi>=0.115`, `uvicorn>=0.30`,
      `pydantic>=2.5`, `boto3` (MinIO S3), `runpod`. Optionally `sqlmodel` (else stdlib `sqlite3`).
      `[tool.setuptools] package-dir = {"training_api" = "src"}`.
- [ ] `apps/training-api/src/__main__.py` — argparse (`--host`, `--port` default **8100**),
      `uvicorn.run("training_api.server:app", ...)`. Clone of `apps/api/src/__main__.py`.
- [ ] `apps/training-api/src/server.py` — FastAPI app + `lifespan` (opens SQLite, builds MinIO
      client into `app.state`), `CORSMiddleware` via a `TRAINING_API_CORS_ORIGINS` env, `GET /health`.
- [ ] Root `pyproject.toml` → add `training-api = { workspace = true }` under `[tool.uv.sources]`.
- [ ] `justfile` → `training-api-dev: uv run --package training-api aoe2-training-api --port 8100`.
- [ ] `apps/dashboard/vite.config.ts` → add proxy: `/datasets`, `/images`, `/annotations`,
      `/classes`, `/train`, `/models` → `http://localhost:8100`.
- [ ] `apps/dashboard/src/App.tsx` → add a sidebar toggle **Arena ⇄ Training**; new
      `src/panels/training/training-view.tsx` placeholder.
- [ ] `uv sync` succeeds; `just training-api-dev` serves `GET /health` → `{status:"ok"}`.

**Acceptance:** both servers run (`:8000` arena, `:8100` training); dashboard shows an empty
Training tab; `curl localhost:8100/health` returns ok.

---

## Phase 1 — Dataset + annotation tracker (SHIP FIRST, standalone value)

**Goal:** answer *"what's labeled, what's unlabeled, what's in dataset vN?"* — no editor, no GPU.

### 1a. SQLite schema — `apps/training-api/src/db.py`
```sql
CREATE TABLE images (
  id INTEGER PRIMARY KEY,
  path TEXT NOT NULL UNIQUE,
  source TEXT NOT NULL CHECK(source IN ('real','synthetic')),
  sha256 TEXT NOT NULL,
  width INTEGER NOT NULL,
  height INTEGER NOT NULL,
  capture_meta_json TEXT,             -- age/biome/army tags (P0.2/P1.1 harvest)
  created_at TEXT NOT NULL
);
CREATE TABLE annotations (
  id INTEGER PRIMARY KEY,
  image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
  class_id INTEGER NOT NULL,          -- 0..59 per classes.yaml
  geom_type TEXT NOT NULL CHECK(geom_type IN ('bbox','polygon')),
  coords_json TEXT NOT NULL,          -- bbox:[x,y,w,h] px | polygon:[[x,y],...] px
  source TEXT NOT NULL CHECK(source IN ('model','human')),
  status TEXT NOT NULL CHECK(status IN ('pending','approved')),
  created_at TEXT NOT NULL
);
CREATE TABLE dataset_versions (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,          -- 'v9','v10',...
  config_json TEXT,                   -- val_split, oversample, include_synthetic
  minio_key TEXT,                     -- slim tarball object key (Phase 3)
  created_at TEXT NOT NULL,
  notes TEXT
);
CREATE TABLE dataset_images (
  dataset_version_id INTEGER NOT NULL REFERENCES dataset_versions(id),
  image_id INTEGER NOT NULL REFERENCES images(id),
  split TEXT NOT NULL CHECK(split IN ('train','val')),
  PRIMARY KEY (dataset_version_id, image_id)
);
CREATE TABLE training_runs (            -- stub now; populated in Phase 4
  id INTEGER PRIMARY KEY,
  dataset_version_id INTEGER REFERENCES dataset_versions(id),
  runpod_pod_id TEXT,
  status TEXT NOT NULL,               -- provisioning|training|done|failed|terminated
  hyperparams_json TEXT,
  metrics_json TEXT,                  -- evaluate_real.py summary
  artifact_urls_json TEXT,
  created_at TEXT NOT NULL
);
```
DB file under a configurable dir (`TRAINING_API_DB`, default `logs/training/tracker.db`).

### 1b. Ingest / seed — `apps/training-api/src/ingest.py`
- [ ] Scan `packages/detection/src/real_screenshots/raw/*.{png,jpg}` → `images(source='real')`
      (compute sha256 + dims via Pillow).
- [ ] Parse `training_data_v9/{train,val}/labels/*.txt`; strip `real_` prefix to map back to raw
      stems; insert `annotations(source='model'|'human', status='approved')` + a
      `dataset_versions('v9')` row with `dataset_images` membership + split.
- [ ] Classes: `from detection…class_mapping import load_classes_yaml` — **never** hardcode the list.
- [ ] Idempotent: re-run upserts by `path`/sha, no duplicates.

### 1c. API — `server.py`
| Method | Path | Returns |
|---|---|---|
| GET | `/classes` | 60-class list `[{id,name,group}]` from `classes.yaml` |
| GET | `/images?status=&class_id=&source=&page=` | paged images + `thumb_url` |
| GET | `/images/{id}` | image meta + its annotations |
| GET | `/datasets` | versions + per-class real/synth counts |
| GET | `/datasets/{id}` | membership + `merge_summary`-shaped stats |
| GET | `/stats` | coverage matrix; flag zero-real-label classes (P1.2's 11) |
| GET | `/thumbs/{id}` / `/raw/{id}` | JPEG (thumbs cached under scratch dir) |

DTOs frozen (`pydantic ConfigDict(frozen=True)` or `@dataclass(slots=True)`) per `apps/api` style.

### 1d. Dashboard — `apps/dashboard/src/panels/training/`
- [ ] `dataset-table.tsx` — thumbnail grid/table; source badge; labeled/unlabeled; class chips;
      filters (status/class/source). shadcn `Table`/`Badge`/`Card`, `lucide-react`.
- [ ] `coverage-stats.tsx` — per-class coverage bars (Recharts); highlight zero-real classes.
- [ ] `src/hooks/use-datasets.ts` + `src/lib/api.ts` client methods (mirror `use-runs.ts`).

**Acceptance (Phase 1 exit):**
- `GET /stats` returns 60 classes, ~220 raw images, v9 membership.
- Dashboard Training tab shows every raw image with labeled/unlabeled + per-class coverage;
  counts match on-disk `training_data_v9` labels on a spot-check of ≥5 stems.
- `pytest apps/training-api` covers `ingest.py` against a small fixture dir.

---

## Phase 2 — In-dashboard annotation editor (largest scope)

**Goal:** create/correct bbox **and** polygon annotations in the browser; replaces CVAT.

### 2a. Canvas — `apps/dashboard/src/panels/training/editor/`
- [ ] Add **`react-konva`** + `konva` to `apps/dashboard/package.json` (self-contained, Vite-friendly).
- [ ] `annotation-canvas.tsx` — Konva `Stage` with image layer + shapes layer:
      - bbox: click-drag to draw; select/move/resize (Transformer).
      - polygon: click to add vertices, close on first-vertex click; drag vertices; insert/delete vertex.
      - delete selected; **undo/redo** via a local reducer (action stack).
      - zoom (wheel) + pan (space-drag); fit-to-view.
- [ ] `class-picker.tsx` — 60 classes from `/classes`; number/letter **hotkeys**; active-class color.
- [ ] `image-queue.tsx` — next/prev, "jump to next unlabeled", progress `n/total`.

### 2b. Model-assisted prelabel
- [ ] `POST /images/{id}/prelabel` — training-api runs current model **in-process** via
      `detection.inference.detector` (fallback: HTTP to a running detection-server `/detect`).
      Map `DetectionResult.bbox` (orig px, x1y1x2y2) + `class_name` → editor boxes
      `source='model', status='pending'`. (Open-vocab bootstrap via `prelabel.py --open-vocab` = later.)

### 2c. Persist
- [ ] `PUT /images/{id}/annotations` — replace-all for the image; normalized geometry → SQLite;
      mark `status='approved'` on human save. `DELETE /annotations/{id}` for single removes.
- [ ] Editor writes the same DB Phase 1 reads → coverage view updates live.

**Acceptance:**
- Prelabel a real screenshot; boxes match a direct detection-server `/detect` on that image.
- Draw a polygon + a bbox, assign classes via hotkeys, save, reload → geometry round-trips.
- Labeling an image flips it to "labeled" in the Phase 1 coverage table without refresh gymnastics.

---

## Phase 3 — Dataset build / export

**Goal:** turn approved annotations into a `training_data_vN/` + a MinIO upload artifact, reusing
`prepare_training.py` **unchanged**.

### Tasks
- [ ] `POST /datasets/build {name, val_split, oversample_real, include_synthetic}`:
      1. Export approved human annotations from SQLite → a **COCO export dir** mimicking CVAT
         (`annotations/instances_default.json` + `images/`). Categories by name from `classes.yaml`.
      2. Call `prepare_training.prepare_training(cvat_export_dirs=[dir], output_dir, synthetic_dir,
         images_dir, val_split, include_synthetic, oversample_real)` → `training_data_vN/`.
      3. Record `dataset_versions` row + `dataset_images` membership from the produced split.
- [ ] `POST /datasets/{id}/pack` → run `pack_dataset_for_runpod.py` (~1GB slim) → upload
      `datasets/vN.tar` to MinIO; store `minio_key`.
- [ ] Dashboard "Build vN" form: the three knobs + a **dry-run** preview
      (reuse `prepare_training(dry_run=True)` → show `merge_summary.json`, zero-real warnings).

**Acceptance:**
- Build a throwaway `v_test` from a few approved images; the resulting `dataset.yaml` +
  `merge_summary.json` + label files are **byte-identical** to a manual `prepare_training.py` run
  on the same COCO dir → proves the editor is a faithful CVAT replacement.
- `datasets/v_test.tar` appears in MinIO and re-downloads intact.

---

## Phase 4 — RunPod one-click training (custom image + MinIO)

**Goal:** dataset vN → trained model + eval, no SSH, pod self-terminates.

### 4a. Trainer image — `docker/aoe2-trainer/`
- [ ] `Dockerfile` FROM CUDA/PyTorch base; bake `ultralytics` + deps + `libgl1 libglib2.0-0`
      (kills per-run apt) + `train_yolo.py` + `evaluate_real.py` + `boto3`/`mc`.
- [ ] `entrypoint.sh` (env: MinIO creds/endpoint, `DATASET_KEY`, `MODEL_NAME`, `EPOCHS/BATCH/IMGSZ`):
      1. pull `datasets/vN.tar` from MinIO, extract, rewrite `dataset.yaml` `path:` → absolute pod path.
      2. `python train_yolo.py --data … --name aoe2_yolo_vN --export-onnx --epochs … --batch … --imgsz …`.
      3. push `best.pt`, `best.onnx`, `results.csv` → MinIO `models/aoe2_yolo_vN/`.
      4. write `models/aoe2_yolo_vN/DONE` sentinel; exit (pod terminates).
- [ ] Build + push once to a registry (GHCR/Docker Hub); store the tag in training-api config.

### 4b. Driver — `apps/training-api/src/runpod_driver.py`
- [ ] `POST /train {dataset_version, model_base, epochs, batch, imgsz}` →
      `runpod.create_pod(image=aoe2-trainer, gpu='NVIDIA GeForce RTX 4090', env={...})`;
      insert `training_runs(status='provisioning')`.
- [ ] `GET /train/{run_id}` — pod status (runpod SDK) + MinIO `DONE` presence.
- [ ] On done: download `best.onnx`/`best.pt` → `packages/detection/src/inference/models/aoe2_yolo_vN.*`;
      **auto-run `evaluate_real.py`** on the vN val split; store metrics; `runpod.terminate_pod`.
- [ ] Secrets: add `RUNPOD_API_KEY` to `.env` + `.env.example`; reuse `MINIO_ROOT_USER/PASSWORD`.

### 4c. Dashboard — Train panel
- [ ] Pick dataset vN + hyperparams; launch; live status + `results.csv` curve (Recharts);
      post-train per-class real-F1 table (`evaluate_real`); **Promote** button → sets the served
      model pointer (`AOE2_DETECTION_MODEL` consumed by `apps/agent/src/config.py`).

**Acceptance:**
- **Local dry run first:** `docker run` the trainer against MinIO + a 5-epoch tiny dataset; model
  lands in MinIO, `DONE` written, container exits.
- Then one real RunPod run end-to-end: model in `inference/models/`, `evaluate_real.py` numbers in
  range (v9 real micro-F1 ≈ 0.67 baseline), pod verified terminated (RunPod dashboard + billing).

---

## Sequencing & dependencies

```
Phase 0 ─► Phase 1 (ship) ─► Phase 2 ─► Phase 3 ─► Phase 4
                    │                        └─ needs MinIO reachable from pod
                    └─ standalone value; can pause here
```
- Phase 1 delivers on its own (the "keep track" pain) — everything after is incremental.
- Phase 2 is the long pole (polygon edit + undo/redo + zoom); react-konva de-risks it.
- Phase 4 gate: prove MinIO↔pod connectivity in the local dry run before wiring RunPod.

## Cross-cutting

- **New env:** `RUNPOD_API_KEY`, `TRAINING_API_DB`, `TRAINING_API_CORS_ORIGINS`, `MINIO_ENDPOINT`
  (+ existing `MINIO_ROOT_USER/PASSWORD`). Document in `.env.example`.
- **Docs:** once shipped, add a `docs/runbooks/` page and retire the manual CVAT/SSH runbook
  (`retrain-detection-v6.md`) — and mirror to `aoe2-llm-wiki` per the doc-sync convention.
- **CVAT stays usable** during the transition; no hard cutover until Phase 2/3 are trusted.

## Risks

- **Editor scope** is the bulk of the effort — Phase 1 ships value first so Phase 2 can be paced.
- **In-process inference** pulls the model into the API process (memory); fine for a single-user
  dev tool, else call detection-server over HTTP.
- **MinIO reachability from RunPod** — expose it (tunnel/public) or use RunPod's own S3-compatible
  network volume as the bus. Confirm in the Phase 4 dry run.
- **Legacy 46-class `training_data/dataset.yaml`** exists — always resolve classes from the 60-class
  `classes.yaml`, never the legacy file.
```
