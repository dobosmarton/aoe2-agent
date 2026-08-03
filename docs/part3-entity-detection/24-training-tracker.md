# Chapter 24 — Detection Training Tracker

`apps/training-api/` is a small FastAPI + SQLite service that makes the detection dataset **visible and editable in place**. It answers the two questions the CVAT round-trip (Chapter 9) could never answer quickly — *which classes are starved of real examples?* and *what exactly is in this image?* — and it hosts a model-assisted review loop that turns detector predictions into training labels without leaving the browser.

Its frontend lives in the arena dashboard under `/training/*` ([Chapter 19](../part7-arena-web/19-web-architecture.md)); the two backends share nothing but the SPA shell.

<aside class="prereqs">

[Chapter 9 — Labeling and Active Learning](./09-labeling-and-active-learning.md) for the CVAT-based workflow this supplements, and [Chapter 7 — Detector Architecture](./07-detector-architecture.md) for the detector the prelabeler drives. SQL basics; FastAPI dependency injection.

</aside>

## Why it exists

The CVAT loop is *export → annotate → export → convert → merge*. It works, but it is batch-shaped: every question about the dataset costs a round-trip through a third-party tool, and the answer is a snapshot that's stale the moment you generate it. Three specific pains drove this service:

1. **No coverage visibility.** "Do we have any real-labeled `monk`?" required grepping label files. Class starvation was discovered *after* a training run underperformed.
2. **Prelabels were fire-and-forget.** `prelabel.py` wrote YOLO `.txt` files for CVAT import. Once imported, the link between "the model predicted this" and "a human accepted it" was lost — you could not later ask how often the model was already right.
3. **Re-running was destructive.** Regenerating prelabels overwrote label files wholesale, including any hand-corrections that had been merged back.

The tracker fixes all three by making the **annotation a first-class row with provenance**, rather than a line in a text file.

## Architecture

```
packages/detection/src/                    apps/training-api/src/
  real_screenshots/raw/*.png  ──┐
  training_data_v9_slim/        ├──> ingest.py ──> SQLite (logs/training/tracker.db)
    {train,val}/{images,labels} ─┘                      │
  training/config/classes.yaml ────> classes.py         │
                                                        │
  inference/ (detector) ──> prelabel_pending.py ────────┤  writes model/pending boxes
                                                        │
                                          server.py ────┴──> HTTP :8100
                                                              │
                                     apps/dashboard /training/* (Vite proxy)
```

Four entry points, one database:

| Module | Role |
|---|---|
| `ingest.py` | Seeds the DB from the raw-screenshot dir + a YOLO dataset version. Idempotent (upserts by path, rebuilds the version). |
| `prelabel_pending.py` | Runs the detector over unreviewed images, writes predictions as `pending` boxes. Idempotent. |
| `server.py` | The HTTP surface (below). |
| `repository.py` | All SQL. Split into `TrackerReader` / `TrackerWriter` protocols so routes depend on the narrow surface they use and tests inject fakes. |

## Data model

`db.py` defines five tables. The tracked entity is **the real screenshot**:

| Table | Holds |
|---|---|
| `images` | One row per screenshot: `path` (unique), `source` (`real`/`synthetic`), `sha256`, dimensions, optional `capture_meta_json`. |
| `annotations` | One row per box: `image_id`, `class_id`, `geom_type` (`bbox`/`polygon`), `coords_json`, `source` (`model`/`human`), `status` (`pending`/`approved`). |
| `dataset_versions` | A named dataset build (`v9`), with `synth_image_count` + `synth_class_counts_json`. |
| `dataset_images` | Which images were in which version, and their `train`/`val` split. |
| `training_runs` | Forward-compatible stub — pod id, hyperparams, metrics, artifact URLs. Not yet written by anything (see [What's not here](#whats-not-here)). |

Synthetic images are deliberately **not** enumerated as annotation rows. They're a bulk generated input measured in thousands of images; their per-class contribution is summarised as a count on `dataset_versions`. Only real screenshots — the scarce, expensive, human-touched ones — get row-level tracking.

### Geometry is absolute pixels, in exactly one place

`geometry.py` is the whole conversion boundary. The canonical unit is **absolute pixels, top-left origin** — matching the detector's `DetectionResult.bbox` and COCO. YOLO's normalized center-form is treated as an *external representation*: `bbox_from_yolo` and `bbox_to_yolo` are the only two functions that know about it.

This is the direct lesson from Chapter 9's conversion traps. When four formats (COCO absolute corner-form, YOLO normalized center-form, the detector's output, the browser's screen coordinates) meet in one system, the defence is a single canonical unit and one module that owns every conversion into and out of it. Every other module — repository, routes, prelabeler, frontend overlay — speaks only absolute pixels.

## The four states of an annotation

`source × status` is a 2×2, and every cell means something different:

| | `pending` | `approved` |
|---|---|---|
| **`model`** | A prediction awaiting review. Not training data. | **The model was already right.** A human looked and confirmed without changing anything. |
| **`human`** | Rare — a hand-drawn box not yet confirmed. | Ground truth: drawn or corrected by a person. |

`apply_update` (`schemas.py:295`) is what maintains the distinction, and its rule is precise:

- Changing `class_id` or the geometry flips `source` to `"human"` — it's a correction.
- Changing **only** `status` leaves `source` alone — it's a confirmation.

That asymmetry is the entire point. Keeping `model`/`approved` distinguishable from `human`/`approved` means you can later measure how often the detector was right on the first pass, per class — which is precisely the signal that tells you whether to keep labeling a class or move on. Collapsing both into "approved" would destroy it, and the information is unrecoverable afterwards.

<aside class="concept" data-title="Why provenance beats a boolean">

The tempting schema is `annotations.is_verified BOOLEAN`. It's smaller and it satisfies today's requirement ("only train on verified boxes").

It also silently answers *no* to every future question. Was this box drawn by a person or accepted from a model? How much of our ground truth is human-authored versus machine-proposed-and-waved-through? Which classes does the detector already nail, so we can stop paying to label them?

The general principle: **when a record is produced by one actor and blessed by another, store both facts.** A single boolean collapses "who made it" and "is it good" into one bit, and you cannot decompose it later — the information was never written down. The cost of keeping them separate is one extra `TEXT CHECK(...)` column; the cost of merging them is a re-labeling campaign.

This is the same reasoning behind git's separate author and committer fields, and behind HTTP's separation of `201 Created` from `200 OK`.

</aside>

## The prelabel → review loop

```
just training-prelabel --conf 0.25
        │
        │  detector runs over images with no approved box
        ▼
  model/pending boxes ────> /training/images lightbox
                                    │
                    ┌───────────────┼───────────────┬──────────────┐
                    ▼               ▼               ▼              ▼
                 approve        reclassify      re-box          reject
              (status only)   (→ human)      (→ human)      (DELETE)
                    └───────────────┴───────────────┘
                                    ▼
                            training data
```

`prelabel_pending.py` is safe to re-run because of one repository method. `set_model_prelabels` (`repository.py:306`) deletes **only** rows matching `source='model' AND status='pending'` before inserting the fresh prediction set. Approved boxes — model or human — and hand-drawn boxes survive untouched. So a reviewer's work is never clobbered by a later prelabel pass, and you can re-prelabel freely as the model improves.

Defaults live at the top of the module: `--conf 0.25`, `imgsz=1280`, and **SAHI off**. That last one is deliberate and load-bearing: the served weights are full-frame trained, and SAHI's tile scale mismatches the training resolution (the same inference-resolution-must-match-training-resolution finding from the detection eval work). A single full-image pass is the correct call for the current model; override it only if a SAHI-aware model is ever served.

The detector is injected behind a one-method `Detector` protocol, so the conversion logic is tested against a fake and the heavy ONNX runtime is imported only inside `main`.

## Seeding: `ingest.py`

`just training-seed` walks two sources and upserts:

1. `real_screenshots/raw/` → `images` rows (`source='real'`, sha256, dimensions read via PIL).
2. `training_data_v9_slim/{train,val}/labels/*.txt` → `annotations` for real images (converted from YOLO via `bbox_from_yolo`), plus aggregate counts for synthetic images.

Defaults are `_DEFAULT_DATASET_DIRNAME = "training_data_v9_slim"` and `_DEFAULT_VERSION_NAME = "v9"` (`ingest.py:36`). Re-running upserts by path and rebuilds the dataset version, so it never duplicates rows. Real labels are matched back to raw images by **canonical stem** — a `__dupN` suffix is stripped, since the dataset build de-duplicates filenames that the raw dir does not.

## URL contract

Served on **:8100** (the arena API is :8000).

| Method | Path | Returns |
|---|---|---|
| `GET` | `/health` | `{"status": "ok"}` |
| `GET` | `/classes` | `list[ClassDto]` — the 60-class schema, read from `classes.yaml`. |
| `GET` | `/images` | `ImagePageDto`. Filters: `labeled`, `class_id`, `source`; paginated (`page`, `page_size` default 60, max 200). |
| `GET` | `/images/{id}` | `ImageDetailDto` — metadata + every annotation. |
| `POST` | `/images/{id}/annotations` | `AnnotationDto`, **201**. Add a box. |
| `PATCH` | `/annotations/{id}` | `AnnotationDto`. Approve / reclassify / re-box. |
| `DELETE` | `/annotations/{id}` | **204**. Reject or remove. |
| `GET` | `/datasets`, `/datasets/{id}` | Dataset version summary / detail with per-class counts. |
| `GET` | `/stats` | `CoverageDto` — the per-class coverage matrix. |
| `GET` | `/thumbs/{id}` | `image/jpeg`, cached, max edge 320px. |
| `GET` | `/raw/{id}` | Original screenshot bytes. |

`PATCH` bodies are all-optional (`AnnotationUpdate`); omitted fields keep their stored value. `geom_type` and `coords` must be supplied **together** — a coords change without its type is a 422, because interpreting four numbers requires knowing whether they're a bbox or a degenerate polygon.

Unknown `class_id` values are rejected with 422 by `_require_known_class` against the `classes.yaml` catalog, so a typo can't create a box in a class that doesn't exist.

### Two implementation details worth knowing

**One SQLite connection per request** (`server.py:99`). Not a process-wide one. Every read runs through `asyncio.to_thread`, so concurrent requests would drive a single connection from several threadpool threads — and sqlite3's prepared-statement cache is per-connection and *not* thread-safe. Two threads binding parameters into the same cached statement raise `SQLITE_MISUSE` ("bad parameter or other API misuse"). Opening a local SQLite file costs microseconds, so per-request connections are the cheap correct answer.

**Thumbnails are written atomically** (`_ensure_thumbnail`, `server.py:294`). Two concurrent requests for the same uncached image both see "missing" and both render. Without care they'd interleave writes to the same path and a reader could be handed a half-written JPEG. The render goes to a private temp file in the *same directory* (so the swap stays on one filesystem), then `Path.replace` — atomic on POSIX and Windows. The loser of the race just overwrites with a byte-identical render.

## The frontend

Two routes under the `TrainingLayout` shell:

### `/training/coverage`

`panels/training/coverage-stats.tsx` renders `/stats`: total / labeled / unlabeled image tiles, and a per-class row showing real instances, synthetic instances, and labeled-image count. The `zero_real_class_ids` field is the headline — classes with synthetic coverage but **no real examples at all** are exactly where the model will fail on real screenshots.

One caveat when reading the matrix: `real_instances` counts every annotation row for a class, **including `pending` ones**. It measures "boxes proposed or confirmed", not "boxes accepted". A class whose count jumped after a prelabel pass hasn't gained ground truth yet.

### `/training/images`

`panels/training/dataset-table.tsx` — a paginated grid of thumbnails with a labeled/unlabeled filter. Filter and page live in the URL (`?labeled=&page=&image=`), so a review session is linkable and reload-survivable; the route `loader` primes the Query cache before render.

Opening an image (`?image=<id>`) mounts the **lightbox** (`image-lightbox.tsx`), which is where the actual review happens:

- **`use-zoom-pan.ts`** — `translate(x,y) scale(z)` with zoom clamped to 1–12×. Necessary because the entities that matter (a lone sheep, a distant scout) are a handful of pixels at 1080p.
- **`use-box-edit.ts`** — pointer-drag box editing with 8 resize handles plus `move`. `startDrag` calls `stopPropagation` so a box drag never reaches the pan handler; `onCommit` fires once on release, and only if the box actually moved.
- **`box-geometry.ts`** — all the math, with no React and no DOM writes, so it's unit-tested directly (`tests/test_training_geometry.py` covers the Python side of the same convention). Its only app import is a *type*, erased at build.
- **`pending-review.tsx`** — the right rail: approve (✓), reclassify, re-box (pencil), reject (trash) per pending annotation.

Write actions all come from one shared hook, `use-annotation-mutations.ts`, owned by the lightbox and passed to both the rail and the box editor. That's what gives the cache-invalidation contract a single owner: every mutation sweeps the `["tracker", …]` key subtree, refreshing the open image detail, the list's labeled counts, and the coverage stats together. Two call sites, one invalidation rule, no drift.

## Running it

```bash
just training-seed              # 1. populate the DB from disk (idempotent)
just training-prelabel          # 2. optional: seed model/pending boxes
just training-api-dev           # 3. API on :8100
just arena-ui-dev               # 4. dashboard on :5173 → /training
```

Step 2 needs the detection extras installed (ONNX runtime + the served weights); steps 1, 3, 4 do not. `just training-prelabel --conf 0.2 --limit 20` narrows a pass while you're calibrating the threshold.

### Configuration

Every environment read happens in `load_config` (`config.py:46`), so the rest of the package takes a plain `TrackerConfig` value. Paths resolve relative to the repo root, not the process CWD.

| Env var | Default |
|---|---|
| `TRAINING_API_DB` | `logs/training/tracker.db` |
| `TRAINING_API_RAW_IMAGES` | `packages/detection/src/real_screenshots/raw` |
| `TRAINING_API_DATASET_ROOT` | `packages/detection/src` |
| `TRAINING_API_CLASSES_YAML` | `packages/detection/src/training/config/classes.yaml` |
| `TRAINING_API_THUMB_CACHE` | `logs/training/thumbs` |
| `TRAINING_API_CORS_ORIGINS` | `http://localhost:5173,http://localhost:8100` |

## What's not here

- **No dataset export.** The tracker records approved boxes; regenerating `training_data_vN/` from them is still `prepare_training.py`'s job (Chapter 9). Closing that loop is the obvious next step.
- **No training-run tracking.** `training_runs` exists as a table and nothing writes to it. Wiring RunPod job status and metrics into it is the Phase 4 idea.
- **No auth.** Same posture as the arena web surface — local-dev tool.
- **No polygon *editing*.** The schema and API accept polygons; the browser editor only manipulates bboxes. Polygons round-trip unmodified.

## Related reading

- [Chapter 9 — Labeling and Active Learning](./09-labeling-and-active-learning.md) — the CVAT workflow this supplements.
- [Chapter 7 — Detector Architecture](./07-detector-architecture.md) — the detector `prelabel_pending.py` drives.
- [Chapter 13 — Class Schema Evolution](../part5-operations/13-class-schema-evolution.md) — where `classes.yaml` comes from.
- [Chapter 19 — Arena Web Architecture](../part7-arena-web/19-web-architecture.md) — the SPA that hosts `/training/*`.
- [ADR 0006 — TanStack Router + Query and React Aria](../adr/0006-tanstack-router-query-react-aria.md) — the frontend decisions this surface forced.
