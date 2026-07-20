# Phase 1 — Dataset + Annotation Tracker (detailed plan)

> First shippable deliverable of the training-automation work. Answers
> *"what's labeled, what's unlabeled, and what's in dataset vN?"* with a new
> `apps/training-api` service + a "Training" section in the dashboard.
> **No annotation editor, no GPU** — those are Phases 2–4.
>
> Parent: [`TRAINING-AUTOMATION-PLAN.md`](./TRAINING-AUTOMATION-PLAN.md)

---

## 0. Skill alignment & house-style reconciliation

Guided by `/clean-code` + `/python-foundations`, reconciled against the existing
repo conventions (clean-code rule #1: *follow standard conventions; be consistent*).

| `python-foundations` says | Repo convention | Phase 1 choice | Why |
|---|---|---|---|
| Python 3.12+, PEP 695 `type`/`def f[T]` | `requires-python >=3.11` | **3.11**, `TypeAlias` + `Literal`, no PEP 695 | Consistency with 8 existing workspace members. |
| `hatchling` build backend | `setuptools` everywhere | **setuptools** (`package-dir={"training_api":"src"}`) | Matches `apps/api/pyproject.toml`. |
| `mypy --strict` | `basedpyright` (`reportAny=error`) | **basedpyright** (repo-wide config) | One type checker; repo already forbids `Any`. |
| `dataclass(frozen=True, slots=True)` | same | **adopt** | — |
| No `Any`; `TypedDict`/`object`/`Literal` | enforced by ruff `ANN` + pyright | **adopt** | — |
| Pydantic at boundary, dataclass internal | matches `apps/api` DTO style | **adopt** | — |
| `Protocol` over ABC; DI | — | **adopt** (repository is a `Protocol`) | — |
| Hypothesis property tests | `hypothesis` already a dev dep | **adopt** | — |

**Net:** repo *toolchain* (3.11 / setuptools / basedpyright), skill *code style*.

---

## 1. Module layout (clean-code: separate concerns, small units, one thing each)

```
apps/training-api/
├── pyproject.toml                 # package "training-api", CLI "aoe2-training-api"
└── src/
    ├── __init__.py
    ├── py.typed                   # PEP 561 typed-package marker
    ├── __main__.py                # argparse + uvicorn.run (clone of apps/api)
    ├── config.py                  # TrackerConfig: env → frozen dataclass (config at high level)
    ├── geometry.py                # BBox / Polygon value objects + (de)serialization + YOLO↔px
    ├── domain.py                  # frozen domain records (ImageRecord, Annotation, ...)
    ├── schemas.py                 # Pydantic response models (API boundary) + mappers
    ├── db.py                      # SQLite connection factory + schema DDL (idempotent)
    ├── repository.py              # TrackerRepository Protocol + SqliteTrackerRepository
    ├── classes.py                 # ClassCatalog — thin wrapper over detection.class_mapping
    ├── ingest.py                  # pure parsers (disk → records) + persist writer
    └── server.py                  # FastAPI wiring: lifespan, DI provider, thin routes
tests/
├── conftest.py                    # in-memory repo + fixtures
├── test_geometry.py              # property: encode∘decode == identity
├── test_ingest.py                # fixture dir → expected records
├── test_repository.py            # coverage/query correctness on seeded :memory: db
└── test_api.py                   # FastAPI TestClient over the frozen contract
```

Each module does one thing; dependency direction is **downward**:
`server → repository → db`, `server → schemas → domain`, `ingest → domain/geometry`.
Routes never touch SQLite directly (Law of Demeter; repository hides storage).

---

## 2. Domain model (typed value objects, no primitives-as-data)

`domain.py` — internal records are frozen + slotted; IDs are `int | None`
(None before insert). Discriminated geometry union enables exhaustive `match`.

```python
from dataclasses import dataclass
from typing import Literal, TypeAlias

ImageSource: TypeAlias = Literal["real", "synthetic"]
AnnotationSource: TypeAlias = Literal["model", "human"]
AnnotationStatus: TypeAlias = Literal["pending", "approved"]
Split: TypeAlias = Literal["train", "val"]
ClassId: TypeAlias = int

@dataclass(frozen=True, slots=True)
class ImageRecord:
    id: int | None
    path: str
    source: ImageSource
    sha256: str
    width: int
    height: int
    capture_meta: "CaptureMeta | None"

@dataclass(frozen=True, slots=True)
class Annotation:
    id: int | None
    image_id: int
    class_id: ClassId
    geometry: "Geometry"          # BBox | Polygon, from geometry.py
    source: AnnotationSource
    status: AnnotationStatus
```

`geometry.py` — canonical unit is **absolute pixels** (matches `DetectionResult`
bbox and COCO; YOLO-normalized conversion lives *only here* — clean-code:
encapsulate boundary conditions in one place):

```python
@dataclass(frozen=True, slots=True)
class BBox:      # top-left origin, pixels
    x: float; y: float; w: float; h: float

@dataclass(frozen=True, slots=True)
class Polygon:
    points: tuple[tuple[float, float], ...]

Geometry: TypeAlias = BBox | Polygon

def geometry_to_json(geom: Geometry) -> str: ...      # match + explicit shape
def geometry_from_json(kind: str, raw: str) -> Geometry: ...
def yolo_line_to_bbox(cx: float, cy: float, w: float, h: float,
                      img_w: int, img_h: int) -> BBox: ...   # normalized → px
```

`capture_meta` is our own optional tag bag → a `TypedDict`, **not** `dict[str, Any]`:

```python
from typing import TypedDict
class CaptureMeta(TypedDict, total=False):
    age: str            # "dark" | "feudal" | ...
    biome: str
    army_comp: str
    source_run_id: str  # P1.1 gameplay-harvest provenance
```

---

## 3. SQLite layer (`db.py`)

- Schema DDL from `TRAINING-AUTOMATION-PLAN.md` §1a, executed idempotently
  (`CREATE TABLE IF NOT EXISTS`) on lifespan startup.
- Connection factory as a **context manager**; `check_same_thread=False`,
  `PRAGMA foreign_keys=ON`, parameterized queries only (no string interpolation —
  security + clean-code clarity).
- Blocking calls wrapped by `asyncio.to_thread` in routes (mirrors `apps/api`'s
  DuckDB offloading).
- DB path from config (`TRAINING_API_DB`, default `logs/training/tracker.db`).

---

## 4. Repository (`repository.py`) — Protocol + SQLite impl (DI)

Interface is a `Protocol` so routes depend on the contract, tests inject a fake
(python-foundations: Protocol default; clean-code: dependency injection). Filters
are a **value object**, not a pile of boolean/optional args (clean-code: prefer
fewer arguments, no flag args):

```python
@dataclass(frozen=True, slots=True)
class ImageFilter:
    status: AnnotationStatus | None = None
    class_id: ClassId | None = None
    source: ImageSource | None = None
    page: int = 0
    page_size: int = 60

class TrackerRepository(Protocol):
    def list_images(self, filt: ImageFilter) -> list[ImageRecord]: ...
    def get_image(self, image_id: int) -> ImageRecord | None: ...
    def annotations_for(self, image_id: int) -> list[Annotation]: ...
    def list_datasets(self) -> list[DatasetSummary]: ...
    def get_dataset(self, dataset_id: int) -> DatasetDetail | None: ...
    def coverage(self) -> CoverageReport: ...
```

`CoverageReport` / `DatasetSummary` are frozen dataclasses carrying per-class
labeled counts (real vs synthetic) and the flagged **zero-real-label classes**
(the 11 rare classes from IMPROVEMENT-PLAN P1.2).

---

## 5. Classes (`classes.py`) — single source of truth, no duplication

Wrap the existing loader; never hardcode the 60 names (clean-code: no needless
repetition; DRY the schema):

```python
from detection.labeling.class_mapping import load_classes_yaml

@dataclass(frozen=True, slots=True)
class ClassInfo:
    id: ClassId
    name: str
    group: str

class ClassCatalog:
    """Loads classes.yaml once; the authoritative 60-class list."""
    def __init__(self, classes_yaml_path: Path) -> None: ...
    def all(self) -> tuple[ClassInfo, ...]: ...
    def name_of(self, class_id: ClassId) -> str: ...
```

(Confirm the exact import path of `load_classes_yaml` during implementation —
exploration placed it in `detection…class_mapping`.)

---

## 6. Ingest (`ingest.py`) — pure parsers + explicit writer

Separate **pure computation** from **IO** (functional style). Parsers return
records; a single writer persists them idempotently.

- `scan_raw_images(raw_dir: Path) -> list[ImageRecord]` — glob real screenshots,
  compute sha256 + dims (Pillow), `source="real"`.
- `parse_yolo_labels(labels_dir, images_dir, catalog) -> list[Annotation]` —
  read `*.txt`, map normalized → `BBox` px via `geometry.yolo_line_to_bbox`,
  strip `real_` prefix to reconnect to raw stems.
- `seed_dataset_v9(repo, dataset_root, catalog) -> None` — register the v9
  `dataset_versions` row + `dataset_images` membership from `{train,val}/images`,
  register synthetic image rows, attach annotations.
- **Idempotency:** upsert `images` by `sha256`/`path`; re-running never duplicates
  (clean-code: repeatable). Ingest is invoked by a `just` recipe / `--seed` flag on
  first boot, logged with counts.

---

## 7. API contract (`server.py` + `schemas.py`)

Routes are thin: validate → call repository → map domain → Pydantic response.
Response models frozen (`ConfigDict(frozen=True)`), built by explicit
`schemas.image_to_dto(record)` mappers (boundary conversion pattern).

| Method | Path | Response model | Notes |
|---|---|---|---|
| GET | `/health` | `HealthDto` | `{status:"ok"}` |
| GET | `/classes` | `list[ClassDto]` | 60 classes from `ClassCatalog` |
| GET | `/images` | `ImagePageDto` | query params → `ImageFilter`; paged + `thumb_url` |
| GET | `/images/{id}` | `ImageDetailDto` | meta + its annotations |
| GET | `/datasets` | `list[DatasetSummaryDto]` | per-class real/synth counts |
| GET | `/datasets/{id}` | `DatasetDetailDto` | membership + split stats |
| GET | `/stats` | `CoverageDto` | coverage matrix; zero-real flags |
| GET | `/thumbs/{id}` | image bytes | cached JPEG under scratch dir |
| GET | `/raw/{id}` | image bytes | original |

FastAPI query params bound to an `ImageFilter` via a dependency (avoids a wide
handler signature). CORS via `TRAINING_API_CORS_ORIGINS`. Unknown ids → 404 with
a typed `NotFoundError` (skill's exception hierarchy), not a bare raise.

---

## 8. Config (`config.py`) — env at the top level

```python
@dataclass(frozen=True, slots=True)
class TrackerConfig:
    db_path: Path
    raw_images_dir: Path
    dataset_root: Path          # packages/detection/src
    classes_yaml: Path
    thumb_cache_dir: Path
    cors_origins: tuple[str, ...]

def load_config(env: Mapping[str, str]) -> TrackerConfig: ...  # pure; testable
```

Pure `load_config(env)` (no global `os.environ` read inside) → trivially unit-testable
(clean-code: no hidden dependencies; pure functions).

---

## 9. Dashboard (`apps/dashboard/src/panels/training/`)

- `training-view.tsx` — section shell (sidebar Arena⇄Training toggle in `App.tsx`).
- `dataset-table.tsx` — thumbnail grid: source badge, labeled/unlabeled, class chips;
  filters (status/class/source). shadcn `Table`/`Badge`/`Card`, `lucide-react`.
- `coverage-stats.tsx` — per-class coverage bars (Recharts, existing dep); highlight
  zero-real classes.
- `src/hooks/use-datasets.ts`, `use-images.ts` — data hooks mirroring `use-runs.ts`.
- `src/lib/training-api.ts` — typed client; TS `strict`; response types mirror the
  Pydantic DTOs (hand-kept in sync — small surface).

---

## 10. Testing (python-foundations: pytest + Hypothesis; clean-code: fast, independent)

- `test_geometry.py` — **property**: `@given` random bboxes/polygons →
  `geometry_from_json(*split(geometry_to_json(g))) == g`; and YOLO→px→YOLO
  roundtrip within float tolerance.
- `test_ingest.py` — a fixture dir (3–4 images + label `.txt`s) → assert exact
  `ImageRecord`/`Annotation` sets; assert idempotency (run twice, same rows).
- `test_repository.py` — seed a `:memory:` SQLite via the real DDL; assert
  `list_images(ImageFilter(status="pending"))`, `coverage()` counts, zero-real flags.
- `test_api.py` — FastAPI `TestClient` with an injected fake repo; assert the frozen
  contract (status codes, response shape, 404 on missing id).
- One assert-per-concept, in-memory, no network → repeatable & fast.

---

## 11. Ordered task checklist

1. [ ] Scaffold `apps/training-api` (pyproject, `__main__`, empty `server.py`,
       `py.typed`); register in root `[tool.uv.sources]`; `uv sync`; `/health` green.
2. [ ] `config.py` + `load_config` + unit test.
3. [ ] `geometry.py` + property tests (do this early — everything depends on it).
4. [ ] `domain.py`, `schemas.py` + mappers.
5. [ ] `db.py` DDL + connection factory.
6. [ ] `repository.py` Protocol + SQLite impl + `test_repository.py`.
7. [ ] `classes.py` over `load_classes_yaml`.
8. [ ] `ingest.py` parsers + writer + `test_ingest.py`; wire a `just training-seed` recipe.
9. [ ] Routes in `server.py` + `test_api.py`.
10. [ ] `justfile` `training-api-dev`; `vite.config.ts` proxy routes.
11. [ ] Dashboard: view shell + `dataset-table` + `coverage-stats` + hooks/client.
12. [ ] `ruff check`, `ruff format`, `basedpyright` all clean; `pytest apps/training-api`.

---

## 12. Acceptance criteria

- `just training-api-dev` serves `:8100`; `GET /health` ok.
- After `just training-seed`: `GET /stats` returns **60 classes**, ~**220** real
  images, and a **v9** dataset with membership; per-class labeled counts match the
  on-disk `training_data_v9` label files on a spot-check of ≥5 stems.
- Dashboard Training tab lists every raw image with labeled/unlabeled + per-class
  coverage; zero-real classes are visibly flagged.
- Full quality gate green: `ruff` (incl. `ANN`), `basedpyright` (no `Any`), and
  `pytest` (unit + property + api) pass. Zero `Any` in the new package.

---

## 13. Explicitly out of scope (later phases)

Annotation editor & `PUT /annotations` (Phase 2) · dataset build/export & MinIO
(Phase 3) · RunPod driver & trainer image (Phase 4). Phase 1 is **read-only over
existing on-disk data** — it introduces no write path to annotations yet, which
keeps it small, safe, and independently shippable.
