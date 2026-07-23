"""FastAPI service for the detection training tracker.

URL contract:
  GET    /health                    -> {"status": "ok"}
  GET    /classes                   -> list[ClassDto]        (the 60-class schema)
  GET    /images                    -> ImagePageDto          (filter: labeled/class_id/source)
  GET    /images/{id}               -> ImageDetailDto        (meta + annotations)
  POST   /images/{id}/annotations   -> AnnotationDto         (add a box; 201)
  PATCH  /annotations/{id}          -> AnnotationDto         (approve / reclassify / re-box)
  DELETE /annotations/{id}          -> 204                   (reject / remove a box)
  GET    /datasets                  -> list[DatasetSummaryDto]
  GET    /datasets/{id}             -> DatasetDetailDto
  GET    /stats                     -> CoverageDto           (per-class coverage matrix)
  GET    /thumbs/{id}               -> image/jpeg            (cached thumbnail)
  GET    /raw/{id}                  -> image bytes           (original screenshot)

The write routes back the prelabel-review loop: a batch prelabeler seeds
`model`/`pending` boxes (see `prelabel_pending.py`), and a reviewer approves,
corrects, or rejects them through PATCH/DELETE.

Config comes from the environment via `load_config`; the lifespan installer is the
single writer to `app.state`, so dependency getters `cast` at that boundary.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from contextlib import asynccontextmanager, closing
from pathlib import Path
from typing import TYPE_CHECKING, cast

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image

from .classes import ClassCatalog
from .config import TrackerConfig, load_config
from .db import connect, init_schema
from .domain import Annotation, ImageFilter, ImageSource
from .repository import SqliteTrackerRepository, TrackerReader, TrackerWriter
from .schemas import (
    AnnotationCreate,
    AnnotationDto,
    AnnotationUpdate,
    ClassDto,
    CoverageDto,
    DatasetDetailDto,
    DatasetSummaryDto,
    ImageDetailDto,
    ImagePageDto,
    annotation_to_dto,
    apply_update,
    class_to_dto,
    coverage_to_dto,
    create_to_annotation,
    dataset_detail_to_dto,
    detail_to_dto,
    page_to_dto,
    summary_to_dto,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

_THUMB_MAX_EDGE = 320


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    config = load_config(os.environ)
    catalog = ClassCatalog(config.classes_yaml)
    # Create the schema once at boot, then drop the connection: request handlers
    # each open their own (see `get_repo`). Nothing process-wide holds a cursor.
    with closing(connect(config.db_path)) as conn:
        init_schema(conn)
    app.state.config = config
    app.state.catalog = catalog
    yield


app = FastAPI(title="AoE2 Detection Training Tracker", lifespan=lifespan)


def _cors_origins() -> list[str]:
    return list(load_config(os.environ).cors_origins)


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["*"],
)


def get_repo(request: Request) -> Iterator[TrackerReader]:
    """Open a repository scoped to this request, closed when the response is done.

    A single process-wide connection is not an option: every route runs its reads
    through `asyncio.to_thread`, so concurrent requests would drive one connection
    from several threadpool threads. sqlite3's prepared-statement cache is shared
    per connection and not thread-safe, so two threads binding parameters into the
    same cached statement raise `SQLITE_MISUSE` ("bad parameter or other API
    misuse"). One connection per request keeps access strictly serialised; opening
    a local SQLite file costs microseconds.
    """
    state = cast("FastAPI", request.app).state
    config = cast("TrackerConfig", state.config)
    catalog = cast("ClassCatalog", state.catalog)
    with closing(connect(config.db_path)) as conn:
        yield SqliteTrackerRepository(conn, catalog)


def get_writer(request: Request) -> Iterator[TrackerWriter]:
    """A request-scoped repository typed to its write surface.

    Same one-connection-per-request rule as `get_repo` (see its docstring); the
    concrete `SqliteTrackerRepository` satisfies both protocols, so this differs
    only in the type the write routes see."""
    state = cast("FastAPI", request.app).state
    config = cast("TrackerConfig", state.config)
    catalog = cast("ClassCatalog", state.catalog)
    with closing(connect(config.db_path)) as conn:
        yield SqliteTrackerRepository(conn, catalog)


def get_catalog(request: Request) -> ClassCatalog:
    return cast("ClassCatalog", cast("FastAPI", request.app).state.catalog)


def _require_known_class(catalog: ClassCatalog, class_id: int) -> None:
    if not catalog.has(class_id):
        raise HTTPException(status_code=422, detail=f"unknown class_id {class_id}")


def get_config(request: Request) -> TrackerConfig:
    return cast("TrackerConfig", cast("FastAPI", request.app).state.config)


def image_filter(
    labeled: bool | None = Query(default=None),
    class_id: int | None = Query(default=None),
    source: ImageSource | None = Query(default=None),
    page: int = Query(default=0, ge=0),
    page_size: int = Query(default=60, ge=1, le=200),
) -> ImageFilter:
    return ImageFilter(
        labeled=labeled,
        class_id=class_id,
        source=source,
        page=page,
        page_size=page_size,
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/classes")
async def classes(catalog: ClassCatalog = Depends(get_catalog)) -> list[ClassDto]:
    return [class_to_dto(info) for info in catalog.all()]


@app.get("/images")
async def images(
    filt: ImageFilter = Depends(image_filter),
    repo: TrackerReader = Depends(get_repo),
) -> ImagePageDto:
    page = await asyncio.to_thread(repo.list_images, filt)
    return page_to_dto(page)


@app.get("/images/{image_id}")
async def image_detail(
    image_id: int,
    repo: TrackerReader = Depends(get_repo),
    catalog: ClassCatalog = Depends(get_catalog),
) -> ImageDetailDto:
    detail = await asyncio.to_thread(repo.get_image, image_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"image {image_id} not found")
    return detail_to_dto(detail, catalog)


@app.get("/datasets")
async def datasets(repo: TrackerReader = Depends(get_repo)) -> list[DatasetSummaryDto]:
    summaries = await asyncio.to_thread(repo.list_datasets)
    return [summary_to_dto(s) for s in summaries]


@app.get("/datasets/{dataset_id}")
async def dataset_detail(
    dataset_id: int, repo: TrackerReader = Depends(get_repo)
) -> DatasetDetailDto:
    detail = await asyncio.to_thread(repo.get_dataset, dataset_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"dataset {dataset_id} not found")
    return dataset_detail_to_dto(detail)


@app.get("/stats")
async def stats(repo: TrackerReader = Depends(get_repo)) -> CoverageDto:
    report = await asyncio.to_thread(repo.coverage)
    return coverage_to_dto(report)


@app.post("/images/{image_id}/annotations", status_code=201)
async def create_annotation(
    image_id: int,
    body: AnnotationCreate,
    repo: TrackerWriter = Depends(get_writer),
    catalog: ClassCatalog = Depends(get_catalog),
) -> AnnotationDto:
    _require_known_class(catalog, body.class_id)
    created = await asyncio.to_thread(_create_annotation, repo, image_id, body)
    return annotation_to_dto(created, catalog)


@app.patch("/annotations/{annotation_id}")
async def update_annotation(
    annotation_id: int,
    body: AnnotationUpdate,
    repo: TrackerWriter = Depends(get_writer),
    catalog: ClassCatalog = Depends(get_catalog),
) -> AnnotationDto:
    if body.class_id is not None:
        _require_known_class(catalog, body.class_id)
    updated = await asyncio.to_thread(_update_annotation, repo, annotation_id, body)
    return annotation_to_dto(updated, catalog)


@app.delete("/annotations/{annotation_id}", status_code=204)
async def delete_annotation(
    annotation_id: int, repo: TrackerWriter = Depends(get_writer)
) -> Response:
    deleted = await asyncio.to_thread(repo.delete_annotation, annotation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"annotation {annotation_id} not found")
    return Response(status_code=204)


def _create_annotation(repo: TrackerWriter, image_id: int, body: AnnotationCreate) -> Annotation:
    if repo.get_image(image_id) is None:
        raise HTTPException(status_code=404, detail=f"image {image_id} not found")
    try:
        annotation = create_to_annotation(image_id, body)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return repo.add_annotation(annotation)


def _update_annotation(
    repo: TrackerWriter, annotation_id: int, body: AnnotationUpdate
) -> Annotation:
    existing = repo.get_annotation(annotation_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"annotation {annotation_id} not found")
    try:
        merged = apply_update(existing, body)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    repo.update_annotation(merged)
    return merged


@app.get("/raw/{image_id}")
async def raw_image(image_id: int, repo: TrackerReader = Depends(get_repo)) -> FileResponse:
    path = await asyncio.to_thread(_image_path, repo, image_id)
    return FileResponse(path)


@app.get("/thumbs/{image_id}")
async def thumbnail(
    image_id: int,
    repo: TrackerReader = Depends(get_repo),
    config: TrackerConfig = Depends(get_config),
) -> FileResponse:
    thumb_path = await asyncio.to_thread(_ensure_thumbnail, repo, config, image_id)
    return FileResponse(thumb_path, media_type="image/jpeg")


def _image_path(repo: TrackerReader, image_id: int) -> Path:
    detail = repo.get_image(image_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"image {image_id} not found")
    return Path(detail.record.path)


def _ensure_thumbnail(repo: TrackerReader, config: TrackerConfig, image_id: int) -> Path:
    """Return the cached thumbnail path, rendering it first if absent.

    The render is written to a private temp file and then `os.replace`d into
    place. Two concurrent requests for the same image both see "missing" and
    both render — without the atomic swap they would interleave writes to the
    same path and a reader could be handed a half-written JPEG. `os.replace` is
    atomic on POSIX and Windows, so a reader sees either the old file or the
    complete new one, never a partial. The loser of the race simply overwrites
    with a byte-identical render.
    """
    source_path = _image_path(repo, image_id)
    config.thumb_cache_dir.mkdir(parents=True, exist_ok=True)
    thumb_path = config.thumb_cache_dir / f"{image_id}.jpg"
    if thumb_path.exists():
        return thumb_path

    # Same directory as the target so the replace stays on one filesystem.
    fd, tmp_name = tempfile.mkstemp(
        dir=config.thumb_cache_dir, prefix=f".{image_id}.", suffix=".jpg.tmp"
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with Image.open(source_path) as image:
            rgb = image.convert("RGB")
            rgb.thumbnail((_THUMB_MAX_EDGE, _THUMB_MAX_EDGE))
            rgb.save(tmp_path, format="JPEG", quality=80)
        tmp_path.replace(thumb_path)  # atomic swap; wraps os.replace
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return thumb_path
