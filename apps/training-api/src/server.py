"""FastAPI service for the detection training tracker.

URL contract (read-only in Phase 1):
  GET /health           -> {"status": "ok"}
  GET /classes          -> list[ClassDto]                (the 60-class schema)
  GET /images           -> ImagePageDto                  (filter: labeled/class_id/source)
  GET /images/{id}      -> ImageDetailDto                (meta + annotations)
  GET /datasets         -> list[DatasetSummaryDto]
  GET /datasets/{id}    -> DatasetDetailDto
  GET /stats            -> CoverageDto                    (per-class coverage matrix)
  GET /thumbs/{id}      -> image/jpeg                     (cached thumbnail)
  GET /raw/{id}         -> image bytes                    (original screenshot)

Config comes from the environment via `load_config`; the lifespan installer is the
single writer to `app.state`, so dependency getters `cast` at that boundary.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager, closing
from pathlib import Path
from typing import TYPE_CHECKING, cast

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image

from .classes import ClassCatalog
from .config import TrackerConfig, load_config
from .db import connect, init_schema
from .domain import ImageFilter, ImageSource
from .repository import SqliteTrackerRepository, TrackerReader
from .schemas import (
    ClassDto,
    CoverageDto,
    DatasetDetailDto,
    DatasetSummaryDto,
    ImageDetailDto,
    ImagePageDto,
    class_to_dto,
    coverage_to_dto,
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
    allow_methods=["GET"],
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


def get_catalog(request: Request) -> ClassCatalog:
    return cast("ClassCatalog", cast("FastAPI", request.app).state.catalog)


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
    source_path = _image_path(repo, image_id)
    config.thumb_cache_dir.mkdir(parents=True, exist_ok=True)
    thumb_path = config.thumb_cache_dir / f"{image_id}.jpg"
    if thumb_path.exists():
        return thumb_path
    with Image.open(source_path) as image:
        rgb = image.convert("RGB")
        rgb.thumbnail((_THUMB_MAX_EDGE, _THUMB_MAX_EDGE))
        rgb.save(thumb_path, format="JPEG", quality=80)
    return thumb_path
