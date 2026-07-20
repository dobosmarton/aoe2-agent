"""API response shapes (the JSON boundary) and mappers from domain records.

Frozen dataclasses serialized by FastAPI — consistent with the arena-web service.
The one transform that earns this layer is flattening the `Geometry` union into an
explicit `{geom_type, coords}` pair so the discriminator survives serialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias, assert_never

from .geometry import BBox, Geometry, GeomType, Polygon, geom_type_of

if TYPE_CHECKING:
    from .classes import ClassCatalog
    from .domain import (
        Annotation,
        ClassCoverage,
        ClassInfo,
        CoverageReport,
        DatasetClassCount,
        DatasetDetail,
        DatasetSummary,
        ImageDetail,
        ImageListing,
        ImagePage,
        ImageRecord,
    )

Coords: TypeAlias = list[float] | list[list[float]]


@dataclass(frozen=True, slots=True)
class ClassDto:
    id: int
    name: str


@dataclass(frozen=True, slots=True)
class AnnotationDto:
    id: int | None
    class_id: int
    class_name: str
    geom_type: GeomType
    coords: Coords
    source: str
    status: str


@dataclass(frozen=True, slots=True)
class ImageRecordDto:
    id: int
    filename: str
    source: str
    width: int
    height: int
    thumb_url: str
    raw_url: str


@dataclass(frozen=True, slots=True)
class ImageListingDto:
    image: ImageRecordDto
    labeled: bool
    annotation_count: int
    class_ids: list[int]


@dataclass(frozen=True, slots=True)
class ImagePageDto:
    items: list[ImageListingDto]
    total: int
    page: int
    page_size: int


@dataclass(frozen=True, slots=True)
class ImageDetailDto:
    image: ImageRecordDto
    annotations: list[AnnotationDto]


@dataclass(frozen=True, slots=True)
class DatasetSummaryDto:
    id: int
    name: str
    created_at: str
    n_real_images: int
    n_synth_images: int
    notes: str | None


@dataclass(frozen=True, slots=True)
class DatasetClassCountDto:
    class_id: int
    name: str
    real_instances: int
    synth_instances: int


@dataclass(frozen=True, slots=True)
class DatasetDetailDto:
    summary: DatasetSummaryDto
    n_train_images: int
    n_val_images: int
    class_counts: list[DatasetClassCountDto]


@dataclass(frozen=True, slots=True)
class ClassCoverageDto:
    class_id: int
    name: str
    real_instances: int
    synth_instances: int
    labeled_images: int


@dataclass(frozen=True, slots=True)
class CoverageDto:
    total_images: int
    labeled_images: int
    unlabeled_images: int
    classes: list[ClassCoverageDto]
    zero_real_class_ids: list[int]


def geometry_coords(geom: Geometry) -> Coords:
    match geom:
        case BBox(x, y, w, h):
            return [x, y, w, h]
        case Polygon(points):
            return [[px, py] for px, py in points]
        case _ as unreachable:
            assert_never(unreachable)


def class_to_dto(info: ClassInfo) -> ClassDto:
    return ClassDto(id=info.id, name=info.name)


def _require_id(record: ImageRecord) -> int:
    if record.id is None:
        raise ValueError("persisted ImageRecord must have an id")
    return record.id


def image_to_dto(record: ImageRecord) -> ImageRecordDto:
    image_id = _require_id(record)
    return ImageRecordDto(
        id=image_id,
        filename=Path(record.path).name,
        source=record.source,
        width=record.width,
        height=record.height,
        thumb_url=f"/thumbs/{image_id}",
        raw_url=f"/raw/{image_id}",
    )


def annotation_to_dto(ann: Annotation, catalog: ClassCatalog) -> AnnotationDto:
    return AnnotationDto(
        id=ann.id,
        class_id=ann.class_id,
        class_name=catalog.name_of(ann.class_id),
        geom_type=geom_type_of(ann.geometry),
        coords=geometry_coords(ann.geometry),
        source=ann.source,
        status=ann.status,
    )


def listing_to_dto(listing: ImageListing) -> ImageListingDto:
    return ImageListingDto(
        image=image_to_dto(listing.record),
        labeled=listing.annotation_count > 0,
        annotation_count=listing.annotation_count,
        class_ids=list(listing.class_ids),
    )


def page_to_dto(page: ImagePage) -> ImagePageDto:
    return ImagePageDto(
        items=[listing_to_dto(item) for item in page.items],
        total=page.total,
        page=page.page,
        page_size=page.page_size,
    )


def detail_to_dto(detail: ImageDetail, catalog: ClassCatalog) -> ImageDetailDto:
    return ImageDetailDto(
        image=image_to_dto(detail.record),
        annotations=[annotation_to_dto(a, catalog) for a in detail.annotations],
    )


def summary_to_dto(summary: DatasetSummary) -> DatasetSummaryDto:
    return DatasetSummaryDto(
        id=summary.id,
        name=summary.name,
        created_at=summary.created_at,
        n_real_images=summary.n_real_images,
        n_synth_images=summary.n_synth_images,
        notes=summary.notes,
    )


def _dataset_class_count_to_dto(count: DatasetClassCount) -> DatasetClassCountDto:
    return DatasetClassCountDto(
        class_id=count.class_id,
        name=count.name,
        real_instances=count.real_instances,
        synth_instances=count.synth_instances,
    )


def dataset_detail_to_dto(detail: DatasetDetail) -> DatasetDetailDto:
    return DatasetDetailDto(
        summary=summary_to_dto(detail.summary),
        n_train_images=detail.n_train_images,
        n_val_images=detail.n_val_images,
        class_counts=[_dataset_class_count_to_dto(c) for c in detail.class_counts],
    )


def _coverage_class_to_dto(coverage: ClassCoverage) -> ClassCoverageDto:
    return ClassCoverageDto(
        class_id=coverage.class_id,
        name=coverage.name,
        real_instances=coverage.real_instances,
        synth_instances=coverage.synth_instances,
        labeled_images=coverage.labeled_images,
    )


def coverage_to_dto(report: CoverageReport) -> CoverageDto:
    return CoverageDto(
        total_images=report.total_images,
        labeled_images=report.labeled_images,
        unlabeled_images=report.unlabeled_images,
        classes=[_coverage_class_to_dto(c) for c in report.classes],
        zero_real_class_ids=list(report.zero_real_class_ids),
    )
