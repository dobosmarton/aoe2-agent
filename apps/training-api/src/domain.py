"""Internal domain records for the tracker.

Frozen + slotted dataclasses passed between the repository, ingest, and route
layers. IDs are `int | None` — `None` before a row is inserted, populated after.
External JSON shapes live in `schemas.py`; these types stay storage-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias, TypedDict

if TYPE_CHECKING:
    from .geometry import Geometry

ClassId: TypeAlias = int
ImageSource: TypeAlias = Literal["real", "synthetic"]
AnnotationSource: TypeAlias = Literal["model", "human"]
AnnotationStatus: TypeAlias = Literal["pending", "approved"]
Split: TypeAlias = Literal["train", "val"]


class CaptureMeta(TypedDict, total=False):
    """Optional provenance tags for a captured frame (all keys optional)."""

    age: str
    biome: str
    army_comp: str
    source_run_id: str


@dataclass(frozen=True, slots=True)
class ImageRecord:
    id: int | None
    path: str
    source: ImageSource
    sha256: str
    width: int
    height: int
    capture_meta: CaptureMeta | None


@dataclass(frozen=True, slots=True)
class Annotation:
    id: int | None
    image_id: int
    class_id: ClassId
    geometry: Geometry
    source: AnnotationSource
    status: AnnotationStatus


@dataclass(frozen=True, slots=True)
class ClassInfo:
    id: ClassId
    name: str


@dataclass(frozen=True, slots=True)
class ImageFilter:
    """Query criteria for `list_images` — one value object instead of a wide
    handler signature (no flag/boolean argument sprawl)."""

    labeled: bool | None = None
    class_id: ClassId | None = None
    source: ImageSource | None = None
    page: int = 0
    page_size: int = 60


@dataclass(frozen=True, slots=True)
class ImageListing:
    """A labeled/unlabeled row in the images table view."""

    record: ImageRecord
    annotation_count: int
    class_ids: tuple[ClassId, ...]


@dataclass(frozen=True, slots=True)
class ImagePage:
    items: tuple[ImageListing, ...]
    total: int
    page: int
    page_size: int


@dataclass(frozen=True, slots=True)
class ImageDetail:
    record: ImageRecord
    annotations: tuple[Annotation, ...]


@dataclass(frozen=True, slots=True)
class DatasetSummary:
    id: int
    name: str
    created_at: str
    n_real_images: int
    n_synth_images: int
    notes: str | None


@dataclass(frozen=True, slots=True)
class DatasetClassCount:
    class_id: ClassId
    name: str
    real_instances: int
    synth_instances: int


@dataclass(frozen=True, slots=True)
class DatasetDetail:
    summary: DatasetSummary
    n_train_images: int
    n_val_images: int
    class_counts: tuple[DatasetClassCount, ...]


@dataclass(frozen=True, slots=True)
class ClassCoverage:
    class_id: ClassId
    name: str
    real_instances: int
    synth_instances: int
    labeled_images: int


@dataclass(frozen=True, slots=True)
class CoverageReport:
    total_images: int
    labeled_images: int
    unlabeled_images: int
    classes: tuple[ClassCoverage, ...]
    zero_real_class_ids: tuple[ClassId, ...]
