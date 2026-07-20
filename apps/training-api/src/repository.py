"""Data access for the tracker.

`TrackerReader` is the read contract the API depends on (Protocol → routes stay
decoupled from SQLite and tests can inject a fake). `SqliteTrackerRepository`
implements it and adds the write methods ingest needs. All driver `Any` leaks are
confined to `_sql`; this module works with concrete types throughout.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Protocol

from ._sql import col_int, col_opt_str, col_str, loads_dict, query_all, query_one
from .db import transaction
from .domain import (
    Annotation,
    CaptureMeta,
    ClassCoverage,
    ClassId,
    CoverageReport,
    DatasetClassCount,
    DatasetDetail,
    DatasetSummary,
    ImageDetail,
    ImageFilter,
    ImageListing,
    ImagePage,
    ImageRecord,
    ImageSource,
    Split,
)
from .geometry import Geometry, GeomType, from_coords_json, geom_type_of, to_coords_json

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Sequence

    from .classes import ClassCatalog


class TrackerReader(Protocol):
    """Read surface consumed by the API routes."""

    def list_images(self, filt: ImageFilter) -> ImagePage: ...
    def get_image(self, image_id: int) -> ImageDetail | None: ...
    def list_datasets(self) -> list[DatasetSummary]: ...
    def get_dataset(self, dataset_id: int) -> DatasetDetail | None: ...
    def coverage(self) -> CoverageReport: ...


class SqliteTrackerRepository:
    def __init__(self, conn: sqlite3.Connection, catalog: ClassCatalog) -> None:
        self._conn = conn
        self._catalog = catalog

    # -- reads ---------------------------------------------------------------

    def list_images(self, filt: ImageFilter) -> ImagePage:
        where, params = _image_where(filt)
        total_row = query_one(
            self._conn, f"SELECT COUNT(*) AS n FROM images i WHERE {where}", params
        )
        total = col_int(total_row, "n") if total_row is not None else 0

        page_params = {**params, "limit": filt.page_size, "offset": filt.page * filt.page_size}
        rows = query_all(
            self._conn,
            f"SELECT * FROM images i WHERE {where} ORDER BY i.path LIMIT :limit OFFSET :offset",
            page_params,
        )
        listings = tuple(self._listing_for(row) for row in rows)
        return ImagePage(items=listings, total=total, page=filt.page, page_size=filt.page_size)

    def get_image(self, image_id: int) -> ImageDetail | None:
        row = query_one(self._conn, "SELECT * FROM images WHERE id = ?", (image_id,))
        if row is None:
            return None
        annotations = tuple(
            _row_to_annotation(a)
            for a in query_all(
                self._conn, "SELECT * FROM annotations WHERE image_id = ? ORDER BY id", (image_id,)
            )
        )
        return ImageDetail(record=_row_to_image(row), annotations=annotations)

    def list_datasets(self) -> list[DatasetSummary]:
        rows = query_all(
            self._conn, "SELECT * FROM dataset_versions ORDER BY created_at DESC, id DESC"
        )
        return [self._summary_for(row) for row in rows]

    def get_dataset(self, dataset_id: int) -> DatasetDetail | None:
        row = query_one(self._conn, "SELECT * FROM dataset_versions WHERE id = ?", (dataset_id,))
        if row is None:
            return None
        summary = self._summary_for(row)
        splits = self._split_counts(dataset_id)
        real_counts = self._real_class_counts(dataset_id)
        synth_counts = _load_synth_counts(row)
        return DatasetDetail(
            summary=summary,
            n_train_images=splits.get("train", 0),
            n_val_images=splits.get("val", 0),
            class_counts=self._merge_class_counts(real_counts, synth_counts),
        )

    def coverage(self) -> CoverageReport:
        total = self._scalar_count("SELECT COUNT(*) AS n FROM images")
        labeled = self._scalar_count("SELECT COUNT(DISTINCT image_id) AS n FROM annotations")
        real_instances = self._class_count_map(
            "SELECT class_id, COUNT(*) AS n FROM annotations GROUP BY class_id"
        )
        labeled_images = self._class_count_map(
            "SELECT class_id, COUNT(DISTINCT image_id) AS n FROM annotations GROUP BY class_id"
        )
        synth_instances = self._latest_synth_counts()

        classes = tuple(
            ClassCoverage(
                class_id=info.id,
                name=info.name,
                real_instances=real_instances.get(info.id, 0),
                synth_instances=synth_instances.get(info.id, 0),
                labeled_images=labeled_images.get(info.id, 0),
            )
            for info in self._catalog.all()
        )
        zero_real = tuple(c.class_id for c in classes if c.real_instances == 0)
        return CoverageReport(
            total_images=total,
            labeled_images=labeled,
            unlabeled_images=total - labeled,
            classes=classes,
            zero_real_class_ids=zero_real,
        )

    # -- writes (ingest) -----------------------------------------------------

    def upsert_image(self, record: ImageRecord) -> int:
        meta_json = json.dumps(record.capture_meta) if record.capture_meta else None
        with transaction(self._conn):
            self._conn.execute(
                """
                INSERT INTO images (path, source, sha256, width, height, capture_meta_json, created_at)
                VALUES (:path, :source, :sha256, :width, :height, :meta, :created_at)
                ON CONFLICT(path) DO UPDATE SET
                    source = excluded.source,
                    sha256 = excluded.sha256,
                    width = excluded.width,
                    height = excluded.height,
                    capture_meta_json = excluded.capture_meta_json
                """,
                {
                    "path": record.path,
                    "source": record.source,
                    "sha256": record.sha256,
                    "width": record.width,
                    "height": record.height,
                    "meta": meta_json,
                    "created_at": _now(),
                },
            )
        row = query_one(self._conn, "SELECT id FROM images WHERE path = ?", (record.path,))
        if row is None:
            raise RuntimeError(f"image row vanished after upsert: {record.path}")
        return col_int(row, "id")

    def replace_annotations(self, image_id: int, annotations: Sequence[Annotation]) -> None:
        with transaction(self._conn):
            self._conn.execute("DELETE FROM annotations WHERE image_id = ?", (image_id,))
            self._conn.executemany(
                """
                INSERT INTO annotations
                    (image_id, class_id, geom_type, coords_json, source, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [_annotation_params(image_id, ann) for ann in annotations],
            )

    def reset_dataset_version(
        self,
        name: str,
        *,
        notes: str | None,
        val_split: float | None,
        synth_image_count: int,
        synth_class_counts: dict[ClassId, int],
    ) -> int:
        with transaction(self._conn):
            self._conn.execute("DELETE FROM dataset_versions WHERE name = ?", (name,))
            self._conn.execute(
                """
                INSERT INTO dataset_versions
                    (name, created_at, notes, val_split, synth_image_count, synth_class_counts_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    name,
                    _now(),
                    notes,
                    val_split,
                    synth_image_count,
                    json.dumps({str(k): v for k, v in synth_class_counts.items()}),
                ),
            )
        row = query_one(self._conn, "SELECT id FROM dataset_versions WHERE name = ?", (name,))
        if row is None:
            raise RuntimeError(f"dataset version row vanished after insert: {name}")
        return col_int(row, "id")

    def add_dataset_image(self, dataset_version_id: int, image_id: int, split: Split) -> None:
        with transaction(self._conn):
            self._conn.execute(
                """
                INSERT INTO dataset_images (dataset_version_id, image_id, split)
                VALUES (?, ?, ?)
                ON CONFLICT(dataset_version_id, image_id) DO UPDATE SET split = excluded.split
                """,
                (dataset_version_id, image_id, split),
            )

    # -- helpers -------------------------------------------------------------

    def _scalar_count(self, sql: str) -> int:
        row = query_one(self._conn, sql)
        return col_int(row, "n") if row is not None else 0

    def _class_count_map(self, sql: str) -> dict[ClassId, int]:
        return {col_int(r, "class_id"): col_int(r, "n") for r in query_all(self._conn, sql)}

    def _listing_for(self, row: sqlite3.Row) -> ImageListing:
        agg = query_one(
            self._conn,
            "SELECT COUNT(*) AS n, GROUP_CONCAT(DISTINCT class_id) AS classes "
            "FROM annotations WHERE image_id = ?",
            (col_int(row, "id"),),
        )
        count = col_int(agg, "n") if agg is not None else 0
        class_ids = _parse_class_concat(col_opt_str(agg, "classes")) if agg is not None else ()
        return ImageListing(record=_row_to_image(row), annotation_count=count, class_ids=class_ids)

    def _summary_for(self, row: sqlite3.Row) -> DatasetSummary:
        n_real = self._scalar_count_for(
            "SELECT COUNT(*) AS n FROM dataset_images WHERE dataset_version_id = ?",
            col_int(row, "id"),
        )
        return DatasetSummary(
            id=col_int(row, "id"),
            name=col_str(row, "name"),
            created_at=col_str(row, "created_at"),
            n_real_images=n_real,
            n_synth_images=col_int(row, "synth_image_count"),
            notes=col_opt_str(row, "notes"),
        )

    def _scalar_count_for(self, sql: str, param: int) -> int:
        row = query_one(self._conn, sql, (param,))
        return col_int(row, "n") if row is not None else 0

    def _split_counts(self, dataset_id: int) -> dict[str, int]:
        return {
            col_str(r, "split"): col_int(r, "n")
            for r in query_all(
                self._conn,
                "SELECT split, COUNT(*) AS n FROM dataset_images "
                "WHERE dataset_version_id = ? GROUP BY split",
                (dataset_id,),
            )
        }

    def _real_class_counts(self, dataset_id: int) -> dict[ClassId, int]:
        return {
            col_int(r, "class_id"): col_int(r, "n")
            for r in query_all(
                self._conn,
                """
                SELECT a.class_id AS class_id, COUNT(*) AS n
                FROM annotations a
                JOIN dataset_images di ON di.image_id = a.image_id
                WHERE di.dataset_version_id = ?
                GROUP BY a.class_id
                """,
                (dataset_id,),
            )
        }

    def _latest_synth_counts(self) -> dict[ClassId, int]:
        row = query_one(
            self._conn,
            "SELECT synth_class_counts_json FROM dataset_versions "
            "ORDER BY created_at DESC, id DESC LIMIT 1",
        )
        return _load_synth_counts(row) if row is not None else {}

    def _merge_class_counts(
        self, real: dict[ClassId, int], synth: dict[ClassId, int]
    ) -> tuple[DatasetClassCount, ...]:
        return tuple(
            DatasetClassCount(
                class_id=info.id,
                name=info.name,
                real_instances=real.get(info.id, 0),
                synth_instances=synth.get(info.id, 0),
            )
            for info in self._catalog.all()
        )


def _image_where(filt: ImageFilter) -> tuple[str, dict[str, object]]:
    clauses = ["1 = 1"]
    params: dict[str, object] = {}
    if filt.source is not None:
        clauses.append("i.source = :source")
        params["source"] = filt.source
    if filt.class_id is not None:
        clauses.append(
            "EXISTS (SELECT 1 FROM annotations a WHERE a.image_id = i.id AND a.class_id = :class_id)"
        )
        params["class_id"] = filt.class_id
    if filt.labeled is True:
        clauses.append("EXISTS (SELECT 1 FROM annotations a WHERE a.image_id = i.id)")
    elif filt.labeled is False:
        clauses.append("NOT EXISTS (SELECT 1 FROM annotations a WHERE a.image_id = i.id)")
    return " AND ".join(clauses), params


def _row_to_image(row: sqlite3.Row) -> ImageRecord:
    meta = _parse_capture_meta(col_opt_str(row, "capture_meta_json"))
    source: ImageSource = "synthetic" if col_str(row, "source") == "synthetic" else "real"
    return ImageRecord(
        id=col_int(row, "id"),
        path=col_str(row, "path"),
        source=source,
        sha256=col_str(row, "sha256"),
        width=col_int(row, "width"),
        height=col_int(row, "height"),
        capture_meta=meta,
    )


def _row_to_annotation(row: sqlite3.Row) -> Annotation:
    geometry: Geometry = from_coords_json(_geom_type(row), col_str(row, "coords_json"))
    source = "human" if col_str(row, "source") == "human" else "model"
    status = "approved" if col_str(row, "status") == "approved" else "pending"
    return Annotation(
        id=col_int(row, "id"),
        image_id=col_int(row, "image_id"),
        class_id=col_int(row, "class_id"),
        geometry=geometry,
        source=source,
        status=status,
    )


def _geom_type(row: sqlite3.Row) -> GeomType:
    # The DB CHECK constraint guarantees one of these two values.
    value = col_str(row, "geom_type")
    if value == "polygon":
        return "polygon"
    return "bbox"


def _annotation_params(image_id: int, ann: Annotation) -> tuple[int, int, str, str, str, str, str]:
    return (
        image_id,
        ann.class_id,
        geom_type_of(ann.geometry),
        to_coords_json(ann.geometry),
        ann.source,
        ann.status,
        _now(),
    )


def _parse_class_concat(value: str | None) -> tuple[ClassId, ...]:
    if not value:
        return ()
    return tuple(sorted(int(part) for part in value.split(",")))


def _parse_capture_meta(raw: str | None) -> CaptureMeta | None:
    if not raw:
        return None
    parsed = loads_dict(raw)
    if parsed is None:
        return None
    meta: CaptureMeta = {}
    for key in ("age", "biome", "army_comp", "source_run_id"):
        value = parsed.get(key)
        if isinstance(value, str):
            meta[key] = value  # type: ignore[literal-required]  # key is a CaptureMeta field
    return meta or None


def _load_synth_counts(row: sqlite3.Row) -> dict[ClassId, int]:
    raw = col_opt_str(row, "synth_class_counts_json")
    if not raw:
        return {}
    parsed = loads_dict(raw)
    if parsed is None:
        return {}
    return {int(k): v for k, v in parsed.items() if isinstance(v, int)}


def _now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()
