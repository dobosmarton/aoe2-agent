"""SQLite storage: schema definition and connection factory.

The tracked entity is the real screenshot. Synthetic images are a generated bulk
input, so their per-class contribution is summarised on `dataset_versions` rather
than enumerated as rows. `training_runs` is a forward-compatible stub for Phase 4.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id             INTEGER PRIMARY KEY,
    path           TEXT NOT NULL UNIQUE,
    source         TEXT NOT NULL CHECK(source IN ('real','synthetic')),
    sha256         TEXT NOT NULL,
    width          INTEGER NOT NULL,
    height         INTEGER NOT NULL,
    capture_meta_json TEXT,
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS annotations (
    id         INTEGER PRIMARY KEY,
    image_id   INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    class_id   INTEGER NOT NULL,
    geom_type  TEXT NOT NULL CHECK(geom_type IN ('bbox','polygon')),
    coords_json TEXT NOT NULL,
    source     TEXT NOT NULL CHECK(source IN ('model','human')),
    status     TEXT NOT NULL CHECK(status IN ('pending','approved')),
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_annotations_image ON annotations(image_id);
CREATE INDEX IF NOT EXISTS idx_annotations_class ON annotations(class_id);

CREATE TABLE IF NOT EXISTS dataset_versions (
    id                      INTEGER PRIMARY KEY,
    name                    TEXT NOT NULL UNIQUE,
    created_at              TEXT NOT NULL,
    notes                   TEXT,
    val_split               REAL,
    synth_image_count       INTEGER NOT NULL DEFAULT 0,
    synth_class_counts_json TEXT
);

CREATE TABLE IF NOT EXISTS dataset_images (
    dataset_version_id INTEGER NOT NULL REFERENCES dataset_versions(id) ON DELETE CASCADE,
    image_id           INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    split              TEXT NOT NULL CHECK(split IN ('train','val')),
    PRIMARY KEY (dataset_version_id, image_id)
);

CREATE TABLE IF NOT EXISTS training_runs (
    id                 INTEGER PRIMARY KEY,
    dataset_version_id INTEGER REFERENCES dataset_versions(id),
    runpod_pod_id      TEXT,
    status             TEXT NOT NULL,
    hyperparams_json   TEXT,
    metrics_json       TEXT,
    artifact_urls_json TEXT,
    created_at         TEXT NOT NULL
);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    """Open a SQLite connection with foreign keys on and row access by name."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Create tables and indexes if absent (idempotent)."""
    conn.executescript(_SCHEMA)
    conn.commit()


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Commit on success, roll back on any exception."""
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
