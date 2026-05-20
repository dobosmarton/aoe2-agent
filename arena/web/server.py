"""FastAPI + SSE backend for replaying arena event logs (Phase 7.1).

URL contract (frozen for future-frontend compatibility):
  GET /health            -> {"status": "ok"}
  GET /runs              -> list[RunSummary], newest first
  GET /events?run_id=X   -> text/event-stream, one event per line

Each SSE line is `data: <payload_json>\\n\\n`, where `<payload_json>` is the
raw column value from the events table (Pydantic-serialised by the writer
in evaluation/event_log.py). The frontend parses it with JSON.parse and
matches on the embedded `kind` discriminator.

Logs root resolution:
  ARENA_LOGS_ROOT env var (test injection) overrides the default
  `logs/arena/` relative to the process CWD.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

if TYPE_CHECKING:
    from collections.abc import Iterator

_DEFAULT_LOGS_ROOT = Path("logs") / "arena"
_DEFAULT_CORS_ORIGINS = ("http://localhost:5173", "http://localhost:8000")


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunSummary:
    """One row in the /runs response — describes a single run within a log file."""

    run_id: str
    db_path: str
    label: str
    n_events: int
    first_ts: str
    last_ts: str


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _logs_root() -> Path:
    override = os.environ.get("ARENA_LOGS_ROOT")
    return Path(override) if override else _DEFAULT_LOGS_ROOT


def _cors_origins() -> list[str]:
    override = os.environ.get("ARENA_WEB_CORS_ORIGINS")
    if override is None:
        return list(_DEFAULT_CORS_ORIGINS)
    return [origin.strip() for origin in override.split(",") if origin.strip()]


def _all_duckdb_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    files = list(root.glob("*/*.duckdb"))
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def _label_from_filename(db_path: Path) -> str:
    # `<label>-<HHMMSS>.duckdb` per arena/__main__.py:_new_db_path.
    stem = db_path.stem
    return stem.split("-", 1)[0] if "-" in stem else stem


def _runs_in_file(db_path: Path) -> list[RunSummary]:
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = conn.execute(
            "SELECT run_id, COUNT(*) AS n, MIN(ts) AS first_ts, MAX(ts) AS last_ts "
            "FROM events GROUP BY run_id ORDER BY MIN(ts)"
        ).fetchall()
    finally:
        conn.close()
    label = _label_from_filename(db_path)
    return [
        RunSummary(
            run_id=str(row[0]),
            db_path=str(db_path),
            label=label,
            n_events=int(row[1]),
            first_ts=_ts_to_str(row[2]),
            last_ts=_ts_to_str(row[3]),
        )
        for row in rows
    ]


def _ts_to_str(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _list_runs(root: Path) -> list[RunSummary]:
    runs: list[RunSummary] = []
    for db_path in _all_duckdb_files(root):
        runs.extend(_runs_in_file(db_path))
    return runs


def _resolve_run(run_id: str, root: Path) -> Path:
    for db_path in _all_duckdb_files(root):
        conn = duckdb.connect(str(db_path), read_only=True)
        try:
            row = conn.execute("SELECT 1 FROM events WHERE run_id=? LIMIT 1", [run_id]).fetchone()
        finally:
            conn.close()
        if row is not None:
            return db_path
    raise HTTPException(status_code=404, detail=f"run_id {run_id!r} not found")


def _stream_events_sync(db_path: Path, run_id: str) -> Iterator[str]:
    # Generator stays synchronous: Starlette's StreamingResponse drives sync
    # iterators on its thread pool, so the event loop is never blocked.
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        cursor = conn.execute("SELECT payload_json FROM events WHERE run_id=? ORDER BY t", [run_id])
        while True:
            row = cursor.fetchone()
            if row is None:
                break
            yield f"data: {row[0]}\n\n"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


app = FastAPI(title="AoE2 Arena Web (event-log replay)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/runs")
async def runs() -> list[dict[str, object]]:
    summaries = await asyncio.to_thread(_list_runs, _logs_root())
    return [asdict(s) for s in summaries]


@app.get("/events")
async def events(run_id: str = Query(..., min_length=1)) -> StreamingResponse:
    db_path = await asyncio.to_thread(_resolve_run, run_id, _logs_root())
    return StreamingResponse(
        _stream_events_sync(db_path, run_id),
        media_type="text/event-stream",
    )
