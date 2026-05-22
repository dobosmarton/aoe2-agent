"""FastAPI + SSE backend for replaying arena event logs.

URL contract (frozen for future-frontend compatibility):
  GET  /health            -> {"status": "ok"}
  GET  /runs              -> list[RunSummary], newest first
  GET  /events?run_id=X   -> text/event-stream, one event per line.
                             Switches to live-tail mode for in-flight runs.
  POST /forks             -> create a child run with mutation patch + N-turn
                             async replay (Phase 9)

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
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

import duckdb
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from arena.web.forks import ForkRequest, ForkResponse, create_fork
from evaluation.event_broker import EventBroker, InProcessEventBroker, RunId, Seq
from evaluation.event_log import stream_cold
from evaluation.fork import ForkError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator


logger = logging.getLogger(__name__)

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
        rows = cast(
            "list[tuple[object, ...]]",
            conn.execute(
                "SELECT run_id, COUNT(*) AS n, MIN(ts) AS first_ts, MAX(ts) AS last_ts "
                "FROM events GROUP BY run_id ORDER BY MIN(ts)"
            ).fetchall(),
        )
    finally:
        conn.close()
    label = _label_from_filename(db_path)
    return [
        RunSummary(
            run_id=str(row[0]),
            db_path=str(db_path),
            label=label,
            n_events=int(cast("int", row[1])),
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


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.broker = InProcessEventBroker()
    fork_tasks: set[asyncio.Task[None]] = set()
    app.state.fork_tasks = fork_tasks
    try:
        yield
    finally:
        # Best-effort cancel of in-flight forks on shutdown.
        tasks: set[asyncio.Task[None]] = app.state.fork_tasks
        for task in list(tasks):
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


app = FastAPI(title="AoE2 Arena Web (event-log replay)", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def get_fork_tasks(request: Request) -> set[asyncio.Task[None]]:
    app_state = cast("FastAPI", request.app).state
    tasks = cast("object", app_state.fork_tasks)
    assert isinstance(tasks, set)
    return cast("set[asyncio.Task[None]]", tasks)


def get_broker(request: Request) -> EventBroker:
    """FastAPI dependency: yields the process-wide `EventBroker`.

    Phase 1 always returns the `InProcessEventBroker` installed at startup;
    Phase C swaps to a Redis/NATS broker — same Protocol, zero handler
    changes."""
    app_state = cast("FastAPI", request.app).state
    broker = cast("object", app_state.broker)
    assert isinstance(broker, InProcessEventBroker)
    return broker


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/runs")
async def runs() -> list[dict[str, object]]:
    summaries = await asyncio.to_thread(_list_runs, _logs_root())
    return [asdict(s) for s in summaries]


async def _stream_from_broker(
    broker: EventBroker,
    run_id: RunId,
) -> AsyncIterator[str]:
    """Phase 2 SSE generator for live runs: drain the broker."""
    async for envelope in broker.stream(run_id, from_seq=Seq(0)):
        yield f"data: {envelope.event.payload.model_dump_json()}\n\n"


def _stream_from_cold(db_path: Path, run_id: RunId) -> Iterator[str]:
    """Phase 2 SSE generator for finalized runs.

    Synchronous so Starlette can drive it on its thread pool — DuckDB
    iteration is blocking, and pretending otherwise via `to_thread`
    would just hide the same cost behind more code. Re-serializes via
    `payload.model_dump_json()` for byte-equivalence with the broker
    path (guarded by `test_payload_roundtrip_is_byte_stable`).
    """
    for envelope in stream_cold(db_path, run_id):
        yield f"data: {envelope.event.payload.model_dump_json()}\n\n"


@app.get("/events")
async def events(
    run_id: str = Query(..., min_length=1),
    broker: EventBroker = Depends(get_broker),
) -> StreamingResponse:
    """Stream events for `run_id` as Server-Sent Events.

    Live runs (`broker.is_open`) read from the broker — zero DuckDB
    opens; immune to writer/reader file-mode collisions.
    Finalized runs fall through to `stream_cold` (read-only DuckDB).
    """
    typed_run = RunId(run_id)
    if broker.is_open(typed_run):
        return StreamingResponse(
            _stream_from_broker(broker, typed_run),
            media_type="text/event-stream",
        )
    db_path = await asyncio.to_thread(_resolve_run, run_id, _logs_root())
    return StreamingResponse(
        _stream_from_cold(db_path, typed_run),
        media_type="text/event-stream",
    )


@app.post("/forks", response_model=ForkResponse)
async def post_forks(
    request: ForkRequest,
    broker: EventBroker = Depends(get_broker),
    fork_tasks: set[asyncio.Task[None]] = Depends(get_fork_tasks),
) -> ForkResponse:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY is not set on the server")
    try:
        return await create_fork(
            request=request,
            api_key=api_key,
            broker=broker,
            logs_root=_logs_root(),
            fork_tasks=fork_tasks,
        )
    except FileNotFoundError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err
    except ForkError as err:
        raise HTTPException(status_code=422, detail=str(err)) from err
