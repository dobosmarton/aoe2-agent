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
from arena.web.live import LiveRunRegistry
from evaluation.event_broker import EventBroker, InProcessEventBroker, RunId, Seq
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


def _broker_enabled() -> bool:
    """Phase 1 feature flag: route /events through the new EventBroker path.

    Off by default. Set ARENA_BROKER_ENABLED=true to opt in. Deleted in
    Phase 2 when the broker becomes the only path.
    """
    return os.environ.get("ARENA_BROKER_ENABLED", "false").lower() in ("1", "true", "yes")


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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.registry = LiveRunRegistry()
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


def get_registry(request: Request) -> LiveRunRegistry:
    app_state = cast("FastAPI", request.app).state
    registry = cast("object", app_state.registry)
    assert isinstance(registry, LiveRunRegistry)
    return registry


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


async def _broker_sse(
    broker: EventBroker,
    run_id: str,
) -> AsyncIterator[str]:
    """Phase 1 SSE generator: drain the broker for the given run.

    Caller (`events()`) must check `broker.is_open(run_id)` before reaching
    this. Same wire format as `_live_event_stream` —
    `data: <payload_json>\\n\\n` — so the frontend can't tell which
    generator served it.
    """
    async for envelope in broker.stream(RunId(run_id), from_seq=Seq(0)):
        yield f"data: {envelope.event.payload.model_dump_json()}\n\n"


async def _live_event_stream(
    registry: LiveRunRegistry,
    run_id: str,
) -> AsyncIterator[str]:
    """Replay any already-written events for the live run, then tail new ones.

    The race window: while we're reading DuckDB rows, the writer may add
    more. Subscribing AFTER the read means we'd miss those rows. The fix
    is to subscribe BEFORE the read and skip any queue event whose t we
    already saw in the DB.
    """
    sub = registry.subscribe(run_id)
    try:
        db_path = await asyncio.to_thread(_resolve_run_optional, run_id, _logs_root())
        max_t_seen = -1
        if db_path is not None:
            for line, last_t in _stream_existing_rows(db_path, run_id):
                yield line
                if last_t > max_t_seen:
                    max_t_seen = last_t
        while True:
            event = await sub.queue.get()
            if event is None:
                break
            if event.t <= max_t_seen:
                continue
            yield f"data: {event.payload.model_dump_json()}\n\n"
    finally:
        registry.unsubscribe(run_id, sub)


def _stream_existing_rows(db_path: Path, run_id: str) -> Iterator[tuple[str, int]]:
    """Yield (sse_line, t) pairs for already-persisted events of `run_id`."""
    with duckdb.connect(str(db_path), read_only=True) as conn:
        cursor = conn.execute(
            "SELECT t, payload_json FROM events WHERE run_id=? ORDER BY t",
            [run_id],
        )
        while True:
            row = cast("tuple[object, ...] | None", cursor.fetchone())
            if row is None:
                break
            yield f"data: {row[1]}\n\n", int(cast("int", row[0]))


def _resolve_run_optional(run_id: str, root: Path) -> Path | None:
    """Like _resolve_run but returns None instead of raising 404."""
    for db_path in _all_duckdb_files(root):
        with duckdb.connect(str(db_path), read_only=True) as conn:
            row = conn.execute("SELECT 1 FROM events WHERE run_id=? LIMIT 1", [run_id]).fetchone()
        if row is not None:
            return db_path
    return None


@app.get("/events")
async def events(
    run_id: str = Query(..., min_length=1),
    registry: LiveRunRegistry = Depends(get_registry),
    broker: EventBroker = Depends(get_broker),
) -> StreamingResponse:
    # Phase 1: when the flag is on, route live runs through the broker.
    # Cold runs (no broker entry, not in registry) still hit DuckDB —
    # that's Phase 2's `stream_cold` work.
    if _broker_enabled() and broker.is_open(RunId(run_id)):
        return StreamingResponse(
            _broker_sse(broker, run_id),
            media_type="text/event-stream",
        )
    if registry.is_live(run_id):
        return StreamingResponse(
            _live_event_stream(registry, run_id),
            media_type="text/event-stream",
        )
    db_path = await asyncio.to_thread(_resolve_run, run_id, _logs_root())
    return StreamingResponse(
        _stream_events_sync(db_path, run_id),
        media_type="text/event-stream",
    )


@app.post("/forks", response_model=ForkResponse)
async def post_forks(
    request: ForkRequest,
    registry: LiveRunRegistry = Depends(get_registry),
    fork_tasks: set[asyncio.Task[None]] = Depends(get_fork_tasks),
) -> ForkResponse:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY is not set on the server")
    try:
        return await create_fork(
            request=request,
            api_key=api_key,
            registry=registry,
            logs_root=_logs_root(),
            fork_tasks=fork_tasks,
        )
    except FileNotFoundError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err
    except ForkError as err:
        raise HTTPException(status_code=422, detail=str(err)) from err
