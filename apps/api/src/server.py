"""FastAPI + SSE backend for replaying arena event logs.

URL contract (frozen for future-frontend compatibility):
  GET  /health            -> {"status": "ok"}
  GET  /runs              -> list[RunSummary], newest first
  GET  /events?run_id=X   -> text/event-stream, one event per line.
                             Switches to live-tail mode for in-flight runs.
  POST /forks             -> create a child run with mutation patch + N-turn
                             async replay (Phase 9)
  GET  /metrics           -> BrokerMetricsSnapshot JSON (Phase 3). Backend-
                             agnostic: dispatches via isinstance(broker, ...)
                             so InProcess and Redis snapshots are surfaced
                             through the same shape.

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
import contextlib
import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, cast

import duckdb
from arena_web.forks import ForkRequest, ForkResponse, create_fork
from evaluation.broker_factory import make_broker
from evaluation.event_broker import (
    BrokerOverflowError,
    EventBroker,
    InProcessEventBroker,
    LiveRun,
    RunId,
    Seq,
)
from evaluation.event_log import stream_cold
from evaluation.fork import ForkError
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator, Sequence


logger = logging.getLogger(__name__)

_DEFAULT_LOGS_ROOT = Path("logs") / "arena"
_DEFAULT_CORS_ORIGINS = ("http://localhost:5173", "http://localhost:8000")
# Shown as a live run's label when the broker recorded no identity (e.g. a
# legacy `b"1"` sentinel). In practice every arena run supplies a real label.
_UNKNOWN_LABEL: Final = "?"


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


RunStatus = Literal["running", "complete"]
"""A run is either live on the broker ("running") or a finalized DuckDB file
("complete"). Mirrors the frontend's `status` union in `lib/events.ts`."""


@dataclass(frozen=True, slots=True)
class RunSummary:
    """One row in the /runs response.

    `status` is "running" for a live run discovered via the broker (the source
    of truth for in-progress runs) or "complete" for a finalized run read from
    a cold DuckDB file. Live rows carry an empty `db_path` — the frontend keys
    and selects by `run_id`, never the path.
    """

    run_id: str
    db_path: str
    label: str
    n_events: int
    first_ts: str
    last_ts: str
    status: RunStatus


# ---------------------------------------------------------------------------
# Reaper — server-side wall-clock tracking for `broker.reap` grace policy.
# ---------------------------------------------------------------------------


_DEFAULT_REAP_GRACE = timedelta(minutes=30)


@dataclass(slots=True)
class _ReaperRegistry:
    """Tracks when runs were closed so the lifespan reaper can drop their
    buffers after a grace period.

    The broker is intentionally time-agnostic — wall-clock policy lives
    here, not in `evaluation/event_broker.py`. Keeps the broker's
    semantics pure and makes the grace period swappable per deployment
    without touching the broker.
    """

    grace_period: timedelta = _DEFAULT_REAP_GRACE
    _closed_at: dict[RunId, datetime] = field(default_factory=dict)

    def mark_closed(self, run_id: RunId) -> None:
        self._closed_at[run_id] = datetime.now(UTC)

    def reap_overdue(self, broker: EventBroker, now: datetime) -> list[RunId]:
        """Reap runs whose close time is older than `now - grace_period`.

        Returns the reaped run_ids so the caller can log them.
        """
        cutoff = now - self.grace_period
        overdue = [rid for rid, t in self._closed_at.items() if t <= cutoff]
        for rid in overdue:
            broker.reap(rid)
            del self._closed_at[rid]
        return overdue


async def _reaper_loop(broker: EventBroker, reaper: _ReaperRegistry) -> None:
    """Background task: scan every `grace_period / 2` and drop overdue runs.

    Sleeping half the grace period bounds the worst-case lateness of a
    reap to 1.5x grace; finer scanning is wasted CPU on this workload.
    """
    interval = max(1.0, reaper.grace_period.total_seconds() / 2)
    while True:
        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            return
        reaped = reaper.reap_overdue(broker, datetime.now(UTC))
        if reaped:
            logger.info("reaped %d run(s): %s", len(reaped), reaped)


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


def _connect_read_only(db_path: Path) -> duckdb.DuckDBPyConnection | None:
    """Open `db_path` read-only, or return None if a live writer holds it.

    DuckDB's single-writer file format refuses a read-only connection while
    another process has the database open for writing (a concurrent
    `arena rank` run writing its own log file). Callers skip such files
    rather than failing the whole request — the file becomes readable again
    once the writer finalizes and releases the lock. Active runs remain
    observable in the meantime via the /events broker (see module docstring).
    """
    try:
        return duckdb.connect(str(db_path), read_only=True)
    except duckdb.IOException as exc:
        logger.warning("skipping locked/unreadable DuckDB %s: %s", db_path, exc)
        return None


def _runs_in_file(db_path: Path) -> list[RunSummary]:
    conn = _connect_read_only(db_path)
    if conn is None:
        return []
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
            status="complete",
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
    # This scans every log file to find the one holding `run_id`. A single
    # file locked by a live writer must not fail the lookup of an unrelated,
    # readable run — skip locked files and keep searching, but remember we
    # did so to pick the right "not found" status below.
    locked = False
    for db_path in _all_duckdb_files(root):
        conn = _connect_read_only(db_path)
        if conn is None:
            locked = True
            continue
        try:
            row = conn.execute("SELECT 1 FROM events WHERE run_id=? LIMIT 1", [run_id]).fetchone()
        finally:
            conn.close()
        if row is not None:
            return db_path
    if locked:
        # The run is in none of the *readable* logs, but a writer-locked log
        # might hold it. Surface 503 (transient — retry once the writer
        # finalizes) rather than 404 (permanent). Live runs normally stream
        # via the broker before reaching this path; this covers cross-process
        # writers the in-memory broker can't see.
        raise HTTPException(
            status_code=503,
            detail=f"run_id {run_id!r} not in readable logs; a log file is "
            "locked by an active writer — retry shortly",
        )
    raise HTTPException(status_code=404, detail=f"run_id {run_id!r} not found")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    broker = make_broker()
    reaper = _ReaperRegistry()
    fork_tasks: set[asyncio.Task[None]] = set()
    app.state.broker = broker
    app.state.reaper = reaper
    app.state.fork_tasks = fork_tasks
    reaper_task = asyncio.create_task(_reaper_loop(broker, reaper))
    try:
        yield
    finally:
        # Stop the reaper before cancelling fork tasks — otherwise it
        # could race with shutdown and reap a run mid-replay.
        reaper_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await reaper_task
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


def get_reaper(request: Request) -> _ReaperRegistry:
    app_state = cast("FastAPI", request.app).state
    reaper = cast("object", app_state.reaper)
    assert isinstance(reaper, _ReaperRegistry)
    return reaper


def get_broker(request: Request) -> EventBroker:
    """FastAPI dependency: yields the process-wide `EventBroker`.

    Returns whichever impl `make_broker()` selected at lifespan startup
    (defaults to `InProcessEventBroker`; set `ARENA_BROKER_BACKEND=redis`
    for `RedisStreamsBroker`). The lifespan installer is the single
    writer to `app.state.broker`, so a runtime `isinstance` check here
    would be hostile to the multi-impl design — `cast` is the right
    boundary: the static type stays `EventBroker`."""
    app_state = cast("FastAPI", request.app).state
    return cast("EventBroker", app_state.broker)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


def _live_summaries(live: Sequence[LiveRun]) -> list[RunSummary]:
    """Map the broker's `LiveRun`s to `/runs` rows. `db_path` is empty (a live
    run has no finalized file yet); `started_at` doubles as `first_ts`/`last_ts`
    — the UI renders `first_ts`, and a real tail timestamp isn't worth a Redis
    round-trip on every list."""
    return [
        RunSummary(
            run_id=lr.run_id,
            db_path="",
            label=lr.label or _UNKNOWN_LABEL,
            n_events=lr.n_events,
            first_ts=lr.started_at or "",
            last_ts=lr.started_at or "",
            status="running",
        )
        for lr in live
    ]


def _merge_runs(live: list[RunSummary], cold: list[RunSummary]) -> list[RunSummary]:
    """Live runs (the broker's authoritative view of in-progress runs) come
    first and shadow any cold DuckDB row for the same run_id — the brief
    close→file-unlock window is the only time both could list the same run."""
    live_ids = {r.run_id for r in live}
    return [*live, *(c for c in cold if c.run_id not in live_ids)]


@app.get("/runs")
async def runs(broker: EventBroker = Depends(get_broker)) -> list[dict[str, object]]:
    # Symmetric with /events: the broker is the source of truth for live runs,
    # the cold DuckDB scan covers finalized ones.
    cold = await asyncio.to_thread(_list_runs, _logs_root())
    live = _live_summaries(await broker.live_runs())
    return [asdict(s) for s in _merge_runs(live, cold)]


async def _stream_from_broker(
    broker: EventBroker,
    run_id: RunId,
    from_seq: Seq = Seq(0),
) -> AsyncIterator[str]:
    """SSE generator for live runs: drain the broker, surface overflow.

    On `BrokerOverflowError` (consumer fell behind the buffer head), emit
    a final SSE event named `overflow` carrying `available_from`, then
    return. The frontend reconnects with `?from_seq=<available_from>`
    and accepts the gap — surfacing the loss is preferable to silently
    serving partial history.
    """
    try:
        async for envelope in broker.stream(run_id, from_seq=from_seq):
            yield f"data: {envelope.event.payload.model_dump_json()}\n\n"
    except BrokerOverflowError as err:
        payload = json.dumps({"available_from": int(err.available_from)})
        yield f"event: overflow\ndata: {payload}\n\n"


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
    from_seq: int = Query(0, ge=0),
    broker: EventBroker = Depends(get_broker),
) -> StreamingResponse:
    """Stream events for `run_id` as Server-Sent Events.

    Live runs (`broker.is_open`) read from the broker — zero DuckDB
    opens; immune to writer/reader file-mode collisions. `from_seq`
    skips already-seen envelopes on reconnect; overflow recovery
    sends `from_seq=available_from` from the previous overflow event.

    Finalized runs fall through to `stream_cold` (read-only DuckDB).
    The cold path ignores `from_seq` — full replay only, since cold
    is the post-mortem case where partial reads aren't worth the
    extra complexity.
    """
    typed_run = RunId(run_id)
    # `is_open_remote` (not `is_open`) is the cross-process liveness signal: the
    # web process never opened this run — a separate CLI process did — so the
    # process-local `is_open` would be False and we'd wrongly fall to the
    # writer-locked DuckDB. For the in-process broker the two coincide.
    if await broker.is_open_remote(typed_run):
        return StreamingResponse(
            _stream_from_broker(broker, typed_run, Seq(from_seq)),
            media_type="text/event-stream",
        )
    db_path = await asyncio.to_thread(_resolve_run, run_id, _logs_root())
    return StreamingResponse(
        _stream_from_cold(db_path, typed_run),
        media_type="text/event-stream",
    )


@app.get("/metrics")
async def metrics(broker: EventBroker = Depends(get_broker)) -> dict[str, int]:
    """Broker operational counters as JSON.

    `metrics()` is intentionally OFF the `EventBroker` Protocol because
    each impl exposes a different counter surface (in-process: pure
    dataclass; Redis: an `await` on Redis state). We dispatch on the
    concrete type here rather than promoting a `BrokerMetrics` Protocol
    — at N=2 impls, an `isinstance` branch is less ceremony than a new
    Protocol. The Redis import is lazy so the slim install
    (`pip install -e .`, no broker-redis extra) never has to import
    `redis` just to start the web server in in-process mode.
    """
    if isinstance(broker, InProcessEventBroker):
        return broker.metrics().to_dict()
    try:
        from evaluation.redis_broker import RedisStreamsBroker
    except ImportError as exc:
        raise HTTPException(503, "broker metrics unavailable") from exc
    if isinstance(broker, RedisStreamsBroker):
        return (await broker.metrics()).to_dict()
    raise HTTPException(503, "broker impl exposes no metrics surface")


@app.post("/forks", response_model=ForkResponse)
async def post_forks(
    request: ForkRequest,
    broker: EventBroker = Depends(get_broker),
    reaper: _ReaperRegistry = Depends(get_reaper),
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
            on_close=reaper.mark_closed,
            logs_root=_logs_root(),
            fork_tasks=fork_tasks,
        )
    except FileNotFoundError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err
    except ForkError as err:
        raise HTTPException(status_code=422, detail=str(err)) from err
