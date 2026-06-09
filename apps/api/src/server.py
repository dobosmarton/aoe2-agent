"""FastAPI + SSE backend for replaying arena event logs.

URL contract (frozen for future-frontend compatibility):
  GET  /health            -> {"status": "ok"}
  GET  /runs              -> list[RunSummary], newest first
  GET  /runs/summaries    -> list[RunMetrics] — per-run end-of-run metrics
                             (profile, final state, cost, turns) for the
                             experiment overview. Finalized runs only.
  GET  /runs/series?db_path=X -> list[RunSeries] — per-turn resource
                             trajectories for every run in one operation's
                             DuckDB file (the overview's per-resource charts).
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
from evaluation.event_log import TurnStartPayload, stream_cold
from evaluation.fork import ForkError
from evaluation.world_sim import AGE_SEQUENCE
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


@dataclass(frozen=True, slots=True)
class RunMetrics:
    """Comparable end-of-run metrics for the dashboard's experiment overview.

    Recomputed from a finalized DuckDB file: the final `WorldStateSnapshot`
    (last `turn_start`), summed LLM cost, and turn count. `profile_name` is the
    racing config that produced the run (None for runs logged before profile
    persistence, or forks). `final_age_index` is the rank of `final_age` in
    `AGE_SEQUENCE` so the frontend can sort by the same lexicographic score as
    `arena.ranking.composite_score` without duplicating the age order.
    """

    run_id: str
    profile_name: str | None
    total_cost_usd: float
    n_turns: int
    final_age: str | None
    final_age_index: int | None
    final_population: int | None
    final_economy: float | None  # food + wood


@dataclass(frozen=True, slots=True)
class RunSeriesPoint:
    """One turn's resource snapshot for the overview's per-resource curves."""

    turn: int
    food: float
    wood: float
    gold: float
    stone: float
    population: int


@dataclass(frozen=True, slots=True)
class RunSeries:
    """Per-turn resource trajectory for one run, labelled by its profile so the
    overview can aggregate runs of the same config into a characteristic curve."""

    run_id: str
    profile_name: str | None
    points: list[RunSeriesPoint]


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


# Decimal places kept when rounding a run's summed LLM cost for display —
# sub-cent precision without surfacing float noise.
_COST_USD_PRECISION = 6


def _parse_turn_start(payload_json: str) -> TurnStartPayload:
    """Validate a `turn_start` `payload_json` back into the Pydantic model it was
    serialized from. Gives strongly-typed `profile_name` / `state` access with
    no hand-rolled dict narrowing — validation at the boundary the JSON came
    from. Shared by the summary and series readers."""
    return TurnStartPayload.model_validate_json(payload_json)


def _metrics_from_last_turn(
    run_id: str, n_turns: int, cost: float, last_turn_json: str | None
) -> RunMetrics:
    """Build a `RunMetrics` from precomputed aggregates + the last turn_start
    payload (JSON string, or None when a run has no snapshot)."""
    profile_name: str | None = None
    final_age: str | None = None
    final_age_index: int | None = None
    final_population: int | None = None
    final_economy: float | None = None
    if last_turn_json is not None:
        payload = _parse_turn_start(last_turn_json)
        profile_name = payload.profile_name
        state = payload.state
        if state is not None:
            final_age = state.age
            final_age_index = AGE_SEQUENCE.index(state.age) if state.age in AGE_SEQUENCE else None
            final_population = state.population
            final_economy = state.food + state.wood
    return RunMetrics(
        run_id=run_id,
        profile_name=profile_name,
        total_cost_usd=round(cost, _COST_USD_PRECISION),
        n_turns=n_turns,
        final_age=final_age,
        final_age_index=final_age_index,
        final_population=final_population,
        final_economy=final_economy,
    )


def _run_metrics_in_file(db_path: Path) -> list[RunMetrics]:
    """Per-run comparable metrics for one DuckDB file.

    Two queries: an aggregate (turn count + summed LLM cost) over every run,
    and a window pick of each run's last `turn_start` payload for the final
    state + profile label. Joined in Python by run_id. Files locked by a live
    writer are skipped — their runs become readable once finalized.
    """
    conn = _connect_read_only(db_path)
    if conn is None:
        return []
    try:
        agg_rows = cast(
            "list[tuple[object, ...]]",
            conn.execute(
                "SELECT run_id, MAX(t) AS n_turns, "
                "SUM(CASE WHEN kind='llm_response' "
                "THEN CAST(json_extract(payload_json, '$.cost_usd') AS DOUBLE) "
                "ELSE 0 END) AS cost "
                "FROM events GROUP BY run_id"
            ).fetchall(),
        )
        last_turn_rows = cast(
            "list[tuple[object, ...]]",
            conn.execute(
                "SELECT run_id, payload_json FROM ("
                "  SELECT run_id, payload_json, "
                "  ROW_NUMBER() OVER (PARTITION BY run_id ORDER BY t DESC, ts DESC) AS rn "
                "  FROM events WHERE kind='turn_start'"
                ") WHERE rn = 1"
            ).fetchall(),
        )
    finally:
        conn.close()
    last_payload: dict[str, str] = {str(row[0]): str(row[1]) for row in last_turn_rows}
    metrics: list[RunMetrics] = []
    for row in agg_rows:
        run_id = str(row[0])
        n_turns = int(cast("int", row[1])) if row[1] is not None else 0
        cost = float(cast("float", row[2])) if row[2] is not None else 0.0
        metrics.append(_metrics_from_last_turn(run_id, n_turns, cost, last_payload.get(run_id)))
    return metrics


def _list_run_metrics(root: Path) -> list[RunMetrics]:
    metrics: list[RunMetrics] = []
    for db_path in _all_duckdb_files(root):
        metrics.extend(_run_metrics_in_file(db_path))
    return metrics


def _resolve_logs_path(db_path: str) -> Path | None:
    """Resolve a client-supplied DuckDB path, but only if it lives under the
    logs root and exists. Guards the `/runs/series` query against path
    traversal — the client echoes a `db_path` we handed it via `/runs`, and we
    refuse anything outside the sandbox."""
    root = _logs_root().resolve()
    try:
        candidate = Path(db_path).resolve()
    except (OSError, ValueError):
        return None
    if root not in candidate.parents or not candidate.is_file():
        return None
    return candidate


def _series_in_file(db_path: Path) -> list[RunSeries]:
    """Per-turn resource trajectories for every run in one DuckDB file.

    Reads each run's `turn_start` rows (which carry the full WorldState
    snapshot) in turn order and projects out the resource fields. One query;
    grouping + JSON parse happen in Python.
    """
    conn = _connect_read_only(db_path)
    if conn is None:
        return []
    try:
        rows = cast(
            "list[tuple[object, ...]]",
            conn.execute(
                "SELECT run_id, t, payload_json FROM events "
                "WHERE kind='turn_start' ORDER BY run_id, t"
            ).fetchall(),
        )
    finally:
        conn.close()

    points_by_run: dict[str, list[RunSeriesPoint]] = {}
    profile_by_run: dict[str, str | None] = {}
    for run_id_obj, t_obj, payload_obj in rows:
        run_id = str(run_id_obj)
        payload = _parse_turn_start(str(payload_obj))
        if run_id not in profile_by_run:
            profile_by_run[run_id] = payload.profile_name
        state = payload.state
        if state is None:
            continue
        points_by_run.setdefault(run_id, []).append(
            RunSeriesPoint(
                turn=int(t_obj) if isinstance(t_obj, int) else 0,
                food=state.food,
                wood=state.wood,
                gold=state.gold,
                stone=state.stone,
                population=state.population,
            )
        )
    return [
        RunSeries(run_id=rid, profile_name=profile_by_run.get(rid), points=pts)
        for rid, pts in points_by_run.items()
    ]


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


@app.get("/runs/summaries")
async def run_summaries() -> list[dict[str, object]]:
    """Per-run comparable metrics for the dashboard's experiment overview.

    Finalized (cold) runs only — a live operation's DuckDB file is held by its
    writer, so its rows appear once it finalizes. Scans every log file once per
    call; the dashboard fetches this on mount, like /runs.
    """
    metrics = await asyncio.to_thread(_list_run_metrics, _logs_root())
    return [asdict(m) for m in metrics]


@app.get("/runs/series")
async def run_series(db_path: str = Query(..., min_length=1)) -> list[dict[str, object]]:
    """Per-turn resource trajectories for every run in one operation's file.

    Scoped to a single DuckDB file (one operation) — the overview passes the
    `db_path` it received from /runs. Returns the unaggregated per-run series so
    the frontend can aggregate by profile however it likes.
    """
    resolved = _resolve_logs_path(db_path)
    if resolved is None:
        raise HTTPException(status_code=404, detail=f"no readable log at {db_path!r}")
    series = await asyncio.to_thread(_series_in_file, resolved)
    return [asdict(s) for s in series]


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
