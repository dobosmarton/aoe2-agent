"""Tests for arena/web/server.py (Phase 7.1)."""

from __future__ import annotations

import importlib
import json
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import duckdb
import pytest
from evaluation.event_log import DuckDBEventSink, Event, MetricPayload, TurnStartPayload
from fastapi.testclient import TestClient

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path


def _make_log(db_path: Path, run_id: str, n_events: int = 3) -> None:
    """Fabricate a DuckDB log file with `n_events` events for `run_id`."""
    conn = duckdb.connect(str(db_path))
    try:
        sink = DuckDBEventSink(conn)
        base_ts = datetime.now(UTC)
        for t in range(1, n_events + 1):
            payload = (
                TurnStartPayload(turn_num=t)
                if t == 1
                else MetricPayload(name=f"m_{t}", value=float(t))
            )
            sink.emit(
                Event(
                    run_id=run_id,
                    agent_id="agent-0",
                    t=t,
                    payload=payload,
                    ts=base_ts,
                )
            )
    finally:
        conn.close()


@pytest.fixture
def logs_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Per-test logs root via env var; re-import server so middleware sees it."""
    root = tmp_path / "logs" / "arena"
    (root / "2026-05-20").mkdir(parents=True)
    monkeypatch.setenv("ARENA_LOGS_ROOT", str(root))
    return root


@pytest.fixture
def client(logs_root: Path) -> Iterator[TestClient]:
    # Re-import to pick up env-var-driven CORS list and a fresh app instance
    # per test (avoids middleware state leakage between fixtures).
    from arena_web import server as server_module

    importlib.reload(server_module)
    with TestClient(server_module.app) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_returns_ok(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# /runs
# ---------------------------------------------------------------------------


def test_runs_lists_known_run(client: TestClient, logs_root: Path) -> None:
    _make_log(logs_root / "2026-05-20" / "race-120000.duckdb", run_id="alpha")
    payload = client.get("/runs").json()
    assert [r["run_id"] for r in payload] == ["alpha"]


def test_runs_is_empty_when_no_logs(client: TestClient) -> None:
    assert client.get("/runs").json() == []


def test_runs_orders_newest_first(client: TestClient, logs_root: Path) -> None:
    older = logs_root / "2026-05-20" / "race-100000.duckdb"
    newer = logs_root / "2026-05-20" / "race-110000.duckdb"
    _make_log(older, run_id="older")
    time.sleep(0.05)  # ensure measurable mtime delta on fast filesystems
    _make_log(newer, run_id="newer")
    payload = client.get("/runs").json()
    assert payload[0]["run_id"] == "newer"


def test_runs_reports_label_from_filename(client: TestClient, logs_root: Path) -> None:
    _make_log(logs_root / "2026-05-20" / "smoke-120000.duckdb", run_id="r1")
    assert client.get("/runs").json()[0]["label"] == "smoke"


# ---------------------------------------------------------------------------
# /events
# ---------------------------------------------------------------------------


def test_events_streams_sse_format(client: TestClient, logs_root: Path) -> None:
    _make_log(logs_root / "2026-05-20" / "race-120000.duckdb", run_id="r1")
    body = client.get("/events", params={"run_id": "r1"}).text
    assert body.startswith("data: {")


def test_events_orders_by_t(client: TestClient, logs_root: Path) -> None:
    _make_log(logs_root / "2026-05-20" / "race-120000.duckdb", run_id="r1", n_events=4)
    lines = [
        line
        for line in client.get("/events", params={"run_id": "r1"}).text.splitlines()
        if line.startswith("data: ")
    ]
    payloads = [json.loads(line.removeprefix("data: ")) for line in lines]
    # First event is TurnStartPayload (kind=turn_start, turn_num=1); the rest
    # are MetricPayloads with values 2..n increasing.
    metric_values = [p["value"] for p in payloads if p["kind"] == "metric"]
    assert metric_values == sorted(metric_values)


def test_events_404_for_unknown_run(client: TestClient, logs_root: Path) -> None:
    _make_log(logs_root / "2026-05-20" / "race-120000.duckdb", run_id="known")
    assert client.get("/events", params={"run_id": "missing"}).status_code == 404


def test_events_payload_is_valid_json(client: TestClient, logs_root: Path) -> None:
    _make_log(logs_root / "2026-05-20" / "race-120000.duckdb", run_id="r1", n_events=2)
    payloads = [
        line.removeprefix("data: ")
        for line in client.get("/events", params={"run_id": "r1"}).text.splitlines()
        if line.startswith("data: ")
    ]
    # If any payload isn't valid JSON, this raises.
    assert all(isinstance(json.loads(p), dict) for p in payloads)


def test_cors_allows_localhost_5173(client: TestClient) -> None:
    response = client.options(
        "/runs",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"


# ---------------------------------------------------------------------------
# /metrics (Phase 3)
# ---------------------------------------------------------------------------


def test_metrics_returns_zeroed_counters_on_fresh_server(client: TestClient) -> None:
    """No traffic yet — all four counters are 0; the route exists and the
    schema matches BrokerMetricsSnapshot's fields."""
    payload = client.get("/metrics").json()
    assert payload == {
        "events_published": 0,
        "events_streamed": 0,
        "streams_dropped": 0,
        "runs_open": 0,
    }


def test_metrics_reflects_broker_publish_activity(
    client: TestClient, build_event: Callable[..., Event]
) -> None:
    """Drive traffic through the app's broker and assert the counters tick.

    Proves the `/metrics` route reads live state — not a cached snapshot
    — by reaching through to `app.state.broker` and running a single
    async batch of publishes via `asyncio.run`."""
    import asyncio

    from arena_web import server as server_module
    from evaluation.event_broker import InProcessEventBroker, RunId

    broker = server_module.app.state.broker
    assert isinstance(broker, InProcessEventBroker)
    run = RunId("m1")
    broker.open_run(run)

    async def publish_batch() -> None:
        for i in range(3):
            await broker.publish(run, build_event(run, t=i))

    asyncio.run(publish_batch())

    payload = client.get("/metrics").json()
    assert payload["events_published"] == 3
    assert payload["runs_open"] == 1

    # Cleanup — `client` is function-scoped (TestClient ctx), but the broker
    # lives in app.state which persists for the duration of the test's
    # TestClient `with` block; explicit close+reap keeps the test
    # hermetic if more metrics tests are added later.
    broker.close_run(run)
    broker.reap(run)
