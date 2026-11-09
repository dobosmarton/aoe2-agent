"""Each run's events land in a JSON-lines file as well as on the console."""

from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING, cast

import structlog
from gameplay_agent.logging_setup import TeeRenderer, configure_logging

if TYPE_CHECKING:
    from pathlib import Path


def test_the_tee_writes_json_and_returns_the_console_line() -> None:
    file = io.StringIO()
    line = TeeRenderer(file)(None, "info", {"event": "press", "key": "h", "level": "info"})
    assert json.loads(file.getvalue()) == {"event": "press", "key": "h", "level": "info"}
    assert "press" in line
    assert "key" in line


def test_a_run_file_is_created_under_the_log_dir(tmp_path: Path) -> None:
    try:
        run_file = configure_logging(tmp_path)
        assert run_file is not None
        assert run_file.parent == tmp_path / "runs"
        assert run_file.suffix == ".jsonl"
        structlog.stdlib.get_logger("x").info("run_started", players=8)
        event = cast("dict[str, object]", json.loads(run_file.read_text().splitlines()[-1]))
        assert (event["event"], event["players"], event["level"]) == ("run_started", 8, "info")
        assert "timestamp" in event
    finally:
        structlog.reset_defaults()  # Other tests read structlog's plain default output.
