"""Console logs for a person, and the same events as JSON lines in a file.

Each run writes `logs/runs/<UTC start time>.jsonl` (one event per line, no colour
codes), so a game can be analysed after it ends without copying the console.
`logs/` is ignored by git: these files hold game state.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, TextIO

import structlog

if TYPE_CHECKING:
    from collections.abc import MutableMapping
    from pathlib import Path

_SHARED_PROCESSORS = (
    structlog.stdlib.add_log_level,
    structlog.processors.TimeStamper(fmt="iso"),
    structlog.processors.StackInfoRenderer(),
    structlog.processors.format_exc_info,
)


class TeeRenderer:
    """Writes each event to the run file as JSON, then renders it for the console."""

    def __init__(self, file: TextIO) -> None:
        self._file = file
        self._json = structlog.processors.JSONRenderer(default=str)
        self._console = structlog.dev.ConsoleRenderer(colors=True)

    def __call__(
        self, logger: object, method_name: str, event_dict: MutableMapping[str, object]
    ) -> str:
        rendered = self._json(logger, method_name, dict(event_dict))
        self._file.write((rendered if isinstance(rendered, str) else rendered.decode()) + "\n")
        return self._console(logger, method_name, event_dict)


def configure_logging(log_dir: Path | None = None) -> Path | None:
    """Console logging, plus a JSON-lines run file under `log_dir` when given.

    Returns the run file's path, or None without one.
    """
    run_file: Path | None = None
    renderer: object = structlog.dev.ConsoleRenderer(colors=True)
    if log_dir is not None:
        runs = log_dir / "runs"
        runs.mkdir(parents=True, exist_ok=True)
        run_file = runs / f"{datetime.now(UTC).strftime('%Y-%m-%dT%H-%M-%SZ')}.jsonl"
        renderer = TeeRenderer(run_file.open("a", encoding="utf-8", buffering=1))
    structlog.configure(
        processors=[*_SHARED_PROCESSORS, renderer],  # pyright: ignore[reportArgumentType]
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    return run_file


__all__ = ["TeeRenderer", "configure_logging"]
