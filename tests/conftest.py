"""Pytest configuration for the evaluation framework.

Registers the `live` marker (used by tests that call the real Anthropic API)
and adds a --runlive flag that gates whether those tests actually run.

Exposes the `build_event` factory fixture used by broker/persister/SSE
tests to construct minimal `turn_start` events without each file
re-declaring its own copy.

The project is installed as a package (`pip install -e .`), so test files
import siblings as `from gameplay_agent.X import Y` directly — no sys.path
manipulation required.
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

# Set BEFORE any gameplay_agent import: `config` is a module-level singleton
# read once at import time. The OpenAI SDK raises at construction when no key is
# present, so a provider built in a test would fail before reaching its stub.
# `setdefault` leaves a real key alone for the --runlive tests.
os.environ.setdefault("AOE2_LLM_API_KEY", "test-key-not-used")

import pytest
from evaluation.event_log import Event, TurnStartPayload

if TYPE_CHECKING:
    from collections.abc import Callable

# Native-GUI deps that don't load on a headless Linux CI runner:
#   - pyautogui reads $DISPLAY at import time → KeyError on Linux without X.
#   - pygetwindow raises NotImplementedError on Linux ("does not support Linux").
# When the real import fails for any reason, install a MagicMock so downstream
# `import pyautogui` / `import pygetwindow` gets a no-op shim. Tests that need
# specific behavior continue to install their own fakes via monkeypatch.
for _mod_name in ("pyautogui", "pygetwindow"):
    try:
        __import__(_mod_name)
    except Exception:
        sys.modules.pop(_mod_name, None)
        sys.modules[_mod_name] = MagicMock()


def pytest_addoption(parser):
    parser.addoption(
        "--runlive",
        action="store_true",
        default=False,
        help="Run live scenario tests (requires AOE2_LLM_API_KEY, costs ~$0.50)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "live: scenario test that calls the real Anthropic API")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--runlive"):
        skip_live = pytest.mark.skip(reason="need --runlive flag")
        for item in items:
            if "live" in item.keywords:
                item.add_marker(skip_live)


# ---------------------------------------------------------------------------
# Shared test factory: minimal `turn_start` Event.
# ---------------------------------------------------------------------------


@pytest.fixture
def build_event() -> Callable[..., Event]:
    """Factory fixture that returns a callable for constructing test Events.

    Defaults satisfy broker/persister/SSE tests; pass `agent_id=` or
    `ts=` explicitly when a test depends on those values (e.g. the
    cross-broker collision test pins a specific 09:00 timestamp).
    """

    def _build(
        run_id: str = "r1",
        t: int = 0,
        *,
        agent_id: str = "agent_x",
        ts: datetime | None = None,
    ) -> Event:
        return Event(
            run_id=run_id,
            agent_id=agent_id,
            t=t,
            payload=TurnStartPayload(turn_num=t),
            ts=ts or datetime(2026, 5, 21, 12, 0, 0, tzinfo=UTC),
        )

    return _build
