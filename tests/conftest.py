"""Pytest configuration for the evaluation framework.

Registers the `live` marker (used by tests that call the real Anthropic API)
and adds a --runlive flag that gates whether those tests actually run.

The project is installed as a package (`pip install -e .`), so test files
import siblings as `from gameplay_agent.X import Y` directly — no sys.path
manipulation required.
"""

import sys
from unittest.mock import MagicMock

import pytest

# pyautogui reads $DISPLAY at import time on Linux. On a headless CI runner
# that raises KeyError during test collection — even for tests that never
# actually drive the mouse. If the real import fails, install a MagicMock
# so any `import pyautogui` downstream gets a no-op shim. Tests that need
# specific pyautogui behavior continue to install their own fakes via
# monkeypatch.setattr(executor, "pyautogui", ...).
try:
    import pyautogui  # noqa: F401
except (KeyError, ImportError):
    sys.modules.pop("pyautogui", None)
    sys.modules["pyautogui"] = MagicMock()


def pytest_addoption(parser):
    parser.addoption(
        "--runlive",
        action="store_true",
        default=False,
        help="Run live scenario tests (requires ANTHROPIC_API_KEY, costs ~$0.50)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "live: scenario test that calls the real Anthropic API")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--runlive"):
        skip_live = pytest.mark.skip(reason="need --runlive flag")
        for item in items:
            if "live" in item.keywords:
                item.add_marker(skip_live)
