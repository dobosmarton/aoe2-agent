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
