"""Pytest configuration for the evaluation framework.

Registers the `live` marker (used by tests that call the real Anthropic API),
adds a --runlive flag that gates whether those tests actually run, and adds
the repo root to sys.path so individual test files can import sibling packages
(`evaluation`, `autoresearch`, `gameplay_agent`) without per-file boilerplate.
"""

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


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
