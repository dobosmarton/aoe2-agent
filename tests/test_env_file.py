"""The `.env` loader that feeds `config` before it builds its global."""

from __future__ import annotations

import os
from pathlib import Path  # noqa: TC003  -- runtime use: tmp_path fixture annotations

import pytest
from gameplay_agent.env_file import load_env_file


@pytest.fixture(autouse=True)
def _isolated_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    """`load_env_file` writes to `os.environ` directly, so hand it a copy.

    `monkeypatch.delenv` cannot undo a key the loader *added*, so isolate the
    whole mapping instead of the one name.
    """
    monkeypatch.setattr(os, "environ", dict(os.environ))


def _write_env(directory: Path, body: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / ".env").write_text(body, encoding="utf-8")


def test_a_value_reaches_the_environment(tmp_path: Path) -> None:
    _write_env(tmp_path, "AOE2_PROBE=from-file\n")
    load_env_file(tmp_path)
    assert os.environ["AOE2_PROBE"] == "from-file"


def test_an_ancestor_file_is_found(tmp_path: Path) -> None:
    """The agent lives in `apps/agent/src`; the documented `.env` sits at the root."""
    _write_env(tmp_path, "AOE2_PROBE=from-root\n")
    nested = tmp_path / "apps" / "agent" / "src"
    nested.mkdir(parents=True)
    load_env_file(nested)
    assert os.environ["AOE2_PROBE"] == "from-root"


def test_an_exported_value_wins(tmp_path: Path) -> None:
    """A one-off `set AOE2_PROBE=...` must still override the file."""
    os.environ["AOE2_PROBE"] = "from-shell"
    _write_env(tmp_path, "AOE2_PROBE=from-file\n")
    load_env_file(tmp_path)
    assert os.environ["AOE2_PROBE"] == "from-shell"


def test_a_missing_file_does_not_raise(tmp_path: Path) -> None:
    """A checkout with no `.env` must still start."""
    load_env_file(tmp_path)


@pytest.mark.parametrize(
    "body",
    ['AOE2_PROBE="quoted"\n', "AOE2_PROBE='quoted'\n"],
    ids=["double-quotes", "single-quotes"],
)
def test_quotes_are_stripped(tmp_path: Path, body: str) -> None:
    _write_env(tmp_path, body)
    load_env_file(tmp_path)
    assert os.environ["AOE2_PROBE"] == "quoted"


@pytest.mark.parametrize(
    "line",
    ["# AOE2_PROBE=commented\n", "AOE2_PROBE\n", "\n"],
    ids=["comment", "no-equals", "blank"],
)
def test_a_non_assignment_line_sets_nothing(tmp_path: Path, line: str) -> None:
    _write_env(tmp_path, line)
    load_env_file(tmp_path)
    assert "AOE2_PROBE" not in os.environ
