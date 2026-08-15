"""Unit tests for gameplay_agent/config.py — determinism knobs and adapter choice.

Uses pytest's `monkeypatch` so env-var mutations are scoped to the test and
do not leak across the suite.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# AOE2_TEMPERATURE
# ---------------------------------------------------------------------------


def test_from_env_parses_temperature(monkeypatch: pytest.MonkeyPatch) -> None:
    from gameplay_agent.config import Config

    monkeypatch.setenv("AOE2_TEMPERATURE", "0.7")
    assert Config.from_env().temperature == 0.7


def test_from_env_temperature_defaults_to_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    from gameplay_agent.config import Config

    monkeypatch.delenv("AOE2_TEMPERATURE", raising=False)
    assert Config.from_env().temperature == 0.0


# ---------------------------------------------------------------------------
# AOE2_SEED
# ---------------------------------------------------------------------------


def test_from_env_parses_seed_as_int(monkeypatch: pytest.MonkeyPatch) -> None:
    from gameplay_agent.config import Config

    monkeypatch.setenv("AOE2_SEED", "42")
    assert Config.from_env().seed == 42


def test_from_env_seed_unset_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    from gameplay_agent.config import Config

    monkeypatch.delenv("AOE2_SEED", raising=False)
    assert Config.from_env().seed is None


def test_from_env_seed_empty_string_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    from gameplay_agent.config import Config

    monkeypatch.setenv("AOE2_SEED", "")
    assert Config.from_env().seed is None


# ---------------------------------------------------------------------------
# AOE2_EXECUTOR_EFFORT
# ---------------------------------------------------------------------------


def test_from_env_parses_executor_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    from gameplay_agent.config import Config

    monkeypatch.setenv("AOE2_EXECUTOR_EFFORT", "high")
    assert Config.from_env().executor_effort == "high"


def test_executor_effort_defaults_to_low(monkeypatch: pytest.MonkeyPatch) -> None:
    from gameplay_agent.config import Config

    monkeypatch.delenv("AOE2_EXECUTOR_EFFORT", raising=False)
    assert Config.from_env().executor_effort == "low"


def test_executor_effort_invalid_falls_back_to_low(monkeypatch: pytest.MonkeyPatch) -> None:
    from gameplay_agent.config import Config

    # xhigh/max are real effort levels but Sonnet 4.6 rejects them, so the
    # parser treats anything outside {low, medium, high} as "low".
    monkeypatch.setenv("AOE2_EXECUTOR_EFFORT", "xhigh")
    assert Config.from_env().executor_effort == "low"


# ---------------------------------------------------------------------------
# Wire selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["anthropic", "openai", "zen"])
def test_every_adapter_name_survives_parsing(name: str) -> None:
    """A name the factory accepts must not be silently rewritten to the default."""
    from gameplay_agent.config import _parse_wire

    assert _parse_wire(name) == name


def test_unknown_wire_name_falls_back_to_the_default() -> None:
    from gameplay_agent.config import _parse_wire

    assert _parse_wire("gemini") == "openai"


def test_missing_wire_name_falls_back_to_the_default() -> None:
    from gameplay_agent.config import _parse_wire

    assert _parse_wire(None) == "openai"


def test_base_url_is_unset_by_default() -> None:
    """Empty means "use the adapter's endpoint"; a URL here would pin every wire."""
    from gameplay_agent.config import Config

    assert Config().llm_base_url == ""
