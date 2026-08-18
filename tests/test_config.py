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


def test_from_env_temperature_defaults_to_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset means "do not send it". A hardcoded 0.0 made every gpt-5.6 call a 400."""
    from gameplay_agent.config import Config

    monkeypatch.delenv("AOE2_TEMPERATURE", raising=False)
    assert Config.from_env().temperature is None


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
# AOE2_LLM_WIRE
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["anthropic", "openai", "zen"])
def test_from_env_accepts_every_adapter_name(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A name the factory has an arm for must survive parsing unchanged."""
    from gameplay_agent.config import Config

    monkeypatch.setenv("AOE2_LLM_WIRE", name)
    assert Config.from_env().llm_wire == name


@pytest.mark.parametrize("raw", ["  ANTHROPIC ", "Anthropic", "anthropic"])
def test_from_env_tolerates_case_and_whitespace(raw: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from gameplay_agent.config import Config

    monkeypatch.setenv("AOE2_LLM_WIRE", raw)
    assert Config.from_env().llm_wire == "anthropic"


def test_unknown_wire_raises_rather_than_falling_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """Silently defaulting would run a whole game on the wrong vendor."""
    from gameplay_agent.config import Config

    monkeypatch.setenv("AOE2_LLM_WIRE", "gemini")
    with pytest.raises(ValueError, match=r"'anthropic'.*'openai'.*'zen'"):
        _ = Config.from_env()


def test_missing_wire_falls_back_to_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from gameplay_agent.config import Config

    monkeypatch.delenv("AOE2_LLM_WIRE", raising=False)
    assert Config.from_env().llm_wire == "openai"


def test_empty_wire_falls_back_to_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """An exported-but-empty var means "unset", not "invalid"."""
    from gameplay_agent.config import Config

    monkeypatch.setenv("AOE2_LLM_WIRE", "   ")
    assert Config.from_env().llm_wire == "openai"


# ---------------------------------------------------------------------------
# AOE2_LLM_BASE_URL
# ---------------------------------------------------------------------------


def test_base_url_is_unset_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty means "use the adapter's endpoint"; a URL here would pin every wire."""
    from gameplay_agent.config import Config

    monkeypatch.delenv("AOE2_LLM_BASE_URL", raising=False)
    assert Config.from_env().llm_base_url == ""


# ---------------------------------------------------------------------------
# Model defaults per wire
# ---------------------------------------------------------------------------


@pytest.fixture
def unpinned_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear the 3 model overrides; a shell export would mask the default."""
    for name in ("AOE2_MODEL", "AOE2_STRATEGIST_MODEL", "AOE2_MEMORY_MODEL"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.usefixtures("unpinned_models")
def test_openai_wire_runs_the_fast_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    from gameplay_agent.config import Config

    monkeypatch.setenv("AOE2_LLM_WIRE", "openai")
    assert Config.from_env().model == "gpt-5.6-luna"


@pytest.mark.usefixtures("unpinned_models")
def test_openai_wire_runs_the_strong_strategist(monkeypatch: pytest.MonkeyPatch) -> None:
    from gameplay_agent.config import Config

    monkeypatch.setenv("AOE2_LLM_WIRE", "openai")
    assert Config.from_env().strategist_model == "gpt-5.6-terra"


@pytest.mark.usefixtures("unpinned_models")
def test_anthropic_wire_runs_the_fast_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    """A gpt name against the Anthropic endpoint fails every call."""
    from gameplay_agent.config import Config

    monkeypatch.setenv("AOE2_LLM_WIRE", "anthropic")
    assert Config.from_env().model == "claude-haiku-4-5"


@pytest.mark.usefixtures("unpinned_models")
def test_anthropic_wire_runs_the_strong_strategist(monkeypatch: pytest.MonkeyPatch) -> None:
    from gameplay_agent.config import Config

    monkeypatch.setenv("AOE2_LLM_WIRE", "anthropic")
    assert Config.from_env().strategist_model == "claude-sonnet-5"


def test_strategist_override_beats_the_wire_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from gameplay_agent.config import Config

    monkeypatch.setenv("AOE2_LLM_WIRE", "openai")
    monkeypatch.setenv("AOE2_STRATEGIST_MODEL", "kimi-k2.7-code")
    assert Config.from_env().strategist_model == "kimi-k2.7-code"


def test_empty_model_falls_back_to_the_wire_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """An exported-but-empty var means "unset", the same rule AOE2_LLM_WIRE follows."""
    from gameplay_agent.config import Config

    monkeypatch.setenv("AOE2_LLM_WIRE", "openai")
    monkeypatch.setenv("AOE2_MODEL", "")
    assert Config.from_env().model == "gpt-5.6-luna"


@pytest.mark.parametrize("wire", ["anthropic", "openai", "zen"])
@pytest.mark.usefixtures("unpinned_models")
def test_every_default_model_has_a_price(wire: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unpriced default reads $0.00, which makes a run look free."""
    from gameplay_agent.config import Config
    from gameplay_agent.providers.pricing import price_for

    monkeypatch.setenv("AOE2_LLM_WIRE", wire)
    resolved = Config.from_env()
    unpriced = [
        name
        for name in (resolved.model, resolved.strategist_model, resolved.memory_model)
        if price_for(name).input == 0.0
    ]
    assert unpriced == []
