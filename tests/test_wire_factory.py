"""Unit tests for gameplay_agent/providers/wire_factory.py and pricing.py.

Mirrors tests/test_broker_factory.py: every selection branch, input tolerance,
and the error message treated as an operator-facing contract.
"""

from __future__ import annotations

import pytest
from gameplay_agent.config import KEY_ENV, Config
from gameplay_agent.providers.base import TokenUsage
from gameplay_agent.providers.pricing import cost_usd, price_for
from gameplay_agent.providers.wire_factory import ZEN_BASE_URL, make_text_completer, make_wire

_WIRE_METHODS = ("tool_turn", "parse_structured", "is_api_error", "is_schema_too_large")


def test_anthropic_branch_returns_an_anthropic_wire() -> None:
    wire = make_wire("anthropic", model="claude-sonnet-4-6", api_key="k")
    assert type(wire).__name__ == "AnthropicWire"
    assert wire.model == "claude-sonnet-4-6"


def test_openai_branch_honours_the_base_url() -> None:
    wire = make_wire("openai", model="gpt-5.6-luna", api_key="k", base_url="http://zen/v1")
    assert type(wire).__name__ == "OpenAIWire"
    assert str(wire.client.base_url).startswith("http://zen/v1")


@pytest.mark.parametrize("name", ["  ANTHROPIC ", "Anthropic", "anthropic"])
def test_name_tolerates_case_and_whitespace(name: str) -> None:
    assert type(make_wire(name, model="m", api_key="k")).__name__ == "AnthropicWire"


@pytest.mark.parametrize("name", ["anthropic", "openai", "zen"])
def test_both_wires_satisfy_the_protocol(name: str) -> None:
    """ChatWire is not runtime_checkable, so conformance is checked by shape."""
    wire = make_wire(name, model="m", api_key="k")
    assert all(callable(getattr(wire, method)) for method in _WIRE_METHODS)


def test_unknown_wire_raises_rather_than_falling_back() -> None:
    """Silently defaulting would run a whole game on the wrong vendor."""
    with pytest.raises(ValueError, match=r"'anthropic'.*'openai'.*'zen'"):
        make_wire("gemini", model="m")


# ---------------------------------------------------------------------------
# Endpoints: `openai` and `zen` share a transport and differ only here
# ---------------------------------------------------------------------------


def test_zen_branch_supplies_its_own_endpoint() -> None:
    """The whole point of the third name: one variable, not a wire plus a URL."""
    wire = make_wire("zen", model="gpt-5.6-luna", api_key="k")
    assert str(wire.endpoint).rstrip("/") == ZEN_BASE_URL


def test_openai_branch_defaults_to_the_vendor_endpoint() -> None:
    wire = make_wire("openai", model="gpt-5.6-luna", api_key="k")
    assert "api.openai.com" in str(wire.endpoint)


def test_explicit_base_url_overrides_the_zen_endpoint() -> None:
    """A staging gateway must still win over the adapter's default."""
    wire = make_wire("zen", model="m", api_key="k", base_url="http://staging/v1")
    assert str(wire.endpoint).startswith("http://staging/v1")


def test_empty_base_url_does_not_reach_the_sdk() -> None:
    """config supplies "" for "unset"; passing it through breaks every request."""
    wire = make_wire("openai", model="m", api_key="k", base_url="")
    assert "api.openai.com" in str(wire.endpoint)


def test_zen_text_completer_shares_the_zen_endpoint() -> None:
    """Forgetting this arm leaves memory extraction pointed at the wrong host."""
    completer = make_text_completer("zen", model="m", api_key="k")
    assert str(completer.client.base_url).rstrip("/") == ZEN_BASE_URL


def test_unknown_text_completer_raises() -> None:
    with pytest.raises(ValueError, match=r"'zen'"):
        make_text_completer("gemini", model="m")


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------


def test_haiku_price_matches_the_published_rate() -> None:
    """Both former tables had this at $0.80/$4.00; the real rate is $1.00/$5.00."""
    price = price_for("claude-haiku-4-5")
    assert (price.input, price.output) == (1.00, 5.00)


def test_luna_is_priced_and_far_cheaper_than_the_sonnet_baseline() -> None:
    usage = TokenUsage(input_tokens=200_000, output_tokens=100_000, cache_read_tokens=700_000)
    luna = cost_usd("gpt-5.6-luna", usage)
    sonnet = cost_usd("claude-sonnet-4-6", usage)
    assert luna == pytest.approx(0.174, abs=1e-3)
    assert sonnet == pytest.approx(2.31, abs=1e-2)


# ---------------------------------------------------------------------------
# Credential resolution
# ---------------------------------------------------------------------------


def test_one_key_serves_whichever_wire_is_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """AOE2_LLM_API_KEY is the only credential; no per-vendor variable exists."""
    monkeypatch.setenv(KEY_ENV, "sk-generic")
    assert Config.from_env().llm_api_key == "sk-generic"


def test_missing_key_is_empty_rather_than_a_stale_vendor_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A leftover ANTHROPIC_API_KEY must not silently satisfy the check."""
    monkeypatch.delenv(KEY_ENV, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-stale")
    assert Config.from_env().llm_api_key == ""


def test_unknown_model_warns_instead_of_silently_costing_nothing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A silent zero is how a new model ends up looking free in results.tsv.

    structlog is configured with a PrintLoggerFactory, so the event lands on
    stdout rather than through stdlib logging.
    """
    assert cost_usd("some-new-model", TokenUsage(input_tokens=1_000_000)) == 0.0
    assert "pricing_unknown_model" in capsys.readouterr().out
