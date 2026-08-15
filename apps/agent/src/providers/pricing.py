"""Per-model token pricing — the single source for every cost figure.

Two divergent tables used to exist: one hardcoded to a single model inside the
executor, and one in the arena that carried no cache rates and silently priced
unknown models at $0.00 (so a newly-added model looked free). Both also had
Claude Haiku 4.5 at $0.80/$4.00 against its actual $1.00/$5.00. This module
replaces both.

Rates are US dollars per million tokens, from the vendors' published first-party
pricing (checked 2026-08-15). `cache_read` and `cache_write` follow the standard
multipliers where a vendor does not publish them separately: reads at 0.1x input,
writes at 1.25x input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import structlog

if TYPE_CHECKING:
    from .base import TokenUsage

log = structlog.stdlib.get_logger()

_PER_MILLION: Final = 1_000_000

# Applied where a vendor does not publish cache rates separately.
_CACHE_READ_MULTIPLIER: Final = 0.1
_CACHE_WRITE_MULTIPLIER: Final = 1.25
_RATE_DECIMALS: Final = 4


@dataclass(frozen=True, slots=True)
class ModelPrice:
    """Dollars per million tokens for one model."""

    input: float
    output: float
    cache_read: float
    cache_write: float


def _standard(input_rate: float, output_rate: float) -> ModelPrice:
    """A price using the default cache multipliers."""
    return ModelPrice(
        input=input_rate,
        output=output_rate,
        cache_read=round(input_rate * _CACHE_READ_MULTIPLIER, _RATE_DECIMALS),
        cache_write=round(input_rate * _CACHE_WRITE_MULTIPLIER, _RATE_DECIMALS),
    )


_PRICES: Final[dict[str, ModelPrice]] = {
    # -- Anthropic ----------------------------------------------------------
    "claude-opus-5": _standard(5.00, 25.00),
    "claude-opus-4-7": _standard(5.00, 25.00),
    # Sonnet 5 runs a $2.00/$10.00 introductory rate through 2026-08-31; the
    # standard rate is used here so cost never reads low after it lapses.
    "claude-sonnet-5": _standard(3.00, 15.00),
    "claude-sonnet-4-6": _standard(3.00, 15.00),
    "claude-haiku-4-5": _standard(1.00, 5.00),
    "claude-haiku-4-5-20251001": _standard(1.00, 5.00),
    # -- OpenAI -------------------------------------------------------------
    # Luna's published cached-input rate is $0.02, matching the 0.1x default.
    "gpt-5.6-luna": _standard(0.20, 1.20),
    "gpt-5.6-terra": _standard(2.00, 12.00),
    # -- Other OpenAI-compatible models reachable through the gateway -------
    # Vendor listings disagree on Kimi ($0.67/$3.40 vs $0.95/$4.00); the higher
    # pair is used so a cost estimate errs high rather than low.
    "kimi-k2.7-code": _standard(0.95, 4.00),
}

_UNKNOWN = ModelPrice(input=0.0, output=0.0, cache_read=0.0, cache_write=0.0)


def price_for(model: str) -> ModelPrice:
    """Return the price for `model`; an unknown one logs and costs $0.00.

    The log matters: a silent zero is how a newly-added model looks free in
    `results.tsv`.
    """
    price = _PRICES.get(model)
    if price is None:
        log.warning("pricing_unknown_model", model=model)
        return _UNKNOWN
    return price


def cost_usd(model: str, usage: TokenUsage) -> float:
    """Cost of `usage` on `model`, in dollars."""
    price = price_for(model)
    return (
        usage.input_tokens * price.input
        + usage.output_tokens * price.output
        + usage.cache_read_tokens * price.cache_read
        + usage.cache_write_tokens * price.cache_write
    ) / _PER_MILLION


__all__ = ["ModelPrice", "cost_usd", "price_for"]
