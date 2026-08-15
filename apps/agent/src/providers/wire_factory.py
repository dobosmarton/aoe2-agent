"""The single point where an LLM wire is chosen by name.

Mirrors `evaluation.broker_factory`: one match arm per implementation, a lazy
import inside each arm so neither SDK becomes mandatory, and a `ValueError`
naming the valid choices rather than a silent fallback — a mistyped wire name
should fail at startup, not quietly run the wrong vendor for a whole game.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, get_args

from .base import WireName

if TYPE_CHECKING:
    from .base import ChatWire
    from .text_wire import TextCompleter

WIRE_ENV: Final = "AOE2_LLM_WIRE"
# Derived from the Literal so the valid set is written once.
_VALID: Final = get_args(WireName)


def make_wire(
    name: str,
    model: str,
    api_key: str = "",
    base_url: str | None = None,
    max_retries: int = 3,
) -> ChatWire:
    """Build the wire called `name` for `model`.

    An empty `api_key` falls back to the SDK's own env lookup. `base_url` only
    matters on the OpenAI wire (OpenCode Zen vs api.openai.com).
    """
    normalized = name.strip().lower()

    if normalized == "anthropic":
        # Local imports: keep each vendor SDK out of the other's dependency path.
        from .wire_anthropic import AnthropicWire

        return AnthropicWire(model=model, api_key=api_key or None, max_retries=max_retries)

    if normalized == "openai":
        from .wire_openai import OpenAIWire

        return OpenAIWire(
            model=model,
            api_key=api_key or None,
            base_url=base_url,
            max_retries=max_retries,
        )

    raise _unknown(name)


def make_text_completer(
    name: str,
    model: str,
    api_key: str = "",
    base_url: str | None = None,
) -> TextCompleter:
    """Build the synchronous text completer called `name` for `model`.

    Same selection rules as `make_wire`, for the blocking prompt-in/text-out
    callers (memory extraction, prompt mutation).
    """
    normalized = name.strip().lower()

    if normalized == "anthropic":
        from .text_wire import AnthropicTextCompleter

        return AnthropicTextCompleter(model=model, api_key=api_key or None)

    if normalized == "openai":
        from .text_wire import OpenAITextCompleter

        return OpenAITextCompleter(model=model, api_key=api_key or None, base_url=base_url)

    raise _unknown(name)


def _unknown(name: str) -> ValueError:
    expected = " or ".join(repr(choice) for choice in _VALID)
    return ValueError(f"unknown {WIRE_ENV}={name!r}; expected {expected}")


__all__ = ["WIRE_ENV", "make_text_completer", "make_wire"]
