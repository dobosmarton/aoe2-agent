"""The single point where an LLM wire is chosen by name.

Mirrors `evaluation.broker_factory`: one match arm per implementation, a lazy
import inside each arm so neither SDK becomes mandatory, and a `ValueError`
naming the valid choices rather than a silent fallback — a mistyped wire name
should fail at startup, not quietly run the wrong vendor for a whole game.

`openai` and `zen` share one transport and differ only in default endpoint, so
picking a gateway is one variable rather than a wire plus a URL.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, get_args

from .base import WireName

if TYPE_CHECKING:
    from .base import ChatWire
    from .text_wire import TextCompleter

WIRE_ENV: Final = "AOE2_LLM_WIRE"
# OpenCode Zen speaks the OpenAI Chat Completions shape, so it reaches GPT-5.6
# Luna, Kimi and GLM over the same transport.
ZEN_BASE_URL: Final = "https://opencode.ai/zen/v1"
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

    An empty `api_key` or `base_url` falls back to the SDK's own default; an
    explicit `base_url` overrides the adapter's endpoint.
    """
    normalized = name.strip().lower()

    if normalized == "anthropic":
        # Local imports: keep each vendor SDK out of the other's dependency path.
        from .wire_anthropic import AnthropicWire

        return AnthropicWire(model=model, api_key=api_key or None, max_retries=max_retries)

    if normalized in ("openai", "zen"):
        from .wire_openai import OpenAIWire

        return OpenAIWire(
            model=model,
            api_key=api_key or None,
            base_url=_endpoint(normalized, base_url),
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

    if normalized in ("openai", "zen"):
        from .text_wire import OpenAITextCompleter

        return OpenAITextCompleter(
            model=model,
            api_key=api_key or None,
            base_url=_endpoint(normalized, base_url),
        )

    raise _unknown(name)


def _endpoint(normalized: str, base_url: str | None) -> str | None:
    """Resolve the endpoint for an OpenAI-compatible arm.

    An explicit `base_url` always wins. Empty must become None, not "": the SDK
    reads None as "use my default" but accepts "" and then fails per request.
    """
    if base_url:
        return base_url
    return ZEN_BASE_URL if normalized == "zen" else None


def _unknown(name: str) -> ValueError:
    expected = ", ".join(repr(choice) for choice in _VALID)
    return ValueError(f"unknown {WIRE_ENV}={name!r}; expected one of {expected}")


__all__ = ["WIRE_ENV", "ZEN_BASE_URL", "make_text_completer", "make_wire"]
