"""The single point where an LLM wire is chosen by name.

Mirrors `evaluation.broker_factory`: one arm per implementation and a lazy import
inside each arm so neither SDK becomes mandatory. Validation is not here — it is
in `config._parse_wire`, which is where a name first arrives as text; by this
point `WireName` makes an invalid one a type error.

`openai` and `zen` share one transport and differ only in default endpoint, so
picking a gateway is one variable rather than a wire plus a URL.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, assert_never

if TYPE_CHECKING:
    from ..config import WireName
    from .base import ChatWire
    from .text_wire import TextCompleter

# OpenCode Zen speaks the OpenAI Chat Completions shape, so it reaches GPT-5.6
# Luna, Kimi and GLM over the same transport.
ZEN_BASE_URL: Final = "https://opencode.ai/zen/v1"


def make_wire(
    name: WireName,
    model: str,
    api_key: str = "",
    base_url: str | None = None,
    max_retries: int = 3,
) -> ChatWire:
    """Build the wire called `name` for `model`.

    An empty `api_key` or `base_url` falls back to the SDK's own default; an
    explicit `base_url` overrides the adapter's endpoint.
    """
    match name:
        case "anthropic":
            # Local imports: keep each vendor SDK out of the other's dependency path.
            from .wire_anthropic import AnthropicWire

            return AnthropicWire(model=model, api_key=api_key or None, max_retries=max_retries)

        case "openai" | "zen":
            from .wire_openai import OpenAIWire

            return OpenAIWire(
                model=model,
                api_key=api_key or None,
                base_url=_openai_endpoint(name, base_url),
                max_retries=max_retries,
            )

        case _ as unreachable:
            assert_never(unreachable)


def make_text_completer(
    name: WireName,
    model: str,
    api_key: str = "",
    base_url: str | None = None,
) -> TextCompleter:
    """Build the synchronous text completer called `name` for `model`.

    Same selection rules as `make_wire`, for the blocking prompt-in/text-out
    callers (memory extraction, prompt mutation).
    """
    match name:
        case "anthropic":
            from .text_wire import AnthropicTextCompleter

            return AnthropicTextCompleter(model=model, api_key=api_key or None)

        case "openai" | "zen":
            from .text_wire import OpenAITextCompleter

            return OpenAITextCompleter(
                model=model,
                api_key=api_key or None,
                base_url=_openai_endpoint(name, base_url),
            )

        case _ as unreachable:
            assert_never(unreachable)


def _openai_endpoint(name: WireName, override: str | None) -> str | None:
    """Override wins; "" must become None, or the SDK builds an empty base URL."""
    return override or (ZEN_BASE_URL if name == "zen" else None)


__all__ = ["ZEN_BASE_URL", "make_text_completer", "make_wire"]
