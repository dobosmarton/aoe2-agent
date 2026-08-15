"""Synchronous plain-text completion, for the callers that need nothing else.

Memory extraction and prompt mutation each make one blocking system+user call and
read back prose — no tools, no structured output, no caching, and no event loop
to await on. Forcing them through the async, tool-oriented `ChatWire` would mean
`asyncio.run` at two sync call sites and a Protocol carrying methods neither uses,
so they get this narrower contract instead.

"""

from __future__ import annotations

from typing import Protocol


class TextCompleter(Protocol):
    """One blocking prompt-in, text-out call."""

    model: str

    def complete(self, system: str, user: str, max_tokens: int) -> str: ...


class AnthropicTextCompleter:
    """`TextCompleter` over the synchronous `anthropic.Anthropic` client."""

    def __init__(self, model: str, api_key: str | None = None) -> None:
        import anthropic

        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key)

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        from anthropic.types import TextBlock

        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in response.content if isinstance(block, TextBlock))


class OpenAITextCompleter:
    """`TextCompleter` over the synchronous `openai.OpenAI` client."""

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None) -> None:
        import openai

        self.model = model
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            max_completion_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""


__all__ = ["AnthropicTextCompleter", "OpenAITextCompleter", "TextCompleter"]
