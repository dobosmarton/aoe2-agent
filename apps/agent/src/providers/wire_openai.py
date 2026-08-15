"""OpenAI-compatible Chat Completions implementation of `ChatWire`.

Targets `/chat/completions`, not the Responses API: it is what every
OpenAI-compatible gateway implements, so one wire reaches OpenCode Zen
(GPT-5.6 Luna, Kimi K2.7, GLM) and api.openai.com.

`prompt_tokens` INCLUDES cached tokens here; Anthropic's `input_tokens` excludes
them, so `_usage_of` subtracts the cached count back out.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import openai
import structlog

from .action_tools import to_openai_tools
from .base import (
    AssistantTurn,
    ChatRequest,
    ModelRefusedError,
    SystemBlock,
    TokenUsage,
    ToolCall,
    ToolOutcome,
    ToolResultsTurn,
    ToolTurnResult,
    UserTurn,
    tool_outcome_json,
)

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion, ChatCompletionMessage

    from .base import ModelT, Turn

log = structlog.stdlib.get_logger()


def _temperature(value: float | None) -> float | openai.NotGiven:
    """Omit when unset. Reasoning models accept only the default (1); sending
    0 returns `unsupported_value` on every call."""
    return openai.NOT_GIVEN if value is None else value


class OpenAIWire:
    """`ChatWire` over `openai.AsyncOpenAI`, pointed at any compatible endpoint."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        # Same as AnthropicWire: SDK-internal 429/5xx backoff, hence no tenacity.
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries,
        )
        # Read back from the client so a None here reports the SDK's own default.
        self.endpoint = str(self.client.base_url)

    # -- Rendering ----------------------------------------------------------

    @staticmethod
    def _render_system(blocks: tuple[SystemBlock, ...]) -> list[dict[str, object]]:
        """Fold the system blocks into one message.

        Order matters: prefix caching keys on it, so the stable core prompt must
        stay ahead of the age block.
        """
        text = "\n\n".join(block.text for block in blocks if block.text)
        return [{"role": "system", "content": text}] if text else []

    @staticmethod
    def _render_tool_call(call: ToolCall) -> dict[str, object]:
        return {
            "id": call.id,
            "type": "function",
            "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
        }

    @staticmethod
    def _render_outcome(outcome: ToolOutcome) -> dict[str, object]:
        return {
            "role": "tool",
            "tool_call_id": outcome.tool_call_id,
            "content": tool_outcome_json(outcome),
        }

    @classmethod
    def _render_turns(cls, turns: tuple[Turn, ...]) -> list[dict[str, object]]:
        """Build the Chat Completions message list from neutral turns."""
        messages: list[dict[str, object]] = []
        for turn in turns:
            if isinstance(turn, UserTurn):
                messages.append({"role": "user", "content": turn.text})
            elif isinstance(turn, AssistantTurn):
                message: dict[str, object] = {"role": "assistant", "content": turn.text or None}
                if turn.tool_calls:
                    message["tool_calls"] = [cls._render_tool_call(c) for c in turn.tool_calls]
                messages.append(message)
            elif isinstance(turn, ToolResultsTurn):
                # One message per outcome — the API pairs them by tool_call_id
                # and rejects a turn that leaves any call unanswered.
                messages.extend(cls._render_outcome(outcome) for outcome in turn.outcomes)
        return messages

    @classmethod
    def _render_messages(cls, request: ChatRequest) -> list[dict[str, object]]:
        return cls._render_system(request.system) + cls._render_turns(request.turns)

    @staticmethod
    def _usage_of(response: ChatCompletion) -> TokenUsage:
        """Normalise usage onto `TokenUsage`, de-overlapping the cached count."""
        usage = response.usage
        if usage is None:
            return TokenUsage()
        details = usage.prompt_tokens_details
        cached = (details.cached_tokens or 0) if details is not None else 0
        return TokenUsage(
            # Anthropic reports uncached input only; subtract so both wires mean
            # the same thing and the shared pricing table stays correct.
            input_tokens=max(usage.prompt_tokens - cached, 0),
            output_tokens=usage.completion_tokens,
            cache_read_tokens=cached,
            # Automatic prefix caching carries no separate write charge.
            cache_write_tokens=0,
        )

    @classmethod
    def _parse_tool_calls(cls, message: ChatCompletionMessage) -> tuple[ToolCall, ...]:
        """Read tool calls, tolerating malformed argument JSON.

        A call whose arguments will not parse is kept with empty arguments rather
        than dropped: dropping it would leave a `tool_call_id` unanswered and the
        next request would be rejected outright.
        """
        return tuple(
            ToolCall(
                id=item.id,
                name=item.function.name,
                arguments=cls._decode(item.function.name, item.function.arguments),
            )
            # The union also admits custom tool calls, which carry no `.function`.
            for item in (message.tool_calls or [])
            if item.type == "function"
        )

    @staticmethod
    def _decode(tool: str, arguments: str) -> dict[str, object]:
        try:
            decoded: object = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            log.warning("openai_tool_arguments_unparsable", tool=tool, raw=arguments[:200])
            return {}
        return decoded if isinstance(decoded, dict) else {}

    # -- ChatWire -----------------------------------------------------------

    async def tool_turn(
        self,
        request: ChatRequest,
        tools: list[dict[str, object]],
    ) -> ToolTurnResult:
        """One `chat.completions.create` call with the action tools attached."""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=self._render_messages(request),  # pyright: ignore[reportArgumentType]
            tools=to_openai_tools(tools),  # pyright: ignore[reportArgumentType]
            temperature=_temperature(request.temperature),
            # Reasoning models reject the classic `max_tokens`. `reasoning_effort`
            # is sent even where support is uncertain: a loud 400 beats silently
            # dropping a knob the operator set (surfaces via the T-533 alarm).
            reasoning_effort=request.effort,
            max_completion_tokens=request.max_tokens,
        )

        choice = response.choices[0]
        calls = self._parse_tool_calls(choice.message)
        return ToolTurnResult(
            text=(choice.message.content or "").strip(),
            # `finish_reason` is the authority, matching the Anthropic wire's
            # use of `stop_reason`.
            tool_calls=calls if choice.finish_reason == "tool_calls" else (),
            usage=self._usage_of(response),
        )

    async def parse_structured(
        self,
        request: ChatRequest,
        schema: type[ModelT],
    ) -> tuple[ModelT, TokenUsage]:
        """One `chat.completions.parse` call constrained to `schema`."""
        response = await self.client.chat.completions.parse(
            model=self.model,
            messages=self._render_messages(request),  # pyright: ignore[reportArgumentType]
            response_format=schema,
            temperature=_temperature(request.temperature),
            reasoning_effort=request.effort,
            max_completion_tokens=request.max_tokens,
        )
        message = response.choices[0].message
        if message.refusal:
            raise ModelRefusedError(f"{self.model} refused the request: {message.refusal}")
        if message.parsed is None:
            raise ValueError(f"{self.model} returned no parsable {schema.__name__}")
        return message.parsed, self._usage_of(response)

    def is_api_error(self, exc: Exception) -> bool:
        return isinstance(exc, openai.APIError)

    def is_schema_too_large(self, exc: Exception) -> bool:
        """Always False — this vendor has no compiled-grammar size cap (cf. F-40)."""
        return False


__all__ = ["OpenAIWire"]
