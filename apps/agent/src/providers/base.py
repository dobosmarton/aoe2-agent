"""Provider-neutral LLM types: the seam between game logic and wire format.

`ExecutorProvider` / `StrategistProvider` hold the game logic; a `ChatWire` holds
everything vendor-specific (message envelopes, tool schema shape, usage field
names, stop-reason spelling, exception classes). Swapping Anthropic for an
OpenAI-compatible endpoint is a `ChatWire` swap, not a second executor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, TypedDict, TypeVar

# Re-exported from config, which is the import leaf — declaring it here would
# cycle back through providers/__init__.
from ..config import WireName as WireName

if TYPE_CHECKING:
    from pydantic import BaseModel


class LLMResult(TypedDict, total=False):
    """Payload from `ExecutorProvider.get_actions`; every key is optional."""

    reasoning: str
    actions: list[dict[str, object]]
    observations: dict[str, object] | None
    actions_already_executed: bool
    success_count: int
    # True when the executor produced NO usable turn — every LLM path (single-
    # shot and its tool-loop fallback) failed and this is a safe-wait no-op.
    # Drives the executor-outage alarm and the llm_error_rate metric (T-533):
    # run 12 logged 90 such turns yet still wrote accepted=true, so the outage
    # was invisible in results.tsv.
    error: bool


# -- Names -------------------------------------------------------------------

# Anthropic accepts low/medium/high here; Sonnet 4.6 rejects xhigh/max (see
# config.EffortLevel). The OpenAI wire maps these onto `reasoning_effort`.
EffortName = Literal["low", "medium", "high"]

ModelT = TypeVar("ModelT", bound="BaseModel")


class ModelRefusedError(RuntimeError):
    """The model declined to answer.

    Both vendors can refuse, but they say so differently — Anthropic sets
    `stop_reason == "refusal"`, OpenAI populates `message.refusal`. Each wire
    detects its own and raises this, so callers handle one exception type.
    """


def text_of_blocks(content: list[dict[str, object]]) -> str:
    """Flatten a list of content blocks to plain text."""
    return "\n\n".join(
        str(block.get("text", "")) for block in content if block.get("type") == "text"
    )


# -- Value objects -----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SystemBlock:
    """One system-prompt segment plus whether it is worth caching.

    `cacheable` is a hint, not a directive: the Anthropic wire turns it into an
    explicit `cache_control` breakpoint, while OpenAI-compatible endpoints cache
    on prefix automatically and ignore it.
    """

    text: str
    cacheable: bool = False


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A model's request to run one tool; handlers narrow `arguments` themselves."""

    id: str
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    """The result of running one `ToolCall`, before wire-specific framing."""

    tool_call_id: str
    success: bool
    detail: str
    entities: tuple[dict[str, object], ...] = ()


def tool_outcome_json(outcome: ToolOutcome) -> str:
    """Serialise an outcome to the JSON the model reads.

    Shared by both wires on purpose: only the envelope differs per vendor. If the
    payload drifted, the two would show the model different tool results and
    cross-vendor comparison would quietly stop being valid.
    """
    payload: dict[str, object] = {"success": outcome.success, "detail": outcome.detail}
    if outcome.entities:
        payload["entities"] = list(outcome.entities)
    return json.dumps(payload)


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Per-call token counts, normalised across vendors.

    Anthropic reports `input_tokens` / `cache_read_input_tokens`; OpenAI reports
    `prompt_tokens` / `prompt_tokens_details.cached_tokens`. Each wire maps its
    own names onto these four so cost accounting has one shape.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )


# -- Conversation turns ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class UserTurn:
    """Plain text from us to the model."""

    text: str


@dataclass(frozen=True, slots=True)
class AssistantTurn:
    """The model's reply: prose, tool requests, or both."""

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True, slots=True)
class ToolResultsTurn:
    """All results for the previous `AssistantTurn` — both vendors want them batched."""

    outcomes: tuple[ToolOutcome, ...]


Turn = UserTurn | AssistantTurn | ToolResultsTurn


# -- Requests ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChatRequest:
    """Everything a wire needs for one call, minus the tools and output schema."""

    system: tuple[SystemBlock, ...]
    turns: tuple[Turn, ...]
    max_tokens: int
    temperature: float
    effort: EffortName = "low"


@dataclass(frozen=True, slots=True)
class ToolTurnResult:
    """One iteration of the agentic tool loop."""

    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    usage: TokenUsage = field(default_factory=TokenUsage)

    @property
    def wants_more_tools(self) -> bool:
        return bool(self.tool_calls)


class ChatWire(Protocol):
    """Vendor-specific transport for one chat model.

    Implementations own their SDK client, their message rendering, and their
    exception taxonomy. They own no game logic.
    """

    model: str
    # Where the calls go, for cost attribution: `openai` and `zen` share a
    # transport, so the class name alone cannot tell their bills apart.
    endpoint: str

    async def tool_turn(
        self,
        request: ChatRequest,
        tools: list[dict[str, object]],
    ) -> ToolTurnResult:
        """Run one tool-loop iteration. `tools` is Anthropic-shaped; wires convert."""
        ...

    async def parse_structured(
        self,
        request: ChatRequest,
        schema: type[ModelT],
    ) -> tuple[ModelT, TokenUsage]:
        """Run one structured-output call validated against `schema`."""
        ...

    def is_api_error(self, exc: Exception) -> bool:
        """Whether `exc` came from the provider rather than our own code."""
        ...

    def is_schema_too_large(self, exc: Exception) -> bool:
        """Whether `exc` means the output schema blew the decoding-grammar limit.

        Anthropic-only (F-40); it routes a turn onto the smaller tool-loop schema.
        """
        ...


__all__ = [
    "AssistantTurn",
    "ChatRequest",
    "ChatWire",
    "EffortName",
    "LLMResult",
    "ModelRefusedError",
    "ModelT",
    "SystemBlock",
    "TokenUsage",
    "ToolCall",
    "ToolOutcome",
    "ToolResultsTurn",
    "ToolTurnResult",
    "Turn",
    "UserTurn",
    "WireName",
    "text_of_blocks",
    "tool_outcome_json",
]
