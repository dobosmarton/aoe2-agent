"""Base LLM provider interface for AoE2 Agent."""

from abc import ABC, abstractmethod
from typing import TypedDict


class LLMResult(TypedDict, total=False):
    """Structured payload returned by `BaseLLMProvider.get_actions`.

    `total=False` makes every key Optional — the provider may omit
    fields when nothing of interest happened that turn (e.g. an empty
    response). Consumers should `.get(...)` with a default rather than
    indexing directly.
    """

    reasoning: str
    actions: list[dict[str, object]]
    observations: dict[str, object] | None
    actions_already_executed: bool
    success_count: int


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def get_actions(
        self,
        context: str,
        width: int = 1920,
        height: int = 1080,
    ) -> LLMResult:
        """
        Send game state to LLM and get actions back.

        Args:
            context: Context string with detected entities, goals, resources, previous actions
            width: Game window width in pixels (for coordinate reference)
            height: Game window height in pixels (for coordinate reference)

        Returns:
            An LLMResult dict with reasoning, actions, observations, etc.
        """
        pass

    @abstractmethod
    def get_system_prompt(self, age: str = "Dark Age") -> str | list[dict[str, object]]:
        """Get the system prompt for this provider.

        Returns either a plain string or a list of content blocks
        (for multi-block system prompts with per-block cache control).
        """
        pass
