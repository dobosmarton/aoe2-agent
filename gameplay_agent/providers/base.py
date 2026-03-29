"""Base LLM provider interface for AoE2 Agent."""

from abc import ABC, abstractmethod
from typing import Any


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def get_actions(
        self,
        context: str,
        width: int = 1920,
        height: int = 1080,
    ) -> dict[str, Any]:
        """
        Send game state to LLM and get actions back.

        Args:
            context: Context string with detected entities, goals, resources, previous actions
            width: Game window width in pixels (for coordinate reference)
            height: Game window height in pixels (for coordinate reference)

        Returns:
            Dictionary containing:
                - reasoning: str - LLM's explanation of what it sees and plans
                - actions: list[dict] - List of action dictionaries to execute
        """
        pass

    @abstractmethod
    def get_system_prompt(self, age: str = "Dark Age") -> str | list[dict]:
        """Get the system prompt for this provider.

        Returns either a plain string or a list of content blocks
        (for multi-block system prompts with per-block cache control).
        """
        pass
