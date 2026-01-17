"""Base LLM provider interface for AoE2 Agent."""

from abc import ABC, abstractmethod
from typing import Any


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def get_actions(
        self,
        screenshot_bytes: bytes,
        context: str = "",
        width: int = 1920,
        height: int = 1080,
    ) -> dict[str, Any]:
        """
        Send screenshot to LLM and get actions back.

        Args:
            screenshot_bytes: JPEG image bytes of the current screen
            context: Optional context string (e.g., game state, previous actions)
            width: Screenshot width in pixels
            height: Screenshot height in pixels

        Returns:
            Dictionary containing:
                - reasoning: str - LLM's explanation of what it sees and plans
                - actions: list[dict] - List of action dictionaries to execute
        """
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this provider."""
        pass
