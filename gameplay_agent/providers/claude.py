"""Claude (Anthropic) LLM provider for AoE2 Agent."""

from pathlib import Path
from typing import Any, Optional

import anthropic
import structlog

from ..config import config
from ..models import LLMResponse
from .base import BaseLLMProvider

log = structlog.get_logger()

# Load system prompt from file
PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"

# Optional game knowledge database for dynamic context injection
try:
    from data.game_knowledge import GameKnowledge, get_db
    GAME_KNOWLEDGE_AVAILABLE = True
except ImportError:
    GAME_KNOWLEDGE_AVAILABLE = False
    log.debug("game_knowledge_not_available", message="Running without dynamic context injection")


class ClaudeProvider(BaseLLMProvider):
    """Anthropic Claude provider implementation."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        use_dynamic_context: bool = True,
    ):
        """
        Initialize Claude provider.

        Args:
            api_key: Anthropic API key (defaults to config/env)
            model: Model to use (defaults to config)
            use_dynamic_context: Whether to use dynamic context injection from game database
        """
        self.api_key = api_key or config.anthropic_api_key
        self.model = model or config.model
        # Use AsyncAnthropic with built-in retry (429/5xx with exponential backoff)
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key, max_retries=3)
        self._system_prompt: str | None = None
        self.use_dynamic_context = use_dynamic_context and GAME_KNOWLEDGE_AVAILABLE
        self._game_db: Optional["GameKnowledge"] = None

        if self.use_dynamic_context:
            try:
                self._game_db = get_db()
                log.info("game_knowledge_initialized")
            except Exception as e:
                log.warning("game_knowledge_init_failed", error=str(e))
                self.use_dynamic_context = False

    def get_system_prompt(self) -> str:
        """Load and return the system prompt."""
        if self._system_prompt is None:
            prompt_file = PROMPTS_DIR / "system.md"
            hotkeys_file = PROMPTS_DIR / "hotkeys.md"
            if prompt_file.exists():
                self._system_prompt = prompt_file.read_text()
                if hotkeys_file.exists():
                    self._system_prompt += "\n\n" + hotkeys_file.read_text()
            else:
                # Fallback minimal prompt
                self._system_prompt = """You are playing Age of Empires 2: Definitive Edition. Your goal is to defeat the enemy AI.

## Output Format
Respond with JSON only:
{
  "reasoning": "What you see and your strategic thinking",
  "observations": {
    "resources": {"food": 0, "wood": 0, "gold": 0, "stone": 0},
    "population": "12/15",
    "age": "Dark Age",
    "idle_tc": true,
    "under_attack": false,
    "events": []
  },
  "actions": [
    {"type": "click", "x": 100, "y": 200, "intent": "What this does"},
    {"type": "right_click", "target_id": "sheep_0", "intent": "Gather from sheep"},
    {"type": "press", "key": "h", "intent": "What this does"}
  ]
}

Action types: click, right_click, press, drag (with x1,y1,x2,y2), wait (with ms)
For click/right_click: use either (x, y) coordinates OR target_id from detected entities.

Play to win!"""
        return self._system_prompt

    def _get_dynamic_context(self, context: str) -> str:
        """Extract game state from context and generate dynamic knowledge context.

        Args:
            context: Memory context string containing game state

        Returns:
            Enhanced context with dynamic game knowledge
        """
        if not self.use_dynamic_context or not self._game_db:
            return context

        # Parse resources and age from context
        resources = {"food": 200, "wood": 200, "gold": 100, "stone": 200}  # Defaults
        age = "dark"

        try:
            # Try to extract resources from context
            import re

            food_match = re.search(r"Food[=:]?\s*(\d+)", context, re.IGNORECASE)
            wood_match = re.search(r"Wood[=:]?\s*(\d+)", context, re.IGNORECASE)
            gold_match = re.search(r"Gold[=:]?\s*(\d+)", context, re.IGNORECASE)
            stone_match = re.search(r"Stone[=:]?\s*(\d+)", context, re.IGNORECASE)

            if food_match:
                resources["food"] = int(food_match.group(1))
            if wood_match:
                resources["wood"] = int(wood_match.group(1))
            if gold_match:
                resources["gold"] = int(gold_match.group(1))
            if stone_match:
                resources["stone"] = int(stone_match.group(1))

            # Extract age
            age_match = re.search(r"(Dark|Feudal|Castle|Imperial)\s*Age", context, re.IGNORECASE)
            if age_match:
                age = age_match.group(1).lower()

        except Exception as e:
            log.debug("context_parse_error", error=str(e))

        # Get dynamic context from database
        try:
            dynamic_context = self._game_db.get_context_for_state(age, resources)
            early_game_tips = self._game_db.get_early_game_priorities()

            # Combine: dynamic context first, then original context
            enhanced_context = f"{dynamic_context}\n{early_game_tips}\n{context}"
            return enhanced_context

        except Exception as e:
            log.warning("dynamic_context_error", error=str(e))
            return context

    def _build_content(
        self, context: str, width: int, height: int
    ) -> list[dict]:
        """Build the message content for Claude.

        Pure text content with YOLO-detected entities, goals, cached resources,
        and game state. No images — all visual info comes from YOLO detection
        and strategist resource readings.
        """
        # Enhance context with dynamic game knowledge
        enhanced_context = self._get_dynamic_context(context)

        # Build text with dimensions info
        center_x = width // 2
        center_y = height // 2
        dimensions_info = f"Game window: {width}x{height} pixels. Center=({center_x},{center_y}). Valid x=0-{width}, y=0-{height}."

        text = f"{dimensions_info}\n\n{enhanced_context}\n\nBased on the detected entities, goals, and resource status above, decide what to do next."

        content = [
            {
                "type": "text",
                "text": text,
            },
        ]

        return content

    async def _call_api(self, content: list[dict]) -> LLMResponse:
        """Call Claude API with structured output parsing.

        Uses messages.parse() to get validated Pydantic output directly.
        SDK handles retry (429/5xx) with exponential backoff automatically.
        """
        response = await self.client.messages.parse(
            model=self.model,
            max_tokens=config.max_tokens,
            system=self.get_system_prompt(),
            messages=[{"role": "user", "content": content}],
            output_format=LLMResponse,
        )
        if response.stop_reason == "refusal":
            raise ValueError("Claude refused the request")
        return response.parsed_output

    async def get_actions(
        self,
        context: str,
        width: int = 1920,
        height: int = 1080,
    ) -> dict[str, Any]:
        """
        Send text context to Claude and get actions back.

        The executor is 100% text-based. All visual information comes from
        YOLO entity detection (text list) and strategist resource readings.

        Args:
            context: Context string with entities, goals, resources, memory
            width: Game window width in pixels
            height: Game window height in pixels

        Returns:
            Dictionary with reasoning, observations, and actions
        """
        content = self._build_content(context, width, height)

        try:
            result = await self._call_api(content)
            log.debug("claude_response", reasoning=result.reasoning[:200])
            return result.model_dump()

        except anthropic.APIError as e:
            log.error("claude_api_error", error=str(e))
            return self._error_response(f"API error: {e}")
        except Exception as e:
            log.error("claude_error", error=str(e))
            return self._error_response(f"Error: {e}")

    def _error_response(self, message: str) -> dict[str, Any]:
        """Return a safe error response with a wait action."""
        return {
            "reasoning": message,
            "observations": {},
            "actions": [{"type": "wait", "ms": 1000, "intent": "Error recovery"}],
        }
