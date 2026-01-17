"""Claude (Anthropic) LLM provider for AoE2 Agent."""

import base64
import json
import re
from pathlib import Path
from typing import Any

import anthropic
import structlog
from pydantic import ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import config
from ..models import LLMResponse
from .base import BaseLLMProvider

log = structlog.get_logger()

# Load system prompt from file
PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"


class ClaudeProvider(BaseLLMProvider):
    """Anthropic Claude provider implementation."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        """
        Initialize Claude provider.

        Args:
            api_key: Anthropic API key (defaults to config/env)
            model: Model to use (defaults to config)
        """
        self.api_key = api_key or config.anthropic_api_key
        self.model = model or config.model
        # Use AsyncAnthropic for proper async support
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
        self._system_prompt: str | None = None

    def get_system_prompt(self) -> str:
        """Load and return the system prompt."""
        if self._system_prompt is None:
            prompt_file = PROMPTS_DIR / "system.md"
            if prompt_file.exists():
                self._system_prompt = prompt_file.read_text()
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
    {"type": "press", "key": "h", "intent": "What this does"}
  ]
}

Action types: click, right_click, press, drag (with x1,y1,x2,y2), wait (with ms)

Play to win!"""
        return self._system_prompt

    def _build_content(
        self, screenshot_bytes: bytes, context: str, width: int, height: int
    ) -> list[dict]:
        """Build the message content for Claude."""
        image_base64 = base64.standard_b64encode(screenshot_bytes).decode("utf-8")

        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_base64,
                },
            },
        ]

        # Build text with dimensions info - include center to help LLM calibrate
        center_x = width // 2
        center_y = height // 2
        dimensions_info = f"Screenshot dimensions: {width}x{height} pixels. Center=({center_x},{center_y}). Valid x=0-{width}, y=0-{height}."

        if context:
            text = f"{dimensions_info}\n\n{context}\n\nWhat should I do next?"
        else:
            text = f"{dimensions_info}\n\nWhat should I do next?"

        content.append({
            "type": "text",
            "text": text,
        })

        return content

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _call_api(self, content: list[dict]) -> str:
        """
        Call Claude API with retry logic.

        Retries up to 3 times with exponential backoff.
        """
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=config.max_tokens,
            system=self.get_system_prompt(),
            messages=[{"role": "user", "content": content}],
        )
        return response.content[0].text

    async def get_actions(
        self,
        screenshot_bytes: bytes,
        context: str = "",
        width: int = 1920,
        height: int = 1080,
    ) -> dict[str, Any]:
        """
        Send screenshot to Claude and get actions back.

        Args:
            screenshot_bytes: JPEG image bytes
            context: Optional context string (memory/game state)
            width: Screenshot width in pixels
            height: Screenshot height in pixels

        Returns:
            Dictionary with reasoning, observations, and actions
        """
        content = self._build_content(screenshot_bytes, context, width, height)

        try:
            response_text = await self._call_api(content)
            log.debug("claude_response", response=response_text[:500])
            return self._parse_response(response_text)

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

    def _parse_response(self, response_text: str) -> dict[str, Any]:
        """
        Parse JSON response from Claude.

        Handles cases where JSON is wrapped in markdown code blocks.
        Validates response with Pydantic models.
        """
        # Try to extract JSON from markdown code block
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON object
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                log.warning("no_json_found", response=response_text[:200])
                return {"reasoning": response_text, "observations": {}, "actions": []}

        try:
            result = json.loads(json_str)

            # Validate with Pydantic
            try:
                validated = LLMResponse.model_validate(result)
                return validated.model_dump()
            except ValidationError as e:
                log.warning("validation_error", errors=str(e.errors()[:3]))
                # Return with original reasoning but empty actions if validation fails
                return {
                    "reasoning": result.get("reasoning", response_text),
                    "observations": result.get("observations", {}),
                    "actions": [],
                }

        except json.JSONDecodeError as e:
            log.warning("json_parse_error", error=str(e), json_str=json_str[:200])
            return {"reasoning": response_text, "observations": {}, "actions": []}
