"""Claude (Anthropic) LLM provider for AoE2 Agent."""

import json
from pathlib import Path
from typing import Any, Optional

import anthropic
import structlog

from ..config import config
from ..executor import execute_action, get_detected_entities
from ..models import LLMResponse, Observations, validate_actions
from .base import BaseLLMProvider

# Pricing per million tokens (claude-sonnet-4-6)
_PRICE_INPUT = 3.00
_PRICE_OUTPUT = 15.00
_PRICE_CACHE_READ = 0.30
_PRICE_CACHE_WRITE = 3.75


def _click_schema(description: str) -> dict:
    """Shared input schema for click and right_click tools."""
    return {
        "type": "object",
        "properties": {
            "x": {"type": "integer", "description": "X coordinate on game screen"},
            "y": {"type": "integer", "description": "Y coordinate on game screen"},
            "target_class": {"type": "string", "description": "Entity class to target nearest of, e.g. 'sheep'"},
            "intent": {"type": "string", "description": description},
        },
        "required": ["x", "y", "intent"],
        "additionalProperties": False,
    }


# Tool definitions for each action type — strict per-tool schemas.
# Each tool has its own enforced schema, preventing field confusion
# that occurred with structured output union types.
_ACTION_TOOLS: list[dict] = [
    {"name": "click", "description": "Left click at screen coordinates. Use for building placement and UI interaction.", "input_schema": _click_schema("What this click does")},
    {"name": "right_click", "description": "Right click at screen coordinates. Use for resource gathering, setting gather points, and unit commands.", "input_schema": _click_schema("What this right click does")},
    {
        "name": "press",
        "description": "Press a keyboard key. Use for hotkeys, queuing units, opening build menus.",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Key to press, e.g. 'h', 'q', '.', ','"},
                "rescan": {"type": "boolean", "description": "Take fresh screenshot+detection after this key press"},
                "modifiers": {"type": "array", "items": {"type": "string"}, "description": "Modifier keys e.g. ['ctrl']"},
                "intent": {"type": "string", "description": "What this key press does"},
            },
            "required": ["key", "intent"],
            "additionalProperties": False,
        },
    },
    {
        "name": "drag",
        "description": "Drag mouse from start to end position.",
        "input_schema": {
            "type": "object",
            "properties": {
                "start_x": {"type": "integer", "description": "Start X coordinate"},
                "start_y": {"type": "integer", "description": "Start Y coordinate"},
                "end_x": {"type": "integer", "description": "End X coordinate"},
                "end_y": {"type": "integer", "description": "End Y coordinate"},
                "intent": {"type": "string"},
            },
            "required": ["start_x", "start_y", "end_x", "end_y", "intent"],
            "additionalProperties": False,
        },
    },
    {
        "name": "wait",
        "description": "Wait for a duration.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ms": {"type": "integer", "description": "Milliseconds to wait (0-5000)"},
                "intent": {"type": "string"},
            },
            "required": ["ms", "intent"],
            "additionalProperties": False,
        },
    },
    {
        "name": "scroll",
        "description": "Scroll mouse wheel for zoom in/out.",
        "input_schema": {
            "type": "object",
            "properties": {
                "clicks": {"type": "integer", "description": "Positive = zoom in, negative = zoom out"},
                "intent": {"type": "string"},
            },
            "required": ["clicks", "intent"],
            "additionalProperties": False,
        },
    },
    {
        "name": "detect",
        "description": "Request full SAHI detection scan. SLOW (~5-10s) — only use when target_class keeps failing.",
        "input_schema": {
            "type": "object",
            "properties": {
                "intent": {"type": "string"},
            },
            "required": ["intent"],
            "additionalProperties": False,
        },
    },
]

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
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_cache_read_tokens: int = 0
        self._total_cache_write_tokens: int = 0

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

Action types: click, right_click, press, drag (with x1,y1,x2,y2), wait (with ms), scroll (with clicks), detect
For click/right_click: MUST include one of: (x, y) coordinates, target_id (e.g. "sheep_0"), or target_class (e.g. "sheep").
For click/right_click: use "x" and "y" fields (NOT "x1"/"y1" — those are for drag only).

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

    async def _execute_tool_call(self, block: object) -> tuple[dict, dict]:
        """Execute a single tool call and build the result payload.

        Returns (action_dict, tool_result_dict).
        """
        action_dict = {"type": block.name, **block.input}  # type: ignore[union-attr]

        result = await execute_action(action_dict)
        log.info("tool_executed", action=block.name, intent=block.input.get("intent", ""),  # type: ignore[union-attr]
                 success=result.success)

        result_data: dict[str, Any] = {"success": result.success, "detail": result.detail}

        # Include fresh entity list after rescan-triggering actions
        if block.name == "press" and block.input.get("rescan"):  # type: ignore[union-attr]
            entities = get_detected_entities()
            result_data["entities"] = [
                {"id": e.get("id", ""), "class": e.get("class", ""), "center": e.get("center", [])}
                for e in entities[:20]
            ]

        tool_result = {
            "type": "tool_result",
            "tool_use_id": block.id,  # type: ignore[union-attr]
            "content": json.dumps(result_data),
        }
        return action_dict, tool_result

    async def _call_api(self, content: list[dict]) -> LLMResponse:
        """Call Claude API in an agentic tool loop.

        Each iteration: model calls one tool → we execute it → feed result
        back → model calls next tool. Loop until model says end_turn or we
        hit max_tool_iterations. The model receives fresh entity positions
        after every camera-moving action.
        """
        messages: list[dict] = [{"role": "user", "content": content}]
        executed_actions: list[dict] = []
        success_count = 0
        reasoning_parts: list[str] = []

        for _ in range(config.max_tool_iterations):
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=config.max_tokens,
                system=self.get_system_prompt(),
                messages=messages,
                tools=_ACTION_TOOLS,
                cache_control={"type": "ephemeral"},
            )

            # Accumulate token usage (cache fields may be None when caching is off)
            usage = response.usage
            self._total_input_tokens += usage.input_tokens
            self._total_output_tokens += usage.output_tokens
            self._total_cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
            self._total_cache_write_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0

            # Single pass: extract text and tool_use blocks
            tool_blocks = []
            for block in response.content:
                if block.type == "text" and block.text.strip():
                    reasoning_parts.append(block.text.strip())
                elif block.type == "tool_use":
                    tool_blocks.append(block)

            if response.stop_reason != "tool_use":
                break

            # Execute each tool call and collect results
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in tool_blocks:
                action_dict, tool_result = await self._execute_tool_call(block)
                executed_actions.append(action_dict)
                tool_results.append(tool_result)
                if json.loads(tool_result["content"]).get("success"):
                    success_count += 1

            messages.append({"role": "user", "content": tool_results})

        result = LLMResponse.model_construct(
            actions=validate_actions(executed_actions),
            observations=Observations(),
            reasoning=" ".join(reasoning_parts),
        )
        result._success_count = success_count  # type: ignore[attr-defined]
        return result

    def _cumulative_cost_usd(self) -> float:
        """Calculate cumulative API cost across all calls."""
        return (
            self._total_input_tokens * _PRICE_INPUT / 1_000_000
            + self._total_output_tokens * _PRICE_OUTPUT / 1_000_000
            + self._total_cache_read_tokens * _PRICE_CACHE_READ / 1_000_000
            + self._total_cache_write_tokens * _PRICE_CACHE_WRITE / 1_000_000
        )

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

            log.info("api_cost",
                     input_tokens=self._total_input_tokens,
                     output_tokens=self._total_output_tokens,
                     cache_read_tokens=self._total_cache_read_tokens,
                     cache_write_tokens=self._total_cache_write_tokens,
                     cumulative_cost_usd=round(self._cumulative_cost_usd(), 4))

            response = result.model_dump()
            response["actions_already_executed"] = True
            response["success_count"] = getattr(result, "_success_count", len(response.get("actions", [])))
            return response

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
