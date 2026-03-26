"""Claude (Anthropic) LLM provider for AoE2 Agent."""

import json
from typing import Any, Optional

import anthropic
import structlog

from ..config import config
from ..executor import execute_action, get_detected_entities
from ..models import BatchResponse, LLMResponse, Observations, validate_actions
from .base import BaseLLMProvider
from .shared import cached_system_block, format_dimensions, load_system_prompt
from .tool_definitions import to_anthropic_tools

_ACTION_TOOLS = to_anthropic_tools()

log = structlog.get_logger()

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
        use_batch_mode: bool = True,
    ):
        """
        Initialize Claude provider.

        Args:
            api_key: Anthropic API key (defaults to config/env)
            model: Model to use (defaults to config)
            use_dynamic_context: Whether to use dynamic context injection from game database
            use_batch_mode: Use single-call batch mode instead of agentic tool loop (faster)
        """
        self.api_key = api_key or config.anthropic_api_key
        self.model = model or config.model
        self.use_batch_mode = use_batch_mode
        # Use AsyncAnthropic with built-in retry (429/5xx with exponential backoff)
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key, max_retries=config.max_retries)
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
            self._system_prompt = load_system_prompt("system.md", "hotkeys.md")
            if not self._system_prompt:
                self._system_prompt = (
                    "You are playing Age of Empires 2: Definitive Edition. "
                    "Your goal is to defeat the enemy AI. Play to win!"
                )
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
        text = f"{format_dimensions(width, height)}\n\n{enhanced_context}\n\nBased on the detected entities, goals, and resource status above, decide what to do next."

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

    async def _call_api_batch(self, content: list[dict]) -> BatchResponse:
        """Call Claude API in single-call batch mode using structured output.

        Uses BatchResponse (no Observations) for a smaller schema and fewer
        output tokens, reducing constrained-decoding latency.
        """
        messages: list[dict] = [{"role": "user", "content": content}]

        response = await self.client.messages.parse(
            model=self.model,
            max_tokens=config.batch_max_tokens,
            system=cached_system_block(self.get_system_prompt()),
            messages=messages,
            output_format=BatchResponse,
        )

        if response.stop_reason == "refusal":
            log.warning("executor_refused")
            return BatchResponse()

        result = response.parsed_output
        log.info("batch_response", action_count=len(result.actions),
                 reasoning=result.reasoning[:150])
        return result

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
                system=cached_system_block(self.get_system_prompt()),
                messages=messages,
                tools=_ACTION_TOOLS,
            )

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
            if self.use_batch_mode:
                result = await self._call_api_batch(content)
                response = result.model_dump()
                # Batch mode returns actions that haven't been executed yet
                response["actions_already_executed"] = False
                return response

            result = await self._call_api(content)
            log.debug("claude_response", reasoning=result.reasoning[:200])
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
