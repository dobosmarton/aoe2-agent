"""Ollama local LLM provider for AoE2 Agent.

Uses Ollama's OpenAI-compatible API for local inference,
eliminating network latency (~100-500ms vs ~1-3s cloud).
"""

import json
from pathlib import Path
from typing import Any

import httpx
import structlog

from ..config import config
from ..models import LLMResponse, validate_actions
from .base import BaseLLMProvider

log = structlog.get_logger()

PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"

# OpenAI-compatible tool definitions (converted from Anthropic format)
_OPENAI_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "click",
            "description": "Left click at screen coordinates. Use for building placement and UI interaction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "X coordinate on game screen"},
                    "y": {"type": "integer", "description": "Y coordinate on game screen"},
                    "target_class": {"type": "string", "description": "Entity class to target nearest of, e.g. 'sheep'"},
                    "intent": {"type": "string", "description": "What this click does"},
                },
                "required": ["x", "y", "intent"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "right_click",
            "description": "Right click at screen coordinates. Use for resource gathering, setting gather points, and unit commands.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "X coordinate on game screen"},
                    "y": {"type": "integer", "description": "Y coordinate on game screen"},
                    "target_class": {"type": "string", "description": "Entity class to target nearest of, e.g. 'sheep'"},
                    "intent": {"type": "string", "description": "What this right click does"},
                },
                "required": ["x", "y", "intent"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "press",
            "description": "Press a keyboard key. Use for hotkeys, queuing units, opening build menus.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Key to press, e.g. 'h', 'q', '.', ','"},
                    "rescan": {"type": "boolean", "description": "Take fresh screenshot+detection after this key press"},
                    "modifiers": {"type": "array", "items": {"type": "string"}, "description": "Modifier keys e.g. ['ctrl']"},
                    "intent": {"type": "string", "description": "What this key press does"},
                },
                "required": ["key", "intent"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "drag",
            "description": "Drag mouse from start to end position.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_x": {"type": "integer", "description": "Start X coordinate"},
                    "start_y": {"type": "integer", "description": "Start Y coordinate"},
                    "end_x": {"type": "integer", "description": "End X coordinate"},
                    "end_y": {"type": "integer", "description": "End Y coordinate"},
                    "intent": {"type": "string"},
                },
                "required": ["start_x", "start_y", "end_x", "end_y", "intent"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wait",
            "description": "Wait for a duration.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ms": {"type": "integer", "description": "Milliseconds to wait (0-5000)"},
                    "intent": {"type": "string"},
                },
                "required": ["ms", "intent"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scroll",
            "description": "Scroll mouse wheel for zoom in/out.",
            "parameters": {
                "type": "object",
                "properties": {
                    "clicks": {"type": "integer", "description": "Positive = zoom in, negative = zoom out"},
                    "intent": {"type": "string"},
                },
                "required": ["clicks", "intent"],
            },
        },
    },
]


class OllamaProvider(BaseLLMProvider):
    """Ollama local LLM provider using OpenAI-compatible API.

    Runs inference locally via Ollama, eliminating network latency.
    Supports tool use via the OpenAI-compatible chat/completions endpoint.
    """

    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
    ):
        self.host = (host or config.ollama_host).rstrip("/")
        self.model = model or config.ollama_model
        self.client = httpx.AsyncClient(timeout=30.0)
        self._system_prompt: str | None = None

    def get_system_prompt(self) -> str:
        if self._system_prompt is None:
            prompt_file = PROMPTS_DIR / "system.md"
            hotkeys_file = PROMPTS_DIR / "hotkeys.md"
            if prompt_file.exists():
                self._system_prompt = prompt_file.read_text()
                if hotkeys_file.exists():
                    self._system_prompt += "\n\n" + hotkeys_file.read_text()
            else:
                self._system_prompt = "You are playing Age of Empires 2. Respond with game actions as tool calls."
        return self._system_prompt

    async def _call_chat(self, messages: list[dict], use_tools: bool = True) -> dict:
        """Call Ollama's OpenAI-compatible chat endpoint."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": config.max_tokens,
            },
        }
        if use_tools:
            payload["tools"] = _OPENAI_TOOLS

        url = f"{self.host}/api/chat"
        resp = await self.client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()

    async def get_actions(
        self,
        context: str,
        width: int = 1920,
        height: int = 1080,
    ) -> dict[str, Any]:
        """Send game state to local Ollama model and get actions back."""
        center_x = width // 2
        center_y = height // 2
        dimensions_info = f"Game window: {width}x{height} pixels. Center=({center_x},{center_y})."

        user_text = f"{dimensions_info}\n\n{context}\n\nBased on the detected entities, goals, and resource status above, decide what to do next. Use the tool functions to execute actions."

        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": user_text},
        ]

        try:
            executed_actions: list[dict] = []

            for _ in range(config.max_tool_iterations):
                result = await self._call_chat(messages)
                msg = result.get("message", {})
                tool_calls = msg.get("tool_calls", [])

                if not tool_calls:
                    # No more tool calls — model is done
                    break

                # Append assistant message with tool calls
                messages.append(msg)

                # Process each tool call
                for tc in tool_calls:
                    func = tc.get("function", {})
                    action_dict = {
                        "type": func.get("name", ""),
                        **func.get("arguments", {}),
                    }
                    executed_actions.append(action_dict)

                    # Feed back a simple success result
                    messages.append({
                        "role": "tool",
                        "content": json.dumps({"success": True, "detail": "ok"}),
                    })

            reasoning = ""
            if result.get("message", {}).get("content"):
                reasoning = result["message"]["content"]

            validated = validate_actions(executed_actions)
            response: dict[str, Any] = {
                "reasoning": reasoning,
                "observations": {},
                "actions": [a.model_dump() if hasattr(a, "model_dump") else a for a in validated],
                "actions_already_executed": False,
            }
            log.info("ollama_response", action_count=len(validated),
                     reasoning=reasoning[:150])
            return response

        except httpx.ConnectError:
            log.error("ollama_connect_error", host=self.host,
                      message="Is Ollama running? Start with: ollama serve")
            return self._error_response(f"Cannot connect to Ollama at {self.host}")
        except Exception as e:
            log.error("ollama_error", error=str(e))
            return self._error_response(f"Ollama error: {e}")

    def _error_response(self, message: str) -> dict[str, Any]:
        return {
            "reasoning": message,
            "observations": {},
            "actions": [{"type": "wait", "ms": 1000, "intent": "Error recovery"}],
        }
