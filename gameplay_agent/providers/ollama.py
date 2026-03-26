"""Ollama local LLM provider for AoE2 Agent.

Uses Ollama's OpenAI-compatible API for local inference,
eliminating network latency (~100-500ms vs ~1-3s cloud).
"""

import json
from typing import Any

import httpx
import structlog

from ..config import config
from ..models import validate_actions
from .base import BaseLLMProvider
from .shared import format_dimensions, load_system_prompt
from .tool_definitions import to_openai_tools

log = structlog.get_logger()

_OPENAI_TOOLS = to_openai_tools()


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
        self.client = httpx.AsyncClient(timeout=config.ollama_timeout)
        self._system_prompt: str | None = None

    def get_system_prompt(self) -> str:
        if self._system_prompt is None:
            self._system_prompt = load_system_prompt("system.md", "hotkeys.md")
            if not self._system_prompt:
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
        user_text = f"{format_dimensions(width, height)}\n\n{context}\n\nBased on the detected entities, goals, and resource status above, decide what to do next. Use the tool functions to execute actions."

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
                    break

                messages.append(msg)

                for tc in tool_calls:
                    func = tc.get("function", {})
                    action_dict = {
                        "type": func.get("name", ""),
                        **func.get("arguments", {}),
                    }
                    executed_actions.append(action_dict)

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
