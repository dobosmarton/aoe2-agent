"""Parallel sub-agent executor for AoE2 Agent.

Splits the executor's work into parallel domain-specific sub-agents
that run concurrently via asyncio.gather(), reducing wall-clock latency
to the time of the slowest sub-agent instead of sequential sum.
"""

import asyncio
from pathlib import Path
from typing import Any

import anthropic
import structlog

from ..config import config
from ..models import LLMResponse, validate_actions
from .base import BaseLLMProvider

log = structlog.get_logger()

PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"

# Domain-specific sub-prompts that focus each sub-agent on its responsibility
_ECONOMY_PROMPT = """You are the ECONOMY sub-agent for an Age of Empires 2 AI. You handle ONLY economy actions:
- Queue villagers at Town Center (H → Q)
- Send idle villagers to gather resources (. → right_click resource)
- Build houses when near population cap
- Build economic buildings (Mill, Lumber Camp, farms)
- Set TC gather points
- Manage food/wood/gold/stone balance

Do NOT handle military units, scouting, or combat. Focus only on economy.
Output 2-5 economy actions. NEVER return 0 actions — at minimum queue a villager (H, Q) and sweep for idles (.).
"""

_MILITARY_PROMPT = """You are the MILITARY sub-agent for an Age of Empires 2 AI. You handle ONLY military and scouting:
- Enable Auto Scout (, → G) if not already done
- Train military units from military buildings
- Send military units to attack or defend
- Research military technologies
- Age advancement (research at TC when ready)

Do NOT handle villager management or economy buildings. Focus only on military and scouting.
If there's nothing military to do (e.g., Dark Age with no enemies), return just 1 action: a wait.
"""


class SubAgentResult:
    """Result from a single sub-agent call."""

    def __init__(self, domain: str, actions: list[dict], reasoning: str):
        self.domain = domain
        self.actions = actions
        self.reasoning = reasoning


class ParallelExecutorProvider(BaseLLMProvider):
    """Parallel sub-agent executor that splits work across concurrent LLM calls.

    Fires economy and military sub-agents in parallel via asyncio.gather().
    Wall-clock latency = max(sub_agent_times) instead of sum(sub_agent_times).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self.api_key = api_key or config.anthropic_api_key
        self.model = model or config.model
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key, max_retries=3)
        self._base_prompt: str | None = None

    def get_system_prompt(self) -> str:
        """Load base game knowledge (hotkeys, rules) shared by all sub-agents."""
        if self._base_prompt is None:
            hotkeys_file = PROMPTS_DIR / "hotkeys.md"
            self._base_prompt = ""
            if hotkeys_file.exists():
                self._base_prompt = hotkeys_file.read_text()
        return self._base_prompt

    def _build_sub_prompt(self, domain_prompt: str) -> str:
        """Build a full sub-agent system prompt: domain focus + hotkeys."""
        return domain_prompt + "\n\n" + self.get_system_prompt()

    async def _call_sub_agent(
        self, domain: str, domain_prompt: str, context: str,
    ) -> SubAgentResult:
        """Call a single domain-specific sub-agent."""
        system_prompt = self._build_sub_prompt(domain_prompt)

        try:
            response = await self.client.messages.parse(
                model=self.model,
                max_tokens=config.max_tokens,
                system=[{
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": context}],
                output_format=LLMResponse,
            )

            if response.stop_reason == "refusal":
                log.warning("sub_agent_refused", domain=domain)
                return SubAgentResult(domain, [], "refused")

            result = response.parsed_output
            actions = [
                a.model_dump() if hasattr(a, "model_dump") else a
                for a in result.actions
            ]
            log.info(
                "sub_agent_response",
                domain=domain,
                action_count=len(actions),
                reasoning=result.reasoning[:100],
            )
            return SubAgentResult(domain, actions, result.reasoning)

        except Exception as e:
            log.error("sub_agent_error", domain=domain, error=str(e))
            return SubAgentResult(domain, [], f"Error: {e}")

    def _merge_actions(self, results: list[SubAgentResult]) -> dict[str, Any]:
        """Merge actions from all sub-agents. Economy actions come first."""
        all_actions: list[dict] = []
        reasoning_parts: list[str] = []

        # Economy first, then military
        for result in sorted(results, key=lambda r: 0 if r.domain == "economy" else 1):
            all_actions.extend(result.actions)
            if result.reasoning:
                reasoning_parts.append(f"[{result.domain}] {result.reasoning}")

        validated = validate_actions(all_actions)
        return {
            "reasoning": " | ".join(reasoning_parts),
            "observations": {},
            "actions": [
                a.model_dump() if hasattr(a, "model_dump") else a
                for a in validated
            ],
            "actions_already_executed": False,
        }

    async def get_actions(
        self,
        context: str,
        width: int = 1920,
        height: int = 1080,
    ) -> dict[str, Any]:
        center_x = width // 2
        center_y = height // 2
        dimensions = f"Game window: {width}x{height} pixels. Center=({center_x},{center_y}). Valid x=0-{width}, y=0-{height}."
        full_context = f"{dimensions}\n\n{context}\n\nDecide what actions to take for your domain."

        # Fire sub-agents in parallel
        results = await asyncio.gather(
            self._call_sub_agent("economy", _ECONOMY_PROMPT, full_context),
            self._call_sub_agent("military", _MILITARY_PROMPT, full_context),
        )

        merged = self._merge_actions(list(results))
        log.info(
            "parallel_executor_merged",
            total_actions=len(merged["actions"]),
            sub_agents=len(results),
        )
        return merged
