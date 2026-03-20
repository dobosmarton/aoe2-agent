"""Strategist LLM provider for goal generation."""

import base64
from pathlib import Path
import anthropic
import structlog
from pydantic import BaseModel, Field

from ..config import config
from ..goals import Goal
from ..memory import GameState

log = structlog.get_logger()

# Load strategist prompt
PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"


class StrategistGoal(BaseModel):
    """A single goal from the strategist."""

    name: str
    type: str = Field(description="'local' or 'global'")
    metric: str
    target: str = Field(description="Target value as string, e.g. '10', '200', 'Feudal Age'")
    priority: int = Field(ge=1, le=10)


class ResourceReadings(BaseModel):
    """Resource readings extracted from screenshot by strategist."""

    food: int = 0
    wood: int = 0
    gold: int = 0
    stone: int = 0
    population: str = "0/0"
    age: str = "Dark Age"


class StrategistResponse(BaseModel):
    """Structured response from the strategist."""

    reasoning: str
    resource_readings: ResourceReadings
    goals: list[StrategistGoal]


class StrategistProvider:
    """Sonnet-powered strategist that creates/updates goals.

    Uses a stronger model for deeper reasoning about strategy.
    Runs every N turns (configured via config.strategist_interval),
    or immediately when an alarm is triggered (e.g. enemy attack).
    """

    def __init__(self, model: str | None = None):
        self.model = model or config.strategist_model
        self.client = anthropic.AsyncAnthropic(api_key=config.anthropic_api_key, max_retries=2)
        self.refresh_interval = config.strategist_interval
        self._system_prompt: str | None = None
        self._last_alarm_turn: int = 0  # Cooldown tracking
        self._has_run: bool = False  # Track first successful run

    def get_system_prompt(self) -> str:
        if self._system_prompt is None:
            prompt_file = PROMPTS_DIR / "strategist.md"
            if prompt_file.exists():
                self._system_prompt = prompt_file.read_text()
            else:
                self._system_prompt = "You are a strategic advisor for an AoE2 AI. Create 3-5 prioritized goals as JSON."
        return self._system_prompt

    def should_run(self, turn: int, alarm: bool = False) -> bool:
        """Check if the strategist should run this turn."""
        if not self._has_run:
            return True  # Keep trying every turn until first success
        if turn % self.refresh_interval == 0:
            return True
        if alarm and (turn - self._last_alarm_turn) >= 3:
            self._last_alarm_turn = turn
            return True
        return False

    async def _call_api(self, content: list[dict]) -> StrategistResponse:
        """Call strategist API with structured output parsing.

        SDK handles retry (429/5xx) with exponential backoff automatically.
        """
        response = await self.client.messages.parse(
            model=self.model,
            max_tokens=768,
            system=self.get_system_prompt(),
            messages=[{"role": "user", "content": content}],
            output_format=StrategistResponse,
        )
        if response.stop_reason == "refusal":
            raise ValueError("Strategist refused the request")
        return response.parsed_output

    async def generate_goals(
        self,
        game_state: GameState,
        current_goals_summary: str,
        detected_entities_summary: str,
        turn: int,
        screenshot_bytes: bytes | None = None,
        alarm: bool = False,
    ) -> tuple[list[Goal], dict]:
        """Ask the strategist to create/update goals based on game state.

        Returns:
            Tuple of (goals, resource_readings_dict)
        """
        alarm_banner = ""
        if alarm:
            alarm_banner = "\n**ALARM: Enemy military units detected! Prioritize defense and military goals.**\n"

        prompt_text = f"""Turn: {turn}
{alarm_banner}
## Current Game State (from previous readings)
- Resources: Food={game_state.resources['food']}, Wood={game_state.resources['wood']}, Gold={game_state.resources['gold']}, Stone={game_state.resources['stone']}
- Population: {game_state.population}/{game_state.population_cap}
- Age: {game_state.current_age}
- Under Attack: {game_state.under_attack}
- Enemy Located: {game_state.enemy_located}

## Detected Entities on Screen
{detected_entities_summary or "No entities detected"}

## Current Goals
{current_goals_summary}

Read the screenshot to get exact current resource values (food, wood, gold, stone), population, and age. Then create 3-5 prioritized goals. Keep goals that are still relevant, replace completed or irrelevant ones."""

        # Build multimodal content: screenshot image + text
        content: list[dict] = []
        if screenshot_bytes:
            image_base64 = base64.standard_b64encode(screenshot_bytes).decode("utf-8")
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_base64,
                },
            })
        content.append({"type": "text", "text": prompt_text})

        try:
            result = await self._call_api(content)
            log.info(
                "strategist_response",
                reasoning=result.reasoning[:150],
                goal_count=len(result.goals),
                resources=result.resource_readings.model_dump(),
            )

            # Convert resource readings to dict
            readings = result.resource_readings.model_dump()

            # Convert to Goal objects
            goals = []
            for strategist_goal in result.goals:
                goal_type = "local" if strategist_goal.type == "local" else "global"
                # Convert numeric target strings back to numbers
                target: str | int | float = strategist_goal.target
                try:
                    target = int(strategist_goal.target)
                except ValueError:
                    try:
                        target = float(strategist_goal.target)
                    except ValueError:
                        pass  # Keep as string (e.g., "Feudal Age")
                goals.append(
                    Goal(
                        name=strategist_goal.name,
                        type=goal_type,
                        metric=strategist_goal.metric,
                        target=target,
                        priority=strategist_goal.priority,
                        created_turn=turn,
                    )
                )
            self._has_run = True
            return goals, readings

        except Exception as e:
            log.error("strategist_error", error=str(e))
            # Return default goals and empty readings on failure
            default_goals = [
                Goal(name="Queue villagers", type="local", metric="population",
                     target=10, priority=9, created_turn=turn),
                Goal(name="Gather food", type="local", metric="food",
                     target=200, priority=8, created_turn=turn),
                Goal(name="Advance to Feudal Age", type="global", metric="age",
                     target="Feudal Age", priority=4, created_turn=turn),
            ]
            return default_goals, {}
