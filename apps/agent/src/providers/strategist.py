"""Strategist LLM provider for goal generation."""

import asyncio
import contextlib
import io
from pathlib import Path
from typing import ClassVar, cast

import anthropic
import structlog
from PIL import Image
from pydantic import BaseModel, Field

from ..config import config
from ..goals import Goal
from ..memory import GameState
from ..resource_ocr import (
    Backend,
    autodetect_calibration,
    calibration_for,
    read_resource_bar,
)

log = structlog.stdlib.get_logger()

# Load strategist prompt
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


class StrategistGoal(BaseModel):
    """A single goal from the strategist."""

    name: str
    type: str = Field(description="'local' or 'global'")
    metric: str
    target: str = Field(description="Target value as string, e.g. '10', '200', 'Feudal Age'")
    priority: int = Field(ge=1, le=10)


class StrategistResponse(BaseModel):
    """Structured response from the strategist.

    Goals only — resources/population/age are read locally via OCR
    (`read_resource_bar`), not by the model.
    """

    reasoning: str
    goals: list[StrategistGoal]


def _clean_readings(ocr: dict) -> dict:
    """Shape `read_resource_bar` output into a readings dict.

    Drops an empty age so a failed age read keeps the last-known age rather than
    overwriting it; resource/population keys are present only when read.
    """
    readings: dict = {}
    for key in ("food", "wood", "gold", "stone", "population"):
        if key in ocr:
            readings[key] = ocr[key]
    if ocr.get("age"):
        readings["age"] = ocr["age"]
    return readings


async def read_hud_readings(screenshot_bytes: bytes) -> dict:
    """Read resources/population/age off the resource bar via local OCR.

    Resolution precedence: a hand-tuned ``calibration.<W>x<H>.yaml`` wins; else
    auto-detect the bar from this frame. Returns cleaned readings (``{}`` when the
    bar can't be localized). No image is sent to any model. Called per-turn by the
    game loop to keep ``game_state`` fresh, and by the strategist for its prompt.
    A bad/undecodable frame returns ``{}`` rather than raising, so one bad capture
    can never kill the game loop (the caller keeps last-known state).
    """
    try:
        with Image.open(io.BytesIO(screenshot_bytes)) as im:
            width, height = im.width, im.height
        calib = calibration_for(width, height)
        if calib is None:
            # Detection runs RapidOCR over the top band — CPU-bound, off the loop.
            calib = await asyncio.to_thread(autodetect_calibration, screenshot_bytes)
            if calib is not None:
                log.info("ocr_autodetect", width=width, height=height, fields=sorted(calib.fields))
        if calib is None:
            log.error("ocr_no_calibration_autodetect_failed", width=width, height=height)
            return {}
        # OCR is sync/CPU-bound — run off the event loop.
        ocr = await asyncio.to_thread(
            read_resource_bar,
            screenshot_bytes,
            calib,
            backend=cast("Backend", config.ocr_backend),
        )
        readings = _clean_readings(ocr)
        log.info("ocr_readings", **readings)
        return readings
    except Exception as e:  # a bad frame must not crash the loop — keep last-known
        log.warning("hud_read_failed", error=str(e))
        return {}


def get_default_goals(turn: int = 0) -> list[Goal]:
    """Return sensible Dark Age starting goals used before the strategist responds."""
    return [
        Goal(
            name="Queue villagers",
            type="local",
            metric="population",
            target=10,
            priority=9,
            created_turn=turn,
        ),
        Goal(
            name="Gather food",
            type="local",
            metric="food",
            target=200,
            priority=8,
            created_turn=turn,
        ),
        Goal(
            name="Advance to Feudal Age",
            type="global",
            metric="age",
            target="Feudal Age",
            priority=4,
            created_turn=turn,
        ),
    ]


class StrategistProvider:
    """Sonnet-powered strategist that creates/updates goals.

    Uses a stronger model for deeper reasoning about strategy.
    Runs every N turns (configured via config.strategist_interval),
    or immediately when an alarm is triggered (e.g. enemy attack).
    """

    def __init__(self, model: str | None = None) -> None:
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
                self._system_prompt = prompt_file.read_text(encoding="utf-8")
            else:
                self._system_prompt = "You are a strategic advisor for an AoE2 AI. Create 3-5 prioritized goals as JSON."
        return self._system_prompt

    # Age-dependent refresh intervals for fresher readings in early game
    _AGE_INTERVALS: ClassVar[dict[str, int]] = {
        "Dark Age": 3,
        "Feudal Age": 5,
    }

    def should_run(self, turn: int, alarm: bool = False, age: str = "") -> bool:
        """Check if the strategist should run this turn."""
        if not self._has_run:
            return True  # Keep trying every turn until first success
        interval = self._AGE_INTERVALS.get(age, self.refresh_interval)
        if turn % interval == 0:
            return True
        if alarm and (turn - self._last_alarm_turn) >= 3:
            self._last_alarm_turn = turn
            return True
        return False

    async def _call_api(self, content: list[dict]) -> StrategistResponse:
        """Call strategist API with structured output parsing.

        SDK handles retry (429/5xx) with exponential backoff automatically.
        """
        # `messages.parse` is the structured-output method — runtime-only,
        # absent from public AsyncMessages stubs. The argument-type union for
        # `messages` shifts between anthropic SDK versions, so we ignore the
        # arg-type check too. The runtime contract is stable; the typing isn't.
        response = await self.client.messages.parse(  # pyright: ignore[reportAttributeAccessIssue]
            model=self.model,
            max_tokens=768,
            temperature=config.temperature,
            system=self.get_system_prompt(),
            messages=[{"role": "user", "content": content}],  # pyright: ignore[reportArgumentType]
            output_format=StrategistResponse,
        )
        if response.stop_reason == "refusal":
            raise ValueError("Strategist refused the request")

        usage = response.usage
        log.info(
            "strategist_usage", input_tokens=usage.input_tokens, output_tokens=usage.output_tokens
        )
        return cast("StrategistResponse", response.parsed_output)

    async def generate_goals(
        self,
        game_state: GameState,
        current_goals_summary: str,
        detected_entities_summary: str,
        turn: int,
        screenshot_bytes: bytes | None = None,
        alarm: bool = False,
    ) -> tuple[list[Goal], dict]:
        """Read the resource bar locally (OCR) and ask the LLM (text-only) for goals.

        Claude vision is not used: resources/population/age come from
        ``read_resource_bar``; the model only reasons over them to set goals.

        Returns:
            Tuple of (goals, resource_readings_dict)
        """
        # 1. Perception — resources/population/age from the resource bar via local
        #    OCR; no screenshot is sent to the model. game_state is the fallback
        #    (the game loop keeps it fresh every turn with the same OCR).
        readings: dict = await read_hud_readings(screenshot_bytes) if screenshot_bytes else {}

        # 2. Reasoning — text-only prompt populated with the locally-read state
        #    (falls back to last-known game_state when a field wasn't read).
        food = readings.get("food", game_state.resources["food"])
        wood = readings.get("wood", game_state.resources["wood"])
        gold = readings.get("gold", game_state.resources["gold"])
        stone = readings.get("stone", game_state.resources["stone"])
        population = readings.get(
            "population", f"{game_state.population}/{game_state.population_cap}"
        )
        age = readings.get("age", game_state.current_age)

        alarm_banner = ""
        if alarm:
            alarm_banner = "\n**ALARM: Enemy military units detected! Prioritize defense and military goals.**\n"

        prompt_text = f"""Turn: {turn}
{alarm_banner}
## Current Game State (resources read from the HUD)
- Resources: Food={food}, Wood={wood}, Gold={gold}, Stone={stone}
- Population: {population}
- Age: {age}
- Under Attack: {game_state.under_attack}
- Enemy Located: {game_state.enemy_located}

## Detected Entities on Screen
{detected_entities_summary or "No entities detected"}

## Current Goals
{current_goals_summary}

Create 3-5 prioritized goals based on the state above. Keep goals that are still relevant, replace completed or irrelevant ones."""

        content: list[dict] = [{"type": "text", "text": prompt_text}]

        try:
            result = await self._call_api(content)
            log.info(
                "strategist_response",
                reasoning=result.reasoning[:150],
                goal_count=len(result.goals),
            )

            # Convert to Goal objects
            goals = []
            for strategist_goal in result.goals:
                goal_type = "local" if strategist_goal.type == "local" else "global"
                # Convert numeric target strings back to numbers
                target: str | int | float = strategist_goal.target
                try:
                    target = int(strategist_goal.target)
                except ValueError:
                    # Try float; if that also fails, keep target as string (e.g., "Feudal Age").
                    with contextlib.suppress(ValueError):
                        target = float(strategist_goal.target)
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
            # OCR already succeeded — return its readings even if goal-gen failed.
            log.error("strategist_error", error=str(e))
            return get_default_goals(turn), readings
