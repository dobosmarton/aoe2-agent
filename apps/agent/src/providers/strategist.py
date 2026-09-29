"""Strategist LLM provider for goal generation."""

import asyncio
import contextlib
import io
from pathlib import Path
from typing import ClassVar, Literal, cast

import structlog
from PIL import Image
from pydantic import BaseModel, Field

from ..config import config
from ..entity_utils import RESOURCE_KINDS, ResourceKind
from ..goals import Goal
from ..memory import GameState
from ..policy.allocation import Allocation
from ..resource_ocr import (
    Backend,
    Calibration,
    ResourceReadings,
    autodetect_calibration,
    calibration_for,
    read_age,
    read_resource_bar,
)
from .base import ChatRequest, ChatWire, SystemBlock, UserTurn, text_of_blocks
from .pricing import cost_usd
from .wire_factory import make_wire

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


class VillagerTargets(BaseModel):
    """Villagers wanted per resource.

    Fixed keys, not a dict: OpenAI structured outputs 400s on an open object.
    Plain ints — a bound would grow the compiled grammar (F-40).
    """

    food: int = 0
    wood: int = 0
    gold: int = 0
    stone: int = 0

    def per_kind(self) -> dict[ResourceKind, int]:
        """The targets keyed the way the policy tier names resources."""
        return {"food": self.food, "wood": self.wood, "gold": self.gold, "stone": self.stone}


class StrategistResponse(BaseModel):
    """Structured response from the strategist.

    Goals and a villager allocation — resources/population/age are read locally
    via OCR (`read_resource_bar`), not by the model.
    """

    reasoning: str
    goals: list[StrategistGoal]
    # All zero falls back to the seeded per-age mix, which the LLM-down path runs on.
    allocation: VillagerTargets = Field(default_factory=VillagerTargets)


def _clean_readings(ocr: dict[str, object]) -> ResourceReadings:
    """Shape `read_resource_bar` output into a readings dict.

    Drops an empty age so a failed age read keeps the last-known age rather than
    overwriting it; resource/population keys are present only when read.
    """
    readings: dict[str, object] = {
        key: ocr[key]
        for key in (
            "food",
            "wood",
            "gold",
            "stone",
            "population",
            "idle_present",
            "idle_count",
            "food_workers",
            "wood_workers",
            "gold_workers",
            "stone_workers",
            "villagers",
        )
        if key in ocr
    }
    if ocr.get("age"):
        readings["age"] = ocr["age"]
    _discard_contradictory_workforce(readings)
    # Per-key value types are enforced where read_resource_bar builds them.
    return cast("ResourceReadings", readings)


def _discard_contradictory_workforce(readings: dict[str, object]) -> None:
    """Leave impossible HUD counts unknown; never replace them with guesses."""
    population = readings.get("population")
    villagers = readings.get("villagers")
    if isinstance(population, str) and isinstance(villagers, int):
        current, separator, _cap = population.partition("/")
        if separator and current.isdigit() and villagers > int(current):
            log.warning(
                "hud_workforce_conflict", field="villagers", value=villagers, population=current
            )
            readings.pop("villagers", None)
            villagers = None
    if not isinstance(villagers, int):
        return
    workers = ("food_workers", "wood_workers", "gold_workers", "stone_workers")
    counts = [readings.get(field) for field in workers]
    if (
        any(isinstance(count, int) and count < 0 for count in counts)
        or sum(count for count in counts if isinstance(count, int)) > villagers
    ):
        log.warning("hud_workforce_conflict", field="workers", villagers=villagers, counts=counts)
        for field in workers:
            readings.pop(field, None)
    idle_count = readings.get("idle_count")
    if isinstance(idle_count, int) and (idle_count < 0 or idle_count > villagers):
        log.warning(
            "hud_workforce_conflict", field="idle_count", value=idle_count, villagers=villagers
        )
        readings.pop("idle_count", None)


# A frame is trusted only when OCR decoded most of the core resources. A frame that
# yields a single field (e.g. only "stone=1", the rest dropped) is a mis-read —
# discarding it keeps last-known state instead of poisoning game_state with garbage.
_CORE_RESOURCE_FIELDS: tuple[str, ...] = ("food", "wood", "gold", "stone")
_MIN_CORE_FIELDS: int = 3


def _is_reliable_frame(readings: ResourceReadings) -> bool:
    """Whether enough core resource fields decoded to trust this OCR frame."""
    return sum(1 for key in _CORE_RESOURCE_FIELDS if key in readings) >= _MIN_CORE_FIELDS


# Turns between RapidOCR age reads on the template backend (which can't read
# text itself). Age changes ~3x/game; between samples the empty age keeps the
# last-known value downstream (_clean_readings drops it, memory ignores falsy).
_AGE_OCR_INTERVAL = 5


def _age_read_due(turn: int | None, *, template_primary: bool) -> bool:
    """Whether this tick pays for the slow RapidOCR age read.

    Only a template-first read needs it (the OCR backends read age inline);
    an unknown turn (standalone/eval callers) always reads.
    """
    if not template_primary:
        return False
    return turn is None or turn % _AGE_OCR_INTERVAL == 0


def _has_fast_templates(calib: Calibration) -> bool:
    """Use calibrated glyphs when shipped for this exact HUD resolution."""
    return calib.template_dir.is_dir()


async def read_hud_readings(
    screenshot_bytes: bytes,
    *,
    turn: int | None = None,
    full_size: tuple[int, int] | None = None,
    mode: Literal["routine", "critical"] = "routine",
) -> tuple[ResourceReadings, Calibration | None]:
    """Read resources/population/age off the resource bar via local OCR.

    Resolution precedence: a hand-tuned ``calibration.<W>x<H>.yaml`` wins; else
    auto-detect the bar from this frame. Returns ``(readings, calibration)`` — the
    cleaned readings plus the calibration actually used, so callers can also draw
    the reading regions (the debug overlay). Both are empty/``None`` when the bar
    can't be localized. No image is sent to any model. Called per-turn by the game
    loop to keep ``game_state`` fresh, and by the strategist for its prompt. A
    bad/undecodable frame returns ``({}, None)`` rather than raising, so one bad
    capture can never kill the game loop (the caller keeps last-known state).
    Critical post-input reads skip age text and slow fallback OCR; an unreadable
    field remains unknown rather than delaying a dependent click.
    """
    try:
        with Image.open(io.BytesIO(screenshot_bytes)) as im:
            width, height = full_size or im.size
        calib = calibration_for(width, height)
        if calib is None:
            # Detection runs RapidOCR over the top band — CPU-bound, off the loop.
            calib = await asyncio.to_thread(autodetect_calibration, screenshot_bytes)
            if calib is not None:
                log.info("ocr_autodetect", width=width, height=height, fields=sorted(calib.fields))
        if calib is None:
            log.error("ocr_no_calibration_autodetect_failed", width=width, height=height)
            return {}, None
        template_primary = config.ocr_backend == "template" or _has_fast_templates(calib)
        backend: Backend = "template" if template_primary else cast("Backend", config.ocr_backend)
        ocr = await asyncio.to_thread(read_resource_bar, screenshot_bytes, calib, backend=backend)
        readings = _clean_readings(ocr)
        if (
            mode == "routine"
            and template_primary
            and (not _is_reliable_frame(readings) or "population" not in readings)
            and config.ocr_backend != "template"
        ):
            # Escalate an unresolved field, rather than paying full-image OCR on
            # every ordinary frame. Keep the calibrated worker digits: the text
            # OCR backend does not read this small HUD font reliably.
            slow = _clean_readings(
                await asyncio.to_thread(
                    read_resource_bar,
                    screenshot_bytes,
                    calib,
                    backend=cast("Backend", config.ocr_backend),
                )
            )
            merged: dict[str, object] = dict(readings)
            for key, value in slow.items():
                merged.setdefault(key, value)
            readings = cast("ResourceReadings", merged)
        if not _is_reliable_frame(readings):
            log.warning("ocr_frame_discarded", fields=sorted(readings))
            return {}, calib
        if mode == "routine" and _age_read_due(turn, template_primary=template_primary):
            age = await asyncio.to_thread(read_age, screenshot_bytes, calib)
            if age:
                readings["age"] = age
        log.info("ocr_readings", **readings)
        return readings, calib
    except Exception as e:  # a bad frame must not crash the loop — keep last-known
        log.warning("hud_read_failed", error=str(e))
        return {}, None


def as_allocation(targets: VillagerTargets) -> Allocation | None:
    """The strategist's allocation, or None to fall back to the seeded mix.

    A fixed-key model is never empty, so the all-zero answer the model gives
    when it has no opinion must still map to None.
    """
    declared = targets.per_kind()
    clean: dict[ResourceKind, int] = {
        kind: declared[kind] for kind in RESOURCE_KINDS if declared[kind] > 0
    }
    return Allocation(targets=clean) if clean else None


def get_default_goals(turn: int = 0) -> list[Goal]:
    """Return sensible Dark Age starting goals used before the strategist responds."""
    return [
        Goal(
            name="Queue villagers",
            type="local",
            metric="villagers",
            target=10,
            priority=9,
            created_turn=turn,
        ),
        Goal(
            name="Gather food",
            type="local",
            metric="food_workers",
            target=3,
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

    def __init__(self, model: str | None = None, wire: ChatWire | None = None) -> None:
        self.model = model or config.strategist_model
        self.wire: ChatWire = wire or make_wire(
            config.llm_wire,
            model=self.model,
            api_key=config.llm_api_key,
            base_url=config.llm_base_url,
            max_retries=2,
        )
        self.refresh_interval = config.strategist_interval
        self._system_prompt: str | None = None
        self._last_alarm_turn: int = 0  # Cooldown tracking
        self._has_run: bool = False  # Track first successful run
        # Latest parsed allocation; None until the model supplies one.
        self.last_allocation: Allocation | None = None

    async def aclose(self) -> None:
        """Release the strategist transport after its task is cancelled."""
        close = getattr(self.wire, "aclose", None)
        if close is not None:
            await close()

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

    # Goals, not tool calls: one structured call is the whole interaction.
    _MAX_TOKENS: ClassVar[int] = 768

    async def _call_api(self, content: list[dict]) -> StrategistResponse:
        """Call the strategist model with structured output parsing.

        SDK-internal retry on 429/5xx; raises `ModelRefusedError` on a decline.
        """
        parsed, usage = await self.wire.parse_structured(
            ChatRequest(
                system=(SystemBlock(text=self.get_system_prompt()),),
                turns=(UserTurn(text=text_of_blocks(content)),),
                max_tokens=self._MAX_TOKENS,
                temperature=config.temperature,
            ),
            StrategistResponse,
        )
        # Nothing else prices these calls, and this is the dearer of the 2 models.
        log.info(
            "strategist_usage",
            model=self.model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=round(cost_usd(self.model, usage), 4),
        )
        return parsed

    async def generate_goals(
        self,
        game_state: GameState,
        current_goals_summary: str,
        detected_entities_summary: str,
        turn: int,
        screenshot_bytes: bytes | None = None,
        alarm: bool = False,
        readings: ResourceReadings | None = None,
        known_buildings: str = "",
    ) -> tuple[list[Goal], ResourceReadings]:
        """Ask the LLM (text-only) for goals from a HUD reading.

        Claude vision is not used: resources/population/age come from
        ``read_resource_bar``; the model only reasons over them to set goals.

        Returns:
            Tuple of (goals, resource_readings)
        """
        # 1. Perception — the game loop passes its per-turn HUD reading (one OCR
        #    pass per frame); only the standalone/eval path (readings=None) OCRs
        #    the screenshot itself. An empty reading is a bad frame, not a reason
        #    to re-OCR — game_state fills the gaps below.
        if readings is None:
            if screenshot_bytes:
                readings, _calib = await read_hud_readings(screenshot_bytes)
            else:
                readings = ResourceReadings()

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
{known_buildings}
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
            self.last_allocation = as_allocation(result.allocation)
            self._has_run = True
            return goals, readings

        except Exception as e:
            # OCR already succeeded — return its readings even if goal-gen failed.
            log.error("strategist_error", error=str(e))
            return get_default_goals(turn), readings
