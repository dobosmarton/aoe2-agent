"""Claude (Anthropic) LLM provider for AoE2 Agent."""

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast

import anthropic
import structlog
from anthropic.types import OutputConfigParam, ToolUseBlock
from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

from ..config import config
from ..entity_utils import ResourceKind
from ..executor import (
    build_menu_steps,
    build_steps,
    default_build_placement,
    execute_action,
    get_detected_entities,
)
from ..models import LLMResponse, Observations, validate_actions
from ..villager_roles import select_worker
from .base import LLMResult
from .claude_tools import _ACTION_TOOLS

# Camera "go to work site" hotkey per source job (see prompts/hotkeys.md): jumps the
# view to the drop-off camp so the workers of that job are on screen to pick from.
_JOB_CAMERA_HOTKEY: dict[ResourceKind, tuple[str, list[str]]] = {
    "wood": ("z", ["ctrl"]),  # Lumber Camp
    "gold": ("g", ["ctrl"]),  # Mining Camp
    "stone": ("g", ["ctrl"]),  # Mining Camp
    "food": ("i", ["ctrl"]),  # Mill
}
# Unknown/omitted job falls back to the Lumber Camp jump (wood is the most common pull).
_DEFAULT_JOB_HOTKEY = _JOB_CAMERA_HOTKEY["wood"]


def _tracker_velocities() -> dict[str, tuple[float, float]]:
    """Best-effort per-entity velocity from the already-initialized detector.

    Reads the local detector singleton if one exists (it does whenever local YOLO
    is running) — never creates one, so the remote/mock paths simply return {} and
    selection falls back to nearest-to-camp. Velocity lets `select_worker` prefer a
    stationary (easy-to-click) worker.
    """
    try:
        from detection.inference.detector import current_detector
    except ImportError:
        return {}
    detector = current_detector()
    if detector is None or detector.tracker is None:
        return {}
    # track.state is a numpy array (indexing returns library-typed Any); the layout
    # is [x, y, vx, vy, w, h] — see EntityTracker.
    return {t.id: (float(t.state[2]), float(t.state[3])) for t in detector.tracker.tracks}


def _target_right_click(inp: dict, intent: object) -> dict[str, object]:
    """Right-click step for a send composite: target_class when given, else raw x/y.

    The executor resolves `target_class` against its detected-entity cache; raw
    coordinates are the LLM's explicit fallback.
    """
    rc_action: dict[str, object] = {"type": "right_click", "intent": intent}
    if "target_class" in inp:
        rc_action["target_class"] = inp["target_class"]
    else:
        rc_action["x"] = inp["x"]
        rc_action["y"] = inp["y"]
    return rc_action


# Pricing per million tokens (claude-sonnet-4-6)
_PRICE_INPUT = 3.00
_PRICE_OUTPUT = 15.00
_PRICE_CACHE_READ = 0.30
_PRICE_CACHE_WRITE = 3.75


log = structlog.stdlib.get_logger()

# Load system prompt from file
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

# Optional game knowledge database for dynamic context injection
try:
    from data.game_knowledge import GameKnowledge, get_db

    GAME_KNOWLEDGE_AVAILABLE = True
except ImportError:
    GAME_KNOWLEDGE_AVAILABLE = False
    log.debug("game_knowledge_not_available", message="Running without dynamic context injection")


class ClaudeProvider:
    """Anthropic Claude provider implementation."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        use_dynamic_context: bool = True,
    ) -> None:
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
        self._core_prompt: str | None = None
        self._age_prompts: dict[str, str] = {}
        # Titles of cross-game memories loaded into the cached system block.
        # Populated in _load_prompts() so game_loop can propagate them onto
        # AgentMemory.memories_loaded for per-turn attribution tracking.
        self.loaded_memory_titles: list[str] = []
        self.use_dynamic_context = use_dynamic_context and GAME_KNOWLEDGE_AVAILABLE
        self._game_db: GameKnowledge | None = None
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

    _AGE_NAMES = ("dark", "feudal", "castle", "imperial")
    _FALLBACK_PROMPT = "You are playing Age of Empires 2: Definitive Edition. Your goal is to defeat the enemy AI. Play to win!"

    def _load_prompts(self) -> None:
        """Load all prompt files once (core + age-specific + cross-game memories)."""
        if self._core_prompt is not None:
            return

        core_file = PROMPTS_DIR / "core.md"
        hotkeys_file = PROMPTS_DIR / "hotkeys.md"

        if core_file.exists():
            self._core_prompt = core_file.read_text(encoding="utf-8")
        else:
            self._core_prompt = self._FALLBACK_PROMPT

        if hotkeys_file.exists():
            self._core_prompt += "\n\n" + hotkeys_file.read_text(encoding="utf-8")

        # Append cross-game memories to the cached core block, so they're paid
        # once per game and hit the prompt cache for every turn after the first.
        # Loaded lazily via MemoryChain — falls back gracefully if the module or
        # the memories/ directory is missing.
        try:
            from gameplay_agent.memory_chain import MemoryChain

            chain = MemoryChain()
            memory_prelude = chain.load_memories(max_tokens=800)
            if memory_prelude:
                self._core_prompt += "\n\n" + memory_prelude
                # Capture titles for downstream attribution (game_loop reads
                # this and copies it onto AgentMemory.memories_loaded).
                self.loaded_memory_titles = [
                    m["title"] for m in chain.list_memories() if m.get("title") and m.get("content")
                ]
                log.info(
                    "cross_game_memories_loaded",
                    chars=len(memory_prelude),
                    titles=self.loaded_memory_titles,
                )
        except ImportError:
            log.debug("memory_chain_unavailable")
        except Exception as e:
            log.warning("memory_load_failed", error=str(e))

        ages_dir = PROMPTS_DIR / "ages"
        for age_name in self._AGE_NAMES:
            age_file = ages_dir / f"{age_name}.md"
            if age_file.exists():
                self._age_prompts[age_name] = age_file.read_text(encoding="utf-8")
            else:
                log.debug("age_prompt_missing", age=age_name)

    def get_system_prompt(self, age: str = "Dark Age") -> list[dict]:
        """Return system prompt as a two-block list for optimal caching.

        Block 1 (core + hotkeys) is stable across all ages — always cached.
        Block 2 (age-specific) changes only on age transitions (3 times per game).
        """
        self._load_prompts()

        # "Dark Age" → "dark", "Feudal Age" → "feudal"
        age_key = age.split()[0].lower() if age else "dark"
        age_content = self._age_prompts.get(age_key, self._age_prompts.get("dark", ""))

        blocks = [
            {
                "type": "text",
                "text": self._core_prompt,
                "cache_control": {"type": "ephemeral"},
            },
        ]
        if age_content:
            # Cache the age block too: it changes only on age-ups (<=3 per game),
            # so every turn within an age reads it from cache instead of re-prefilling.
            blocks.append(
                {
                    "type": "text",
                    "text": age_content,
                    "cache_control": {"type": "ephemeral"},
                }
            )

        return blocks

    @staticmethod
    def _extract_age(context: str) -> str:
        """Extract current age from context string."""
        match = re.search(r"(Dark|Feudal|Castle|Imperial)\s*Age", context, re.IGNORECASE)
        return f"{match.group(1)} Age" if match else "Dark Age"

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
        resources: dict[str, object] = {
            "food": 200,
            "wood": 200,
            "gold": 100,
            "stone": 200,
        }  # Defaults
        age = "dark"

        try:
            # Try to extract resources from context
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
            return f"{dynamic_context}\n{early_game_tips}\n{context}"

        except Exception as e:
            log.warning("dynamic_context_error", error=str(e))
            return context

    def _build_content(self, context: str, width: int, height: int) -> list[dict]:
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

        return [
            {
                "type": "text",
                "text": text,
            },
        ]

    # -- Shared helpers --------------------------------------------------------

    ENTITY_RESULT_LIMIT = 20

    def _entity_snapshot(self) -> list[dict]:
        """Return truncated entity list for tool results."""
        return [
            {"id": e.get("id", ""), "class": e.get("class", ""), "center": e.get("center", [])}
            for e in get_detected_entities()[: self.ENTITY_RESULT_LIMIT]
        ]

    def _make_tool_result(
        self,
        block: ToolUseBlock,
        success: bool,
        detail: str,
        *,
        include_entities: bool = False,
    ) -> dict:
        """Build the tool_result dict returned to Claude."""
        result_data: dict[str, object] = {"success": success, "detail": detail}
        if include_entities:
            result_data["entities"] = self._entity_snapshot()
        return {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": json.dumps(result_data),
        }

    @staticmethod
    def _dump_actions(result: LLMResponse) -> list[dict[str, object]]:
        """Serialize each action individually.

        Composite action dicts don't match the Action union type, so we serialize
        per-action to avoid PydanticSerializationUnexpectedValue warnings.
        """
        actions: list[dict[str, object]] = []
        for a in result.actions:
            if isinstance(a, BaseModel):
                actions.append(a.model_dump())
            elif isinstance(a, dict):
                actions.append(a)
        return actions

    @staticmethod
    def _observations_dict(result: LLMResponse) -> dict[str, object]:
        """Dump observations, tolerating a non-model placeholder."""
        if hasattr(result.observations, "model_dump"):
            return result.observations.model_dump()
        return {}

    @staticmethod
    def _serialize_response(result: LLMResponse) -> LLMResult:
        """Serialize a tool-loop response.

        The loop already executed the actions, so actions_already_executed is
        True and the game loop only records them.
        """
        actions = ClaudeProvider._dump_actions(result)
        return LLMResult(
            reasoning=result.reasoning,
            observations=ClaudeProvider._observations_dict(result),
            actions=actions,
            actions_already_executed=True,
            success_count=result._success_count if result._success_count else len(actions),
        )

    @staticmethod
    def _serialize_single_shot(result: LLMResponse) -> LLMResult:
        """Serialize a single-shot response.

        The actions have NOT run yet, so actions_already_executed is False and the
        game loop executes them via the standard executor path. Deliberately not
        _serialize_response, which marks actions already-executed and would
        silently no-op this path.
        """
        return LLMResult(
            reasoning=result.reasoning,
            observations=ClaudeProvider._observations_dict(result),
            actions=ClaudeProvider._dump_actions(result),
            actions_already_executed=False,
        )

    async def _run_steps(self, composite_name: str, steps: list[dict]) -> tuple[bool, str]:
        """Execute action steps sequentially, stop on first failure."""
        for step in steps:
            r = await execute_action(step)
            log.info(
                "composite_step",
                composite=composite_name,
                action=step["type"],
                key=step.get("key", ""),
                success=r.success,
            )
            if not r.success:
                return False, f"failed at {step['intent']}"
        return True, "ok"

    # -- Composite tool handlers ------------------------------------------------
    # Execute multi-step sequences locally, avoiding intermediate API roundtrips.

    _COMPOSITE_NAMES: ClassVar[set[str]] = {
        "build",
        "send_villager",
        "queue_villager",
        "reassign_villager",
    }

    async def _run_composite(
        self,
        block: ToolUseBlock,
        name: str,
        steps: list[dict],
        *,
        include_entities: bool = True,
    ) -> tuple[dict, dict]:
        """Run a composite's steps and package the (action_dict, tool_result) pair.

        The tail every composite handler shares: execute the steps, echo the tool
        input back as the recorded action dict, wrap success/detail for Claude.
        """
        success, detail = await self._run_steps(name, steps)
        action_dict = {"type": name, **block.input}
        tool_result = self._make_tool_result(
            block, success, detail, include_entities=include_entities
        )
        return action_dict, tool_result

    async def _execute_build(self, block: ToolUseBlock) -> tuple[dict, dict]:
        """Composite: press . → q (econ menu) → building_key → click(x,y).

        Uses the LLM-provided coordinates directly.  The executor's
        built-in retry logic (_handle_click) tries 4 nearby offsets
        (±80px) if placement fails.  If the spot is truly blocked,
        the LLM will choose a different position next turn.
        """
        inp = block.input
        intent = str(inp.get("intent", "Build"))
        # x,y are optional: the text-only model can't see open ground, so when it
        # omits them we auto-place near the town centre. See default_build_placement.
        x, y = inp.get("x"), inp.get("y")
        placement = (
            (cast("int", x), cast("int", y))
            if x is not None and y is not None
            else default_build_placement()
        )
        steps = build_steps(cast("str", inp["building_key"]), intent, placement)
        return await self._run_composite(block, "build", steps)

    async def _execute_send_villager(self, block: ToolUseBlock) -> tuple[dict, dict]:
        """Composite: press . (with rescan) → right_click target.

        The "." press moves the camera, so we rescan to get fresh entity
        positions before right-clicking.
        """
        inp = block.input
        intent = inp.get("intent", "Send villager")
        steps: list[dict] = [
            {
                "type": "press",
                "key": ".",
                "rescan": True,
                "intent": f"Select idle villager ({intent})",
            },
            _target_right_click(inp, intent),
        ]
        return await self._run_composite(block, "send_villager", steps)

    async def _execute_send_all_idle(self, block: ToolUseBlock) -> tuple[dict, dict]:
        """Composite: Shift-. (select ALL idle) → right_click target.

        Dispatches every idle villager in one action. Mirrors send_villager but
        uses the select-all hotkey so no idle count is needed.
        """
        inp = block.input
        intent = inp.get("intent", "Send all idle villagers")
        steps: list[dict] = [
            {
                "type": "press",
                "key": ".",
                "modifiers": ["shift"],
                "rescan": True,
                "intent": f"Select ALL idle villagers ({intent})",
            },
            _target_right_click(inp, intent),
        ]
        return await self._run_composite(block, "send_all_idle", steps)

    async def _execute_queue_villager(self, block: ToolUseBlock) -> tuple[dict, dict]:
        """Composite: press h → press q."""
        inp = block.input
        intent = inp.get("intent", "Queue villager")
        steps: list[dict] = [
            {"type": "press", "key": "h", "intent": f"Go to TC ({intent})"},
            {"type": "press", "key": "q", "intent": f"Queue villager ({intent})"},
        ]
        return await self._run_composite(block, "queue_villager", steps, include_entities=False)

    async def _execute_reassign_villager(self, block: ToolUseBlock) -> tuple[dict, dict]:
        """Composite: jump to a work site → pick a working villager → build.

        Two phases because the worker's screen position only exists AFTER the camera
        jump: (1) run the go-to-camp press with a rescan, then (2) read the fresh
        detections, choose a worker of `from_job` (stationary-first when tracker
        velocities are available), and select→build→place. Falls back to selecting
        the highest-confidence villager on screen if the job model finds none.
        """
        inp = block.input
        intent = str(inp.get("intent", "Reassign villager"))
        # LLM boundary: the tool schema restricts from_job to the resource kinds, so
        # narrow the raw string here; an off-schema value keeps today's behavior
        # (default hotkey jump → no candidates → highest-confidence villager).
        from_job = cast("ResourceKind", str(inp.get("from_job", "wood")))
        building_key = str(inp.get("building_key", "a"))  # 'a' = Farm in the econ menu
        action_dict = {"type": "reassign_villager", **inp}

        # Phase 1 — jump the camera to the source work site and re-detect.
        goto_key, goto_mods = _JOB_CAMERA_HOTKEY.get(from_job, _DEFAULT_JOB_HOTKEY)
        ok, detail = await self._run_steps(
            "reassign_villager",
            [
                {
                    "type": "press",
                    "key": goto_key,
                    "modifiers": goto_mods,
                    "rescan": True,
                    "intent": f"Go to {from_job} work site ({intent})",
                }
            ],
        )
        if not ok:
            return action_dict, self._make_tool_result(block, False, detail, include_entities=True)

        # Phase 2 — pick a worker from the fresh view, then select → build → place.
        worker_click = select_worker(
            cast("list[object]", get_detected_entities()),
            from_job,
            velocities=_tracker_velocities(),
        )
        if worker_click is not None:
            select_step: dict = {
                "type": "click",
                "x": worker_click[0],
                "y": worker_click[1],
                "intent": f"Select {from_job} villager ({intent})",
            }
        else:
            # No worker of that job resolved — grab the best villager on screen.
            select_step = {
                "type": "click",
                "target_class": "villager",
                "intent": f"Select villager ({intent})",
            }
        steps = [
            select_step,
            *build_menu_steps(
                building_key,
                intent,
                default_build_placement(),
                menu_intent=f"Open economic build menu ({intent})",
            ),
        ]
        ok, detail = await self._run_steps("reassign_villager", steps)
        return action_dict, self._make_tool_result(block, ok, detail, include_entities=True)

    # -- Tool dispatch ---------------------------------------------------------

    _COMPOSITE_HANDLERS: ClassVar[dict[str, str]] = {
        "build": "_execute_build",
        "send_villager": "_execute_send_villager",
        "send_all_idle": "_execute_send_all_idle",
        "queue_villager": "_execute_queue_villager",
        "reassign_villager": "_execute_reassign_villager",
    }

    async def _execute_tool_call(self, block: ToolUseBlock) -> tuple[dict, dict]:
        """Execute a single tool call and build the result payload."""
        tool_name = block.name
        handler_name = self._COMPOSITE_HANDLERS.get(tool_name)
        if handler_name:
            handler = cast(
                "Callable[[ToolUseBlock], Awaitable[tuple[dict, dict]]]",
                getattr(self, handler_name),
            )
            return await handler(block)

        block_input = cast("dict[str, object]", block.input)
        action_dict = {"type": tool_name, **block_input}
        result = await execute_action(action_dict)
        intent = block_input.get("intent", "")
        log.info(
            "tool_executed",
            action=tool_name,
            intent=intent if isinstance(intent, str) else "",
            success=result.success,
        )
        include_entities = tool_name == "press" and bool(block_input.get("rescan"))
        tool_result = self._make_tool_result(
            block, result.success, result.detail, include_entities=include_entities
        )
        return action_dict, tool_result

    @staticmethod
    def _apply_moving_cache_breakpoint(messages: list[dict]) -> None:
        """Mark the latest message so the conversation prefix hits the cache.

        Strips any prior message-level breakpoint, then caches the last content
        block of the most recent message. Only dict blocks are touched —
        assistant turns are appended as SDK content objects, which must not be
        mutated. One moving breakpoint keeps the total at three (two system
        blocks plus this), within the four-breakpoint API limit.
        """
        for msg in messages:
            content = msg["content"]
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        block.pop("cache_control", None)
        last_content = messages[-1]["content"]
        if isinstance(last_content, list) and last_content and isinstance(last_content[-1], dict):
            last_content[-1]["cache_control"] = {"type": "ephemeral"}

    async def _call_api(self, content: list[dict], age: str = "Dark Age") -> LLMResponse:
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
        system_prompt = self.get_system_prompt(age)
        output_config: OutputConfigParam = {"effort": config.executor_effort}

        for _ in range(config.max_tool_iterations):
            # Cache the conversation prefix so iterations 2..N read it back
            # instead of re-prefilling the whole growing message list each call.
            self._apply_moving_cache_breakpoint(messages)
            # The anthropic SDK types these args with strict TypedDicts
            # (MessageParam, ToolUnionParam, etc.). Our dicts are runtime-
            # equivalent; matching the TypedDicts everywhere upstream is more
            # churn than the safety buys, so we tell pyright to skip these
            # three argument-type checks specifically.
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                system=system_prompt,  # pyright: ignore[reportArgumentType]
                messages=messages,  # pyright: ignore[reportArgumentType]
                tools=_ACTION_TOOLS,  # pyright: ignore[reportArgumentType]
                output_config=output_config,
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
                parsed_content = cast("object", json.loads(tool_result["content"]))
                if isinstance(parsed_content, dict) and parsed_content.get("success"):
                    success_count += 1

            messages.append({"role": "user", "content": tool_results})

        # Validate standard actions; keep composite actions as-is (already executed).
        _COMPOSITE_NAMES = self._COMPOSITE_NAMES
        validated = validate_actions(
            [a for a in executed_actions if a.get("type") not in _COMPOSITE_NAMES]
        )
        composite = [a for a in executed_actions if a.get("type") in _COMPOSITE_NAMES]
        result = LLMResponse.model_construct(
            actions=validated + composite,
            observations=Observations(),
            reasoning=" ".join(reasoning_parts),
        )
        result._success_count = success_count
        return result

    async def _call_single_shot(self, content: list[dict], age: str = "Dark Age") -> LLMResult:
        """Fast path for routine turns: one structured-output call, no tool loop.

        Returns actions for the game loop to execute (actions_already_executed is
        False), keeping coordinate resolution and failure tracking on the existing
        executor path. Composite tools and mid-turn rescans are unavailable here;
        turns that need them route to the tool loop via _use_single_shot.
        """
        system_prompt = self.get_system_prompt(age)
        # parse() merges output_format into output_config.format, so we get
        # structured output and the effort knob in a single call.
        output_config: OutputConfigParam = {"effort": config.executor_effort}
        response = await self.client.messages.parse(  # pyright: ignore[reportAttributeAccessIssue]
            model=self.model,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            system=system_prompt,  # pyright: ignore[reportArgumentType]
            messages=[{"role": "user", "content": content}],  # pyright: ignore[reportArgumentType]
            output_format=LLMResponse,
            output_config=output_config,
        )
        usage = response.usage
        self._total_input_tokens += usage.input_tokens
        self._total_output_tokens += usage.output_tokens
        self._total_cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
        self._total_cache_write_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0
        parsed = cast("LLMResponse", response.parsed_output)
        return self._serialize_single_shot(parsed)

    def _cumulative_cost_usd(self) -> float:
        """Calculate cumulative API cost across all calls."""
        return (
            self._total_input_tokens * _PRICE_INPUT / 1_000_000
            + self._total_output_tokens * _PRICE_OUTPUT / 1_000_000
            + self._total_cache_read_tokens * _PRICE_CACHE_READ / 1_000_000
            + self._total_cache_write_tokens * _PRICE_CACHE_WRITE / 1_000_000
        )

    def _log_api_cost(self) -> None:
        """Emit running token and cost totals (shared by both executor paths)."""
        log.info(
            "api_cost",
            input_tokens=self._total_input_tokens,
            output_tokens=self._total_output_tokens,
            cache_read_tokens=self._total_cache_read_tokens,
            cache_write_tokens=self._total_cache_write_tokens,
            cumulative_cost_usd=round(self._cumulative_cost_usd(), 4),
        )

    # Context phrases that force the tool loop: combat needs mid-turn rescans and
    # composite tools, which the single-shot Action union can't express.
    _INTERACTIVE_SIGNALS: ClassVar[tuple[str, ...]] = (
        "under attack: true",
        "under_attack: true",
        "defend",
        "housed (cannot",
    )

    def _use_single_shot(self, context: str) -> bool:
        """Whether this turn can skip the tool loop for one structured call.

        Routine turns take the fast single-shot path; combat and housing
        emergencies stay on the tool loop (see _INTERACTIVE_SIGNALS) because
        they need mid-turn rescans and composite tools the single-shot Action
        union can't express.
        """
        lowered = context.lower()
        return not any(signal in lowered for signal in self._INTERACTIVE_SIGNALS)

    async def get_actions(
        self,
        context: str,
        width: int = 1920,
        height: int = 1080,
    ) -> LLMResult:
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
        age = self._extract_age(context)

        try:
            if self._use_single_shot(context):
                payload = await self._call_single_shot(content, age=age)
            else:
                result = await self._call_api(content, age=age)
                log.debug("claude_response", age=age, reasoning=result.reasoning[:200])
                payload = self._serialize_response(result)
            self._log_api_cost()
            return payload

        except anthropic.APIError as e:
            log.error("claude_api_error", error=str(e))
            return self._error_response(f"API error: {e}")
        except Exception as e:
            log.error("claude_error", error=str(e))
            return self._error_response(f"Error: {e}")

    def _error_response(self, message: str) -> LLMResult:
        """Return a safe error response with a wait action."""
        return LLMResult(
            reasoning=message,
            observations={},
            actions=[{"type": "wait", "ms": 1000, "intent": "Error recovery"}],
        )
