"""Executor provider: turns game context into actions.

Holds the game logic — prompt assembly, the composite tool handlers, the
single-shot/tool-loop routing — and delegates every vendor detail to a
`ChatWire`, selected by `AOE2_LLM_WIRE`.
"""

import re
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast

import structlog
from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

from ..config import config
from ..entity_utils import ResourceKind
from ..executor import (
    ECON_MENU,
    STALE_COORDS_DETAIL,
    build_menu_steps,
    build_rejection,
    build_steps,
    execute_action,
    get_detected_entities,
)
from ..models import LLMResponse, Observations, validate_actions
from ..villager_roles import select_worker
from .action_tools import _ACTION_TOOLS
from .base import (
    AssistantTurn,
    ChatRequest,
    ChatWire,
    LLMResult,
    SystemBlock,
    TokenUsage,
    ToolCall,
    ToolOutcome,
    ToolResultsTurn,
    Turn,
    UserTurn,
    text_of_blocks,
)
from .pricing import cost_usd
from .wire_factory import make_wire

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


def _target_right_click(inp: dict, intent: object) -> dict[str, object] | None:
    """Right-click step for a send composite; None when only raw x/y were given.

    The executor resolves `target_class` against the post-rescan entity cache.
    Raw coordinates are refused: the preceding idle-select re-centers the
    camera, so a spot computed from the previous frame lands on arbitrary
    terrain (run 8, F-33 — villagers walked to random places).
    """
    if "target_class" not in inp:
        return None
    return {"type": "right_click", "target_class": inp["target_class"], "intent": intent}


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


class ExecutorProvider:
    """Turns game context into actions, over whichever wire is configured."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        use_dynamic_context: bool = True,
        wire: ChatWire | None = None,
    ) -> None:
        """
        Initialize the executor provider.

        Args:
            api_key: Credential for the default wire (defaults to config)
            model: Model to use (defaults to config)
            use_dynamic_context: Whether to use dynamic context injection from game database
            wire: Transport to talk to the model over. Defaults to whatever
                `AOE2_LLM_WIRE` selects. Pass one explicitly to run the same
                game logic against another vendor.
        """
        self.api_key = api_key or config.llm_api_key
        self.model = model or config.model
        self.wire: ChatWire = wire or make_wire(
            config.llm_wire,
            model=self.model,
            api_key=self.api_key,
            base_url=config.llm_base_url,
        )
        self._core_prompt: str | None = None
        self._age_prompts: dict[str, str] = {}
        # Titles of cross-game memories loaded into the cached system block.
        # Populated in _load_prompts() so game_loop can propagate them onto
        # AgentMemory.memories_loaded for per-turn attribution tracking.
        self.loaded_memory_titles: list[str] = []
        self.use_dynamic_context = use_dynamic_context and GAME_KNOWLEDGE_AVAILABLE
        self._game_db: GameKnowledge | None = None
        self._usage = TokenUsage()

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

    def get_system_prompt(self, age: str = "Dark Age") -> tuple[SystemBlock, ...]:
        """Return the system prompt as two cacheable blocks.

        Block 1 (core + hotkeys) is stable across all ages — always cached.
        Block 2 (age-specific) changes only on age transitions (3 times per game),
        so every turn within an age reads it back instead of re-prefilling.

        """
        self._load_prompts()

        # "Dark Age" → "dark", "Feudal Age" → "feudal"
        age_key = age.split()[0].lower() if age else "dark"
        age_content = self._age_prompts.get(age_key, self._age_prompts.get("dark", ""))

        blocks = [SystemBlock(text=self._core_prompt or "", cacheable=True)]
        if age_content:
            blocks.append(SystemBlock(text=age_content, cacheable=True))
        return tuple(blocks)

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

    def _entity_snapshot(self) -> tuple[dict[str, object], ...]:
        """Return truncated entity list for tool results."""
        return tuple(
            {"id": e.get("id", ""), "class": e.get("class", ""), "center": e.get("center", [])}
            for e in get_detected_entities()[: self.ENTITY_RESULT_LIMIT]
        )

    def _make_tool_result(
        self,
        block: ToolCall,
        success: bool,
        detail: str,
        *,
        include_entities: bool = False,
    ) -> ToolOutcome:
        """Package one tool's result; the wire frames it for its own API."""
        return ToolOutcome(
            tool_call_id=block.id,
            success=success,
            detail=detail,
            entities=self._entity_snapshot() if include_entities else (),
        )

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
        actions = ExecutorProvider._dump_actions(result)
        return LLMResult(
            reasoning=result.reasoning,
            observations=ExecutorProvider._observations_dict(result),
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
            observations=ExecutorProvider._observations_dict(result),
            actions=ExecutorProvider._dump_actions(result),
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
        block: ToolCall,
        name: str,
        steps: list[dict],
        *,
        include_entities: bool = True,
    ) -> tuple[dict, ToolOutcome]:
        """Run a composite's steps and package the (action_dict, tool_result) pair.

        The tail every composite handler shares: execute the steps, echo the tool
        input back as the recorded action dict, wrap success/detail for Claude.
        """
        success, detail = await self._run_steps(name, steps)
        action_dict = {"type": name, **block.arguments}
        tool_result = self._make_tool_result(
            block, success, detail, include_entities=include_entities
        )
        return action_dict, tool_result

    async def _execute_build(self, block: ToolCall) -> tuple[dict, ToolOutcome]:
        """Composite: press . (rescan) → q (econ menu) → building_key → place.

        Placement is resolved by the executor AT CLICK TIME, after the "."
        select has re-centered the camera — LLM coordinates are not accepted
        because they were computed from the pre-jump frame (run 8, F-33: the
        mill rose wherever the idle villager stood).
        """
        inp = block.arguments
        intent = str(inp.get("intent", "Build"))
        building_key = cast("str", inp["building_key"])
        menu = str(inp.get("menu") or ECON_MENU)
        rejection = build_rejection(building_key, intent, menu=menu)
        if rejection is not None:
            action_dict = {"type": "build", **inp}
            return action_dict, self._make_tool_result(block, False, rejection)
        steps = build_steps(building_key, intent, menu=menu)
        return await self._run_composite(block, "build", steps)

    def _refuse_raw_send(self, block: ToolCall, name: str) -> tuple[dict, ToolOutcome]:
        """Failure result for a send composite that gave only raw coordinates."""
        return {"type": name, **block.arguments}, self._make_tool_result(
            block, False, STALE_COORDS_DETAIL
        )

    async def _execute_send_villager(self, block: ToolCall) -> tuple[dict, ToolOutcome]:
        """Composite: press . (with rescan) → right_click target.

        The "." press moves the camera, so we rescan to get fresh entity
        positions before right-clicking. Raw x/y are refused (F-33).
        """
        inp = block.arguments
        intent = inp.get("intent", "Send villager")
        right_click = _target_right_click(inp, intent)
        if right_click is None:
            return self._refuse_raw_send(block, "send_villager")
        steps: list[dict] = [
            {
                "type": "press",
                "key": ".",
                "rescan": True,
                "intent": f"Select idle villager ({intent})",
            },
            right_click,
        ]
        return await self._run_composite(block, "send_villager", steps)

    async def _execute_send_all_idle(self, block: ToolCall) -> tuple[dict, ToolOutcome]:
        """Composite: Shift-. (select ALL idle) → right_click target.

        Dispatches every idle villager in one action. Mirrors send_villager but
        uses the select-all hotkey so no idle count is needed.
        """
        inp = block.arguments
        intent = inp.get("intent", "Send all idle villagers")
        right_click = _target_right_click(inp, intent)
        if right_click is None:
            return self._refuse_raw_send(block, "send_all_idle")
        steps: list[dict] = [
            {
                "type": "press",
                "key": ".",
                "modifiers": ["shift"],
                "rescan": True,
                "intent": f"Select ALL idle villagers ({intent})",
            },
            right_click,
        ]
        return await self._run_composite(block, "send_all_idle", steps)

    async def _execute_queue_villager(self, block: ToolCall) -> tuple[dict, ToolOutcome]:
        """Composite: one villager order through the executor's ledger gate.

        A single `queue_villager` action (not raw h+q presses) so the order
        target and food gate apply to LLM-initiated queues too — raw presses
        were invisible to the brake and over-ordered 40 villagers (F-38).
        """
        inp = block.arguments
        steps: list[dict] = [
            {"type": "queue_villager", "intent": str(inp.get("intent", "Queue villager"))}
        ]
        return await self._run_composite(block, "queue_villager", steps, include_entities=False)

    async def _execute_reassign_villager(self, block: ToolCall) -> tuple[dict, ToolOutcome]:
        """Composite: jump to a work site → pick a working villager → build.

        Two phases because the worker's screen position only exists AFTER the camera
        jump: (1) run the go-to-camp press with a rescan, then (2) read the fresh
        detections, choose a worker of `from_job` (stationary-first when tracker
        velocities are available), and select→build→place. Falls back to selecting
        the highest-confidence villager on screen if the job model finds none.
        """
        inp = block.arguments
        intent = str(inp.get("intent", "Reassign villager"))
        # LLM boundary: the tool schema restricts from_job to the resource kinds, so
        # narrow the raw string here; an off-schema value keeps today's behavior
        # (default hotkey jump → no candidates → highest-confidence villager).
        from_job = cast("ResourceKind", str(inp.get("from_job", "wood")))
        building_key = str(inp.get("building_key", "a"))  # 'a' = Farm in the econ menu
        action_dict = {"type": "reassign_villager", **inp}

        # Same gates as the plain build composite (prerequisite / cost / headroom):
        # reassigning a worker to a build that can't exist wastes the whole jump.
        # Note this reads the PRE-jump frame, so a camp key ("r"/"e") can be skipped
        # for a resource the phase-1 jump would have brought into view. Accepted:
        # the cost is one turn with a logged reason, and the common call is "a".
        rejection = build_rejection(building_key, intent)
        if rejection is not None:
            return action_dict, self._make_tool_result(block, False, rejection)

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

    async def _execute_tool_call(self, block: ToolCall) -> tuple[dict, ToolOutcome]:
        """Execute a single tool call and build the result payload."""
        tool_name = block.name
        handler_name = self._COMPOSITE_HANDLERS.get(tool_name)
        if handler_name:
            handler = cast(
                "Callable[[ToolCall], Awaitable[tuple[dict, ToolOutcome]]]",
                getattr(self, handler_name),
            )
            return await handler(block)

        block_input = cast("dict[str, object]", block.arguments)
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

    def _record_usage(self, usage: TokenUsage) -> None:
        """Fold one call's tokens into the running per-game total."""
        self._usage = self._usage + usage

    async def _call_api(self, content: list[dict], age: str = "Dark Age") -> LLMResponse:
        """Run the agentic tool loop.

        Each iteration: model calls one tool → we execute it → feed the result
        back → model calls the next tool. Loop until the model ends its turn or
        we hit max_tool_iterations. The model receives fresh entity positions
        after every camera-moving action.
        """
        turns: list[Turn] = [UserTurn(text=text_of_blocks(content))]
        executed_actions: list[dict] = []
        success_count = 0
        reasoning_parts: list[str] = []
        system = self.get_system_prompt(age)

        for _ in range(config.max_tool_iterations):
            reply = await self.wire.tool_turn(
                ChatRequest(
                    system=system,
                    turns=tuple(turns),
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    effort=config.executor_effort,
                ),
                _ACTION_TOOLS,
            )
            self._record_usage(reply.usage)
            if reply.text:
                reasoning_parts.append(reply.text)

            if not reply.wants_more_tools:
                break

            turns.append(AssistantTurn(text=reply.text, tool_calls=reply.tool_calls))
            outcomes: list[ToolOutcome] = []
            for call in reply.tool_calls:
                action_dict, outcome = await self._execute_tool_call(call)
                executed_actions.append(action_dict)
                outcomes.append(outcome)
                if outcome.success:
                    success_count += 1

            turns.append(ToolResultsTurn(outcomes=tuple(outcomes)))

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

    # Follow-up when a single-shot turn narrates a plan but emits zero actions —
    # 6-7 turns/game were wasted this way (run reviews, F-7/F-19). One bounded
    # retry converts most of them into real actions.
    _EMPTY_ACTIONS_NUDGE: ClassVar[str] = (
        "You returned zero actions. Respond again and EXECUTE your stated plan: "
        "emit at least one concrete action (press/build/click/right_click), or a "
        "single wait action if you genuinely intend to do nothing this turn."
    )

    async def _parse_single_shot(self, turns: tuple[Turn, ...], age: str) -> LLMResponse:
        """One structured-output call; accumulates token usage."""
        parsed, usage = await self.wire.parse_structured(
            ChatRequest(
                system=self.get_system_prompt(age),
                turns=turns,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                effort=config.executor_effort,
            ),
            LLMResponse,
        )
        self._record_usage(usage)
        return parsed

    async def _call_single_shot(self, content: list[dict], age: str = "Dark Age") -> LLMResult:
        """Fast path for routine turns: one structured-output call, no tool loop.

        Returns actions for the game loop to execute (actions_already_executed is
        False), keeping coordinate resolution and failure tracking on the existing
        executor path. Composite tools and mid-turn rescans are unavailable here;
        turns that need them route to the tool loop via _use_single_shot.
        A zero-action response gets ONE nudged retry before the game loop's
        hardcoded fallback takes over.
        """
        turns: tuple[Turn, ...] = (UserTurn(text=text_of_blocks(content)),)
        parsed = await self._parse_single_shot(turns, age)
        if not parsed.actions:
            log.warning("single_shot_empty_actions_retried", reasoning=parsed.reasoning[:120])
            turns = (
                *turns,
                AssistantTurn(text=parsed.reasoning or "(no actions)"),
                UserTurn(text=self._EMPTY_ACTIONS_NUDGE),
            )
            parsed = await self._parse_single_shot(turns, age)
        return self._serialize_single_shot(parsed)

    def _cumulative_cost_usd(self) -> float:
        """Cumulative API cost across all calls, priced for the served model."""
        return cost_usd(self.model, self._usage)

    def _log_api_cost(self) -> None:
        """Emit running token and cost totals (shared by both executor paths)."""
        log.info(
            "api_cost",
            model=self.wire.model,
            endpoint=self.wire.endpoint,
            input_tokens=self._usage.input_tokens,
            output_tokens=self._usage.output_tokens,
            cache_read_tokens=self._usage.cache_read_tokens,
            cache_write_tokens=self._usage.cache_write_tokens,
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
                payload = await self._single_shot_or_tool_loop(content, age)
            else:
                result = await self._call_api(content, age=age)
                log.debug("claude_response", age=age, reasoning=result.reasoning[:200])
                payload = self._serialize_response(result)
            self._log_api_cost()
            return payload

        except Exception as e:
            if self.wire.is_api_error(e):
                log.error("llm_api_error", error=str(e))
                return self._error_response(f"API error: {e}")
            log.error("llm_error", error=str(e))
            return self._error_response(f"Error: {e}")

    async def _single_shot_or_tool_loop(self, content: list[dict], age: str) -> LLMResult:
        """Single-shot fast path, falling back to the tool loop on a 400.

        The single-shot path sends LLMResponse's schema to Anthropic structured
        output as a constrained-decoding grammar; if that grammar exceeds the
        server's size limit, every call 400s with "compiled grammar is too
        large" and the turn is otherwise lost (run 12, F-40: 85/95 turns burned
        this way). The tool-loop path uses a smaller schema surface
        (_ACTION_TOOLS) that stayed under the limit that entire run, so on a
        BadRequestError we retry THIS turn there instead of returning a no-op
        wait. Non-400 API errors propagate to the caller's handler unchanged.
        """
        try:
            return await self._call_single_shot(content, age=age)
        except Exception as e:
            if not self.wire.is_schema_too_large(e):
                raise
            log.warning("single_shot_bad_request_fallback_tool_loop", error=str(e))
            result = await self._call_api(content, age=age)
            return self._serialize_response(result)

    def _error_response(self, message: str) -> LLMResult:
        """Return a safe error response with a wait action.

        error=True marks the turn as an executor no-op so the game loop can
        alarm on an outage and count it in llm_error_rate (T-533) — otherwise a
        run where every executor call 400s looks identical to a healthy one in
        results.tsv (run 12, F-40).
        """
        return LLMResult(
            reasoning=message,
            observations={},
            actions=[{"type": "wait", "ms": 1000, "intent": "Error recovery"}],
            error=True,
        )
