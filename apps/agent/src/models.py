"""Pydantic models for action validation."""

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    Discriminator,
    Field,
    PrivateAttr,
    ValidationError,
    field_validator,
    model_validator,
)

# Screen-coordinate and wait ceilings. These are enforced by field_validators
# (below) rather than Field(ge=, le=) ON PURPOSE: a bounded integer in a
# pydantic Field emits minimum/maximum into the JSON schema, and the executor's
# single-shot path feeds that schema to Anthropic structured output as a
# constrained-decoding grammar. Each numeric range compiles to a large
# digit-by-digit automaton, and 22 such bounds across the Action union pushed
# the compiled grammar over Anthropic's size limit — every executor turn 400'd
# with "compiled grammar is too large" (run 12, F-40). A field_validator
# enforces the SAME range at parse time (dict validation and structured-output
# parsing both run it) while keeping the field an unbounded int in the schema,
# so the grammar stays small. The real on-screen bound is enforced downstream
# anyway (the executor rejects off-map clicks); these ceilings are the 8K-screen
# outer sanity limit.
_MAX_X = 7680
_MAX_Y = 4320
_MAX_WAIT_MS = 5000


def _in_range(value: int | None, hi: int, name: str) -> int | None:
    """Enforce 0 <= value <= hi for a coordinate/duration; pass None through.

    Kept schema-free (a validator, not Field bounds) so the constrained-decoding
    grammar stays small — see the module note above (F-40).
    """
    if value is not None and not (0 <= value <= hi):
        raise ValueError(f"{name} must be in [0, {hi}]")
    return value


class PointTargetAction(BaseModel):
    """Base for actions that target a point via coordinates, entity ID, or class.

    Can specify either:
    - x, y coordinates directly
    - target_id referencing a detected entity (resolved to coordinates at execution)
    - target_class to target the nearest entity of that class
    """

    x: int | None = Field(default=None, description="Screen x pixel, 0.._MAX_X")
    y: int | None = Field(default=None, description="Screen y pixel, 0.._MAX_Y")
    target_id: str | None = Field(
        default=None, description="Entity ID from detection, e.g. 'sheep_0'"
    )
    target_class: str | None = Field(
        default=None, description="Entity class to target nearest of, e.g. 'sheep'"
    )
    intent: str = ""

    @field_validator("x")
    @classmethod
    def _check_x(cls, v: int | None) -> int | None:
        return _in_range(v, _MAX_X, "x")

    @field_validator("y")
    @classmethod
    def _check_y(cls, v: int | None) -> int | None:
        return _in_range(v, _MAX_Y, "y")

    def _targeting_provided(self) -> bool:
        """Whether the action names a point (subclasses may add other modes)."""
        has_coords = self.x is not None and self.y is not None
        return has_coords or self.target_id is not None or self.target_class is not None

    @model_validator(mode="after")
    def check_coords_or_target(self) -> "PointTargetAction":
        """Ensure the action can be resolved to a point at execution time."""
        if not self._targeting_provided():
            raise ValueError("Must provide (x, y) coordinates, target_id, or target_class")
        return self


class ClickAction(PointTargetAction):
    """Left click action."""

    type: Literal["click"]
    building_key: str | None = Field(
        default=None,
        description="Econ build-menu key when this click places a building — carried "
        "through validation so the executor can verify the placement landed",
    )
    auto_placement: bool = Field(
        default=False,
        description="Resolve the placement to open ground AT CLICK TIME — coordinates "
        "computed before a camera move land on arbitrary terrain (run 8, F-33)",
    )

    def _targeting_provided(self) -> bool:
        return self.auto_placement or super()._targeting_provided()


class RightClickAction(PointTargetAction):
    """Right click action."""

    type: Literal["right_click"]


# Keys that open the game menu or pause the game — never what the agent means:
# Escape with nothing to cancel OPENS the menu (run 8, F-32), F10 IS the menu,
# F3 is pause. UI state is cleared by selecting the TC ('h') instead.
_GAME_PAUSING_KEYS: frozenset[str] = frozenset({"escape", "esc", "f10", "f3"})


class PressAction(BaseModel):
    """Keyboard press action."""

    type: Literal["press"]
    key: str = Field(min_length=1, max_length=20)
    modifiers: list[str] = Field(
        default_factory=list, description="Modifier keys, e.g. ['ctrl', 'shift']"
    )
    rescan: bool = Field(
        default=False, description="Take fresh screenshot+detection after this key press"
    )
    intent: str = ""

    @field_validator("key")
    @classmethod
    def validate_key(cls, v: str) -> str:
        """Validate key is a valid pyautogui key."""
        # Common valid keys for pyautogui
        valid_special_keys = {
            "enter",
            "return",
            "space",
            "tab",
            "escape",
            "esc",
            "backspace",
            "delete",
            "del",
            "up",
            "down",
            "left",
            "right",
            "home",
            "end",
            "pageup",
            "pagedown",
            "ctrl",
            "control",
            "alt",
            "shift",
            "win",
            "command",
            "f1",
            "f2",
            "f3",
            "f4",
            "f5",
            "f6",
            "f7",
            "f8",
            "f9",
            "f10",
            "f11",
            "f12",
            "insert",
            "pause",
            "capslock",
            "numlock",
            "scrolllock",
            "printscreen",
        }

        key_lower = v.lower()

        if key_lower in _GAME_PAUSING_KEYS:
            raise ValueError(
                f"key '{v}' opens the game menu / pauses the game — "
                "press 'h' (select TC) to clear UI state instead"
            )

        # Single character keys are always valid (letters, numbers, symbols)
        if len(v) == 1:
            return v

        # Check if it's a valid special key
        if key_lower in valid_special_keys:
            return key_lower

        # Allow function keys with numbers
        if key_lower.startswith("f") and key_lower[1:].isdigit():
            return key_lower

        raise ValueError(f"Invalid key: {v}")


class DragAction(BaseModel):
    """Mouse drag action."""

    type: Literal["drag"]
    start_x: int
    start_y: int
    end_x: int
    end_y: int
    intent: str = ""

    # Range-enforced via validators (not Field bounds) to keep the grammar small — F-40.
    @field_validator("start_x", "end_x")
    @classmethod
    def _check_x(cls, v: int) -> int:
        return _in_range(v, _MAX_X, "x")  # type: ignore[return-value]

    @field_validator("start_y", "end_y")
    @classmethod
    def _check_y(cls, v: int) -> int:
        return _in_range(v, _MAX_Y, "y")  # type: ignore[return-value]


class WaitAction(BaseModel):
    """Wait/delay action."""

    type: Literal["wait"]
    ms: int  # Max 5 second wait; range-enforced below (schema-free, F-40)
    intent: str = ""

    @field_validator("ms")
    @classmethod
    def _check_ms(cls, v: int) -> int:
        return _in_range(v, _MAX_WAIT_MS, "ms")  # type: ignore[return-value]


class ScrollAction(BaseModel):
    """Mouse scroll action (for zoom in/out)."""

    type: Literal["scroll"]
    clicks: int = Field(
        description="Positive = scroll up (zoom in), negative = scroll down (zoom out)"
    )
    x: int | None = Field(default=None, description="Screen x pixel, 0.._MAX_X")
    y: int | None = Field(default=None, description="Screen y pixel, 0.._MAX_Y")
    intent: str = ""

    @field_validator("x")
    @classmethod
    def _check_x(cls, v: int | None) -> int | None:
        return _in_range(v, _MAX_X, "x")

    @field_validator("y")
    @classmethod
    def _check_y(cls, v: int | None) -> int | None:
        return _in_range(v, _MAX_Y, "y")


class DetectAction(BaseModel):
    """Request full SAHI detection scan for accurate entity detection."""

    type: Literal["detect"]
    intent: str = ""


class BuildAction(BaseModel):
    """Build a structure via the economic build menu, auto-placed near the Town Center.

    Coordinate-free on purpose: x,y are omitted because the text-only model can't see
    open ground — the executor picks the placement (near the TC, with retry). Available
    on the fast single-shot path too, so routine turns can build without the tool loop.
    """

    type: Literal["build"]
    building_key: str = Field(
        min_length=1,
        max_length=2,
        description="Build hotkey: q=House, w=Mill, e=Mining Camp, r=Lumber Camp, a=Farm",
    )
    intent: str = ""


class QueueVillagerAction(BaseModel):
    """Queue one villager at the Town Center (select TC → q).

    A first-class action (not raw presses) so the executor's order ledger and
    villager-target gate see EVERY queue attempt — run 11 (F-38) over-ordered
    to 40 villagers because raw h+q presses were invisible to any brake.
    """

    type: Literal["queue_villager"]
    intent: str = ""


# Discriminated union ensures the JSON schema uses oneOf + discriminator on "type",
# preventing the model from confusing field names across action types
# (e.g., using DragAction's x1/y1 for ClickAction instead of x/y).
Action = Annotated[
    ClickAction
    | RightClickAction
    | PressAction
    | BuildAction
    | QueueVillagerAction
    | DragAction
    | WaitAction
    | ScrollAction
    | DetectAction,
    Discriminator("type"),
]


class Observations(BaseModel):
    """Game observations extracted by LLM."""

    resources: dict[str, int] = Field(default_factory=dict)
    population: str = ""
    age: str = ""
    idle_tc: bool = False
    under_attack: bool = False
    game_state: Literal["playing", "victory", "defeat", "menu"] = "playing"
    events: list[str] = Field(default_factory=list)


_ACTION_TYPE_MAP: dict[str, type[Action]] = {
    "click": ClickAction,
    "right_click": RightClickAction,
    "press": PressAction,
    "build": BuildAction,
    "queue_villager": QueueVillagerAction,
    "drag": DragAction,
    "wait": WaitAction,
    "scroll": ScrollAction,
    "detect": DetectAction,
}


def validate_action(action_dict: dict) -> Action | None:
    """Validate a single action dictionary.

    Returns validated action or None if invalid.
    """
    model_class = _ACTION_TYPE_MAP.get(action_dict.get("type", ""))
    if not model_class:
        return None
    try:
        return model_class.model_validate(action_dict)
    except ValidationError:
        return None


def validate_actions(actions: list[dict]) -> list[Action]:
    """Validate a list of action dicts, filtering out invalid ones."""
    return [a for raw in actions if (a := validate_action(raw)) is not None]


class LLMResponse(BaseModel):
    """Complete LLM response with validation.

    Field order matters: structured output generates fields sequentially.
    Actions first ensures they get generated before reasoning consumes
    the token budget.

    The field_validator on actions individually validates each action and
    silently drops invalid ones, so messages.parse() succeeds even when
    the LLM produces some malformed actions.
    """

    actions: list[Action] = Field(default_factory=list)
    observations: Observations = Field(default_factory=Observations)
    reasoning: str = ""

    # Side-channel: how many actions actually succeeded when the executor ran
    # the composite-tool loop locally. Set by ClaudeProvider._call_api after
    # tool execution; read by _serialize_response. PrivateAttr keeps it out
    # of model serialization and Pydantic field validation.
    _success_count: int = PrivateAttr(default=0)

    @field_validator("actions", mode="before")
    @classmethod
    def salvage_valid_actions(cls, v: list) -> list:
        """Validate actions individually, dropping invalid ones.

        Without this, a single bad action (e.g. right_click with no coords)
        fails the entire LLMResponse validation and messages.parse() raises,
        discarding all valid actions in the response.
        """
        if not isinstance(v, list):
            return v
        validated = []
        for item in v:
            if isinstance(item, dict):
                action = validate_action(item)
                if action is not None:
                    validated.append(action)
            else:
                validated.append(item)
        return validated
