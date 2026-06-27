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


class PointTargetAction(BaseModel):
    """Base for actions that target a point via coordinates, entity ID, or class.

    Can specify either:
    - x, y coordinates directly
    - target_id referencing a detected entity (resolved to coordinates at execution)
    - target_class to target the nearest entity of that class
    """

    x: int | None = Field(default=None, ge=0, le=7680)
    y: int | None = Field(default=None, ge=0, le=4320)
    target_id: str | None = Field(
        default=None, description="Entity ID from detection, e.g. 'sheep_0'"
    )
    target_class: str | None = Field(
        default=None, description="Entity class to target nearest of, e.g. 'sheep'"
    )
    intent: str = ""

    @model_validator(mode="after")
    def check_coords_or_target(self) -> "PointTargetAction":
        """Ensure either coordinates, target_id, or target_class is provided."""
        has_coords = self.x is not None and self.y is not None
        has_target = self.target_id is not None
        has_class = self.target_class is not None
        if not has_coords and not has_target and not has_class:
            raise ValueError("Must provide (x, y) coordinates, target_id, or target_class")
        return self


class ClickAction(PointTargetAction):
    """Left click action."""

    type: Literal["click"]


class RightClickAction(PointTargetAction):
    """Right click action."""

    type: Literal["right_click"]


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
    start_x: int = Field(ge=0, le=7680)
    start_y: int = Field(ge=0, le=4320)
    end_x: int = Field(ge=0, le=7680)
    end_y: int = Field(ge=0, le=4320)
    intent: str = ""


class WaitAction(BaseModel):
    """Wait/delay action."""

    type: Literal["wait"]
    ms: int = Field(ge=0, le=5000)  # Max 5 second wait
    intent: str = ""


class ScrollAction(BaseModel):
    """Mouse scroll action (for zoom in/out)."""

    type: Literal["scroll"]
    clicks: int = Field(
        description="Positive = scroll up (zoom in), negative = scroll down (zoom out)"
    )
    x: int | None = Field(default=None, ge=0, le=7680)
    y: int | None = Field(default=None, ge=0, le=4320)
    intent: str = ""


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


# Discriminated union ensures the JSON schema uses oneOf + discriminator on "type",
# preventing the model from confusing field names across action types
# (e.g., using DragAction's x1/y1 for ClickAction instead of x/y).
Action = Annotated[
    ClickAction
    | RightClickAction
    | PressAction
    | BuildAction
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
