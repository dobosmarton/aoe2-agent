"""Pydantic models for action validation."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator


class ClickAction(BaseModel):
    """Left click action."""

    type: Literal["click"]
    x: int = Field(ge=0, le=7680)  # Support up to 8K resolution
    y: int = Field(ge=0, le=4320)
    intent: str = ""


class RightClickAction(BaseModel):
    """Right click action."""

    type: Literal["right_click"]
    x: int = Field(ge=0, le=7680)  # Support up to 8K resolution
    y: int = Field(ge=0, le=4320)
    intent: str = ""


class PressAction(BaseModel):
    """Keyboard press action."""

    type: Literal["press"]
    key: str = Field(min_length=1, max_length=20)
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
    x1: int = Field(ge=0, le=7680)  # Support up to 8K resolution
    y1: int = Field(ge=0, le=4320)
    x2: int = Field(ge=0, le=7680)
    y2: int = Field(ge=0, le=4320)
    intent: str = ""


class WaitAction(BaseModel):
    """Wait/delay action."""

    type: Literal["wait"]
    ms: int = Field(ge=0, le=5000)  # Max 5 second wait
    intent: str = ""


# Union type for all actions
Action = ClickAction | RightClickAction | PressAction | DragAction | WaitAction


class Observations(BaseModel):
    """Game observations extracted by LLM."""

    resources: dict[str, int] = Field(default_factory=dict)
    population: str = ""
    age: str = ""
    idle_tc: bool = False
    under_attack: bool = False
    events: list[str] = Field(default_factory=list)


class LLMResponse(BaseModel):
    """Complete LLM response with validation."""

    reasoning: str
    observations: Observations = Field(default_factory=Observations)
    actions: list[Action] = Field(default_factory=list)


def validate_action(action_dict: dict) -> Action | None:
    """
    Validate a single action dictionary.

    Returns validated action or None if invalid.
    """
    action_type = action_dict.get("type")

    type_map = {
        "click": ClickAction,
        "right_click": RightClickAction,
        "press": PressAction,
        "drag": DragAction,
        "wait": WaitAction,
    }

    model_class = type_map.get(action_type)
    if not model_class:
        return None

    try:
        return model_class.model_validate(action_dict)
    except Exception:
        return None


def validate_actions(actions: list[dict]) -> list[Action]:
    """
    Validate a list of action dictionaries.

    Returns list of valid actions, filtering out invalid ones.
    """
    validated = []
    for action_dict in actions:
        action = validate_action(action_dict)
        if action:
            validated.append(action)
    return validated
