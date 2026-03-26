"""Canonical tool definitions for AoE2 Agent actions.

Single source of truth — format converters produce Anthropic or OpenAI
tool schemas from the same base definitions.
"""


def _click_properties(intent_desc: str) -> dict:
    """Shared properties for click and right_click tools."""
    return {
        "x": {"type": "integer", "description": "X coordinate on game screen"},
        "y": {"type": "integer", "description": "Y coordinate on game screen"},
        "target_class": {"type": "string", "description": "Entity class to target nearest of, e.g. 'sheep'"},
        "intent": {"type": "string", "description": intent_desc},
    }


# Each entry: (name, description, properties, required_fields)
TOOL_SCHEMAS: list[dict] = [
    {
        "name": "click",
        "description": "Left click at screen coordinates. Use for building placement and UI interaction.",
        "properties": _click_properties("What this click does"),
        "required": ["x", "y", "intent"],
    },
    {
        "name": "right_click",
        "description": "Right click at screen coordinates. Use for resource gathering, setting gather points, and unit commands.",
        "properties": _click_properties("What this right click does"),
        "required": ["x", "y", "intent"],
    },
    {
        "name": "press",
        "description": "Press a keyboard key. Use for hotkeys, queuing units, opening build menus.",
        "properties": {
            "key": {"type": "string", "description": "Key to press, e.g. 'h', 'q', '.', ','"},
            "rescan": {"type": "boolean", "description": "Take fresh screenshot+detection after this key press"},
            "modifiers": {"type": "array", "items": {"type": "string"}, "description": "Modifier keys e.g. ['ctrl']"},
            "intent": {"type": "string", "description": "What this key press does"},
        },
        "required": ["key", "intent"],
    },
    {
        "name": "drag",
        "description": "Drag mouse from start to end position.",
        "properties": {
            "start_x": {"type": "integer", "description": "Start X coordinate"},
            "start_y": {"type": "integer", "description": "Start Y coordinate"},
            "end_x": {"type": "integer", "description": "End X coordinate"},
            "end_y": {"type": "integer", "description": "End Y coordinate"},
            "intent": {"type": "string"},
        },
        "required": ["start_x", "start_y", "end_x", "end_y", "intent"],
    },
    {
        "name": "wait",
        "description": "Wait for a duration.",
        "properties": {
            "ms": {"type": "integer", "description": "Milliseconds to wait (0-5000)"},
            "intent": {"type": "string"},
        },
        "required": ["ms", "intent"],
    },
    {
        "name": "scroll",
        "description": "Scroll mouse wheel for zoom in/out.",
        "properties": {
            "clicks": {"type": "integer", "description": "Positive = zoom in, negative = zoom out"},
            "intent": {"type": "string"},
        },
        "required": ["clicks", "intent"],
    },
    {
        "name": "detect",
        "description": "Request full SAHI detection scan. SLOW (~5-10s) — only use when target_class keeps failing.",
        "properties": {
            "intent": {"type": "string"},
        },
        "required": ["intent"],
    },
]


def to_anthropic_tools(schemas: list[dict] | None = None) -> list[dict]:
    """Convert tool schemas to Anthropic tool format."""
    schemas = schemas or TOOL_SCHEMAS
    return [
        {
            "name": s["name"],
            "description": s["description"],
            "input_schema": {
                "type": "object",
                "properties": s["properties"],
                "required": s["required"],
                "additionalProperties": False,
            },
        }
        for s in schemas
    ]


def to_openai_tools(schemas: list[dict] | None = None) -> list[dict]:
    """Convert tool schemas to OpenAI function-calling format."""
    schemas = schemas or TOOL_SCHEMAS
    return [
        {
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s["description"],
                "parameters": {
                    "type": "object",
                    "properties": s["properties"],
                    "required": s["required"],
                },
            },
        }
        for s in schemas
    ]
