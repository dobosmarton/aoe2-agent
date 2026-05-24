"""Anthropic tool-schema definitions for the executor's tool-use loop.

These are pure data: 8 standard one-step actions (click, right_click, press,
drag, wait, scroll, detect, plus the placement variants) and 3 composite
sequences (build, send_villager, queue_villager). The composite tools collapse
common multi-step UI flows into a single tool call so the model doesn't pay
per-step API roundtrip latency for predictable sequences.

Strict per-tool input schemas are intentional — the previous structured-output
union approach allowed field confusion (e.g. a `click` action getting `key`
fields), and per-tool schemas eliminate that class of bug at the SDK boundary.
"""


def _click_schema(description: str) -> dict:
    """Shared input schema for click and right_click tools."""
    return {
        "type": "object",
        "properties": {
            "x": {"type": "integer", "description": "X coordinate on game screen"},
            "y": {"type": "integer", "description": "Y coordinate on game screen"},
            "target_class": {
                "type": "string",
                "description": "Entity class to target nearest of, e.g. 'sheep'",
            },
            "intent": {"type": "string", "description": description},
        },
        "required": ["x", "y", "intent"],
        "additionalProperties": False,
    }


# Tool definitions for each action type — strict per-tool schemas.
# Each tool has its own enforced schema, preventing field confusion
# that occurred with structured output union types.
_ACTION_TOOLS: list[dict] = [
    {
        "name": "click",
        "description": "Left click at screen coordinates. Use for building placement and UI interaction.",
        "input_schema": _click_schema("What this click does"),
    },
    {
        "name": "right_click",
        "description": "Right click at screen coordinates. Use for resource gathering, setting gather points, and unit commands.",
        "input_schema": _click_schema("What this right click does"),
    },
    {
        "name": "press",
        "description": "Press a keyboard key. Use for hotkeys, queuing units, opening build menus.",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Key to press, e.g. 'h', 'q', '.', ','"},
                "rescan": {
                    "type": "boolean",
                    "description": "Take fresh screenshot+detection after this key press",
                },
                "modifiers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Modifier keys e.g. ['ctrl']",
                },
                "intent": {"type": "string", "description": "What this key press does"},
            },
            "required": ["key", "intent"],
            "additionalProperties": False,
        },
    },
    {
        "name": "drag",
        "description": "Drag mouse from start to end position.",
        "input_schema": {
            "type": "object",
            "properties": {
                "start_x": {"type": "integer", "description": "Start X coordinate"},
                "start_y": {"type": "integer", "description": "Start Y coordinate"},
                "end_x": {"type": "integer", "description": "End X coordinate"},
                "end_y": {"type": "integer", "description": "End Y coordinate"},
                "intent": {"type": "string"},
            },
            "required": ["start_x", "start_y", "end_x", "end_y", "intent"],
            "additionalProperties": False,
        },
    },
    {
        "name": "wait",
        "description": "Wait for a duration.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ms": {"type": "integer", "description": "Milliseconds to wait (0-5000)"},
                "intent": {"type": "string"},
            },
            "required": ["ms", "intent"],
            "additionalProperties": False,
        },
    },
    {
        "name": "scroll",
        "description": "Scroll mouse wheel for zoom in/out.",
        "input_schema": {
            "type": "object",
            "properties": {
                "clicks": {
                    "type": "integer",
                    "description": "Positive = zoom in, negative = zoom out",
                },
                "intent": {"type": "string"},
            },
            "required": ["clicks", "intent"],
            "additionalProperties": False,
        },
    },
    {
        "name": "detect",
        "description": "Request full SAHI detection scan. SLOW (~5-10s) — only use when target_class keeps failing.",
        "input_schema": {
            "type": "object",
            "properties": {
                "intent": {"type": "string"},
            },
            "required": ["intent"],
            "additionalProperties": False,
        },
    },
    # --- Composite tools (multi-step sequences, no intermediate API roundtrips) ---
    {
        "name": "build",
        "description": "Composite: select idle villager → open economic build menu → press building_key → place at (x,y). MUCH faster than individual steps. Building keys: q=House, w=Mill, e=Mining Camp, r=Lumber Camp, a=Farm. ALWAYS use this instead of press(.)+press(q)+press(key)+click() separately.",
        "input_schema": {
            "type": "object",
            "properties": {
                "building_key": {
                    "type": "string",
                    "description": "Hotkey for the building: q=House, w=Mill, e=Mining Camp, r=Lumber Camp, a=Farm",
                },
                "x": {"type": "integer", "description": "X coordinate for placement"},
                "y": {"type": "integer", "description": "Y coordinate for placement"},
                "intent": {"type": "string", "description": "What you are building and why"},
            },
            "required": ["building_key", "x", "y", "intent"],
            "additionalProperties": False,
        },
    },
    {
        "name": "send_villager",
        "description": "Composite: select idle villager (press .) → right_click target. MUCH faster than press(.)+right_click() separately. Use target_class for resources (e.g. 'sheep', 'tree', 'berry_bush') or x,y for specific locations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "x": {"type": "integer", "description": "X coordinate to right-click"},
                "y": {"type": "integer", "description": "Y coordinate to right-click"},
                "target_class": {
                    "type": "string",
                    "description": "Entity class to target (e.g. 'sheep', 'tree', 'berry_bush')",
                },
                "intent": {
                    "type": "string",
                    "description": "Where you are sending the villager and why",
                },
            },
            "required": ["intent"],
            "additionalProperties": False,
        },
    },
    {
        "name": "queue_villager",
        "description": "Composite: go to TC (press h) → queue villager (press q). MUCH faster than individual steps. Use this instead of doing press(h)+press(q) separately.",
        "input_schema": {
            "type": "object",
            "properties": {
                "intent": {"type": "string", "description": "Why queuing this villager"},
            },
            "required": ["intent"],
            "additionalProperties": False,
        },
    },
]
