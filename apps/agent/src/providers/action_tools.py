"""Tool schemas for deliberate combat and catalog-guarded recovery.

Economic and production descriptions derive from the shared action catalog;
raw input tools remain for tactical interaction only.

Schemas are written in Anthropic's shape and converted for OpenAI-compatible
endpoints by `to_openai_tools` at the bottom of this module.

Strict per-tool input schemas are intentional — the previous structured-output
union approach allowed field confusion (e.g. a `click` action getting `key`
fields), and per-tool schemas eliminate that class of bug at the SDK boundary.
"""

from typing import cast

from ..policy.catalog import BUILDINGS, RESEARCH, UNITS

_BUILDING_OPTIONS = "; ".join(
    f"{menu}+{key}: {spec.description} ({spec.price('wood')} wood)"
    for (menu, key), spec in BUILDINGS.items()
)
_RESEARCH_OPTIONS = "; ".join(f"{name}: {spec.description}" for name, spec in RESEARCH.items())
_UNIT_OPTIONS = "; ".join(
    f"{name}: {spec.description}" for name, spec in UNITS.items() if name != "villager"
)
_ECON_BUILDING_OPTIONS = ", ".join(
    f"{key}={spec.subject}" for (menu, key), spec in BUILDINGS.items() if menu == "q"
)


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
        "description": "Tactical/navigation key only. Purchases must use named build, research, train_unit, or queue_villager tools.",
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
        "description": (
            "Build one catalogued structure. The executor selects a worker, refreshes after "
            "camera movement, checks feasibility, and places it. Bindings: "
            f"{_BUILDING_OPTIONS}."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "menu": {
                    "type": "string",
                    "enum": ["q", "w", "v"],
                    "description": "Build menu: q=economic, w=military, v=advanced",
                },
                "building_key": {
                    "type": "string",
                    "description": "Key within that menu — see the tool description",
                },
                "intent": {"type": "string", "description": "What you are building and why"},
            },
            "required": ["menu", "building_key", "intent"],
            "additionalProperties": False,
        },
    },
    {
        "name": "research",
        "description": (
            "Research one named catalogued technology. The executor checks current age, "
            "completed prerequisites, available resources, and pending purchases before "
            f"the spending key. Options: {_RESEARCH_OPTIONS}."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tech": {
                    "type": "string",
                    "enum": list(RESEARCH),
                    "description": "Technology to research",
                },
                "intent": {"type": "string", "description": "Why you are researching it"},
            },
            "required": ["tech", "intent"],
            "additionalProperties": False,
        },
    },
    {
        "name": "send_villager",
        "description": "Composite: select idle villager (press .) → right_click target. MUCH faster than press(.)+right_click() separately. target_class only (e.g. 'sheep', 'tree', 'berry_bush') — selecting the villager moves the camera, so coordinates you compute now would land on the wrong terrain.",
        "input_schema": {
            "type": "object",
            "properties": {
                "target_class": {
                    "type": "string",
                    "description": "Entity class to target (e.g. 'sheep', 'tree', 'berry_bush')",
                },
                "intent": {
                    "type": "string",
                    "description": "Where you are sending the villager and why",
                },
            },
            "required": ["target_class", "intent"],
            "additionalProperties": False,
        },
    },
    {
        "name": "send_all_idle",
        "description": "Composite: select ALL idle villagers (Shift-.) → right_click target. Dispatches every idle villager at once in a single action — use this instead of repeating send_villager when several villagers are idle. target_class only — the select moves the camera, so pre-computed coordinates would be stale.",
        "input_schema": {
            "type": "object",
            "properties": {
                "target_class": {
                    "type": "string",
                    "description": "Entity class to send all idle villagers to (e.g. 'tree', 'sheep')",
                },
                "intent": {
                    "type": "string",
                    "description": "Where you are sending the idle villagers and why",
                },
            },
            "required": ["target_class", "intent"],
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
    {
        "name": "reassign_villager",
        "description": (
            "Pull a gathering villager into a catalog-guarded economic build. The camera "
            "refreshes before worker selection and placement. Economic bindings: "
            f"{_ECON_BUILDING_OPTIONS}."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "from_job": {
                    "type": "string",
                    "description": "Which worker to pull: 'wood', 'gold', 'stone', or 'food'",
                },
                "building_key": {
                    "type": "string",
                    "description": f"Economic building key: {_ECON_BUILDING_OPTIONS}",
                },
                "intent": {
                    "type": "string",
                    "description": "Which worker you are pulling and what you are building",
                },
            },
            "required": ["from_job", "building_key", "intent"],
            "additionalProperties": False,
        },
    },
]

_ACTION_TOOLS.extend(
    [
        {
            "name": "train_unit",
            "description": (
                "Train one catalogued military unit. Its cost and population slot remain "
                f"committed until observed settlement. Options: {_UNIT_OPTIONS}."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "unit": {
                        "type": "string",
                        "enum": [name for name in UNITS if name != "villager"],
                    },
                    "intent": {"type": "string"},
                },
                "required": ["unit", "intent"],
                "additionalProperties": False,
            },
        },
        {
            "name": "assign_idle",
            "description": "Select one idle worker, refresh, then find the resource in the new view.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "resource": {
                        "type": "string",
                        "enum": ["food", "wood", "gold", "stone"],
                    },
                    "intent": {"type": "string"},
                },
                "required": ["resource", "intent"],
                "additionalProperties": False,
            },
        },
    ]
)

_RECOVERY_TOOLS = [
    tool
    for tool in _ACTION_TOOLS
    if tool["name"] in {"build", "research", "queue_villager", "train_unit", "assign_idle", "wait"}
]


# -- OpenAI strict-mode conversion -------------------------------------------
#
# OpenAI strict mode demands every `properties` key appear in `required`, with
# "optional" expressed as a nullable union. Numeric bounds ARE allowed here — it
# is Anthropic's constrained decoding that rejects them, which is why models.py
# enforces ranges via field_validator instead (F-40).


def _strictify(schema: dict[str, object]) -> dict[str, object]:
    """Return `schema` with every property required, optionals made nullable."""
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return dict(schema)

    required = schema.get("required")
    already_required = set(required) if isinstance(required, list) else set()

    converted: dict[str, object] = {}
    for name, raw in properties.items():
        if not isinstance(raw, dict):
            converted[name] = raw
            continue
        prop = _strictify(raw)
        if name not in already_required:
            prop["type"] = _nullable(prop.get("type"))
        converted[name] = prop

    return {
        **schema,
        "properties": converted,
        "required": list(properties.keys()),
        "additionalProperties": False,
    }


def _nullable(declared: object) -> object:
    """Widen a JSON Schema `type` to also admit null."""
    if isinstance(declared, str):
        return [declared, "null"]
    if isinstance(declared, list) and "null" not in declared:
        return [*declared, "null"]
    return declared


def to_openai_tools(tools: list[dict[str, object]]) -> list[dict[str, object]]:
    """Convert the Anthropic tool list to OpenAI strict function definitions."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": _strictify(cast("dict[str, object]", tool["input_schema"])),
                "strict": True,
            },
        }
        for tool in tools
    ]


__all__ = ["to_openai_tools"]
