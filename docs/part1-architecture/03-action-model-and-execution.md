# Chapter 3: Action Model and Execution

The agent's output is a list of actions that must be validated, resolved to screen coordinates, and executed as real mouse/keyboard inputs. Pydantic models enforce structural correctness, and a coordinate resolution system bridges YOLO detection with physical action execution.

## 3.1 Action Types

The agent has **seven base action types** (Pydantic-validated in `apps/agent/src/models.py`) and **three composite tools** (defined in `apps/agent/src/providers/claude.py`) that bundle multi-step sequences to eliminate API roundtrips.

### Base Actions

All base action types are defined as Pydantic models in `apps/agent/src/models.py`.

### PointTargetAction (base class)

`ClickAction` and `RightClickAction` share a common base that handles all three targeting modes:

```python
class PointTargetAction(BaseModel):
    x: Optional[int] = Field(default=None, ge=0, le=7680)
    y: Optional[int] = Field(default=None, ge=0, le=4320)
    target_id: Optional[str] = Field(default=None, description="Entity ID from detection, e.g. 'sheep_0'")
    target_class: Optional[str] = Field(default=None, description="Entity class to target nearest of, e.g. 'sheep'")
    intent: str = ""
```

A `@model_validator` enforces that at least one of `(x, y)`, `target_id`, or `target_class` is provided.

### ClickAction

Left click at a position. Inherits targeting from `PointTargetAction`:

```python
class ClickAction(PointTargetAction):
    type: Literal["click"]
```

### RightClickAction

Identical to ClickAction but with `type: Literal["right_click"]`. Used for move commands, gather orders, and attack-move.

### PressAction

Keyboard key press. Includes a whitelist validator for valid keys:

```python
class PressAction(BaseModel):
    type: Literal["press"]
    key: str = Field(min_length=1, max_length=20)
    modifiers: list[str] = Field(default_factory=list)
    rescan: bool = Field(default=False, description="Take fresh screenshot+detection after this key press")
```

Single characters pass through directly. Multi-character strings are validated against a set of ~30 special keys (`enter`, `escape`, `f1`-`f12`, `space`, arrow keys, modifiers). Invalid keys raise a `ValueError`.

### DragAction

Mouse drag with start and end coordinates. Used for box-selecting units:

```python
class DragAction(BaseModel):
    type: Literal["drag"]
    x1: int = Field(ge=0, le=7680)
    y1: int = Field(ge=0, le=4320)
    x2: int = Field(ge=0, le=7680)
    y2: int = Field(ge=0, le=4320)
```

### WaitAction

Async delay between dependent actions. Capped at 5 seconds:

```python
class WaitAction(BaseModel):
    type: Literal["wait"]
    ms: int = Field(ge=0, le=5000)
```

### ScrollAction

Mouse scroll for zoom in/out. Optional position to scroll at:

```python
class ScrollAction(BaseModel):
    type: Literal["scroll"]
    clicks: int  # Positive = scroll up (zoom in), negative = scroll down (zoom out)
    x: Optional[int] = Field(default=None, ge=0, le=7680)
    y: Optional[int] = Field(default=None, ge=0, le=4320)
```

### DetectAction

Requests a full SAHI detection scan for accurate entity detection. No parameters required:

```python
class DetectAction(BaseModel):
    type: Literal["detect"]
    intent: str = ""
```

### Union Type

```python
Action = ClickAction | RightClickAction | PressAction | DragAction | WaitAction | ScrollAction | DetectAction
```

### LLMResponse

The complete response structure validated by Pydantic:

```python
class LLMResponse(BaseModel):
    actions: list[Action] = Field(default_factory=list)
    observations: Observations = Field(default_factory=Observations)
    reasoning: str = ""
```

Field order matters: `actions` first ensures structured output generates them before reasoning consumes the token budget. `Observations` tracks resources, population, age, idle_tc, under_attack, game_state, and events.

### Composite Tools

Composite tools execute multi-step hotkey sequences locally without intermediate API roundtrips. They are defined as Claude tool_use tools in `_ACTION_TOOLS` and handled by dedicated methods in `ClaudeProvider`. Each composite calls `_run_steps()` which executes sub-actions sequentially via `execute_action()`, stopping on the first failure.

**`build(building_key, x, y)`** — Select idle villager → open economic build menu → press building_key → click placement. Building keys: q=House, w=Mill, e=Mining Camp, r=Lumber Camp, a=Farm. Saves 3 API roundtrips (~9s) per building.

**`send_villager(target_class or x, y)`** — Select idle villager → right_click target. Accepts `target_class` (e.g. "sheep", "tree") or raw coordinates. Saves 1 roundtrip (~3s).

**`queue_villager()`** — Go to TC (press h) → queue villager (press q). Saves 1 roundtrip (~3s).

Composite actions bypass Pydantic validation (they are already executed by the time `_call_api` returns). The `_COMPOSITE_NAMES` set ensures they pass through `validate_actions()` unchanged.

Shared helpers eliminate repetition across handlers:
- `_run_steps(composite_name, steps)` — executes steps, logs each, returns `(success, detail)`
- `_entity_snapshot()` — returns truncated entity list (capped at `ENTITY_RESULT_LIMIT = 20`)
- `_make_tool_result(block, success, detail, include_entities)` — builds the tool_result dict for Claude

## 3.2 Triple Targeting: Coordinates, target_id, target_class

The LLM can specify click/right-click positions in three ways:

**Direct coordinates** -- the LLM estimates pixel positions from the screenshot:
```json
{"type": "right_click", "x": 920, "y": 460, "intent": "Gather from sheep"}
```

**Entity ID reference** -- the LLM uses a detection ID from the entity list:
```json
{"type": "right_click", "target_id": "sheep_0", "intent": "Gather from sheep"}
```

**Entity class reference** -- the LLM targets the nearest entity of a given class:
```json
{"type": "right_click", "target_class": "sheep", "intent": "Gather from nearest sheep"}
```

> **Key Insight**: The `target_id` and `target_class` mechanisms bridge vision detection and action execution. The LLM says `"target_id": "sheep_0"` or `"target_class": "sheep"` and the executor resolves it to exact pixel coordinates from the detection cache. This avoids the LLM needing to estimate precise pixel positions for small moving entities -- a task where even advanced vision models are unreliable.

## 3.3 Coordinate Resolution

When the game loop runs detection, entities are cached in the executor module via `set_detected_entities()`:

```python
_detected_entities: list[dict] = []

def set_detected_entities(entities: list) -> None:
    global _detected_entities
    _detected_entities = [
        e.to_dict() if hasattr(e, 'to_dict') else e
        for e in entities
    ]
```

The unified resolver `_resolve_coords()` tries three strategies in order:

1. **target_id** — linear search for matching entity ID, return center coordinates
2. **target_class** — linear search for first entity of that class, return center coordinates
3. **(x, y)** — use raw coordinates directly

```python
def _resolve_coords(action_dict: dict) -> tuple[str, tuple[int, int] | None]:
    """Returns (error_detail, coords). error_detail is non-empty on failure."""
    target_id = action_dict.get("target_id")
    if target_id:
        coords = _resolve_target_id(str(target_id))
        if coords is None:
            return (f"target_id '{target_id}' not found", None)
        return ("", coords)

    target_class = action_dict.get("target_class")
    if target_class:
        coords = _resolve_target_class(str(target_class))
        if coords is None:
            return (f"target_class '{target_class}' not found", None)
        return ("", coords)

    x, y = action_dict.get("x"), action_dict.get("y")
    if x is not None and y is not None:
        return ("", (int(x), int(y)))

    return ("no coordinates, target_id, or target_class provided", None)
```

Entity IDs follow the pattern `{class_name}_{counter}` (e.g., `sheep_0`, `villager_1`, `town_center_0`). IDs persist across detection frames via IoU-based matching — if an entity overlaps >40% with a same-class entity from the previous frame, it keeps the same ID. New entities get a globally unique counter that never resets. This means `sheep_0` remains `sheep_0` across turns as long as it's visible, giving the LLM a stable reference.

If resolution fails (entity not found), the action returns `ActionResult(success=False, detail=...)`.

## 3.4 Coordinate Translation

Screenshots capture the game window at its screen position. The LLM sees coordinates relative to the screenshot (0,0 = top-left of game window). But pyautogui operates in screen-absolute coordinates.

The executor re-fetches the window position **before each individual action** via `get_game_window_rect()`:

```python
global _window_offset
rect = get_game_window_rect()
if rect:
    _window_offset = (rect[0], rect[1])
```

Then each action applies the offset via `_translate()`:

```python
def _translate(x: int, y: int) -> tuple[int, int]:
    return (x + _window_offset[0], y + _window_offset[1])
```

This handles cases where the game window moves during a batch (e.g., OS repositioning).

If the window rect is unavailable, offset defaults to `(0, 0)`, which works for fullscreen games.

## 3.5 Execution Pipeline

The executor uses a dispatch pattern. Each action type has a dedicated async handler, registered in `_ACTION_HANDLERS`:

```python
_ACTION_HANDLERS: dict[str, Callable] = {
    "click": _handle_click,
    "right_click": _handle_right_click,
    "press": _handle_press,
    "drag": _handle_drag,
    "scroll": _handle_scroll,
    "detect": _handle_detect,
    "wait": _handle_wait,
}
```

`execute_actions()` orchestrates the batch:

1. **Ensure focus** — activates game window, retries once if it fails
2. **Execute sequentially** — iterates through actions, calling `execute_action()` for each
3. **Dispatch** — looks up handler in `_ACTION_HANDLERS`, returns `ActionResult(False, ...)` for unknown types
4. **Per-action window offset** — each action re-fetches the window position before translating coordinates
5. **Returns results** — list of `ActionResult(success, detail)` per action

Each handler dispatches to pyautogui:

| Action | pyautogui Call | Notes |
|--------|---------------|-------|
| `click` | `pyautogui.click(x, y)` | With building placement retry logic |
| `right_click` | `pyautogui.rightClick(x, y)` | After coordinate translation |
| `press` | `pyautogui.press(key)` or `pyautogui.hotkey(*modifiers, key)` | Supports modifiers; optional rescan after |
| `drag` | `pyautogui.moveTo()` + `pyautogui.drag()` | 200ms drag duration |
| `scroll` | `pyautogui.scroll(clicks)` | Optional x, y position |
| `detect` | Calls `_rescan_full_fn()` | Full SAHI detection scan |
| `wait` | `asyncio.sleep(ms / 1000)` | Async, does not block event loop |

### pyautogui Configuration

```python
pyautogui.FAILSAFE = False   # Disable corner-abort safety
pyautogui.PAUSE = 0.02       # 20ms between pyautogui calls (default is 100ms)
```

`FAILSAFE = False` is necessary because the game is fullscreen -- the mouse frequently visits screen corners during gameplay. The default 100ms pause is reduced to 20ms for snappier action sequences.

## 3.6 Action Validation Utilities

Two helper functions for ad-hoc validation in `apps/agent/src/models.py`:

**`validate_action(action_dict)`** — validates a single action dict against a type map. Returns a Pydantic model or `None`.

**`validate_actions(actions)`** — batch validation, filters out invalid actions silently. Returns only the valid ones.

The executor uses `validate_action()` for any action that arrives as a raw dict rather than a pre-validated Pydantic model.

## 3.7 Coordinate Bounds

All coordinate fields enforce bounds: `ge=0, le=7680` for x, `ge=0, le=4320` for y. This supports up to 8K resolution (7680x4320) and catches obviously invalid coordinates from LLM hallucination. At typical 1920x1080 resolution, coordinates outside the screen are still accepted by the model -- the pyautogui call may click outside the game window but won't crash.

---

## Summary

- 7 base action types with Pydantic validation: click, right_click, press, drag, wait, scroll, detect
- 3 composite tools: build, send_villager, queue_villager — bundle multi-step sequences to eliminate API roundtrips
- `PointTargetAction` base class for shared triple-targeting logic (coordinates, target_id, target_class)
- Unified `_resolve_coords()` resolver tries target_id → target_class → (x, y)
- `_ACTION_HANDLERS` dispatch pattern maps base action types to async handler functions
- `_COMPOSITE_HANDLERS` dict maps composite tools to dedicated handler methods
- Shared helpers (`_run_steps`, `_entity_snapshot`, `_make_tool_result`) eliminate repetition
- Coordinate translation from screenshot-relative to screen-absolute
- Sequential execution with configurable inter-action delay

## Related Topics

- [Chapter 2: Game Loop Pipeline](./02-game-loop-pipeline.md) -- where actions are requested and executed
- [Chapter 5: Prompt Engineering](../part2-llm-integration/05-prompt-engineering.md) -- how the LLM learns the action format
- [Chapter 7: Detector Architecture](../part3-entity-detection/07-detector-architecture.md) -- how entity IDs are generated
