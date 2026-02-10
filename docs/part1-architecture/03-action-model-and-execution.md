# Chapter 3: Action Model and Execution

The agent's output is a list of actions that must be validated, resolved to screen coordinates, and executed as real mouse/keyboard inputs. Pydantic models enforce structural correctness, and a target_id resolution system bridges YOLO detection with physical action execution.

## 3.1 The Five Action Types

All action types are defined as Pydantic models in `src/models.py:8-149`:

### ClickAction (`models.py:8-29`)

Left click at a position. Supports two targeting modes:

```python
class ClickAction(BaseModel):
    type: Literal["click"]
    x: Optional[int] = Field(default=None, ge=0, le=7680)
    y: Optional[int] = Field(default=None, ge=0, le=4320)
    target_id: Optional[str] = Field(default=None)
    intent: str = ""
```

A `@model_validator` enforces that either `(x, y)` or `target_id` is provided -- not neither.

### RightClickAction (`models.py:32-53`)

Identical structure to ClickAction but with `type: Literal["right_click"]`. Used for move commands, gather orders, and attack-move.

### PressAction (`models.py:56-126`)

Keyboard key press. Includes a whitelist validator for valid keys:

```python
class PressAction(BaseModel):
    type: Literal["press"]
    key: str = Field(min_length=1, max_length=20)
```

Single characters pass through directly. Multi-character strings are validated against a set of ~30 special keys (`enter`, `escape`, `f1`-`f12`, `space`, arrow keys, modifiers). Invalid keys raise a `ValueError`.

### DragAction (`models.py:129-137`)

Mouse drag with start and end coordinates. Used for box-selecting units:

```python
class DragAction(BaseModel):
    type: Literal["drag"]
    x1: int = Field(ge=0, le=7680)
    y1: int = Field(ge=0, le=4320)
    x2: int = Field(ge=0, le=7680)
    y2: int = Field(ge=0, le=4320)
```

### WaitAction (`models.py:140-145`)

Async delay between dependent actions. Capped at 5 seconds:

```python
class WaitAction(BaseModel):
    type: Literal["wait"]
    ms: int = Field(ge=0, le=5000)
```

### Union Type (`models.py:149`)

```python
Action = ClickAction | RightClickAction | PressAction | DragAction | WaitAction
```

### LLMResponse (`models.py:163-168`)

The complete response structure validated by Pydantic:

```python
class LLMResponse(BaseModel):
    reasoning: str
    observations: Observations = Field(default_factory=Observations)
    actions: list[Action] = Field(default_factory=list)
```

Where `Observations` tracks resources, population, age, idle_tc, under_attack, and events.

## 3.2 Dual Targeting: Coordinates vs target_id

The LLM can specify click positions in two ways:

**Direct coordinates** -- the LLM estimates pixel positions from the screenshot:
```json
{"type": "right_click", "x": 920, "y": 460, "intent": "Gather from sheep"}
```

**Entity reference** -- the LLM uses a detection ID from the entity list:
```json
{"type": "right_click", "target_id": "sheep_0", "intent": "Gather from sheep"}
```

> **Key Insight**: The `target_id` mechanism bridges vision detection and action execution. The LLM says `"target_id": "sheep_0"` and the executor resolves it to exact pixel coordinates from the detection cache. This avoids the LLM needing to estimate precise pixel positions for small moving entities -- a task where even advanced vision models are unreliable.

## 3.3 Target ID Resolution

When the game loop runs detection, entities are cached in the executor module (`src/executor.py:22-40`):

```python
_detected_entities: list[dict] = []

def set_detected_entities(entities: list) -> None:
    global _detected_entities
    _detected_entities = [
        e.to_dict() if hasattr(e, 'to_dict') else e
        for e in entities
    ]
```

Resolution searches this cache linearly (`src/executor.py:43-57`):

```python
def resolve_target_id(target_id: str) -> Optional[tuple[int, int]]:
    for entity in _detected_entities:
        if entity.get("id") == target_id:
            center = entity.get("center")
            if center:
                return (int(center[0]), int(center[1]))
    return None
```

Entity IDs follow the pattern `{class_name}_{counter}` (e.g., `sheep_0`, `villager_1`, `town_center_0`). Counters reset each detection cycle, so IDs are only valid within the current iteration.

If resolution fails (entity not found), the action is skipped with a warning log.

## 3.4 Coordinate Translation

Screenshots capture the game window at its screen position. The LLM sees coordinates relative to the screenshot (0,0 = top-left of game window). But pyautogui operates in screen-absolute coordinates.

The executor translates before every action batch (`src/executor.py:184-188`):

```python
rect = get_game_window_rect()
if rect:
    _window_offset = (rect[0], rect[1])
```

Then each action applies the offset (`src/executor.py:95-96`):

```python
def translate(x: int, y: int) -> tuple[int, int]:
    return (x + _window_offset[0], y + _window_offset[1])
```

If the window rect is unavailable, offset defaults to `(0, 0)`, which works for fullscreen games.

## 3.5 Execution Pipeline

`execute_actions()` at `src/executor.py:172-203`:

1. **Get window position** -- updates `_window_offset` for coordinate translation
2. **Ensure focus** -- activates game window, retries once if it fails
3. **Execute sequentially** -- iterates through actions, calling `execute_action()` for each
4. **Count successes** -- returns the number of actions that executed without error

Each action type dispatches to pyautogui:

| Action | pyautogui Call | Notes |
|--------|---------------|-------|
| `click` | `pyautogui.click(x, y)` | Left click after coordinate translation |
| `right_click` | `pyautogui.rightClick(x, y)` | Right click after coordinate translation |
| `press` | `pyautogui.press(key)` | Direct key press |
| `drag` | `pyautogui.moveTo()` + `pyautogui.drag()` | 200ms drag duration |
| `wait` | `asyncio.sleep(ms / 1000)` | Async, does not block event loop |

### pyautogui Configuration (`executor.py:16-17`)

```python
pyautogui.FAILSAFE = False   # Disable corner-abort safety
pyautogui.PAUSE = 0.02       # 20ms between pyautogui calls (default is 100ms)
```

`FAILSAFE = False` is necessary because the game is fullscreen -- the mouse frequently visits screen corners during gameplay. The default 100ms pause is reduced to 20ms for snappier action sequences.

## 3.6 Action Validation Utilities

Two helper functions for ad-hoc validation (`src/models.py:171-208`):

**`validate_action(action_dict)`** -- validates a single action dict against the type map. Returns a Pydantic model or `None`.

**`validate_actions(actions)`** -- batch validation, filters out invalid actions silently. Returns only the valid ones.

The executor uses `validate_action()` for any action that arrives as a raw dict rather than a pre-validated Pydantic model (`executor.py:83-88`).

## 3.7 Coordinate Bounds

All coordinate fields enforce bounds: `ge=0, le=7680` for x, `ge=0, le=4320` for y. This supports up to 8K resolution (7680x4320) and catches obviously invalid coordinates from LLM hallucination. At typical 1920x1080 resolution, coordinates outside the screen are still accepted by the model -- the pyautogui call may click outside the game window but won't crash.

---

## Summary

- 5 action types with Pydantic validation: click, right_click, press, drag, wait
- Dual targeting via (x,y) coordinates or target_id entity references
- Entity IDs resolved from detection cache at execution time
- Coordinate translation from screenshot-relative to screen-absolute
- Sequential execution with 50ms inter-action delay

## Related Topics

- [Chapter 2: Game Loop Pipeline](./02-game-loop-pipeline.md) -- where actions are requested and executed
- [Chapter 5: Prompt Engineering](../part2-llm-integration/05-prompt-engineering.md) -- how the LLM learns the action format
- [Chapter 7: Detector Architecture](../part3-entity-detection/07-detector-architecture.md) -- how entity IDs are generated
