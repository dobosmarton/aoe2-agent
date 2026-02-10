# Chapter 5: Prompt Engineering

The system prompt (`prompts/system.md`, 83 lines) is the agent's rulebook. It teaches Claude how to interpret AoE2 screenshots, which hotkeys are available, how to structure multi-turn operations, and the exact JSON output format expected.

## 5.1 Prompt Structure

The prompt is organized into 7 sections:

| Section | Lines | Purpose |
|---------|-------|---------|
| Your Capabilities | 1-8 | What the agent can do (see, click, remember, target) |
| Camera Movement Rules | 9-16 | Critical constraint about coordinate invalidation |
| Action Patterns | 18-30 | Multi-turn recipes for common operations |
| Using Detected Entities | 31-41 | target_id vs coordinate targeting |
| Output Format | 43-61 | JSON schema with example |
| Action Types + Hotkeys | 63-75 | Reference for available actions and keys |
| Action Limits | 77-82 | Constraints on actions per turn |

## 5.2 Camera Movement Rules (The Critical Constraint)

Lines 9-16 of `prompts/system.md`:

```markdown
## CRITICAL: Camera Movement Rules

**Hotkeys that MOVE the camera:**
- H: Selects TC and centers camera on it
- . (period): Selects idle villager and centers camera on them

**Rule: If you use H or ., END your turn immediately after.**
Do NOT click in the same turn - coordinates become invalid after camera moves!
```

This is the most important constraint in the entire prompt. When the camera moves, every pixel position on screen changes. If the LLM presses H (go to Town Center) and then tries to click at coordinates from the pre-move screenshot, it will click on the wrong location.

> **Key Insight**: Complex game operations must be split across multiple turns because camera-moving hotkeys invalidate all coordinates. A house build requires Turn 1 (press period to select villager, camera moves, turn ends) then Turn 2 (see new viewport, press Q,Q to open build menu, click placement location). This is a non-obvious constraint that the LLM must internalize.

## 5.3 Multi-Turn Action Patterns

Lines 18-30 define recipes for common operations:

**Build a house (2 turns):**
```
Turn 1: Press . (selects idle villager, moves camera) → STOP
Turn 2: Press Q, Q (open build menu, select house), Click on clear ground
```

**Queue a villager (1 turn):**
```
Press H (select TC, camera moves but action doesn't need coordinates), Press Q → STOP
```

Note the queue-villager pattern works in 1 turn because pressing Q after H doesn't require clicking on the screen -- it's a purely keyboard action. The camera moves but coordinates are never used.

**Gather food (2 turns):**
```
Turn 1: Press . (select idle villager) → STOP
Turn 2: Right-click on target_id (e.g., "sheep_0") or sheep coordinates
```

## 5.4 Entity Targeting

Lines 31-41 teach the LLM to prefer target_id when detection is available:

```json
{"type": "right_click", "target_id": "sheep_0", "intent": "Gather from sheep"}
```

And fall back to coordinates when it isn't:

```json
{"type": "right_click", "x": 920, "y": 460, "intent": "Gather from sheep at coordinates"}
```

The LLM sees detected entities in the context as a list with IDs and coordinates, so it can reference them by name.

## 5.5 Output Format Specification

Lines 43-61 define the JSON contract:

```json
{
  "reasoning": "What you see and strategic thinking",
  "observations": {
    "resources": {"food": 0, "wood": 0, "gold": 0, "stone": 0},
    "population": "5/10",
    "age": "Dark Age",
    "idle_tc": true,
    "housed": false,
    "under_attack": false,
    "events": []
  },
  "actions": [
    {"type": "press", "key": ".", "intent": "Select idle villager"}
  ]
}
```

**reasoning** -- free-form text explaining what the LLM sees in the screenshot and its strategic thinking. This is logged and stored in memory for context in future turns.

**observations** -- structured game state extracted from the screenshot. These feed back into the memory system (see [Chapter 6](./06-context-injection.md)) to track resources, population, and alerts across turns.

**actions** -- ordered list of actions to execute sequentially. Each has a `type`, parameters, and an `intent` string for logging.

## 5.6 Hotkey Reference

Lines 70-75:

| Key | Effect |
|-----|--------|
| H | Select Town Center, center camera |
| Q | Queue villager (in TC) / Economic build menu (with villager) |
| . | Select idle villager, center camera |
| , | Select idle military unit, center camera |
| W | Military build menu (with villager) |

These are the AoE2:DE default hotkeys. The set is deliberately small -- the prompt focuses on Dark Age / early Feudal Age operations where these keys cover most actions.

## 5.7 Action Limits

Lines 77-80:

- **3-5 actions per turn** maximum -- prevents the LLM from generating long action sequences that become unreliable
- **200-500ms wait** between dependent steps -- gives the game UI time to respond
- **Focus on ONE task per turn** -- reduces the chance of mid-sequence errors

These limits keep each turn atomic and recoverable. If an action fails, the next iteration sees the current state and can retry.

## 5.8 Prompt Loading Mechanism

The prompt is loaded from disk at `src/providers/claude.py:65-98`:

```python
def get_system_prompt(self) -> str:
    if self._system_prompt is None:
        prompt_file = PROMPTS_DIR / "system.md"
        if prompt_file.exists():
            self._system_prompt = prompt_file.read_text()
        else:
            self._system_prompt = """...(fallback)..."""
    return self._system_prompt
```

The prompt is lazily loaded on first API call and cached for the session. Editing `prompts/system.md` requires restarting the agent to pick up changes.

The fallback inline prompt (lines 73-97) is a minimal version that teaches only the JSON format and basic action types -- enough to run but without the strategic depth of the full prompt.

---

## Summary

- 83-line system prompt teaching game mechanics, hotkeys, action format, and constraints
- Camera movement rule is the most critical constraint: end turn after H or period
- Multi-turn patterns split complex operations to handle coordinate invalidation
- 3-5 actions per turn, one task per turn, with waits between dependent steps
- Loaded from disk with inline fallback

## Related Topics

- [Chapter 4: Provider Pattern](./04-provider-pattern.md) -- how the prompt is loaded and used
- [Chapter 6: Context Injection](./06-context-injection.md) -- what additional context accompanies the prompt
- [Chapter 3: Action Model & Execution](../part1-architecture/03-action-model-and-execution.md) -- how the output format maps to execution
