# Chapter 5: Prompt Engineering

The system prompt (`prompts/system.md`, ~320 lines + `prompts/hotkeys.md` appended) is the agent's rulebook. It teaches Claude the game mechanics, available composite tools, multi-step action patterns, hotkey reference, and strategic priorities.

## 5.1 Prompt Structure

The prompt is organized into major sections:

| Section | Purpose |
|---------|---------|
| Your Capabilities | What the agent can do (detect, click, remember, target, rescan) |
| Active Goals | How to follow strategist-provided goals by priority |
| EVERY TURN Checklist | 11-point priority checklist (idle villagers, housing, food, etc.) |
| Multi-Task Actions | Recipes using composite tools (build, send_villager, queue_villager) |
| Smart Targeting | rescan, target_class, fallback patterns, modifiers |
| Handling Failed Actions | How to react to action failures |
| Action Types | 7 base + 3 composite tool types |
| Hotkeys | Full AoE2 hotkey reference (appended from hotkeys.md) |
| Building Placement | Placement rules and constraints |
| Action Limits | 3-7 actions per turn |

## 5.2 Rescan and Coordinate Freshness

Camera-moving hotkeys (H, .) invalidate all screen coordinates. The prompt teaches the LLM to use `rescan: true` on press actions for these keys, which triggers a fresh screenshot + detection cycle. After rescan, entity coordinates are updated in the detection cache, and the LLM receives fresh entity positions in the tool result.

Composite tools handle this automatically — for example, `build` executes the full press-click sequence without intermediate rescans since the placement coordinates are pre-determined.

## 5.3 Composite Tool Patterns

The prompt defines recipes using composite tools for common operations:

**Build a house (1 tool call):**
```json
{"type": "build", "building_key": "q", "x": 1500, "y": 800, "intent": "Build house"}
```

**Queue a villager (1 tool call):**
```json
{"type": "queue_villager", "intent": "Queue villager"}
```

**Send idle villager to resource (1 tool call):**
```json
{"type": "send_villager", "target_class": "sheep", "intent": "Send villager to gather sheep"}
```

These composite tools replaced the old multi-turn patterns where operations had to be split across turns due to camera movement. The composites execute the full sequence internally without intermediate API roundtrips.

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

A comprehensive hotkey reference is appended from `prompts/hotkeys.md` (~113 lines). Key hotkeys for Dark Age:

| Key | Effect |
|-----|--------|
| H | Select Town Center, center camera |
| Q | Queue villager (at TC) / Economic build menu (with villager selected) |
| . | Select idle villager, center camera |
| , | Select idle military unit, center camera |
| W | Military build menu (with villager, Feudal Age+) |
| G | Auto Scout (when military unit selected) |

The hotkey file covers navigation, TC commands, villager build menus (economic, military, more buildings), and unit commands.

## 5.7 Action Limits

- **3-7 actions per turn** — with composite tools, each tool call does more work so fewer calls are needed
- **Multi-task turns encouraged** — queue villagers + build houses + sweep idle villagers in ONE turn using composite tools
- **Rescan after camera-moving keys** — ensures fresh coordinates for subsequent clicks

## 5.8 Prompt Loading Mechanism

The prompt is loaded from disk in `ClaudeProvider.get_system_prompt()`:

```python
def get_system_prompt(self) -> str:
    if self._system_prompt is None:
        prompt_file = PROMPTS_DIR / "system.md"
        hotkeys_file = PROMPTS_DIR / "hotkeys.md"
        if prompt_file.exists():
            self._system_prompt = prompt_file.read_text()
            if hotkeys_file.exists():
                self._system_prompt += "\n\n" + hotkeys_file.read_text()
    return self._system_prompt
```

The system prompt and hotkey reference are concatenated and lazily loaded on first API call, then cached for the session. The system prompt uses Anthropic's `cache_control` for prompt caching, reducing cost on subsequent API calls. Editing prompt files requires restarting the agent.

A fallback inline prompt provides minimal JSON format and action types — enough to run but without strategic depth.

---

## Summary

- ~320-line system prompt + ~113-line hotkey reference teaching game mechanics, composite tools, and strategic priorities
- Composite tools (build, send_villager, queue_villager) collapse multi-step sequences into single tool calls
- 11-point EVERY TURN checklist drives prioritized decision-making
- Rescan mechanism handles coordinate freshness after camera-moving keys
- 3-7 actions per turn with multi-task turns encouraged
- Loaded from disk with prompt caching and inline fallback

## Related Topics

- [Chapter 4: Provider Pattern](./04-provider-pattern.md) -- how the prompt is loaded and used
- [Chapter 6: Context Injection](./06-context-injection.md) -- what additional context accompanies the prompt
- [Chapter 3: Action Model & Execution](../part1-architecture/03-action-model-and-execution.md) -- how the output format maps to execution
