# Chapter 5: Prompt Engineering

The system prompt (`prompts/system.md`, ~320 lines + `prompts/hotkeys.md` appended) is the agent's rulebook. It teaches Claude the game mechanics, available composite tools, multi-step action patterns, hotkey reference, and strategic priorities.

<aside class="prereqs">

Comfortable with the basic LLM API loop (system prompt + user message → text response). The two callouts in this chapter introduce [structured output](../glossary.md#s) (§5.5) and [prompt caching](../glossary.md#p) (§5.8) from scratch.

</aside>

## 5.1 Prompt Structure

The prompt is organized into major sections:

| Section | Purpose |
|---------|---------|
| Your Capabilities | What the agent can do (detect, click, remember, target, rescan) |
| Active Goals | How to follow strategist-provided goals by priority |
| EVERY TURN Checklist | 11-point priority checklist (idle villagers, housing, food, etc.) |
| Multi-Task Actions | Recipes using the `build` action + composite tools (send_villager, queue_villager) |
| Smart Targeting | rescan, target_class, fallback patterns, modifiers |
| Handling Failed Actions | How to react to action failures |
| Action Types | 8 base + 2 composite tool types |
| Hotkeys | Full AoE2 hotkey reference (appended from hotkeys.md) |
| Building Placement | Placement rules and constraints |
| Action Limits | 3-7 actions per turn |

## 5.2 Rescan and Coordinate Freshness

Camera-moving hotkeys (H, .) invalidate all screen coordinates. The prompt teaches the LLM to use `rescan: true` on press actions for these keys, which triggers a fresh screenshot + detection cycle. After rescan, entity coordinates are updated in the detection cache, and the LLM receives fresh entity positions in the tool result.

Composite tools (and the `build` action) handle this automatically — for example, `build` executes the full press-click sequence without intermediate rescans, with placement auto-derived near the Town Center.

## 5.3 Composite Tool Patterns

The prompt defines recipes using composite tools for common operations:

**Build a house (coordinate-free — the executor auto-places near the TC):**
```json
{"type": "build", "building_key": "q", "intent": "Build house"}
```

**Queue a villager (1 tool call):**
```json
{"type": "queue_villager", "intent": "Queue villager"}
```

**Send idle villager to resource (1 tool call):**
```json
{"type": "send_villager", "target_class": "sheep", "intent": "Send villager to gather sheep"}
```

These actions replaced the old multi-turn patterns where operations had to be split across turns due to camera movement. They execute the full sequence internally without intermediate API roundtrips — and `build` is now a first-class action, so it works on the fast single-shot path too, not only the tool loop.

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

<aside class="concept" data-title="Structured output (why we ask for JSON, not prose)">

When you let an LLM reply in free-form text you have to parse it with regex — which works until the model phrases its answer differently and silently breaks your action loop. **Structured output** flips the contract: the prompt declares an exact JSON schema, the model is constrained (or strongly nudged) to emit only that shape, and your code can `json.loads` the response straight into a Pydantic model.

There are three common ways to get this guarantee, in order of strictness:

1. **Tool use / function calling** (Anthropic, OpenAI) — the strictest. The model emits a tool call with a JSON-Schema-validated argument blob. We use this for our composite tools.
2. **`response_format: json_object` / `messages.parse(response_model=...)`** — the model is forced into JSON-mode and the SDK validates against a Pydantic model.
3. **Prompted JSON in plain text** — what you see above in section 5.5. Cheapest and most flexible, but you still need a `try/except json.JSONDecodeError` and a retry path.

We use a mix: tool-use for actions that must execute (no room for malformed output), prompted JSON for the strategist's reasoning blob (where a one-time parse failure is recoverable). The trade-off is **strictness vs. expressiveness**: stricter schemas catch more bugs but cap what the model can say.

</aside>

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

The prompt is loaded from disk in `ExecutorProvider.get_system_prompt(age)`, which returns a **list of cacheable content blocks** (not a single string):

```python
def get_system_prompt(self, age: str = "Dark Age") -> list[dict]:
    self._load_prompts()
    age_content = self._age_prompts.get(age.split()[0].lower(), ...)
    blocks = [
        {"type": "text", "text": self._core_prompt,          # core + hotkeys + memories
         "cache_control": {"type": "ephemeral"}},
    ]
    if age_content:
        blocks.append({"type": "text", "text": age_content,  # age-specific guidance
                       "cache_control": {"type": "ephemeral"}})
    return blocks
```

Block 1 (core rules + hotkey reference + cross-game memories) is stable for the whole game and is cached on every call. Block 2 (age-specific guidance) changes only on age-ups (≤3 times per game) and is **also** cached, so every turn within an age reads it from cache instead of re-prefilling. Prompts are lazily loaded once and cached for the session; editing prompt files requires restarting the agent.

A fallback inline prompt provides minimal JSON format and action types — enough to run but without strategic depth.

<aside class="concept" data-title="Prompt caching (why the same 320-line prompt isn't billed 100×)">

Every turn we send the same ~320-line system prompt plus a turn-specific user message. Without caching, the provider would tokenize and re-process all 320 lines every call — both billing you for them and adding ~100–300ms of TTFT.

**Prompt caching** marks a prefix of the request as cacheable. On the next call within the TTL (5 minutes on Anthropic, configurable up to 1h), the provider reuses its internal KV-cache for that prefix. You pay **~10% of the normal input price** for cached tokens, and TTFT drops by ~50–80% on long prompts.

The trade-off: writing to the cache costs **~25% extra** the first time, the cache key is the *exact prefix* (a single-character change invalidates it), and only blocks ≥1024 tokens are cacheable. So the pattern is: put stable content (system prompt, tool definitions, hotkey reference) **first** with `cache_control: {"type": "ephemeral"}`, then put turn-volatile content (the turn's entity list, last action result) **after** it. Verify it's working by checking `usage.cache_read_input_tokens` in the response — a value > 0 means you got a hit.

For our agent, this turns a 320-line prompt from a per-turn cost into a once-per-5-min cost, which is the difference between "viable for live play" and "burns a dollar per game."

</aside>

The `cache_control` markers are set in `providers/executor_provider.py`: two on the system blocks (above), plus a **moving breakpoint** on the most recent message in the executor's tool loop (`_apply_moving_cache_breakpoint`) so iterations 2–7 read the growing conversation from cache rather than re-prefilling it. See [Chapter 4: Provider Pattern](./04-provider-pattern.md) for the exact API call shape.

---

## Summary

- ~320-line system prompt + ~113-line hotkey reference teaching game mechanics, composite tools, and strategic priorities
- `build` is a first-class coordinate-free action; composite tools (send_villager, queue_villager) collapse multi-step sequences into single tool calls
- 11-point EVERY TURN checklist drives prioritized decision-making
- Rescan mechanism handles coordinate freshness after camera-moving keys
- 3-7 actions per turn with multi-task turns encouraged
- Loaded from disk with prompt caching and inline fallback

## Related Topics

- [Chapter 4: Provider Pattern](./04-provider-pattern.md) -- how the prompt is loaded and used
- [Chapter 6: Context Injection](./06-context-injection.md) -- what additional context accompanies the prompt
- [Chapter 3: Action Model & Execution](../part1-architecture/03-action-model-and-execution.md) -- how the output format maps to execution
