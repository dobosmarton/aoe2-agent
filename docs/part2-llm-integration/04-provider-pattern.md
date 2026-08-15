# Chapter 4: Provider Pattern

The agent splits LLM communication in two: **one executor holds the game logic**, and a **wire** holds everything vendor-specific. Switching between Anthropic and any OpenAI-compatible endpoint (OpenCode Zen, api.openai.com) is a wire swap, not a second copy of the ~900-line executor.

<aside class="prereqs">

Python Protocols and async/await. If "tool use" or "function calling" is new, jump to the agentic-tool-loop deep dive at the end of §4.3 first.

</aside>

## 4.1 The Wire Contract

`apps/agent/src/providers/base.py` defines a `Protocol` (structural, no inheritance) plus the frozen value objects that cross it:

```python
class ChatWire(Protocol):
    model: str

    async def tool_turn(self, request: ChatRequest,
                        tools: list[dict[str, object]]) -> ToolTurnResult: ...
    async def parse_structured(self, request: ChatRequest,
                               schema: type[ModelT]) -> tuple[ModelT, TokenUsage]: ...
    def is_api_error(self, exc: Exception) -> bool: ...
    def is_schema_too_large(self, exc: Exception) -> bool: ...
```

The last two matter: they move **exception classification** behind the seam, so the executor never names a vendor's error classes. `is_schema_too_large` is true only on Anthropic, whose constrained decoding has a compiled-grammar size cap — that is what routes a turn off the single-shot path onto the smaller tool-loop schema (run 12, F-40).

Conversations cross the seam as neutral turns — `UserTurn`, `AssistantTurn`, `ToolResultsTurn` — and each wire renders its own message list from them. That is where the vendors genuinely differ:

| Concern | `wire_anthropic.py` | `wire_openai.py` |
|---|---|---|
| System prompt | list of blocks with explicit `cache_control` | one message; caching automatic on prefix |
| Tools | `{name, description, input_schema}` | `{type: "function", function: {...}, strict: true}` |
| Tool results | all batched into one user turn | **one `tool` message per `tool_call_id`** |
| Wants more tools | `stop_reason == "tool_use"` | `finish_reason == "tool_calls"` |
| Usage | `input_tokens` excludes cached | `prompt_tokens` **includes** cached — subtracted back out |
| Refusal | `stop_reason == "refusal"` | `message.refusal` |

Both raise `ModelRefusedError` so callers handle one exception type.

## 4.2 Wire Selection

`apps/agent/src/providers/wire_factory.py` is the single place a wire is chosen by name — one match arm per implementation, a lazy import inside each so neither SDK is mandatory, and a `ValueError` naming the valid choices rather than a silent fallback:

```python
wire = make_wire(config.llm_wire, model=config.model,
                 api_key=..., base_url=config.llm_base_url)
```

Three names are valid. `openai` and `zen` share `wire_openai` and differ only in default endpoint, so picking a gateway is one variable rather than a wire plus a URL:

| `AOE2_LLM_WIRE` | Wire | Default endpoint |
|---|---|---|
| `openai` (default) | `OpenAIWire` | `https://api.openai.com/v1` |
| `zen` | `OpenAIWire` | `https://opencode.ai/zen/v1` |
| `anthropic` | `AnthropicWire` | `https://api.anthropic.com` |

`AOE2_LLM_BASE_URL` overrides the endpoint on either OpenAI-compatible arm; empty means "use the adapter's own". The valid set is the `WireName` Literal in `config.py`, and both `_parse_wire` and the `--wire` CLI choices derive from it, so a fourth adapter is a one-line change.

Selected by env or CLI:

```bash
AOE2_LLM_WIRE=zen AOE2_LLM_API_KEY=sk-... AOE2_MODEL=gpt-5.6-luna just agent
python -m gameplay_agent.main --wire zen
```

There is deliberately **no registry of provider classes** — one executor serves every model. `make_text_completer` is the synchronous sibling for the plain prompt-in/text-out callers (memory extraction, prompt mutation); it carries the same arms, so a new adapter must be added to both.

Per-model pricing lives in one table, `providers/pricing.py`, shared by the agent and the arena. An unknown model logs `pricing_unknown_model` rather than silently costing $0.00. The table is keyed by model alone, so a gateway that resells a model at its own rate is priced at the first-party rate — the `api_cost` log event carries `endpoint` so a figure stays attributable.

## 4.3 Executor Implementation

`apps/agent/src/providers/executor_provider.py` -- the one production executor, holding only game logic.

### Initialization

```python
class ExecutorProvider:
    def __init__(self, api_key=None, model=None, use_dynamic_context=True, wire=None):
        self.model = model or config.model
        # Defaults to whatever AOE2_LLM_WIRE selects; inject one to override.
        self.wire: ChatWire = wire or make_wire(config.llm_wire, model=self.model, ...)
        self.use_dynamic_context = use_dynamic_context and GAME_KNOWLEDGE_AVAILABLE
```

- Delegates every API call to its `ChatWire`; the SDK client lives there
- Lazily loads the system prompt on first access
- Optionally initializes the game knowledge database for dynamic context injection

### System Prompt Loading (`executor_provider.py`)

Loads and concatenates `prompts/core.md` + `prompts/hotkeys.md`, then layers the age-specific `prompts/ages/<age>.md` at request time. If a file doesn't exist, falls back to a minimal inline prompt that teaches the JSON output format and basic action types. See [Chapter 5](./05-prompt-engineering.md) for prompt content.

### Content Building (`executor_provider._build_content`)

The executor is **text-only** — no screenshot — so all visual information arrives as the YOLO entity list plus the strategist's cached resource readings. (The strategist itself is also text-only: it produces those readings by OCR-ing the resource bar locally — `resource_ocr.py`, RapidOCR — not via a Claude vision call.) `_build_content()` assembles a single text content block:

1. Enhances the context with dynamic game knowledge (affordable units/buildings) when the knowledge DB is available
2. Prepends a dimensions line: `"Game window: 1920x1080 pixels. Center=(960,540). ..."`
3. Returns `[{"type": "text", "text": ...}]`

### The two executor paths

The executor runs whatever `config.model` names, over `config.llm_wire`. `get_actions()` routes each turn to one of two paths via `_use_single_shot(context)`:

- **Single-shot (routine turns).** `_call_single_shot()` makes **one** `parse_structured` call — no tool loop. The returned actions are handed to the game loop to execute (`actions_already_executed=False`). This is the fast, cheap path for ordinary economy turns.
- **Agentic tool loop (interactive turns).** `_call_api()` runs `wire.tool_turn(..., _ACTION_TOOLS)` up to `config.max_tool_iterations` (7) times: Claude calls a tool, the host executes it, the result is fed back, and composite tools (`send_villager`, `queue_villager`) run multi-step sequences within one iteration. Used when the turn needs mid-turn rescans or the composite tools the single-shot `Action` union can't express. (`build` is now in the single-shot `Action` union too, so routine turns build directly; the loop still exposes a `build` tool that runs the same shared `build_steps()` sequence.)

`_use_single_shot` keeps the loop for combat/housing emergencies — it scans the context for signals like `under attack: true`, `defend`, `housed (cannot` — and takes the single-shot path otherwise.

```python
async def _parse_single_shot(self, turns: tuple[Turn, ...], age: str) -> LLMResponse:
    parsed, usage = await self.wire.parse_structured(
        ChatRequest(
            system=self.get_system_prompt(age),
            turns=turns,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            effort=config.executor_effort,
        ),
        LLMResponse,
    )
    self._record_usage(usage)
    return parsed
```

Both paths share the **effort knob** (`config.executor_effort`, default `low`, env `AOE2_EXECUTOR_EFFORT`), carried on `ChatRequest.effort` and rendered as `output_config` on Anthropic or `reasoning_effort` on OpenAI: a low effort trims latency and consolidates tool calls on Sonnet 4.6 (the SDK rejects `xhigh`/`max` for this tier, so the config type is `Literal["low","medium","high"]`). `parse_structured` returns a validated `LLMResponse` Pydantic model directly; the tool loop assembles an `LLMResponse` from the executed tool calls. Both paths use prompt caching — see [Chapter 5 §5.8](./05-prompt-engineering.md).

### Error Recovery (`executor_provider._error_response`)

On any API or parsing failure (either path), `get_actions()` returns a safe fallback:

```python
def _error_response(self, message: str) -> LLMResult:
    return LLMResult(
        reasoning=message,
        observations={},
        actions=[{"type": "wait", "ms": 1000, "intent": "Error recovery"}],
    )
```

A 1-second wait action keeps the loop running while the transient error resolves.

<details class="deep-dive">
<summary>Deep dive — Agentic tool loops, and why composite tools save you a fortune</summary>

**The shape of an agentic loop.** When an LLM is given tools, the API stops being request/response and starts being a state machine. Each turn looks like this:

```
user message  →  LLM thinks  →  emits tool_call  →  host runs tool  →
                 tool_result fed back  →  LLM thinks again  →  emits next tool_call  →
                 ...  →  LLM emits "stop" (text-only response)
```

Every arrow that says "LLM thinks" is a **full API roundtrip**. You pay the full input cost for the entire conversation so far (including all prior tool calls and results — which is why prompt caching matters so much here), plus output tokens, plus ~2–4 seconds of latency. The `max_tool_iterations = 7` cap in our executor exists because, without it, a confused model can spiral into 20+ iterations and burn through dollars.

**Why host-expanded actions change the math.** A naive "build a house" sequence is four tool calls — `press('.')` to select an idle villager, `press('q')` for the economic build menu, `press('q')` to pick the house, then `click()` for placement. That's 4 roundtrips × ~3s = ~12 seconds of wall-clock per house. Instead, `build` is a single **coordinate-free** action (`{type:"build", building_key:"q"}`) that the *host* expands into that sequence via `build_steps()` — auto-placing near the Town Center — collapsing it back to **one** decision: ~3 seconds and ~one-quarter of the tokens. And because `build` is a first-class action (not a tool-loop-only composite), routine single-shot turns get the same saving; the remaining composites `send_villager`/`queue_villager` work the same way inside the loop. The model loses no flexibility (it can still fall back to primitives when needed), but the common case is cheap.

**Versus the alternatives.**

- **ReAct** (Yao et al., 2022) interleaves free-form *thought*-tokens between tool calls. More inspectable, but more output tokens and the thoughts are not validated by any schema.
- **Plan-and-execute** asks the LLM to write a full plan upfront and then executes it without re-prompting. Faster for predictable tasks; brittle when the world changes between plan and execution — exactly our situation.
- **Single-shot structured output** (what our chapter shows: `parse_structured` returning an `LLMResponse` with a list of actions) sidesteps the loop entirely: one API call, one response, the host runs each action and feeds nothing back to the LLM until the next turn. Cheapest and most predictable, but the model can't react mid-turn to a tool's success or failure.

We actually use a **hybrid that switches per turn**: routine turns take the single-shot path (one `parse_structured`, no roundtrips), while combat/housing turns — the ones that need mid-execution feedback (a rescan whose result changes what to click next) or composite tools — take the agentic tool loop. The router (`_use_single_shot`) keeps the predictable case cheap and reserves the expensive loop for the turns that genuinely need it.

**Mental model for the cost.** A useful rule of thumb: at Claude Sonnet rates, every tool roundtrip on a fully primed conversation costs roughly the same as one second of GPT-running-flat-out — pennies, but they add up. If your agent feels expensive, the lever is almost always "reduce the number of roundtrips," not "switch to a cheaper model."

</details>

## 4.4 Adding a New Vendor

You add a **wire**, not a provider — the executor stays as it is.

1. Create `apps/agent/src/providers/wire_<vendor>.py` satisfying `ChatWire`
2. Render `SystemBlock`/`Turn` values onto that API's message shape, and map its usage fields onto `TokenUsage`
3. Classify its exceptions in `is_api_error` / `is_schema_too_large`
4. Add the name to the `WireName` Literal in `config.py` — `_parse_wire` and the CLI choices follow automatically
5. Add one arm to **both** `make_wire` and `make_text_completer` in `providers/wire_factory.py`, with a lazy import so the SDK stays optional
6. Add the model's rates to `providers/pricing.py`

An OpenAI-compatible endpoint needs only steps 4 and 5: give it a name on the existing `wire_openai` arm with its own default URL, as `zen` does.

The game loop, memory system, executor, and detection pipeline never see a vendor -- they interact only through `LLMResult`.

---

## Summary

- One executor (`ExecutorProvider`) holding game logic; a `ChatWire` Protocol holding every vendor detail
- Per-turn router (`_use_single_shot`) — a single structured call for routine turns, an agentic tool loop for combat/housing, with the tool loop as the fallback when a vendor rejects the larger schema
- Shared `effort` knob (`config.executor_effort`, default `low`): `output_config` on Anthropic, `reasoning_effort` on OpenAI
- Error recovery returns a safe wait action with `error=True`, feeding the executor-outage alarm and `llm_error_rate` rather than crashing
- Vendor choice is one env var; pricing lives in one shared table

## Related Topics

- [Chapter 5: Prompt Engineering](./05-prompt-engineering.md) -- the system prompt content
- [Chapter 6: Context Injection](./06-context-injection.md) -- how dynamic context enhances the prompt
- [Chapter 1: System Overview](../part1-architecture/01-system-overview.md) -- graceful degradation for optional dependencies
