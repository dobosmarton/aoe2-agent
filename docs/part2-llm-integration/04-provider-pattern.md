# Chapter 4: Provider Pattern

The agent abstracts LLM communication behind a provider interface. Currently only Claude is implemented, but the pattern allows adding OpenAI, Gemini, or local models without touching the game loop.

<aside class="prereqs">

Python ABCs and async/await. If "tool use" or "function calling" is new, jump to the agentic-tool-loop deep dive at the end of §4.3 first.

</aside>

## 4.1 The Abstract Interface

`apps/agent/src/providers/base.py:7-37` defines the contract:

```python
class BaseLLMProvider(ABC):
    @abstractmethod
    async def get_actions(
        self,
        screenshot_bytes: bytes,
        context: str = "",
        width: int = 1920,
        height: int = 1080,
    ) -> dict[str, Any]:
        """Returns dict with 'reasoning', 'actions', and optionally 'observations'."""
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this provider."""
        pass
```

Two methods: `get_actions()` takes a screenshot and context, returns structured output. `get_system_prompt()` returns the provider-specific prompt. Both are required for any new provider.

## 4.2 Provider Registration

Providers are registered in a simple dict at `apps/agent/src/main.py:31-44`:

```python
def create_provider(provider_name: str):
    providers = {
        "claude": ClaudeProvider,
        # "openai": OpenAIProvider,
        # "gemini": GeminiProvider,
    }
    if provider_name not in providers:
        available = ", ".join(providers.keys())
        raise ValueError(f"Unknown provider: {provider_name}. Available: {available}")
    return providers[provider_name]()
```

Selected via CLI: `python -m gameplay_agent --provider claude`.

## 4.3 Claude Provider Implementation

`apps/agent/src/providers/claude.py:32-291` -- the only production provider.

### Initialization (`claude.py:35-63`)

```python
class ClaudeProvider(BaseLLMProvider):
    def __init__(self, api_key=None, model=None, use_dynamic_context=True):
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
        self._system_prompt: str | None = None
        self.use_dynamic_context = use_dynamic_context and GAME_KNOWLEDGE_AVAILABLE
        self._game_db: Optional["GameKnowledge"] = None
```

- Uses `AsyncAnthropic` for non-blocking API calls
- Lazily loads the system prompt on first access
- Optionally initializes the game knowledge database for dynamic context injection

### System Prompt Loading (`claude.py:65-98`)

Loads from `prompts/system.md` on disk. If the file doesn't exist, falls back to a minimal inline prompt that teaches the JSON output format and basic action types. See [Chapter 5](./05-prompt-engineering.md) for prompt content.

### Content Building (`claude.py:_build_content`)

The executor is **text-only** — no screenshot — so all visual information arrives as the YOLO entity list plus the strategist's cached resource readings. `_build_content()` assembles a single text content block:

1. Enhances the context with dynamic game knowledge (affordable units/buildings) when the knowledge DB is available
2. Prepends a dimensions line: `"Game window: 1920x1080 pixels. Center=(960,540). ..."`
3. Returns `[{"type": "text", "text": ...}]`

### The two executor paths (`claude.py`)

The executor runs `claude-sonnet-4-6` (`config.model`). `get_actions()` routes each turn to one of two paths via `_use_single_shot(context)`:

- **Single-shot (routine turns).** `_call_single_shot()` makes **one** `messages.parse()` call — no tool loop. The returned actions are handed to the game loop to execute (`actions_already_executed=False`). This is the fast, cheap path for ordinary economy turns.
- **Agentic tool loop (interactive turns).** `_call_api()` runs `messages.create(..., tools=_ACTION_TOOLS)` up to `config.max_tool_iterations` (7) times: Claude calls a tool, the host executes it, the result is fed back, and composite tools (`build`, `send_villager`) run multi-step sequences within one iteration. Used when the turn needs mid-turn rescans or composite tools the single-shot `Action` union can't express.

`_use_single_shot` keeps the loop for combat/housing emergencies — it scans the context for signals like `under attack: true`, `defend`, `housed (cannot` — and takes the single-shot path otherwise.

```python
async def _call_single_shot(self, content, age="Dark Age") -> LLMResult:
    output_config: OutputConfigParam = {"effort": config.executor_effort}
    response = await self.client.messages.parse(
        model=self.model,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        system=self.get_system_prompt(age),
        messages=[{"role": "user", "content": content}],
        output_format=LLMResponse,   # SDK merges this into output_config.format
        output_config=output_config,
    )
    # ... accumulate usage ...
    return self._serialize_single_shot(response.parsed_output)
```

Both paths share the **effort knob** (`config.executor_effort`, default `low`, env `AOE2_EXECUTOR_EFFORT`) passed as `output_config={"effort": ...}`: a low effort trims latency and consolidates tool calls on Sonnet 4.6 (the SDK rejects `xhigh`/`max` for this tier, so the config type is `Literal["low","medium","high"]`). `messages.parse()` returns a validated `LLMResponse` Pydantic model directly (requires `anthropic>=0.84.0`); the tool loop assembles an `LLMResponse` from the executed tool calls. Both paths use prompt caching — see [Chapter 5 §5.8](./05-prompt-engineering.md).

### Error Recovery (`claude.py:_error_response`)

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

**Why composite tools change the math.** A naive "build a house" sequence is four tool calls — `press('q')`, then `press('q')` for the house menu, then `rescan: true` (because pressing a hotkey may have moved the camera), then `click(x, y)`. That's 4 roundtrips × ~3s = ~12 seconds of wall-clock per house. By wrapping that recipe into one composite `build(building_key='q', x, y)` tool that the *host* executes as a sequence, we collapse it back to **one** roundtrip — ~3 seconds and ~one-quarter of the tokens. The model loses no flexibility (it can still fall back to primitives when needed), but the common case is cheap.

**Versus the alternatives.**

- **ReAct** (Yao et al., 2022) interleaves free-form *thought*-tokens between tool calls. More inspectable, but more output tokens and the thoughts are not validated by any schema.
- **Plan-and-execute** asks the LLM to write a full plan upfront and then executes it without re-prompting. Faster for predictable tasks; brittle when the world changes between plan and execution — exactly our situation.
- **Single-shot structured output** (what our chapter shows: `messages.parse` returning an `LLMResponse` with a list of actions) sidesteps the loop entirely: one API call, one response, the host runs each action and feeds nothing back to the LLM until the next turn. Cheapest and most predictable, but the model can't react mid-turn to a tool's success or failure.

We actually use a **hybrid that switches per turn**: routine turns take the single-shot path (one `messages.parse`, no roundtrips), while combat/housing turns — the ones that need mid-execution feedback (a rescan whose result changes what to click next) or composite tools — take the agentic tool loop. The router (`_use_single_shot`) keeps the predictable case cheap and reserves the expensive loop for the turns that genuinely need it.

**Mental model for the cost.** A useful rule of thumb: at Claude Sonnet rates, every tool roundtrip on a fully primed conversation costs roughly the same as one second of GPT-running-flat-out — pennies, but they add up. If your agent feels expensive, the lever is almost always "reduce the number of roundtrips," not "switch to a cheaper model."

</details>

## 4.4 Adding a New Provider

1. Create `apps/agent/src/providers/new_provider.py` implementing `BaseLLMProvider`
2. Implement `get_actions()` to accept screenshot bytes and return the standard dict
3. Implement `get_system_prompt()` with an appropriate prompt for the model
4. Register in `create_provider()` at `apps/agent/src/main.py:33`
5. Add to `--choices` in the argparse definition at `apps/agent/src/main.py:83`

The game loop, memory system, executor, and detection pipeline are provider-agnostic -- they only interact through the `get_actions()` return value.

---

## Summary

- Abstract `BaseLLMProvider` with two required methods
- Claude implementation: AsyncAnthropic executor on `claude-sonnet-4-6` with a per-turn router (`_use_single_shot`) — single-shot `messages.parse` for routine turns, an agentic `messages.create` tool loop for combat/housing
- Shared `effort` knob (`config.executor_effort`, default `low`) via `output_config`; structured output via `messages.parse` (requires `anthropic>=0.84.0`)
- Error recovery returns a safe wait action rather than crashing
- Provider-agnostic game loop enables model switching

## Related Topics

- [Chapter 5: Prompt Engineering](./05-prompt-engineering.md) -- the system prompt content
- [Chapter 6: Context Injection](./06-context-injection.md) -- how dynamic context enhances the prompt
- [Chapter 1: System Overview](../part1-architecture/01-system-overview.md) -- graceful degradation for optional dependencies
