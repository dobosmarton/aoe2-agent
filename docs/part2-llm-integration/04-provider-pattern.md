# Chapter 4: Provider Pattern

The agent abstracts LLM communication behind a provider interface. Currently only Claude is implemented, but the pattern allows adding OpenAI, Gemini, or local models without touching the game loop.

## 4.1 The Abstract Interface

`src/providers/base.py:7-37` defines the contract:

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

Providers are registered in a simple dict at `src/main.py:31-44`:

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

Selected via CLI: `python -m src.main --provider claude`.

## 4.3 Claude Provider Implementation

`src/providers/claude.py:32-291` -- the only production provider.

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

### Content Building (`claude.py:155-190`)

`_build_content()` assembles the user message:

1. Base64-encodes the JPEG screenshot as a vision image block
2. Enhances context with dynamic game knowledge (if available)
3. Appends dimensions info: `"Screenshot dimensions: 1920x1080 pixels. Center=(960,540). Valid x=0-1920, y=0-1080."`
4. Ends with `"What should I do next?"`

### API Call with Retry (`claude.py:192-209`)

```python
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
async def _call_api(self, content: list[dict]) -> str:
    response = await self.client.messages.create(
        model=self.model,
        max_tokens=config.max_tokens,
        system=self.get_system_prompt(),
        messages=[{"role": "user", "content": content}],
    )
    return response.content[0].text
```

Uses tenacity's `@retry` decorator: 3 attempts with exponential backoff (1s, 2s, 4s delays). Handles transient API errors (rate limits, network timeouts) without crashing the game loop.

### Response Parsing (`claude.py:252-290`)

A two-tier JSON extraction strategy:

1. **Try markdown code fences first**: `re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)`
2. **Fall back to raw JSON**: `re.search(r"\{.*\}", response, re.DOTALL)`
3. **If no JSON found**: return reasoning as the raw response text with no actions

After extraction, Pydantic validation acts as a safety net:

```python
try:
    validated = LLMResponse.model_validate(result)
    return validated.model_dump()
except ValidationError:
    # Return reasoning but drop all actions
    return {"reasoning": result.get("reasoning", response_text), "observations": ..., "actions": []}
```

> **Key Insight**: The two-tier extraction handles Claude's tendency to wrap JSON in markdown code fences. Pydantic validation is a safety net -- if actions are malformed, they're dropped rather than executed. This prevents garbage actions (e.g., negative coordinates, missing keys) from reaching pyautogui.

### Error Recovery (`claude.py:244-250`)

On any API or parsing failure, `_error_response()` returns a safe fallback:

```python
def _error_response(self, message: str) -> dict[str, Any]:
    return {
        "reasoning": message,
        "observations": {},
        "actions": [{"type": "wait", "ms": 1000, "intent": "Error recovery"}],
    }
```

A 1-second wait action keeps the loop running while the transient error resolves.

## 4.4 Adding a New Provider

1. Create `src/providers/new_provider.py` implementing `BaseLLMProvider`
2. Implement `get_actions()` to accept screenshot bytes and return the standard dict
3. Implement `get_system_prompt()` with an appropriate prompt for the model
4. Register in `create_provider()` at `src/main.py:33`
5. Add to `--choices` in the argparse definition at `src/main.py:83`

The game loop, memory system, executor, and detection pipeline are provider-agnostic -- they only interact through the `get_actions()` return value.

---

## Summary

- Abstract `BaseLLMProvider` with two required methods
- Claude implementation: AsyncAnthropic, retry logic, two-tier JSON parsing
- Error recovery returns a safe wait action rather than crashing
- Provider-agnostic game loop enables model switching

## Related Topics

- [Chapter 5: Prompt Engineering](./05-prompt-engineering.md) -- the system prompt content
- [Chapter 6: Context Injection](./06-context-injection.md) -- how dynamic context enhances the prompt
- [Chapter 1: System Overview](../part1-architecture/01-system-overview.md) -- graceful degradation for optional dependencies
