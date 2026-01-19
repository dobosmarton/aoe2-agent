# AoE2 LLM Agent

A pure vision-based AI agent that plays Age of Empires 2: Definitive Edition using LLMs.

## Overview

The agent uses a simple loop:
1. **Capture** - Take a screenshot of the game window
2. **Think** - Send to LLM with game context, get back actions
3. **Act** - Execute mouse/keyboard commands
4. **Remember** - Update memory with observations
5. **Repeat**

The LLM sees the raw screenshot and outputs structured JSON with reasoning, observations, and actions. No OCR, no hardcoded coordinates, no predefined action vocabulary.

## Requirements

- Windows 10/11 with AoE2:DE installed
- Python 3.11+
- Anthropic API key

## Installation

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Set your Anthropic API key:

```bash
set ANTHROPIC_API_KEY=your-key-here
```

Optional environment variables:
- `AOE2_MODEL` - Claude model to use (default: claude-sonnet-4-5-20250929)
- `AOE2_LOOP_DELAY` - Seconds between decisions (default: 2.0)
- `AOE2_SAVE_SCREENSHOTS` - Save screenshots to logs/ (default: true)

## Usage

### Run the Agent

```bash
# Start the agent (runs until Ctrl+C)
python -m src.main

# Run specific number of iterations
python -m src.main --iterations 10

# Test mode - single iteration, no action execution
python -m src.main --test

# Specify provider (currently only claude available)
python -m src.main --provider claude
```

### Test Screen Capture

```python
from src.screen import capture_screenshot, save_screenshot

# Capture returns (bytes, width, height)
screenshot_bytes, width, height = capture_screenshot()
save_screenshot(screenshot_bytes, "test.jpg")
```

## Project Structure

```
agent/
├── src/                        # Core agent implementation
│   ├── main.py                 # Entry point, CLI argument parsing
│   ├── game_loop.py            # Main capture→think→act loop
│   ├── screen.py               # Screenshot capture (targets game window)
│   ├── executor.py             # Mouse/keyboard action execution
│   ├── config.py               # Settings via Pydantic
│   ├── window.py               # AoE2 window detection and focus
│   ├── memory.py               # Agent memory and game state tracking
│   ├── models.py               # Pydantic models for action validation
│   └── providers/
│       ├── base.py             # Provider interface (BaseLLMProvider)
│       └── claude.py           # Anthropic Claude implementation
├── detection/                  # YOLO entity detection system
│   ├── inference/              # Runtime detection (detector.py, models/)
│   ├── training/               # Training pipeline & config
│   ├── extraction/             # Sprite & screenshot extraction
│   ├── testing/                # Validation scripts
│   └── docs/                   # Detection documentation
├── data/                       # Game knowledge database
│   ├── game_knowledge.py       # Structured AoE2 game rules
│   └── aoe2.db                 # SQLite database
├── prompts/
│   └── system.md               # System prompt with game knowledge
├── logs/                       # Screenshots saved here
└── requirements.txt
```

See [detection/README.md](detection/README.md) for details on the entity detection system.

## Architecture

### Entity Detection (Optional)

When available, YOLO-based detection provides structured game understanding:

```python
from detection import get_detector

detector = get_detector()
entities = detector.detect(screenshot_bytes)
# Returns: [{"id": "sheep_0", "class": "sheep", "center": (x, y), "confidence": 0.95}, ...]
```

The LLM can reference entities by ID instead of pixel coordinates:
```json
{"type": "right_click", "target_id": "sheep_0"}
```

The executor automatically resolves IDs to screen coordinates.

### Memory System

The agent maintains memory across turns:
- **Working Memory** - Last 10 turns with reasoning and actions
- **Game State** - Tracked resources, population, age, alerts
- **Context Building** - Memory is summarized and sent to LLM each turn

### Action Validation

Actions are validated using Pydantic models before execution:
- `click` - Left click at (x, y)
- `right_click` - Right click at (x, y)
- `press` - Keyboard key press
- `drag` - Mouse drag from (x1, y1) to (x2, y2)
- `wait` - Delay in milliseconds

Coordinates are automatically translated from screenshot-relative to screen-absolute based on window position.

### Window Management

The agent automatically:
- Finds the AoE2 window by title
- Captures only the game window (not full screen)
- Ensures the game is focused before executing actions
- Falls back gracefully if window detection fails

## Adding New Providers

Create a new file in `src/providers/` implementing `BaseLLMProvider`:

```python
from .base import BaseLLMProvider

class MyProvider(BaseLLMProvider):
    async def get_actions(
        self,
        screenshot_bytes: bytes,
        context: str = "",
        width: int = 1920,
        height: int = 1080,
    ) -> dict:
        # Send screenshot to your LLM
        # Return {"reasoning": "...", "observations": {...}, "actions": [...]}
        pass

    def get_system_prompt(self) -> str:
        return "..."
```

Then register it in `src/main.py` in the `create_provider()` function.

## License

MIT
