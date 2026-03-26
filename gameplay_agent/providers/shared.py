"""Shared utilities for LLM providers."""

from functools import lru_cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"


@lru_cache(maxsize=8)
def load_system_prompt(filename: str, *extra_files: str) -> str:
    """Load and concatenate prompt files from the prompts directory.

    Returns empty string if the primary file is not found.
    Results are cached — prompt files do not change at runtime.
    """
    primary = PROMPTS_DIR / filename
    if not primary.exists():
        return ""
    parts = [primary.read_text()]
    for name in extra_files:
        path = PROMPTS_DIR / name
        if path.exists():
            parts.append(path.read_text())
    return "\n\n".join(parts)


def format_dimensions(width: int, height: int) -> str:
    """Format game window dimensions for LLM context."""
    center_x = width // 2
    center_y = height // 2
    return f"Game window: {width}x{height} pixels. Center=({center_x},{center_y}). Valid x=0-{width}, y=0-{height}."


def cached_system_block(text: str) -> list[dict]:
    """Build an Anthropic system message block with prompt caching."""
    return [{
        "type": "text",
        "text": text,
        "cache_control": {"type": "ephemeral"},
    }]
