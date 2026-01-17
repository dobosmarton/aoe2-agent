"""Configuration settings for the AoE2 LLM Agent."""

import os
from pathlib import Path
from pydantic import BaseModel


class Config(BaseModel):
    """Agent configuration."""

    # Screenshot settings
    screenshot_quality: int = 85  # JPEG quality (1-100)

    # LLM settings
    anthropic_api_key: str = ""
    model: str = "claude-sonnet-4-5-20250929"
    max_tokens: int = 1024

    # Timing settings
    loop_delay: float = 2.0  # Seconds between decisions
    action_delay: float = 0.05  # Seconds between actions

    # Logging
    log_dir: Path = Path("logs")
    save_screenshots: bool = True

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        return cls(
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            model=os.environ.get("AOE2_MODEL", "claude-sonnet-4-5-20250929"),
            loop_delay=float(os.environ.get("AOE2_LOOP_DELAY", "2.0")),
            save_screenshots=os.environ.get("AOE2_SAVE_SCREENSHOTS", "true").lower() == "true",
        )


# Global config instance
config = Config.from_env()
