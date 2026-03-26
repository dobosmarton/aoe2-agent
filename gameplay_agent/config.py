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
    model: str = "claude-sonnet-4-6"  # Executor: better instruction following
    max_tokens: int = 1536
    max_tool_iterations: int = 7  # Max tool calls per game turn in agentic loop
    strategist_model: str = "claude-sonnet-4-6"  # Strategist: deeper reasoning
    strategist_interval: int = 10  # Run strategist every N turns

    # Detection settings
    detection_imgsz: int = 1280  # YOLO inference resolution (higher = more detections, slower)
    adaptive_sahi: bool = True   # Use adaptive SAHI (fast scan + targeted SAHI on entity clusters)
    full_sahi_interval: int = 5  # Force full SAHI scan every N turns
    detection_host: str = ""     # Remote CoreML server URL (e.g., "http://192.168.64.1:8420")

    # Timing settings
    loop_delay: float = 1.0  # Seconds between decisions
    action_delay: float = 0.05  # Seconds between actions

    # Logging
    log_dir: Path = Path("logs")
    save_screenshots: bool = True

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        return cls(
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            model=os.environ.get("AOE2_MODEL", "claude-sonnet-4-6"),
            strategist_model=os.environ.get("AOE2_STRATEGIST_MODEL", "claude-sonnet-4-6"),
            strategist_interval=int(os.environ.get("AOE2_STRATEGIST_INTERVAL", "10")),
            loop_delay=float(os.environ.get("AOE2_LOOP_DELAY", "1.0")),
            save_screenshots=os.environ.get("AOE2_SAVE_SCREENSHOTS", "true").lower() == "true",
            detection_host=os.environ.get("AOE2_DETECTION_HOST", ""),
        )


# Global config instance
config = Config.from_env()
