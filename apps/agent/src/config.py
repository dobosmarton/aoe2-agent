"""Configuration settings for the AoE2 LLM Agent."""

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

EffortLevel = Literal["low", "medium", "high"]  # Sonnet 4.6 rejects xhigh/max


def _parse_optional_int(value: str | None) -> int | None:
    """Parse an optional integer env var. Treats missing or empty as None."""
    if value is None or value == "":
        return None
    return int(value)


def _parse_effort(value: str | None) -> EffortLevel:
    """Parse the executor effort env var. Unknown or missing falls back to "low"."""
    if value in ("low", "medium", "high"):
        return value
    return "low"


class Config(BaseModel):
    """Agent configuration."""

    # Screenshot settings
    screenshot_quality: int = 85  # JPEG quality (1-100)

    # LLM settings
    anthropic_api_key: str = ""
    model: str = "claude-sonnet-4-6"  # Executor: better instruction following
    max_tokens: int = 1536
    max_tool_iterations: int = 7  # Max tool calls per game turn in agentic loop
    executor_effort: EffortLevel = "low"  # output_config effort for the executor tool loop
    strategist_model: str = "claude-sonnet-4-6"  # Strategist: deeper reasoning
    strategist_interval: int = 10  # Run strategist every N turns

    # Determinism knobs (Phase 3). Pin model snapshots via AOE2_MODEL /
    # AOE2_STRATEGIST_MODEL to a dated form (e.g. claude-sonnet-4-6-2026-XX-XX)
    # rather than the floating family alias for reproducible runs.
    temperature: float = 0.0  # Anthropic Messages API temperature (0.0 = lowest variance)
    seed: int | None = None  # Local RNG seed; None = OS entropy (today's behavior)

    # Detection settings
    detection_imgsz: int = 1280  # YOLO inference resolution (higher = more detections, slower)
    adaptive_sahi: bool = True  # Use adaptive SAHI (fast scan + targeted SAHI on entity clusters)
    full_sahi_interval: int = 5  # Force full SAHI scan every N turns
    detection_host: str = ""  # Remote CoreML server URL (e.g., "http://192.168.64.1:8420")

    # Timing settings
    loop_delay: float = (
        0.3  # Seconds between decisions (pipeline latency provides additional pacing)
    )
    action_delay: float = 0.05  # Seconds between actions

    # Phase 2 tuning
    pipeline_commit_max: int = 2  # S6: actions committed per pipelined turn (tail discarded)

    # Logging
    log_dir: Path = Path("logs")
    save_screenshots: bool = True

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        return cls(
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            model=os.environ.get("AOE2_MODEL", "claude-sonnet-4-6"),
            executor_effort=_parse_effort(os.environ.get("AOE2_EXECUTOR_EFFORT")),
            strategist_model=os.environ.get("AOE2_STRATEGIST_MODEL", "claude-sonnet-4-6"),
            strategist_interval=int(os.environ.get("AOE2_STRATEGIST_INTERVAL", "10")),
            loop_delay=float(os.environ.get("AOE2_LOOP_DELAY", "0.3")),
            save_screenshots=os.environ.get("AOE2_SAVE_SCREENSHOTS", "true").lower() == "true",
            detection_host=os.environ.get("AOE2_DETECTION_HOST", ""),
            temperature=float(os.environ.get("AOE2_TEMPERATURE", "0.0")),
            seed=_parse_optional_int(os.environ.get("AOE2_SEED")),
            pipeline_commit_max=int(os.environ.get("AOE2_PIPELINE_COMMIT_MAX", "2")),
        )


# Global config instance
config = Config.from_env()
