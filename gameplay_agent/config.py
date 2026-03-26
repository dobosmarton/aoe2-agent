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
    max_tokens: int = 512  # Token budget for agentic tool loop
    batch_max_tokens: int = 384  # Token budget for batch mode (smaller schema = fewer tokens needed)
    max_tool_iterations: int = 4  # Max tool calls per game turn in agentic loop
    strategist_model: str = "claude-sonnet-4-6"  # Strategist: deeper reasoning
    strategist_interval: int = 10  # Run strategist every N turns

    # Provider settings
    provider: str = "claude"  # "claude" or "ollama"
    max_retries: int = 3  # Anthropic client retry count (429/5xx with exponential backoff)
    strategist_max_tokens: int = 768  # Strategist uses more tokens for deeper reasoning
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    ollama_timeout: float = 30.0  # Ollama HTTP request timeout in seconds

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
            provider=os.environ.get("AOE2_PROVIDER", "claude"),
            max_retries=int(os.environ.get("AOE2_MAX_RETRIES", "3")),
            batch_max_tokens=int(os.environ.get("AOE2_BATCH_MAX_TOKENS", "384")),
            strategist_max_tokens=int(os.environ.get("AOE2_STRATEGIST_MAX_TOKENS", "768")),
            ollama_host=os.environ.get("AOE2_OLLAMA_HOST", "http://localhost:11434"),
            ollama_model=os.environ.get("AOE2_OLLAMA_MODEL", "qwen2.5:7b"),
            ollama_timeout=float(os.environ.get("AOE2_OLLAMA_TIMEOUT", "30.0")),
        )


# Global config instance
config = Config.from_env()
