"""Configuration settings for the AoE2 LLM Agent."""

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

EffortLevel = Literal["low", "medium", "high"]  # Sonnet 4.6 rejects xhigh/max

# Villager build menus whose slot layout has been verified in-game on THIS
# machine. Only the economic menu ships verified; the military ("w") and
# more-buildings ("v") menus stay rejected until the VM check, because a wrong
# slot doesn't no-op — it builds whatever occupies that position (runs 6-7,
# 14 outposts). See executor._BUILD_ENTRIES.
_DEFAULT_VERIFIED_BUILD_MENUS: frozenset[str] = frozenset({"q"})


def _parse_optional_int(value: str | None) -> int | None:
    """Parse an optional integer env var. Treats missing or empty as None."""
    if value is None or value == "":
        return None
    return int(value)


def _parse_menu_keys(value: str | None) -> frozenset[str]:
    """Parse a comma-separated build-menu allowlist. Missing = economic only."""
    if value is None:
        return _DEFAULT_VERIFIED_BUILD_MENUS
    return frozenset(part.strip().lower() for part in value.split(",") if part.strip())


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
    # Resource bar is read locally (Claude vision dropped). Backend: rapidocr
    # (pip-only, runs on onnxruntime) | tesseract (needs binary) | template.
    ocr_backend: str = "rapidocr"  # AOE2_OCR_BACKEND

    # Determinism knobs (Phase 3). Pin model snapshots via AOE2_MODEL /
    # AOE2_STRATEGIST_MODEL to a dated form (e.g. claude-sonnet-4-6-2026-XX-XX)
    # rather than the floating family alias for reproducible runs.
    temperature: float = 0.0  # Anthropic Messages API temperature (0.0 = lowest variance)
    seed: int | None = None  # Local RNG seed; None = OS entropy (today's behavior)

    # Detection settings
    # Inference resolution must match the served model's training res. v9
    # (YOLO26n) is trained at 1280 and serves at 1280 for better small-object
    # recall (villagers, sheep, mills). NOTE: the ONNX must be a *static* 1280
    # export (export_onnx.py --no-dynamic) — the CoreML/ANE provider cannot build
    # a plan for a dynamic graph at 1280. A static graph accepts only this one
    # size, so every /detect request uses it (see remote_detector.detect_fast_multi).
    # History: v6 was trained @640; upscaling it to 1280 tanked F1 (0.42→0.21) —
    # that penalty is a train/inference scale mismatch, not a property of 1280.
    detection_imgsz: int = 1280
    # Served YOLO model — SINGLE SOURCE OF TRUTH for the version the agent runs
    # locally (detection.inference.detector resolves this name in its bundled
    # models dir). The remote detection server picks its model via --model at
    # launch; keep the two in sync. detection_imgsz above must match this
    # model's training resolution (v9 trained @1280).
    detection_model: str = "aoe2_yolo_v9"  # AOE2_DETECTION_MODEL
    adaptive_sahi: bool = False  # SAHI hurts v6 at retina res (scale mismatch); single-pass wins
    full_sahi_interval: int = 5  # Force full SAHI scan every N turns (only if adaptive_sahi=True)
    detection_host: str = ""  # Remote CoreML server URL (e.g., "http://192.168.64.1:8420")

    # Build menus the executor may open (T-544). Set
    # AOE2_VERIFIED_BUILD_MENUS="q,v" once the VM check confirms the
    # more-buildings layout; until then the Castle age-up waits.
    verified_build_menus: frozenset[str] = _DEFAULT_VERIFIED_BUILD_MENUS

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
            ocr_backend=os.environ.get("AOE2_OCR_BACKEND", "rapidocr"),
            loop_delay=float(os.environ.get("AOE2_LOOP_DELAY", "0.3")),
            save_screenshots=os.environ.get("AOE2_SAVE_SCREENSHOTS", "true").lower() == "true",
            detection_host=os.environ.get("AOE2_DETECTION_HOST", ""),
            detection_model=os.environ.get("AOE2_DETECTION_MODEL", "aoe2_yolo_v9"),
            temperature=float(os.environ.get("AOE2_TEMPERATURE", "0.0")),
            seed=_parse_optional_int(os.environ.get("AOE2_SEED")),
            pipeline_commit_max=int(os.environ.get("AOE2_PIPELINE_COMMIT_MAX", "2")),
            verified_build_menus=_parse_menu_keys(os.environ.get("AOE2_VERIFIED_BUILD_MENUS")),
        )


# Global config instance
config = Config.from_env()
