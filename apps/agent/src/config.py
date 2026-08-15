"""Configuration settings for the AoE2 LLM Agent."""

import os
from pathlib import Path
from typing import Final, Literal, get_args

from pydantic import BaseModel

EffortLevel = Literal["low", "medium", "high"]  # Sonnet 4.6 rejects xhigh/max
WireName = Literal["anthropic", "openai", "zen"]

# The one credential, whichever vendor is serving.
KEY_ENV = "AOE2_LLM_API_KEY"
WIRE_ENV = "AOE2_LLM_WIRE"
# Derived from the Literal so the valid set is written once.
_WIRES: Final[tuple[WireName, ...]] = get_args(WireName)


def _parse_optional_int(value: str | None) -> int | None:
    """Parse an optional integer env var. Treats missing or empty as None."""
    if value is None or value == "":
        return None
    return int(value)


def _parse_optional_float(value: str | None) -> float | None:
    """Parse an optional float env var. Treats missing or empty as None."""
    if value is None or not value.strip():
        return None
    return float(value)


def _parse_effort(value: str | None) -> EffortLevel:
    """Parse the executor effort env var. Unknown or missing falls back to "low"."""
    if value in ("low", "medium", "high"):
        return value
    return "low"


def _parse_wire(value: str | None) -> WireName:
    """Parse the wire env var. Missing defaults to openai; an unknown name raises.

    Unlike the other parsers, this one refuses to fall back: a wrong effort costs
    a knob, a wrong vendor costs a whole run. Config is built at import, so a bad
    value fails every module that imports it — which is the point.
    """
    if value is None or not value.strip():
        return "openai"
    normalized = value.strip().lower()
    if normalized not in _WIRES:
        expected = ", ".join(repr(wire) for wire in _WIRES)
        raise ValueError(f"unknown {WIRE_ENV}={value!r}; expected one of {expected}")
    return normalized


class Config(BaseModel):
    """Agent configuration."""

    # Screenshot settings
    screenshot_quality: int = 85  # JPEG quality (1-100)

    # LLM settings
    llm_wire: WireName = "openai"  # AOE2_LLM_WIRE; each adapter has its own endpoint
    # Empty means "use the adapter's own endpoint" — only set this to reach a
    # non-default host (a staging gateway, a self-hosted proxy).
    llm_base_url: str = ""  # AOE2_LLM_BASE_URL
    llm_api_key: str = ""  # AOE2_LLM_API_KEY — the only credential
    model: str = "gpt-5.6-luna"  # Executor
    max_tokens: int = 1536
    max_tool_iterations: int = 7  # Max tool calls per game turn in agentic loop
    executor_effort: EffortLevel = "low"  # reasoning effort for the executor tool loop
    strategist_model: str = "gpt-5.6-luna"  # Strategist: deeper reasoning
    strategist_interval: int = 10  # Run strategist every N turns
    # Post-game memory extraction and prompt mutation: a cheap model is enough.
    memory_model: str = "gpt-5.6-luna"  # AOE2_MEMORY_MODEL
    # Resource bar is read locally (Claude vision dropped). Backend: rapidocr
    # (pip-only, runs on onnxruntime) | tesseract (needs binary) | template.
    ocr_backend: str = "rapidocr"  # AOE2_OCR_BACKEND

    # Determinism knobs (Phase 3). Pin model snapshots via AOE2_MODEL /
    # AOE2_STRATEGIST_MODEL to a dated form rather than the floating family
    # alias for reproducible runs.
    # Unset means "do not send it", so each model applies its own default. The
    # gpt-5.6 family rejects every value but 1, so a hardcoded 0.0 made every
    # call a 400 (run 2026_08_15_1: 88 of 88 failed).
    temperature: float | None = None  # AOE2_TEMPERATURE
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
            llm_wire=_parse_wire(os.environ.get("AOE2_LLM_WIRE")),
            llm_base_url=os.environ.get("AOE2_LLM_BASE_URL", ""),
            llm_api_key=os.environ.get(KEY_ENV, ""),
            model=os.environ.get("AOE2_MODEL", "gpt-5.6-luna"),
            executor_effort=_parse_effort(os.environ.get("AOE2_EXECUTOR_EFFORT")),
            strategist_model=os.environ.get("AOE2_STRATEGIST_MODEL", "gpt-5.6-luna"),
            strategist_interval=int(os.environ.get("AOE2_STRATEGIST_INTERVAL", "10")),
            memory_model=os.environ.get("AOE2_MEMORY_MODEL", "gpt-5.6-luna"),
            ocr_backend=os.environ.get("AOE2_OCR_BACKEND", "rapidocr"),
            loop_delay=float(os.environ.get("AOE2_LOOP_DELAY", "0.3")),
            save_screenshots=os.environ.get("AOE2_SAVE_SCREENSHOTS", "true").lower() == "true",
            detection_host=os.environ.get("AOE2_DETECTION_HOST", ""),
            detection_model=os.environ.get("AOE2_DETECTION_MODEL", "aoe2_yolo_v9"),
            temperature=_parse_optional_float(os.environ.get("AOE2_TEMPERATURE")),
            seed=_parse_optional_int(os.environ.get("AOE2_SEED")),
            pipeline_commit_max=int(os.environ.get("AOE2_PIPELINE_COMMIT_MAX", "2")),
        )


# Global config instance
config = Config.from_env()
