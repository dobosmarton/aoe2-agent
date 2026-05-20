"""ConfigProfile YAML schema for the synthetic arena racing harness (Phase 6).

A profile YAML enumerates named variants, each specifying the LLM knobs
that meaningfully vary behaviour in the synth tier. The loader produces
a frozen `RaceConfig` consumable by `arena.race.race()`.

Example YAML (arena/profiles/v1.yaml):
    turns: 50
    profiles:
      - name: haiku-deterministic
        model: claude-haiku-4-5-20251001
        temperature: 0.0
        seed: 42
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from pathlib import Path


class ConfigProfile(BaseModel):
    """One racing variant — a named slice of LLM knobs for the synth tier."""

    model_config = ConfigDict(frozen=True)

    name: str
    model: str = "claude-haiku-4-5-20251001"
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    seed: int | None = None


class RaceConfig(BaseModel):
    """Top-level YAML structure consumed by `just arena-race`."""

    model_config = ConfigDict(frozen=True)

    turns: int = Field(default=50, ge=1)
    profiles: list[ConfigProfile] = Field(min_length=2)

    @classmethod
    def from_yaml(cls, path: Path) -> RaceConfig:
        import yaml

        raw: object = yaml.safe_load(path.read_text())  # pyright: ignore[reportAny]
        return cls.model_validate(raw)
