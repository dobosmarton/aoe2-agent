"""Validated boundaries for the first supported screen-control game setup."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from pathlib import Path

Relationship = Literal["own", "ally", "enemy"]
PlayerColor = Literal["blue", "red", "green", "yellow", "cyan", "purple", "gray", "orange"]


class PlayerSlot(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    number: int = Field(ge=1, le=8)
    name: str = Field(min_length=1)
    color: PlayerColor
    team: int = Field(ge=1)
    relationship: Relationship


class GameProfile(BaseModel):
    """One roster and HUD/hotkey setup; no implied team from a unit's color."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    civilization: Literal["Magyars"]
    map_name: Literal["Arabia"]
    resources: Literal["standard"]
    population_limit: Literal[200]
    locked_teams: Literal[True]
    language: Literal["en"]
    capture_width: Literal[3024]
    capture_height: Literal[1672]
    ui_scale: float = Field(gt=0)
    hotkey_profile: str = Field(min_length=1)
    game_data_version: str = Field(min_length=1)
    players: tuple[PlayerSlot, ...]
    roster_verified: bool = False
    hotkeys_verified: bool = False
    ownership_verified: bool = False

    @model_validator(mode="after")
    def validate_roster(self) -> GameProfile:
        if len(self.players) != 8:
            raise ValueError("the 4v4 roster must contain exactly eight player slots")
        if {player.number for player in self.players} != set(range(1, 9)):
            raise ValueError("the roster must record each player number once")
        if len({player.color for player in self.players}) != 8:
            raise ValueError("the roster must record each color once")
        if [player.relationship for player in self.players].count("own") != 1:
            raise ValueError("the roster must identify exactly one controlled player")
        if [player.relationship for player in self.players].count("ally") != 3:
            raise ValueError("the roster must identify three allies")
        if [player.relationship for player in self.players].count("enemy") != 4:
            raise ValueError("the roster must identify four enemies")
        own = next(player for player in self.players if player.relationship == "own")
        if own.number != 1 or own.color != "blue":
            raise ValueError("the controlled first-profile player must be blue player 1")
        if any(
            (player.team == own.team) != (player.relationship != "enemy") for player in self.players
        ):
            raise ValueError("locked-team relationships contradict recorded team numbers")
        return self


def load_profile(path: Path) -> GameProfile:
    """Parse one explicitly supplied, recorded lobby/profile file."""
    return GameProfile.model_validate_json(path.read_text(encoding="utf-8"))


__all__ = ["GameProfile", "PlayerSlot", "load_profile"]
