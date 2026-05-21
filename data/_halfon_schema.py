"""Pydantic boundary models for the halfon.aoe2.se JSON API.

Validated once at the network boundary (`HalfonResponse.model_validate_json`)
so the rest of the codebase deals in typed `HalfonEntity` instances rather
than `dict[str, object]` with hand-rolled `_as_int` / `_as_str` narrowing.

The API itself is loosely typed (any field may be missing or have an
unexpected runtime type), so every field has a default and we set
`extra="ignore"` to tolerate fields we don't model.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class HalfonCost(BaseModel):
    """Resource cost for a unit or building."""

    model_config = ConfigDict(extra="ignore")

    food: int = 0
    wood: int = 0
    gold: int = 0
    stone: int = 0


class HalfonEntity(BaseModel):
    """One entity (unit or building) from the halfon dataset.

    The halfon JSON groups units and buildings together under
    `units_buildings`; classification happens downstream via heuristics
    in `data.game_knowledge._is_building`.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    name: str = ""
    localised_name: str = ""
    localized_name: str = ""  # halfon spells it both ways across endpoints
    hit_points: int = 0
    attack: int = 0
    melee_armor: int = 0
    pierce_armor: int = 0
    range: int = 0
    train_time: int = 0
    build_time: int = 0
    line_of_sight: int = 4
    garrison_capacity: int = 0
    type: int = 0
    # `class` is a Python keyword — accept it as input via alias, expose as class_.
    class_: int = Field(0, alias="class")
    age: str | int | None = None
    cost: HalfonCost = Field(default_factory=HalfonCost)

    @property
    def display_name(self) -> str:
        """The first non-empty of (localised_name, localized_name, name)."""
        return self.localised_name or self.localized_name or self.name


class HalfonResponse(BaseModel):
    """Root response from halfon.aoe2.se/data/units_buildings_techs.de.json."""

    model_config = ConfigDict(extra="ignore")

    units_buildings: dict[str, HalfonEntity] = Field(default_factory=dict)
