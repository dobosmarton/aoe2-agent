"""WorldState dataclass — the canonical mid-game state of the synth arena.

Pure data; the simulator (economy, rendering, perception projection) lives
in `evaluation.world_sim` and operates on this type.
"""

from __future__ import annotations

from dataclasses import dataclass

AGE_SEQUENCE: list[str] = ["Dark Age", "Feudal Age", "Castle Age", "Imperial Age"]


@dataclass
class WorldState:
    food: float
    wood: float
    gold: float
    stone: float
    population: int
    pop_cap: int
    age: str  # "Dark Age" | "Feudal Age" | "Castle Age" | "Imperial Age"
    buildings: list[str]  # may contain duplicates (e.g. multiple houses)
    villager_queue: list[int]  # countdown ticks remaining per pending villager
    age_up_ticks_remaining: int  # 0 = not in progress
    turn: int = 0
