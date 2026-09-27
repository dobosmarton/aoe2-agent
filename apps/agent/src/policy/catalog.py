"""Named, bounded actions shared by policy choices and execution adapters.

Bindings describe the shipped AoE2 hotkey profile. They still require an
in-game smoke test: static metadata cannot prove a player's layout matches it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Resource = Literal["food", "wood", "gold", "stone"]
ActionKind = Literal["wait", "build", "research", "train", "assign", "handoff"]


@dataclass(frozen=True, slots=True)
class ActionSpec:
    id: str
    description: str
    kind: ActionKind
    cost: tuple[tuple[Resource, int], ...] = ()
    age: str = "Dark Age"
    requires: frozenset[str] = frozenset()
    subject: str = ""
    menu: str = ""
    key: str = ""
    goto_key: str = ""
    goto_modifiers: tuple[str, ...] = ()
    unique: bool = False

    def price(self, resource: Resource) -> int:
        return next((amount for kind, amount in self.cost if kind == resource), 0)


SPECS: tuple[ActionSpec, ...] = (
    ActionSpec("wait", "Preserve resources while the economy gathers.", "wait"),
    ActionSpec(
        "advance_to_feudal",
        "Research Feudal Age after the Dark Age buildings.",
        "research",
        (("food", 500),),
        subject="feudal_age",
        key="z",
        goto_key="h",
    ),
    ActionSpec(
        "advance_to_castle",
        "Research Castle Age after two Feudal buildings.",
        "research",
        (("food", 800), ("gold", 200)),
        age="Feudal Age",
        subject="castle_age",
        key="z",
        goto_key="h",
    ),
    ActionSpec(
        "advance_to_imperial",
        "Research Imperial Age after two Castle buildings.",
        "research",
        (("food", 1000), ("gold", 800)),
        age="Castle Age",
        subject="imperial_age",
        key="z",
        goto_key="h",
    ),
    ActionSpec(
        "build_house",
        "Add population capacity when production needs room.",
        "build",
        (("wood", 25),),
        subject="house",
        menu="q",
        key="q",
    ),
    ActionSpec(
        "build_mill",
        "Build a mill to unlock farms and Feudal progress.",
        "build",
        (("wood", 100),),
        subject="mill",
        menu="q",
        key="w",
        unique=True,
    ),
    ActionSpec(
        "build_lumber_camp",
        "Build a lumber camp at a visible forest.",
        "build",
        (("wood", 100),),
        subject="lumber_camp",
        menu="q",
        key="r",
        unique=True,
    ),
    ActionSpec(
        "build_mining_camp",
        "Build a mining camp at a visible mine.",
        "build",
        (("wood", 100),),
        subject="mining_camp",
        menu="q",
        key="e",
        unique=True,
    ),
    ActionSpec(
        "build_farm",
        "Build one farm when the mill stands and food sources are scarce.",
        "build",
        (("wood", 60),),
        subject="farm",
        menu="q",
        key="a",
        requires=frozenset({"mill"}),
    ),
    ActionSpec(
        "build_barracks",
        "Build a barracks for infantry and Feudal military access.",
        "build",
        (("wood", 175),),
        subject="barracks",
        menu="w",
        key="q",
        unique=True,
    ),
    ActionSpec(
        "build_archery_range",
        "Build an archery range after a barracks.",
        "build",
        (("wood", 175),),
        age="Feudal Age",
        requires=frozenset({"barracks"}),
        subject="archery_range",
        menu="w",
        key="w",
        unique=True,
    ),
    ActionSpec(
        "build_stable",
        "Build a stable after a barracks.",
        "build",
        (("wood", 175),),
        age="Feudal Age",
        requires=frozenset({"barracks"}),
        subject="stable",
        menu="w",
        key="e",
        unique=True,
    ),
    ActionSpec(
        "build_blacksmith",
        "Build a blacksmith for Castle prerequisites and upgrades.",
        "build",
        (("wood", 150),),
        age="Feudal Age",
        subject="blacksmith",
        menu="q",
        key="s",
        unique=True,
    ),
    ActionSpec(
        "build_market",
        "Build a market for Castle prerequisites and trade.",
        "build",
        (("wood", 175),),
        age="Feudal Age",
        subject="market",
        menu="v",
        key="d",
        unique=True,
    ),
    ActionSpec(
        "build_siege_workshop",
        "Build a siege workshop in Castle Age.",
        "build",
        (("wood", 200),),
        age="Castle Age",
        subject="siege_workshop",
        menu="w",
        key="r",
        unique=True,
    ),
    ActionSpec(
        "build_monastery",
        "Build a monastery in Castle Age.",
        "build",
        (("wood", 175),),
        age="Castle Age",
        subject="monastery",
        menu="w",
        key="f",
        unique=True,
    ),
    ActionSpec(
        "queue_villager",
        "Queue one villager when food and housing permit.",
        "train",
        (("food", 50),),
        subject="villager",
        key="q",
        goto_key="h",
    ),
    ActionSpec(
        "train_militia",
        "Train one militia at a barracks.",
        "train",
        (("food", 50), ("gold", 20)),
        subject="militia",
        key="q",
        goto_key="b",
        goto_modifiers=("ctrl",),
        requires=frozenset({"barracks"}),
    ),
    ActionSpec(
        "train_spearman",
        "Train one spearman at a barracks.",
        "train",
        (("food", 35), ("wood", 25)),
        age="Feudal Age",
        subject="spearman",
        key="w",
        goto_key="b",
        goto_modifiers=("ctrl",),
        requires=frozenset({"barracks"}),
    ),
    ActionSpec(
        "train_archer",
        "Train one archer at an archery range.",
        "train",
        (("wood", 25), ("gold", 45)),
        age="Feudal Age",
        subject="archer",
        key="q",
        goto_key="a",
        goto_modifiers=("ctrl",),
        requires=frozenset({"archery_range"}),
    ),
    ActionSpec(
        "train_skirmisher",
        "Train one skirmisher at an archery range.",
        "train",
        (("food", 25), ("wood", 35)),
        age="Feudal Age",
        subject="skirmisher",
        key="w",
        goto_key="a",
        goto_modifiers=("ctrl",),
        requires=frozenset({"archery_range"}),
    ),
    ActionSpec(
        "train_scout_cavalry",
        "Train one scout cavalry at a stable.",
        "train",
        (("food", 80),),
        age="Feudal Age",
        subject="scout_cavalry",
        key="q",
        goto_key="l",
        goto_modifiers=("ctrl",),
        requires=frozenset({"stable"}),
    ),
    ActionSpec(
        "train_knight",
        "Train one knight at a stable.",
        "train",
        (("food", 60), ("gold", 75)),
        age="Castle Age",
        subject="knight",
        key="w",
        goto_key="l",
        goto_modifiers=("ctrl",),
        requires=frozenset({"stable"}),
    ),
    ActionSpec(
        "research_loom",
        "Research Loom at the Town Center.",
        "research",
        (("gold", 50),),
        subject="loom",
        key="a",
        goto_key="h",
    ),
    ActionSpec(
        "research_wheelbarrow",
        "Research Wheelbarrow at the Town Center.",
        "research",
        (("food", 175), ("wood", 50)),
        age="Feudal Age",
        subject="wheelbarrow",
        key="s",
        goto_key="h",
    ),
    ActionSpec(
        "research_horse_collar",
        "Research Horse Collar at the mill.",
        "research",
        (("food", 75), ("wood", 75)),
        age="Feudal Age",
        subject="horse_collar",
        key="q",
        goto_key="i",
        goto_modifiers=("ctrl",),
        requires=frozenset({"mill"}),
    ),
    ActionSpec(
        "research_double_bit_axe",
        "Research Double-Bit Axe at the lumber camp.",
        "research",
        (("food", 100), ("wood", 50)),
        age="Feudal Age",
        subject="double_bit_axe",
        key="q",
        goto_key="z",
        goto_modifiers=("ctrl",),
        requires=frozenset({"lumber_camp"}),
    ),
    ActionSpec(
        "research_gold_mining",
        "Research Gold Mining at the mining camp.",
        "research",
        (("food", 100), ("wood", 75)),
        age="Feudal Age",
        subject="gold_mining",
        key="q",
        goto_key="g",
        goto_modifiers=("ctrl",),
        requires=frozenset({"mining_camp"}),
    ),
    ActionSpec("assign_food", "Send one idle villager to visible food.", "assign", subject="food"),
    ActionSpec("assign_wood", "Send one idle villager to visible wood.", "assign", subject="wood"),
    ActionSpec("assign_gold", "Send one idle villager to visible gold.", "assign", subject="gold"),
    ActionSpec(
        "assign_stone", "Send one idle villager to visible stone.", "assign", subject="stone"
    ),
    ActionSpec(
        "tactical_handoff", "Ask the deliberate controller to use the existing army.", "handoff"
    ),
)

BY_ID = {spec.id: spec for spec in SPECS}
BUILDINGS = {(spec.menu, spec.key): spec for spec in SPECS if spec.kind == "build"}
RESEARCH = {spec.subject: spec for spec in SPECS if spec.kind == "research"}
UNITS = {spec.subject: spec for spec in SPECS if spec.kind == "train"}

AGE_ORDER = ("Dark Age", "Feudal Age", "Castle Age", "Imperial Age")
DARK_BUILDINGS = frozenset({"mill", "lumber_camp", "mining_camp", "barracks"})
FEUDAL_BUILDINGS = frozenset({"archery_range", "stable", "blacksmith", "market"})
CASTLE_BUILDINGS = frozenset({"siege_workshop", "monastery", "university", "castle"})
