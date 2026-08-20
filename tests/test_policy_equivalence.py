"""Golden-file regression guard for the policy engine.

Recorded at the commit where the engine was proven equivalent to the deleted
`reactive.py`: exact match on 3015 affordable states, and a strict subsequence
of reactive's output on all 6000. The only difference was the reservation step,
which drops builds the executor would have rejected anyway.

The sweep caught the mill/camp exclusivity bug during the rewrite, when two
independent rules emitted both builds where the `if/elif` emitted one.

Regenerate with `python -m tests.test_policy_equivalence` after a DELIBERATE
behavior change, and review the diff.
"""

from __future__ import annotations

import itertools
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

from gameplay_agent.memory import GameState
from gameplay_agent.policy.engine import decide
from gameplay_agent.policy.state import from_game_state

from tests.factories import make_entity as _ent

GOLDEN = Path(__file__).parent / "golden" / "policy_decisions.json"

_AGES = ("Dark Age", "Feudal Age", "Castle Age", "Imperial Age")
_BUILDING_SETS = (
    frozenset(),
    frozenset({"mill"}),
    frozenset({"mill", "lumber_camp"}),
    frozenset({"mill", "lumber_camp", "mining_camp"}),
    frozenset({"lumber_camp"}),
)
_SEED = 7
_CASES_PER_COMBINATION = 60


def _entity_sets() -> tuple[list[object], list[object]]:
    return [], [
        _ent("town_center", (0, 0)),
        _ent("sheep", (10, 10)),
        _ent("tree", (20, 20)),
        _ent("gold_mine", (30, 30)),
    ]


# Hand-picked states on rule boundaries. The random sweep hit the mill/camp
# exclusivity bug in only 2 of 1200 cases, which is too thin to depend on.
@dataclass(frozen=True, slots=True)
class _EdgeCase:
    """One boundary state, named so the golden key stays readable."""

    name: str
    age: str
    population: int
    cap: int
    food: int
    wood: int
    buildings: frozenset[str] = frozenset()

    def to_game_state(self) -> GameState:
        return GameState(
            resources={"food": self.food, "wood": self.wood, "gold": 100, "stone": 200},
            population=self.population,
            population_cap=self.cap,
            current_age=self.age,
            buildings_seen=self.buildings,
            villagers_ordered=10,
        )


_DARK = "Dark Age"
_MILL = frozenset({"mill"})
_BOTH_PREREQS = frozenset({"mill", "lumber_camp"})

_EDGE_CASES: tuple[_EdgeCase, ...] = (
    # Both prereqs missing and wood covers BOTH builds, so only the trigger —
    # not the reservation — can keep the camp from riding along with the mill.
    _EdgeCase("prep-both-missing", _DARK, 12, 30, 200, 400),
    _EdgeCase("prep-both-missing-rich", _DARK, 20, 30, 200, 999),
    # Mill stands, camp does not: the camp must now fire.
    _EdgeCase("prep-camp-next", _DARK, 12, 30, 200, 400, _MILL),
    # Below the prep population gate.
    _EdgeCase("prep-below-pop", _DARK, 11, 30, 200, 400),
    # Age-up boundary, both prereqs standing.
    _EdgeCase("age-up-affordable", _DARK, 22, 30, 500, 200, _BOTH_PREREQS),
    _EdgeCase("age-up-one-food-short", _DARK, 22, 30, 499, 200, _BOTH_PREREQS),
    # Housed, with wood for a house and a prep build at once.
    _EdgeCase("housed-and-prep-pending", _DARK, 30, 30, 200, 400),
)


def _cases() -> list[tuple[str, list[dict[str, object]]]]:
    """Every (state, actions) pair, keyed by a stable description of the state."""
    rng = random.Random(_SEED)
    entity_sets = _entity_sets()
    out: list[tuple[str, list[dict[str, object]]]] = []
    for case in _EDGE_CASES:
        fresh = from_game_state(case.to_game_state(), captured_at=time.monotonic())
        actions = decide([], fresh, alarm=False)
        out.append((f"edge|{case.name}", actions))
    for age, buildings in itertools.product(_AGES, _BUILDING_SETS):
        for _ in range(_CASES_PER_COMBINATION):
            cap = rng.choice([5, 10, 15, 25, 30, 200])
            state = GameState(
                resources={
                    "food": rng.choice([0, 40, 60, 200, 520, 900]),
                    "wood": rng.choice([0, 15, 60, 80, 100, 200, 400]),
                    "gold": rng.choice([0, 90, 200]),
                    "stone": 200,
                },
                population=rng.randint(1, cap),
                population_cap=cap,
                current_age=age,
                buildings_seen=buildings,
                villagers_ordered=rng.choice([4, 10, 29, 30, 34, 35, 60]),
                idle_present=rng.choice([None, False, True]),
                idle_count=rng.choice([None, 0, 1, 3, 8]),
                idle_streak=rng.choice([0, 5]),
            )
            entities = entity_sets[rng.randrange(len(entity_sets))]
            key = (
                f"{age}|{','.join(sorted(buildings))}|pop={state.population}/{cap}"
                f"|food={state.resources['food']}|wood={state.resources['wood']}"
                f"|gold={state.resources['gold']}|ord={state.villagers_ordered}"
                f"|idle={state.idle_present},{state.idle_count},{state.idle_streak}"
                f"|ents={len(entities)}"
            )
            fresh = from_game_state(state, captured_at=time.monotonic())
            out.append((key, decide(entities, fresh, alarm=False)))
    return out


def test_engine_matches_the_recorded_golden_set() -> None:
    recorded = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert dict(_cases()) == recorded


def test_golden_set_covers_every_shipped_rule() -> None:
    """A rule absent from every case is untested by this harness."""
    from gameplay_agent.policy.engine import registry

    fired = {str(action.get("intent", "")) for _, actions in _cases() for action in actions}
    for rule in registry():
        intents = {str(a.get("intent", "")).split(" (headroom")[0] for a in rule.actions}
        assert any(i in f for i in intents for f in fired), rule.id


def _regenerate() -> None:
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(dict(_cases()), indent=1, sort_keys=True)
    GOLDEN.write_text(f"{body}\n", encoding="utf-8")  # trailing newline: pre-commit hook
    print(f"wrote {GOLDEN}")


if __name__ == "__main__":
    _regenerate()
