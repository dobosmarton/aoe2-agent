"""Programmatic facts bound the choices exposed to the dynamic policy."""

from gameplay_agent.policy.candidates import feasible_candidates
from gameplay_agent.policy.state import PolicyState


def _ids(state: PolicyState) -> set[str]:
    return {candidate.id for candidate in feasible_candidates(state)}


def test_unaffordable_actions_are_never_exposed_to_the_model() -> None:
    assert _ids(PolicyState(population=4, population_cap=10)) == {"wait"}


def test_exact_feudal_cost_and_prerequisites_stay_in_code() -> None:
    almost_ready = PolicyState(
        food=499,
        population=20,
        population_cap=30,
        buildings_seen=frozenset({"mill", "lumber_camp"}),
    )
    ready = PolicyState(
        food=500,
        population=20,
        population_cap=30,
        buildings_seen=frozenset({"mill", "lumber_camp"}),
    )

    assert "advance_to_feudal" not in _ids(almost_ready)
    assert "advance_to_feudal" in _ids(ready)


def test_house_requires_known_cap_worker_and_low_headroom() -> None:
    unknown = PolicyState(population=0, population_cap=0, wood=25, idle_present=True)
    without_worker = PolicyState(population=5, population_cap=10, wood=25)
    premature = PolicyState(population=5, population_cap=10, wood=25, idle_present=True)
    useful = PolicyState(population=6, population_cap=10, wood=25, idle_present=True)

    assert "build_house" not in _ids(unknown)
    assert "build_house" not in _ids(without_worker)
    assert "build_house" not in _ids(premature)
    assert "build_house" in _ids(useful)


def test_pending_house_is_hidden_until_settlement_finishes() -> None:
    pending = PolicyState(
        population=6,
        population_cap=10,
        wood=25,
        idle_present=True,
        pending_buildings=frozenset({"house"}),
    )
    settled = PolicyState(population=6, population_cap=10, wood=25, idle_present=True)

    assert ("build_house" in _ids(pending), "build_house" in _ids(settled)) == (False, True)


def test_pending_idle_assignment_hides_other_idle_assignments() -> None:
    state = PolicyState(
        population=4,
        population_cap=5,
        idle_present=True,
        assignment_pending=True,
        visible_classes=frozenset({"sheep", "tree"}),
    )

    assert not {"assign_food", "assign_wood"} & _ids(state)


def test_render_returns_fresh_mutable_action_dictionaries() -> None:
    house = next(
        candidate
        for candidate in feasible_candidates(
            PolicyState(population=4, population_cap=5, wood=25, idle_present=True)
        )
        if candidate.id == "build_house"
    )

    first = house.render()
    first[0]["intent"] = "changed"

    assert house.render()[0]["intent"] == "Build house (TypeSafe)"
