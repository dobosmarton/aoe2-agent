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


def test_house_requires_a_known_cap_and_executor_safe_headroom() -> None:
    unknown = PolicyState(population=0, population_cap=0, wood=25)
    too_early = PolicyState(population=5, population_cap=10, wood=25)
    useful = PolicyState(population=6, population_cap=10, wood=25)

    assert "build_house" not in _ids(unknown)
    assert "build_house" not in _ids(too_early)
    assert "build_house" in _ids(useful)


def test_render_returns_fresh_mutable_action_dictionaries() -> None:
    house = next(
        candidate
        for candidate in feasible_candidates(PolicyState(population=4, population_cap=5, wood=25))
        if candidate.id == "build_house"
    )

    first = house.render()
    first[0]["intent"] = "changed"

    assert house.render()[0]["intent"] == "Build house (TypeSafe)"
