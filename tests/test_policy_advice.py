"""The handoff between the asynchronous advisor and synchronous actor."""

from __future__ import annotations

from gameplay_agent.policy.advice import (
    PolicyAdvice,
    PolicyAdviceStore,
    readonly_probabilities,
)
from gameplay_agent.policy.candidates import ActionCandidate, feasible_candidates
from gameplay_agent.policy.state import PolicyState


def _advice(
    *,
    tick: int = 3,
    captured_at: float = 10.0,
    action: str = "build_house",
    action_confidence: float = 0.9,
    allocation_confidence: float = 0.8,
) -> PolicyAdvice:
    return PolicyAdvice(
        source_tick=tick,
        source_captured_at=captured_at,
        created_at=captured_at + 0.1,
        model="jev-1.13.0",
        candidate_ids=frozenset({"wait", "build_house"}),
        action_choice=action,
        action_confidence=action_confidence,
        action_probabilities=readonly_probabilities({action: action_confidence}),
        allocation_focus="wood",
        allocation_confidence=allocation_confidence,
        allocation_probabilities=readonly_probabilities({"wood": allocation_confidence}),
        input_tokens=42,
    )


def _candidates() -> tuple[ActionCandidate, ...]:
    return feasible_candidates(PolicyState(population=4, population_cap=5, wood=25))


def test_action_advice_is_consumed_once_while_allocation_remains_available() -> None:
    store = PolicyAdviceStore()
    store.publish(_advice())

    first = store.consume(_candidates(), now=10.5, ttl_seconds=2.0, minimum_confidence=0.65)
    second = store.consume(_candidates(), now=10.6, ttl_seconds=2.0, minimum_confidence=0.65)

    assert first.status == "applicable"
    assert first.action is not None and first.action.id == "build_house"
    assert second.status == "consumed"
    assert second.action is None
    assert second.allocation_focus == "wood"


def test_stale_advice_requests_the_rule_fallback() -> None:
    store = PolicyAdviceStore()
    store.publish(_advice())

    resolution = store.inspect(_candidates(), now=12.1, ttl_seconds=2.0, minimum_confidence=0.65)

    assert resolution.status == "stale"
    assert resolution.should_fallback


def test_low_action_confidence_requests_the_rule_fallback() -> None:
    store = PolicyAdviceStore()
    store.publish(_advice(action_confidence=0.4))

    resolution = store.inspect(_candidates(), now=10.5, ttl_seconds=2.0, minimum_confidence=0.65)

    assert resolution.status == "low_confidence"
    assert resolution.should_fallback


def test_low_allocation_confidence_does_not_reject_a_confident_action() -> None:
    store = PolicyAdviceStore()
    store.publish(_advice(allocation_confidence=0.4))

    resolution = store.inspect(_candidates(), now=10.5, ttl_seconds=2.0, minimum_confidence=0.65)

    assert resolution.status == "applicable"
    assert resolution.action is not None
    assert resolution.allocation_focus is None


def test_advice_cannot_select_an_action_that_is_no_longer_feasible() -> None:
    store = PolicyAdviceStore()
    store.publish(_advice())
    changed = feasible_candidates(PolicyState(population=4, population_cap=10, wood=0))

    resolution = store.inspect(changed, now=10.5, ttl_seconds=2.0, minimum_confidence=0.65)

    assert resolution.status == "candidate_unavailable"
    assert resolution.should_fallback


def test_older_response_cannot_replace_newer_advice() -> None:
    store = PolicyAdviceStore()
    store.publish(_advice(tick=4, action="wait"))
    store.publish(_advice(tick=3))

    assert store.latest() is not None
    assert store.latest().source_tick == 4
    assert store.latest().action_choice == "wait"
