"""TypeSafe adapter tests use the SDK models but never call the service."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

import pytest
from gameplay_agent.policy.advice import PolicyGoal, PolicyRequest
from gameplay_agent.policy.candidates import feasible_candidates
from gameplay_agent.policy.state import PolicyState
from gameplay_agent.providers.policy import PolicyAdvisorError
from gameplay_agent.providers.typesafe_policy import TypeSafePolicyAdvisor
from typesafe_sdk import ChoiceAnswer, SystemOneResponse, Usage

if TYPE_CHECKING:
    from typesafe_sdk import JSONContent, Question


class FakeTypeSafeClient:
    def __init__(self, response: SystemOneResponse) -> None:
        self.response = response
        self.calls: list[tuple[JSONContent, Mapping[str, Question], str | None]] = []
        self.closed = False

    async def system_one(
        self,
        state: JSONContent,
        questions: Mapping[str, Question],
        *,
        model: str | None = None,
    ) -> SystemOneResponse:
        self.calls.append((state, questions, model))
        return self.response

    async def aclose(self) -> None:
        self.closed = True


def _as_mapping(value: object) -> Mapping[str, object]:
    """Narrow an SDK JSON value after validating the shape under test."""
    assert isinstance(value, Mapping)
    return cast("Mapping[str, object]", value)


def _request(state: PolicyState) -> PolicyRequest:
    return PolicyRequest(
        source_tick=7,
        source_captured_at=20.0,
        state=state,
        candidates=feasible_candidates(state),
        goals=(
            PolicyGoal(
                name="Reach Feudal",
                metric="age",
                target="Feudal Age",
                priority=9,
                progress=0.0,
            ),
        ),
    )


def _response(action: str = "build_house") -> SystemOneResponse:
    return SystemOneResponse(
        model="jev-1.13.0",
        usage=Usage(input_tokens=91, output_tokens=12),
        answers={
            "economy_action": ChoiceAnswer(
                choice=action,
                confidence=0.91,
                probabilities={action: 0.91, "wait": 0.09},
            ),
            "allocation_focus": ChoiceAnswer(
                choice="wood",
                confidence=0.78,
                probabilities={"wood": 0.78, "balanced": 0.22},
            ),
        },
    )


def test_adapter_batches_action_and_allocation_into_one_request() -> None:
    state = PolicyState(population=4, population_cap=5, wood=100, food=200)
    client = FakeTypeSafeClient(_response())
    advisor = TypeSafePolicyAdvisor(api_key="unused", model="jev-1.13.0", client=client)

    advice = asyncio.run(advisor.advise(_request(state)))

    assert advice.action_choice == "build_house"
    assert advice.allocation_focus == "wood"
    assert advice.input_tokens == 91
    assert advice.output_tokens == 12
    assert len(client.calls) == 1
    request_state, questions, model = client.calls[0]
    request_payload = _as_mapping(request_state)
    game = _as_mapping(request_payload["game"])
    resources = _as_mapping(game["resources"])
    active_goals = request_payload["active_goals"]
    assert isinstance(active_goals, list)
    first_goal = _as_mapping(active_goals[0])

    assert set(questions) == {"economy_action", "allocation_focus"}
    assert resources["wood"] == 100
    assert first_goal["name"] == "Reach Feudal"
    assert model == "jev-1.13.0"


def test_adapter_requires_a_server_side_credential() -> None:
    with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
        _ = TypeSafePolicyAdvisor(api_key="  ", model="jev-1.13.0")


def test_adapter_rejects_a_choice_outside_reviewed_candidates() -> None:
    state = PolicyState(population=4, population_cap=5, wood=100, food=200)
    client = FakeTypeSafeClient(_response(action="delete_town_center"))
    advisor = TypeSafePolicyAdvisor(api_key="unused", model="jev-1.13.0", client=client)

    with pytest.raises(PolicyAdvisorError, match="unavailable action"):
        _ = asyncio.run(advisor.advise(_request(state)))


def test_only_wait_is_selected_locally_without_an_action_question() -> None:
    state = PolicyState(population=4, population_cap=10)
    response = SystemOneResponse(
        model="jev-1.13.0",
        usage=Usage(input_tokens=35),
        answers={
            "allocation_focus": ChoiceAnswer(
                choice="balanced",
                confidence=0.8,
                probabilities={"balanced": 0.8},
            )
        },
    )
    client = FakeTypeSafeClient(response)
    advisor = TypeSafePolicyAdvisor(api_key="unused", model="jev-1.13.0", client=client)

    advice = asyncio.run(advisor.advise(_request(state)))

    assert advice.action_choice == "wait"
    assert set(client.calls[0][1]) == {"allocation_focus"}


def test_client_is_closed_at_the_lifecycle_boundary() -> None:
    client = FakeTypeSafeClient(_response())
    advisor = TypeSafePolicyAdvisor(api_key="unused", model="jev-1.13.0", client=client)

    asyncio.run(advisor.aclose())

    assert client.closed
