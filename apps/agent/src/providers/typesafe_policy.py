"""TypeSafe System One adapter for bounded economic policy judgments."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Protocol, cast

from typesafe_sdk import (
    AsyncTypeSafeClient,
    Choice,
    ChoiceAnswer,
    JSONContent,
    Question,
    RetryPolicy,
    SystemOneResponse,
    TypeSafeError,
)

from ..policy.advice import PolicyAdvice, readonly_probabilities
from ..policy.allocation import ALLOCATION_FOCI, is_famine
from .policy import PolicyAdvisorError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ..policy.advice import PolicyRequest
    from ..policy.allocation import AllocationFocus
    from ..policy.candidates import ActionCandidate, CandidateId

_ACTION_QUESTION = "economy_action"
_ALLOCATION_QUESTION = "allocation_focus"
_REQUEST_TIMEOUT_SECONDS = 2.0

_ALLOCATION_CRITERIA: Mapping[str, str] = {
    "balanced": "Keep the normal age-appropriate balance across useful resources.",
    "food": "Bias new or idle villagers toward food for production and age advancement.",
    "wood": "Bias new or idle villagers toward wood for buildings, farms and infrastructure.",
    "gold": "Bias new or idle villagers toward gold for age advancement and military units.",
    "stone": "Bias new or idle villagers toward stone for castles, towers or extra Town Centers.",
}


class _SystemOneClient(Protocol):
    async def system_one(
        self,
        state: JSONContent,
        questions: Mapping[str, Question],
        *,
        model: str | None = None,
    ) -> SystemOneResponse: ...

    async def aclose(self) -> None: ...


class TypeSafePolicyAdvisor:
    """Ask Jev two narrow questions and convert the response to domain data."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        client: _SystemOneClient | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("TYPESAFE_API_KEY is required for the actor policy")
        self._model = model
        if client is None:
            sdk_client = AsyncTypeSafeClient(
                api_key=api_key,
                model=model,
                retry=RetryPolicy(max_retries=0),
                timeout=_REQUEST_TIMEOUT_SECONDS,
            )
            # The SDK overload returns SystemOneResponse when response_model is
            # omitted. The Protocol records the narrower contract used here.
            self._client = cast("_SystemOneClient", sdk_client)
        else:
            self._client = client

    async def advise(self, request: PolicyRequest) -> PolicyAdvice:
        """Evaluate all useful questions for one frame in a single request."""
        try:
            response = await self._client.system_one(
                _request_state(request),
                _questions(request),
                model=self._model,
            )
        except TypeSafeError as exc:
            raise PolicyAdvisorError(str(exc)) from exc

        action = _action_answer(request, response)
        allocation = _required_choice(response, _ALLOCATION_QUESTION)
        candidate_ids: frozenset[CandidateId] = frozenset(
            candidate.id for candidate in request.candidates
        )
        return PolicyAdvice(
            source_tick=request.source_tick,
            source_captured_at=request.source_captured_at,
            created_at=time.monotonic(),
            model=response.model,
            candidate_ids=candidate_ids,
            action_choice=_candidate_id(action.choice, candidate_ids),
            action_confidence=action.confidence,
            action_probabilities=readonly_probabilities(action.probabilities),
            allocation_focus=_allocation_focus(allocation.choice),
            allocation_confidence=allocation.confidence,
            allocation_probabilities=readonly_probabilities(allocation.probabilities),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    async def aclose(self) -> None:
        await self._client.aclose()


def _questions(request: PolicyRequest) -> dict[str, Question]:
    questions: dict[str, Question] = {
        _ALLOCATION_QUESTION: Choice(
            instructions=(
                "Choose the resource focus that best advances `active_goals` from the current "
                "`game` state. The result changes only the normal allocation bias; emergency "
                "food protection remains deterministic."
            ),
            criteria=_ALLOCATION_CRITERIA,
        )
    }
    if len(request.candidates) > 1:
        questions[_ACTION_QUESTION] = Choice(
            instructions=(
                "Choose the single available economic action that best advances `active_goals` "
                "from the current `game` state. Every option is executable now. Choose `wait` "
                "when spending now would delay a more important goal."
            ),
            criteria={
                candidate.id: _candidate_description(candidate) for candidate in request.candidates
            },
        )
    return questions


def _request_state(request: PolicyRequest) -> JSONContent:
    state = request.state
    payload: dict[str, object] = {
        "game": {
            "age": state.age,
            "resources": {
                "food": state.food,
                "wood": state.wood,
                "gold": state.gold,
                "stone": state.stone,
            },
            "population": {
                "current": state.population,
                "capacity": state.population_cap,
                "villagers_ordered": state.villagers_ordered,
            },
            "buildings": sorted(state.buildings_seen),
            "villager_jobs": dict(state.villager_jobs),
            "idle_villagers_present": state.idle_present,
        },
        "computed_signals": {
            "food_crisis": is_famine(state),
            "population_blocked": state.population_cap > 0
            and state.villagers_ordered >= state.population_cap,
        },
        "active_goals": [
            {
                "name": goal.name,
                "metric": goal.metric,
                "target": goal.target,
                "priority": goal.priority,
                "progress": goal.progress,
            }
            for goal in request.goals
        ],
    }
    # Every nested value above is JSON-compatible; the cast bridges the SDK's
    # recursive JSON alias without weakening internal types to Any.
    return cast("JSONContent", payload)


def _candidate_description(candidate: ActionCandidate) -> str:
    if not candidate.cost:
        return candidate.description
    cost = ", ".join(f"{amount} {resource}" for resource, amount in candidate.cost.items())
    return f"{candidate.description} Immediate cost: {cost}."


def _action_answer(request: PolicyRequest, response: SystemOneResponse) -> ChoiceAnswer:
    if len(request.candidates) == 1:
        only = request.candidates[0]
        return ChoiceAnswer(choice=only.id, confidence=1.0, probabilities={only.id: 1.0})
    return _required_choice(response, _ACTION_QUESTION)


def _required_choice(response: SystemOneResponse, name: str) -> ChoiceAnswer:
    answer = response.choices.get(name)
    if answer is None:
        raise PolicyAdvisorError(f"TypeSafe response omitted {name!r}")
    return answer


def _candidate_id(choice: str, available: frozenset[CandidateId]) -> CandidateId:
    if choice not in available:
        raise PolicyAdvisorError(f"TypeSafe selected unavailable action {choice!r}")
    return cast("CandidateId", choice)


def _allocation_focus(choice: str) -> AllocationFocus:
    if choice not in ALLOCATION_FOCI:
        raise PolicyAdvisorError(f"TypeSafe selected unknown allocation focus {choice!r}")
    return cast("AllocationFocus", choice)


__all__ = ["TypeSafePolicyAdvisor"]
