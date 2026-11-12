"""The facts the strategist plans from: the situation, the gap to the next age, and the rules.

Everything here is read from the agent's own records or the action catalog. None
of it says what to do next; the strategist works that out from the objective.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .entity_utils import GATHER_CLASSES_BY_KIND, RESOURCE_KINDS
from .policy.allocation import gatherable_kinds
from .policy.catalog import AGE_ORDER, AGE_UPS, BY_ID, SPECS

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .policy.catalog import ActionSpec, Resource
    from .policy.state import PolicyState

_PLANNED_KINDS = frozenset({"build", "train", "research"})
_UNREAD = "not read"


def strategist_brief(state: PolicyState, recent_failures: Sequence[str]) -> str:
    """The situation, the next age's gap, and the actions this age allows."""
    sections = [
        "## Situation (from the agent's records)",
        *_situation_lines(state, recent_failures),
        "",
        "## Next age",
        _age_gap_line(state),
        "",
        "## Actions the agent can take in this age",
        *_rule_lines(state.age),
    ]
    return "\n".join(sections)


def _situation_lines(state: PolicyState, recent_failures: Sequence[str]) -> list[str]:
    lines = [_villager_line(state), _population_line(state)]
    lines.append(f"Buildings standing: {_names(state.buildings_seen)}.")
    if state.building_purchases:
        lines.append(f"Paid for, not yet seen finished: {_names(state.building_purchases)}.")
    unpaid = state.pending_buildings - state.building_purchases
    if unpaid:
        lines.append(f"Placed, payment not yet seen: {_names(unpaid)}.")
    food_in_view = state.visible_classes & GATHER_CLASSES_BY_KIND["food"]
    lines.append(f"Natural food in view (sheep, berries, ...): {_names(food_in_view)}.")
    lines.append(
        f"Resources the agent gathers in this age: {', '.join(gatherable_kinds(state.age))}."
    )
    if recent_failures:
        lines.append("Recent failed actions: " + "; ".join(recent_failures) + ".")
    return lines


def _villager_line(state: PolicyState) -> str:
    if state.villagers is None:
        return "Villagers: count not read."
    jobs = ", ".join(f"{kind} {state.villager_jobs.get(kind, _UNREAD)}" for kind in RESOURCE_KINDS)
    idle = "unknown" if state.idle_count is None else str(state.idle_count)
    return f"Villagers: {state.villagers} ({jobs}, idle {idle})."


def _population_line(state: PolicyState) -> str:
    if not state.population_known:
        return "Population: not read."
    return (
        f"Population: {state.population}/{state.population_cap}, "
        f"with {state.pending_population} more in production."
    )


def _age_gap_line(state: PolicyState) -> str:
    age_up = AGE_UPS.get(state.age)
    if not state.age_known or age_up is None:
        return "No further age to reach." if state.age_known else "Current age not read."
    spec = BY_ID[age_up.action_id]
    costs = ", ".join(f"{price} {kind} (have {_banked(state, kind)})" for kind, price in spec.cost)
    standing = state.buildings_seen & age_up.buildings
    missing = max(age_up.buildings_needed - len(standing), 0)
    verdict = "requirement met" if missing == 0 else f"{missing} more needed"
    return (
        f"{_next_age(state.age)} costs {costs}. It also needs any {age_up.buildings_needed} of "
        f"{_names(age_up.buildings)}: {len(standing)} standing ({_names(standing)}), {verdict}."
    )


def _banked(state: PolicyState, kind: Resource) -> int | str:
    return getattr(state, kind) if kind in state.known_resources else _UNREAD


def _next_age(age: str) -> str:
    return AGE_ORDER[AGE_ORDER.index(age) + 1]


def _rule_lines(age: str) -> list[str]:
    allowed = AGE_ORDER[: AGE_ORDER.index(age) + 1] if age in AGE_ORDER else AGE_ORDER[:1]
    return [
        _rule_line(spec) for spec in SPECS if spec.kind in _PLANNED_KINDS and spec.age in allowed
    ]


def _rule_line(spec: ActionSpec) -> str:
    cost = ", ".join(f"{price} {kind}" for kind, price in spec.cost) or "free"
    needs = f"; needs {_names(spec.requires)}" if spec.requires else ""
    return f"- {spec.id}: {cost}{needs}."


def _names(items: frozenset[str]) -> str:
    return ", ".join(sorted(items)) or "none"


__all__ = ["strategist_brief"]
