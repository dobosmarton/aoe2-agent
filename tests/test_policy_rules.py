"""The rule registry: loader, restricted evaluator, and resource reservation."""

from __future__ import annotations

import textwrap
from pathlib import Path  # noqa: TC003  -- runtime use: tmp_path fixture annotations

import pytest
from gameplay_agent.policy.engine import decide, matched_actions, registry
from gameplay_agent.policy.rules import RuleError, load_rules
from gameplay_agent.policy.state import PolicyState


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "rules.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def _rule(when: str, **extra: object) -> str:
    """A one-rule file. `when` is emitted as a YAML block so quoting is free."""
    fields = "".join(f"\n          {k}: {v}" for k, v in extra.items())
    return f"""
        - id: probe
          when: |-
            {when}
          then: {{type: queue_villager, intent: probe}}
          weight: 10{fields}
    """


# ---------------------------------------------------------------------------
# The evaluator rejects anything outside the allowed grammar — at LOAD time
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('x')",
        "open('/etc/passwd')",
        "population.__class__",
        "[population for _ in [1]]",
        "population * 2 > 4",
        "{'a': 1}",
    ],
    ids=["import", "open", "attribute", "comprehension", "multiply", "dict-literal"],
)
def test_evaluator_rejects_unsafe_expressions(tmp_path: Path, expression: str) -> None:
    with pytest.raises(RuleError):
        load_rules([_write(tmp_path, _rule(expression))])


def test_evaluator_rejects_an_unknown_state_field(tmp_path: Path) -> None:
    with pytest.raises(RuleError, match="unknown state field"):
        load_rules([_write(tmp_path, _rule("hitpoints > 3"))])


def test_a_bad_rule_fails_at_load_not_at_evaluation(tmp_path: Path) -> None:
    """Startup is the only safe place to fail — never mid-game."""
    with pytest.raises(RuleError):
        load_rules([_write(tmp_path, _rule("population >"))])


# ---------------------------------------------------------------------------
# The grammar the seed rules actually need
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("population >= 12", True),
        ("population < 12", False),
        ("age == 'Dark Age'", True),
        ("age != 'Dark Age'", False),
        ("'mill' in buildings_seen", True),
        ("'dock' not in buildings_seen", True),
        ("(population_cap - population) <= 2", False),
        ("population > 5 and 'mill' in buildings_seen", True),
        ("population > 99 or age == 'Dark Age'", True),
        ("not idle_present", True),
    ],
)
def test_supported_grammar(tmp_path: Path, expression: str, expected: bool) -> None:
    rule = load_rules([_write(tmp_path, _rule(expression))])[0]
    state = PolicyState(
        age="Dark Age", population=20, population_cap=30, buildings_seen=frozenset({"mill"})
    )
    assert rule.matches(state) is expected


def test_intent_placeholders_are_filled_from_state(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        - id: probe
          when: "population > 0"
          then: {type: build, building_key: q, intent: "headroom {population}/{population_cap}"}
          weight: 10
        """,
    )
    rule = load_rules([path])[0]
    rendered = rule.render(PolicyState(population=14, population_cap=15))
    assert rendered[0]["intent"] == "headroom 14/15"


def test_then_accepts_a_list_of_actions(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        - id: probe
          when: "population > 0"
          then:
            - {type: press, key: h, intent: first}
            - {type: press, key: z, intent: second}
          weight: 10
        """,
    )
    assert len(load_rules([path])[0].actions) == 2


# ---------------------------------------------------------------------------
# Reservation (plan 1.3)
# ---------------------------------------------------------------------------


@pytest.fixture
def two_builds(tmp_path: Path) -> Path:
    return _write(
        tmp_path,
        """
        - id: cheap_but_lower
          when: "population > 0"
          then: {type: build, building_key: r, intent: lower}
          weight: 10
          cost: {wood: 100}
        - id: preferred
          when: "population > 0"
          then: {type: build, building_key: w, intent: higher}
          weight: 90
          cost: {wood: 100}
        """,
    )


def test_reservation_keeps_only_what_the_balance_can_pay(two_builds: Path) -> None:
    actions = matched_actions(PolicyState(population=1, wood=100), load_rules([two_builds]))
    assert len(actions) == 1


def test_reservation_keeps_the_higher_weight_rule(two_builds: Path) -> None:
    assert (
        matched_actions(PolicyState(population=1, wood=100), load_rules([two_builds]))[0]["intent"]
        == "higher"
    )


def test_reservation_admits_both_when_affordable(two_builds: Path) -> None:
    assert len(matched_actions(PolicyState(population=1, wood=200), load_rules([two_builds]))) == 2


def test_reservation_logs_every_drop(two_builds: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A dropped rule replaces the executor's rejection message, so it must
    leave a trace rather than vanish. structlog prints to stdout here."""
    matched_actions(PolicyState(population=1, wood=100), load_rules([two_builds]))
    assert "policy_rule_unaffordable" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Staleness and the alarm short-circuit
# ---------------------------------------------------------------------------


def test_a_stale_snapshot_skips_the_rule(tmp_path: Path) -> None:
    path = _write(tmp_path, _rule("population > 0", max_state_age_ms=1000))
    stale = PolicyState(population=1, captured_at=0.0)  # monotonic 0 is long past
    assert matched_actions(stale, load_rules([path])) == []


def test_alarm_emits_nothing() -> None:
    assert decide([], PolicyState(population=20, population_cap=30), alarm=True) == []


# ---------------------------------------------------------------------------
# Immutability — Phase 3 shares one snapshot across three loops
# ---------------------------------------------------------------------------


def test_a_frozen_state_rejects_scalar_assignment() -> None:
    with pytest.raises(AttributeError):
        PolicyState(population=1).food = 5  # pyright: ignore[reportAttributeAccessIssue]


def test_villager_jobs_cannot_be_mutated_through_the_mapping() -> None:
    """`frozen=True` alone left a plain dict writable inside a "frozen" state."""
    state = PolicyState(population=1)
    with pytest.raises(TypeError):
        state.villager_jobs["food"] = 999  # pyright: ignore[reportIndexIssue]


def test_villager_jobs_copies_the_caller_s_dict() -> None:
    """A later write by the caller must not reach a snapshot already taken."""
    from gameplay_agent.memory import GameState
    from gameplay_agent.policy.state import from_game_state

    jobs = {"food": 3}
    state = from_game_state(GameState(), villager_jobs=jobs)
    jobs["food"] = 999
    assert state.villager_jobs["food"] == 3


# ---------------------------------------------------------------------------
# The shipped registry
# ---------------------------------------------------------------------------


def test_shipped_registry_loads() -> None:
    assert registry()


def test_shipped_registry_is_sorted_by_descending_weight() -> None:
    weights = [rule.weight for rule in registry()]
    assert weights == sorted(weights, reverse=True)


def test_shipped_rules_have_unique_ids() -> None:
    ids = [rule.id for rule in registry()]
    assert len(ids) == len(set(ids))


def test_shipped_actions_pass_model_validation() -> None:
    """`then` must survive the same validation the executor applies."""
    from gameplay_agent.models import validate_actions

    for rule in registry():
        rendered = rule.render(PolicyState(population=1, population_cap=5))
        assert len(validate_actions(rendered)) == len(rendered), rule.id
