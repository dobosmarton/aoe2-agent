"""The rule registry: a `Rule` dataclass, a YAML loader, and a safe evaluator."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, cast

import structlog
import yaml

from .state import PolicyState

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable

log = structlog.stdlib.get_logger()

# The frozen seed corpus is the live registry until Phase 4 needs a writable one.
RULES_DIR = Path(__file__).parent.parent.parent / "knowledge" / "seed" / "rules"

# Rule files the engine loads. `allocation.yaml` and `safety_floor.yaml` are
# read by other tiers, not by the engine.
RULE_FILES: tuple[str, ...] = ("dark_age.yaml", "feudal_age.yaml")

_STATE_FIELDS: frozenset[str] = frozenset(f.name for f in fields(PolicyState))

# Everything a `when` expression may contain. A rule is authored by the
# strategist from Phase 4, so anything outside this set is rejected at load.
_ALLOWED_NODES: tuple[type[ast.AST], ...] = (
    ast.Expression,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.UnaryOp,
    ast.Not,
    ast.Compare,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
)


class RuleError(ValueError):
    """A rule that cannot be loaded. Raised at startup, never mid-game."""


@dataclass(frozen=True, slots=True)
class Rule:
    """One trigger, one or more actions, and the weight that orders them."""

    id: str
    when: str
    tree: ast.Expression  # parsed and validated by `_build_rule`
    actions: tuple[dict[str, object], ...]
    weight: int
    cost: dict[str, int] = field(default_factory=dict)
    max_state_age_ms: float | None = None
    provenance: str = ""
    enabled: bool = True
    # Whether the idle rotation should bank wood toward this build (F-34).
    bank_wood: bool = False

    def matches(self, state: PolicyState) -> bool:
        """Whether this rule's trigger holds for `state`."""
        return bool(_evaluate(self.tree.body, state))

    def is_fresh(self, state: PolicyState) -> bool:
        """Whether the snapshot is recent enough for this rule to act on."""
        return self.max_state_age_ms is None or state.age_ms <= self.max_state_age_ms

    def render(self, state: PolicyState) -> list[dict[str, object]]:
        """This rule's actions, with `intent` placeholders filled from `state`."""
        return [_render_intent(action, state) for action in self.actions]


def _field_values(state: PolicyState) -> dict[str, object]:
    """Every readable field, typed as `object` so no `Any` escapes the boundary."""
    return {name: _field(state, name) for name in _STATE_FIELDS}


def _field(state: PolicyState, name: str) -> object:
    # cast: dynamic attribute access is inherently Any; `name` is checked
    # against _STATE_FIELDS at load time, so the attribute always exists.
    return cast("object", getattr(state, name))


def _render_intent(action: dict[str, object], state: PolicyState) -> dict[str, object]:
    intent = action.get("intent")
    if not isinstance(intent, str) or "{" not in intent:
        return dict(action)
    return {**action, "intent": intent.format(**_field_values(state))}


def _check_nodes(tree: ast.Expression, rule_id: str) -> None:
    """Reject any construct outside `_ALLOWED_NODES`, and any unknown field."""
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise RuleError(f"rule {rule_id!r}: {type(node).__name__} is not allowed in `when`")
        if isinstance(node, ast.Name) and node.id not in _STATE_FIELDS:
            raise RuleError(f"rule {rule_id!r}: unknown state field {node.id!r}")


def _compile(expression: str, rule_id: str) -> ast.Expression:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise RuleError(f"rule {rule_id!r}: cannot parse `when`: {exc}") from exc
    _check_nodes(tree, rule_id)
    return tree


def _evaluate(node: ast.expr, state: PolicyState) -> object:
    """Walk a validated tree. Only `_ALLOWED_NODES` reach here."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return _field(state, node.id)
    if isinstance(node, ast.UnaryOp):
        return not _evaluate(node.operand, state)
    if isinstance(node, ast.BoolOp):
        values = (_evaluate(v, state) for v in node.values)
        return all(values) if isinstance(node.op, ast.And) else any(values)
    if isinstance(node, ast.BinOp):
        left, right = (
            _as_number(_evaluate(node.left, state)),
            _as_number(_evaluate(node.right, state)),
        )
        return left + right if isinstance(node.op, ast.Add) else left - right
    if isinstance(node, ast.Compare):
        return _compare(node, state)
    raise RuleError(f"unreachable node {type(node).__name__}")


def _as_number(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuleError(f"arithmetic needs an int, got {type(value).__name__}")
    return value


def _compare(node: ast.Compare, state: PolicyState) -> bool:
    left = _evaluate(node.left, state)
    for op, comparator in zip(node.ops, node.comparators, strict=True):
        right = _evaluate(comparator, state)
        if not _apply_comparison(op, left, right):
            return False
        left = right
    return True


def _apply_comparison(op: ast.cmpop, left: object, right: object) -> bool:
    if isinstance(op, ast.Eq):
        return left == right
    if isinstance(op, ast.NotEq):
        return left != right
    if isinstance(op, ast.In):
        return _contains(right, left)
    if isinstance(op, ast.NotIn):
        return not _contains(right, left)
    return _apply_ordering(op, left, right)


def _apply_ordering(op: ast.cmpop, left: object, right: object) -> bool:
    a, b = _as_number(left), _as_number(right)
    if isinstance(op, ast.Lt):
        return a < b
    if isinstance(op, ast.LtE):
        return a <= b
    if isinstance(op, ast.Gt):
        return a > b
    if isinstance(op, ast.GtE):
        return a >= b
    raise RuleError(f"unsupported comparison {type(op).__name__}")


def _contains(container: object, item: object) -> bool:
    """`'mill' in buildings_seen` — the only shape the seed rules use."""
    if isinstance(container, str):
        return isinstance(item, str) and item in container
    if isinstance(container, (frozenset, set, tuple, list)):
        members: Collection[object] = container
        return item in members
    raise RuleError(f"`in` needs a collection, got {type(container).__name__}")


def _as_actions(raw: object, rule_id: str) -> tuple[dict[str, object], ...]:
    """`then` accepts one action or a list; age-up emits two presses."""
    items = raw if isinstance(raw, list) else [raw]
    actions: list[dict[str, object]] = []
    for item in items:
        if not isinstance(item, dict):
            raise RuleError(f"rule {rule_id!r}: each `then` entry must be a mapping")
        actions.append({str(k): v for k, v in item.items()})
    if not actions:
        raise RuleError(f"rule {rule_id!r}: `then` is empty")
    return tuple(actions)


def _as_cost(raw: object, rule_id: str) -> dict[str, int]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise RuleError(f"rule {rule_id!r}: `cost` must be a mapping")
    return {str(k): int(v) for k, v in raw.items() if isinstance(v, (int, float, str))}


def _build_rule(raw: dict[str, object]) -> Rule:
    rule_id = str(raw.get("id", ""))
    if not rule_id:
        raise RuleError("a rule is missing `id`")
    when = raw.get("when")
    if not isinstance(when, str) or not when.strip():
        raise RuleError(f"rule {rule_id!r}: `when` must be a non-empty string")
    return Rule(
        id=rule_id,
        when=when,
        tree=_compile(when, rule_id),
        actions=_as_actions(raw.get("then"), rule_id),
        weight=int(str(raw.get("weight", 0))),
        cost=_as_cost(raw.get("cost"), rule_id),
        max_state_age_ms=_as_age_limit(raw.get("max_state_age_ms")),
        provenance=str(raw.get("provenance", "")),
        enabled=bool(raw.get("enabled", True)),
        bank_wood=bool(raw.get("bank_wood", False)),
    )


def _as_age_limit(raw: object) -> float | None:
    return float(raw) if isinstance(raw, (int, float)) else None


def load_rules(paths: Iterable[Path] | None = None) -> tuple[Rule, ...]:
    """Load the registry, highest weight first. Raises `RuleError` on a bad rule."""
    files = list(paths) if paths is not None else [RULES_DIR / name for name in RULE_FILES]
    rules: list[Rule] = []
    for path in files:
        # cast: yaml.safe_load is Any; the isinstance below is the real gate.
        raw = cast("object", yaml.safe_load(path.read_text(encoding="utf-8")))
        if not isinstance(raw, list):
            raise RuleError(f"{path.name}: expected a list of rules")
        entries: list[object] = raw
        rules.extend(_build_rule(e) for e in entries if isinstance(e, dict))
    rules.sort(key=lambda rule: -rule.weight)
    log.debug("policy_rules_loaded", count=len(rules), files=[p.name for p in files])
    return tuple(rules)
