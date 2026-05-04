"""Assertion DSL for scenario evaluation.

Each function takes (executed_actions, expected_value, context) and returns a
list of failure strings. Empty list = assertion passed.

Property-dict matching uses subset semantics: an action is considered a match
for `{"type": "press", "key": "h"}` if all expected keys are present with
equal values, regardless of extra fields on the action.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

REASONING_PREVIEW_CHARS = 300
ACTION_DISPLAY_KEYS = ("key", "building_key", "target_class", "target_id", "x", "y")


def _matches(action: dict, pattern: dict) -> bool:
    """Subset match: every key in pattern must equal the same key in action."""
    for key, expected in pattern.items():
        if action.get(key) != expected:
            return False
    return True


def _format_action(action: dict) -> str:
    """One-line action repr for failure messages."""
    parts = [f"type={action.get('type')!r}"]
    for key in ACTION_DISPLAY_KEYS:
        if key in action:
            parts.append(f"{key}={action[key]!r}")
    return "{" + ", ".join(parts) + "}"


def _format_action_list(actions: Iterable[dict]) -> str:
    items = list(actions)
    if not items:
        return "(no actions)"
    return "\n  - " + "\n  - ".join(_format_action(action) for action in items)


def _preview(reasoning: str) -> str:
    return reasoning[:REASONING_PREVIEW_CHARS] if reasoning else "(empty)"


# ---------------------------------------------------------------------------
# Action-list assertions
# ---------------------------------------------------------------------------


def must_include(actions: list[dict], pattern: dict, **_) -> list[str]:
    """Action list contains at least one action matching the pattern (anywhere)."""
    if not isinstance(pattern, dict):
        return [f"must_include expected a dict, got {type(pattern).__name__}"]
    if any(_matches(action, pattern) for action in actions):
        return []
    return [
        f"must_include FAILED — no action matched {pattern!r}.\n"
        f"  Actual actions:{_format_action_list(actions)}"
    ]


def must_include_first(actions: list[dict], patterns: list[dict], **_) -> list[str]:
    """First N actions match exactly (ordered prefix)."""
    if not isinstance(patterns, list):
        return [f"must_include_first expected a list, got {type(patterns).__name__}"]
    for index, pattern in enumerate(patterns):
        if index >= len(actions):
            return [
                f"must_include_first FAILED — expected {len(patterns)} prefix actions, "
                f"got only {len(actions)}.\n  Actual:{_format_action_list(actions)}"
            ]
        if not _matches(actions[index], pattern):
            return [
                f"must_include_first FAILED at index {index} — expected {pattern!r}, "
                f"got {_format_action(actions[index])}.\n"
                f"  Full actions:{_format_action_list(actions)}"
            ]
    return []


def must_not_include(actions: list[dict], pattern: dict, **_) -> list[str]:
    """Action list contains NO action matching the pattern."""
    if not isinstance(pattern, dict):
        return [f"must_not_include expected a dict, got {type(pattern).__name__}"]
    for action in actions:
        if _matches(action, pattern):
            return [
                f"must_not_include FAILED — found forbidden action {_format_action(action)} "
                f"matching {pattern!r}.\n  Full actions:{_format_action_list(actions)}"
            ]
    return []


def _count_matches(actions: list[dict], spec: dict, default_n: int) -> tuple[int, int, dict]:
    """Return (expected_n, actual_count, pattern) extracted from a count spec."""
    expected_n = int(spec.get("n", default_n))
    pattern = {key: value for key, value in spec.items() if key != "n"}
    actual_count = sum(1 for action in actions if _matches(action, pattern))
    return expected_n, actual_count, pattern


def count_at_least(actions: list[dict], spec: dict, **_) -> list[str]:
    """At least N actions match the type/pattern."""
    expected_n, actual, pattern = _count_matches(actions, spec, default_n=1)
    if actual < expected_n:
        return [
            f"count_at_least FAILED — expected ≥ {expected_n} actions matching {pattern!r}, "
            f"got {actual}.\n  Full actions:{_format_action_list(actions)}"
        ]
    return []


def count_at_most(actions: list[dict], spec: dict, **_) -> list[str]:
    """At most N actions match the type/pattern (n: 0 forbids the type entirely)."""
    expected_n, actual, pattern = _count_matches(actions, spec, default_n=0)
    if actual > expected_n:
        return [
            f"count_at_most FAILED — expected ≤ {expected_n} actions matching {pattern!r}, "
            f"got {actual}.\n  Full actions:{_format_action_list(actions)}"
        ]
    return []


def differs_from_baseline_by(
    actions: list[dict],
    spec: dict,
    *,
    baseline_actions: list[dict] | None = None,
    **_,
) -> list[str]:
    """Differential assertion: this variant's action list differs from the baseline.

    The first variant in a scenario IS the baseline; subsequent variants
    compare against it. Spec accepts:
      must_include: pattern        # appears in this variant, NOT in baseline
      must_not_include: pattern    # appears in baseline, NOT in this variant

    Both check directions are independent — supply either or both.
    """
    if baseline_actions is None:
        return [
            "differs_from_baseline_by FAILED — no baseline available. "
            "This assertion only works on a NON-FIRST variant within a scenario "
            "that has multiple variants. The first variant is treated as the baseline."
        ]
    if not isinstance(spec, dict):
        return [f"differs_from_baseline_by expected a dict, got {type(spec).__name__}"]

    failures: list[str] = []

    if "must_include" in spec:
        pattern = spec["must_include"]
        in_variant = any(_matches(action, pattern) for action in actions)
        in_baseline = any(_matches(action, pattern) for action in baseline_actions)
        if not in_variant or in_baseline:
            failures.append(
                f"differs_from_baseline_by.must_include FAILED — pattern {pattern!r} "
                f"should appear in this variant but NOT baseline. "
                f"Got: variant={in_variant}, baseline={in_baseline}."
            )

    if "must_not_include" in spec:
        pattern = spec["must_not_include"]
        in_variant = any(_matches(action, pattern) for action in actions)
        in_baseline = any(_matches(action, pattern) for action in baseline_actions)
        if in_variant or not in_baseline:
            failures.append(
                f"differs_from_baseline_by.must_not_include FAILED — pattern {pattern!r} "
                f"should appear in baseline but NOT this variant. "
                f"Got: variant={in_variant}, baseline={in_baseline}."
            )

    return failures


# ---------------------------------------------------------------------------
# Reasoning + memory assertions
# ---------------------------------------------------------------------------

_APPLIED_RE = re.compile(r"\[applied:\s*([^\]]+)\]", re.IGNORECASE)


def _extract_applied_titles(reasoning: str) -> list[str]:
    """Extract titles from any `[applied: t1, t2]` tag in reasoning.

    Was anchored to start-of-string (`re.match`), but the model often emits
    the tag inside a numbered list or after a header (e.g. `**Plan:**\n1. [applied: ...]`).
    Position is incidental — the contract is "model self-reports applied
    memories somewhere in its response" — so we search instead of match.
    Multiple `[applied: ...]` tags are unioned.
    """
    titles: list[str] = []
    for match in _APPLIED_RE.finditer(reasoning or ""):
        titles.extend(t.strip() for t in match.group(1).split(",") if t.strip())
    # Preserve first-seen order while deduping (deterministic for assertions).
    seen: set[str] = set()
    return [t for t in titles if not (t in seen or seen.add(t))]


def applied_memories(reasoning: str, expected: list[str], **_) -> list[str]:
    """Reasoning's `[applied: ...]` tag names exactly these titles (set-equal)."""
    actual = _extract_applied_titles(reasoning)
    if set(actual) != set(expected):
        return [
            f"applied_memories FAILED — expected {sorted(expected)!r}, "
            f"got {sorted(actual)!r}.\n  Reasoning: {_preview(reasoning)}"
        ]
    return []


def applied_memories_subset(reasoning: str, expected: list[str], **_) -> list[str]:
    """Reasoning's `[applied: ...]` tag names AT LEAST these titles (extras allowed)."""
    actual = set(_extract_applied_titles(reasoning))
    missing = set(expected) - actual
    if missing:
        return [
            f"applied_memories_subset FAILED — missing {sorted(missing)!r} from tag.\n"
            f"  Tagged: {sorted(actual)!r}.\n  Reasoning: {_preview(reasoning)}"
        ]
    return []


def reasoning_contains(reasoning: str, expected: str, **_) -> list[str]:
    """Reasoning string contains the substring (case-insensitive)."""
    if not isinstance(expected, str):
        return [f"reasoning_contains expected a string, got {type(expected).__name__}"]
    if expected.lower() not in (reasoning or "").lower():
        return [
            f"reasoning_contains FAILED — substring {expected!r} not found.\n"
            f"  Reasoning: {_preview(reasoning)}"
        ]
    return []


def reasoning_excludes(reasoning: str, expected: str, **_) -> list[str]:
    """Reasoning string does NOT contain the substring (case-insensitive)."""
    if not isinstance(expected, str):
        return [f"reasoning_excludes expected a string, got {type(expected).__name__}"]
    if expected.lower() in (reasoning or "").lower():
        return [
            f"reasoning_excludes FAILED — forbidden substring {expected!r} appeared.\n"
            f"  Reasoning: {_preview(reasoning)}"
        ]
    return []


# Dispatch — `expected:` block in fixture maps directly to assertion names

_ACTION_ASSERTIONS = {
    "must_include": must_include,
    "must_include_first": must_include_first,
    "must_not_include": must_not_include,
    "count_at_least": count_at_least,
    "count_at_most": count_at_most,
    "differs_from_baseline_by": differs_from_baseline_by,
}
_REASONING_ASSERTIONS = {
    "applied_memories": applied_memories,
    "applied_memories_subset": applied_memories_subset,
    "reasoning_contains": reasoning_contains,
    "reasoning_excludes": reasoning_excludes,
}

# Assertions that consume a LIST as a single value (don't iterate the list).
# Anything not in this set, when given a list, runs once per item — useful
# for `must_not_include: [a, b, c]` style.
_TAKES_LIST_AS_VALUE = {
    "must_include_first",
    "applied_memories",
    "applied_memories_subset",
}


def _normalize_value(key: str, value):
    """Wrap scalars in a list so the caller can always iterate.

    Lists become [item, item, ...] (one assertion call per item) UNLESS the
    assertion takes a list as its value (`must_include_first`, `applied_memories`).
    """
    if isinstance(value, list) and key not in _TAKES_LIST_AS_VALUE:
        return value
    return [value]


def evaluate(
    expected: dict,
    *,
    actions: list[dict],
    reasoning: str,
    baseline_actions: list[dict] | None = None,
) -> list[str]:
    """Evaluate every assertion in `expected:`. Returns aggregated failure messages.

    `baseline_actions` is the first variant's executed actions, used by
    `differs_from_baseline_by`. None for non-variant fixtures or for the
    baseline variant itself.

    For example:
        must_not_include:
          - {type: press, key: b}
          - {type: press, key: t}
    runs `must_not_include` twice (one per item).
    `applied_memories: [a, b]` runs once with the whole list.
    """
    failures: list[str] = []

    for key, value in expected.items():
        action_fn = _ACTION_ASSERTIONS.get(key)
        reasoning_fn = _REASONING_ASSERTIONS.get(key)
        if action_fn is None and reasoning_fn is None:
            failures.append(f"unknown assertion key: {key!r}")
            continue

        for item in _normalize_value(key, value):
            if action_fn is not None:
                failures.extend(action_fn(actions, item, baseline_actions=baseline_actions))
            else:
                failures.extend(reasoning_fn(reasoning, item))  # type: ignore[misc]

    return failures
