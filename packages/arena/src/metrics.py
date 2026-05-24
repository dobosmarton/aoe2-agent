"""Metrics extraction and ranking table for the synth arena (Phase 6).

Reads directly from in-memory `SynthLoopResult` — no DuckDB query needed.
`summarise(results)` returns a plain-text ranked table + per-variant detail
section suitable for stdout.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arena.race import VariantResult


@dataclass(frozen=True, slots=True)
class VariantMetrics:
    """Per-variant summary extracted from a completed race."""

    name: str
    final_food: float
    final_wood: float
    final_pop: int
    final_age: str
    buildings: tuple[str, ...]
    total_actions: int
    action_counts: tuple[tuple[str, int], ...]
    total_cost_usd: float
    turns_completed: int


def _action_type(action: dict[str, object]) -> str:
    raw = action.get("type", "")
    return str(raw) if raw else "<none>"


def extract_metrics(result: VariantResult) -> VariantMetrics:
    """Extract summary metrics from a single VariantResult."""
    turns = result.loop_result.turns
    final_state = turns[-1].state_after if turns else None
    counter: Counter[str] = Counter()
    total_actions = 0
    for turn in turns:
        for action in turn.actions:
            counter[_action_type(action)] += 1
            total_actions += 1
    return VariantMetrics(
        name=result.profile.name,
        final_food=final_state.food if final_state is not None else 0.0,
        final_wood=final_state.wood if final_state is not None else 0.0,
        final_pop=final_state.population if final_state is not None else 0,
        final_age=final_state.age if final_state is not None else "unknown",
        buildings=tuple(final_state.buildings) if final_state is not None else (),
        total_actions=total_actions,
        action_counts=tuple(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))),
        total_cost_usd=result.loop_result.total_cost_usd,
        turns_completed=len(turns),
    )


def _format_buildings(buildings: tuple[str, ...]) -> str:
    if not buildings:
        return "<none>"
    counts = Counter(buildings)
    return ", ".join(f"{name}x{n}" if n > 1 else name for name, n in sorted(counts.items()))


def _format_action_counts(counts: tuple[tuple[str, int], ...]) -> str:
    if not counts:
        return "<none>"
    return ", ".join(f"{name}:{n}" for name, n in counts)


def summarise(results: list[VariantResult]) -> str:
    """Format a ranked table + per-variant detail (highest final_pop first)."""
    rows = sorted(
        [extract_metrics(r) for r in results],
        key=lambda m: m.final_pop,
        reverse=True,
    )
    header = (
        f"{'Rank':>4}  {'Profile':<22}  {'Age':<12}  "
        f"{'Pop':>4}  {'Food':>6}  {'Wood':>6}  {'Bldgs':>5}  "
        f"{'Acts':>5}  {'Cost($)':>8}  {'Turns':>6}"
    )
    separator = "-" * len(header)
    lines = [header, separator]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            f"{rank:>4}  {row.name:<22}  {row.final_age:<12}  "
            f"{row.final_pop:>4}  {row.final_food:>6.0f}  {row.final_wood:>6.0f}  "
            f"{len(row.buildings):>5}  {row.total_actions:>5}  "
            f"{row.total_cost_usd:>8.4f}  {row.turns_completed:>6}"
        )
    lines.append("")
    lines.append("Per-variant detail:")
    for row in rows:
        lines.append(
            f"  {row.name}: actions=[{_format_action_counts(row.action_counts)}] "
            f"buildings=[{_format_buildings(row.buildings)}]"
        )
    return "\n".join(lines)
