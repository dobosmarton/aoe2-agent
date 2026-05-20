"""Metrics extraction and ranking table for the synth arena (Phase 6).

Reads directly from in-memory `SynthLoopResult` — no DuckDB query needed.
`summarise(results)` returns a plain-text ranked table suitable for stdout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arena.race import VariantResult


@dataclass(frozen=True, slots=True)
class VariantMetrics:
    """Per-variant summary extracted from a completed race."""

    name: str
    final_food: float
    final_pop: int
    final_age: str
    total_cost_usd: float
    turns_completed: int


def extract_metrics(result: VariantResult) -> VariantMetrics:
    """Extract summary metrics from a single VariantResult."""
    turns = result.loop_result.turns
    final_state = turns[-1].state_after if turns else None
    return VariantMetrics(
        name=result.profile.name,
        final_food=final_state.food if final_state is not None else 0.0,
        final_pop=final_state.population if final_state is not None else 0,
        final_age=final_state.age if final_state is not None else "unknown",
        total_cost_usd=result.loop_result.total_cost_usd,
        turns_completed=len(turns),
    )


def summarise(results: list[VariantResult]) -> str:
    """Format a ranked table of variant results (highest final_pop first)."""
    rows = sorted(
        [extract_metrics(r) for r in results],
        key=lambda m: m.final_pop,
        reverse=True,
    )
    header = f"{'Rank':>4}  {'Profile':<22}  {'Age':<12}  {'Pop':>4}  {'Food':>7}  {'Cost($)':>8}  {'Turns':>6}"
    separator = "-" * len(header)
    lines = [header, separator]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            f"{rank:>4}  {row.name:<22}  {row.final_age:<12}  "
            f"{row.final_pop:>4}  {row.final_food:>7.0f}  "
            f"{row.total_cost_usd:>8.4f}  {row.turns_completed:>6}"
        )
    return "\n".join(lines)
