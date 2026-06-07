"""Game-trace capture for reflective prompt optimization (A2).

Turns a finished game's working memory + score into a compact, serializable
trace the reflective mutator can reason over (what specifically happened, turn
by turn, and how the 5 score components landed). Persisted as JSON next to the
experiment ledger so a later tournament can reflect on prior games.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

import structlog

if TYPE_CHECKING:
    from gameplay_agent.memory import AgentMemory

    from .metrics import GameScore

log = structlog.stdlib.get_logger()

_TRACES_DIR = Path(__file__).parent.parent / "experiments" / "traces"


@dataclass(frozen=True, slots=True)
class TurnTrace:
    """One turn's reasoning, actions, and post-action verification."""

    iteration: int
    reasoning: str
    actions: str
    verification: str


@dataclass(frozen=True, slots=True)
class GameTrace:
    """A whole game distilled for reflection."""

    turns: list[TurnTrace]
    components: dict[str, float]  # survival/population/age/economy/action_success
    composite: float
    end_reason: str


def _summarize_actions(actions: list[dict]) -> str:
    parts: list[str] = []
    for a in actions[:4]:
        if not isinstance(a, dict):
            continue
        arg = a.get("key") or a.get("target_id") or a.get("target_class") or ""
        parts.append(f"{a.get('type', '?')}({arg})")
    return ", ".join(parts)


def build_game_trace(memory: AgentMemory, score: GameScore) -> GameTrace:
    """Distill an AgentMemory + GameScore into a serializable GameTrace."""
    turns = [
        TurnTrace(
            iteration=t.iteration,
            reasoning=t.reasoning[:150],
            actions=_summarize_actions(t.actions),
            verification=t.verification,
        )
        for t in memory.working_memory
    ]
    components = {
        "survival": score.survival,
        "population": score.population,
        "age": score.age,
        "economy": score.economy,
        "action_success": score.action_success,
    }
    end_reason = str(score.raw_metrics.get("game_end_reason", ""))
    return GameTrace(
        turns=turns, components=components, composite=score.composite, end_reason=end_reason
    )


def format_trace_excerpt(trace: GameTrace, max_turns: int = 12) -> str:
    """Render a trace as prompt text: component vector + recent turns."""
    header = (
        f"Composite {trace.composite:.3f} | "
        + ", ".join(f"{k}={v:.2f}" for k, v in trace.components.items())
        + f" | end={trace.end_reason or 'unknown'}"
    )
    lines = [header, "Turns:"]
    for t in trace.turns[-max_turns:]:
        suffix = f" | {t.verification}" if t.verification else ""
        lines.append(f"  T{t.iteration}: {t.reasoning} | actions: {t.actions}{suffix}")
    return "\n".join(lines)


def _trace_to_dict(trace: GameTrace) -> dict[str, object]:
    return {
        "turns": [
            {
                "iteration": t.iteration,
                "reasoning": t.reasoning,
                "actions": t.actions,
                "verification": t.verification,
            }
            for t in trace.turns
        ],
        "components": trace.components,
        "composite": trace.composite,
        "end_reason": trace.end_reason,
    }


def _trace_from_dict(data: dict[str, object]) -> GameTrace:
    raw_turns = data.get("turns")
    turns: list[TurnTrace] = []
    if isinstance(raw_turns, list):
        for item in raw_turns:
            if isinstance(item, dict):
                turns.append(
                    TurnTrace(
                        iteration=int(cast("int", item.get("iteration", 0))),
                        reasoning=str(item.get("reasoning", "")),
                        actions=str(item.get("actions", "")),
                        verification=str(item.get("verification", "")),
                    )
                )
    raw_components = data.get("components")
    components: dict[str, float] = {}
    if isinstance(raw_components, dict):
        components = {str(k): float(cast("float", v)) for k, v in raw_components.items()}
    return GameTrace(
        turns=turns,
        components=components,
        composite=float(cast("float", data.get("composite", 0.0))),
        end_reason=str(data.get("end_reason", "")),
    )


def save_trace(
    trace: GameTrace, game_id: str | None = None, traces_dir: Path | None = None
) -> Path:
    """Persist a trace as JSON. Uses a timestamped filename when no id given."""
    directory = traces_dir or _TRACES_DIR
    directory.mkdir(parents=True, exist_ok=True)
    name = game_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    path = directory / f"{name}.json"
    path.write_text(json.dumps(_trace_to_dict(trace), indent=2))
    return path


def load_recent_traces(n: int, traces_dir: Path | None = None) -> list[GameTrace]:
    """Load the most recent N traces (by filename), oldest first."""
    directory = traces_dir or _TRACES_DIR
    if not directory.exists():
        return []
    files = sorted(directory.glob("*.json"))[-n:]
    traces: list[GameTrace] = []
    for f in files:
        try:
            raw = cast("object", json.loads(f.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(raw, dict):
            traces.append(_trace_from_dict(raw))
    return traces
