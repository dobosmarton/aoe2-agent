"""Goal tracking logger for AoE2 LLM Agent."""

from datetime import datetime
from pathlib import Path

from .goals import Goal


class GoalLogger:
    """Writes structured goal tracking to logs/goals.log."""

    def __init__(self, log_dir: Path):
        self.log_path = log_dir / "goals.log"
        log_dir.mkdir(parents=True, exist_ok=True)
        # Write header
        with open(self.log_path, "a") as f:
            f.write(f"\n=== Game started {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    def _write(self, text: str) -> None:
        with open(self.log_path, "a") as f:
            f.write(text + "\n")

    def log_goals_created(self, turn: int, goals: list[Goal]) -> None:
        lines = [f"[T{turn:03d}] STRATEGIST: Created {len(goals)} goals"]
        for goal in goals:
            type_label = f"{goal.type.upper():<6}"
            lines.append(
                f"  [{type_label} P{goal.priority:<2}] {goal.name} "
                f"(metric={goal.metric}, target={goal.target})"
            )
        self._write("\n".join(lines))

    def log_progress(self, turn: int, goals: list[Goal], reward: dict) -> None:
        total = reward.get("total", 0.0)
        lines = [f"[T{turn:03d}] PROGRESS: reward={total:+.3f}"]
        for goal in goals:
            if goal.completed or goal.failed:
                continue
            pct = int(goal.progress * 100)
            if isinstance(goal.target, (int, float)):
                current = int(goal.progress * goal.target)
                progress_str = f"{current}/{int(goal.target)} ({pct}%)"
            else:
                progress_str = f"{pct}%"
            lines.append(f"  {goal.name}: {progress_str}")
        self._write("\n".join(lines))

    def log_goal_completed(self, turn: int, goal: Goal) -> None:
        turns_taken = turn - goal.created_turn
        self._write(
            f"[T{turn:03d}] COMPLETED: \"{goal.name}\" in {turns_taken} turns"
        )

    def log_goal_failed(self, turn: int, goal: Goal, reason: str = "") -> None:
        reason_str = f" -- {reason}" if reason else ""
        self._write(
            f"[T{turn:03d}] FAILED: \"{goal.name}\"{reason_str}"
        )

    def log_strategist_update(
        self, turn: int, old_goals: list[Goal], new_goals: list[Goal]
    ) -> None:
        old_names = {g.name for g in old_goals}
        new_names = {g.name for g in new_goals}
        added = new_names - old_names
        removed = old_names - new_names
        kept = old_names & new_names

        lines = [
            f"[T{turn:03d}] STRATEGIST UPDATE: "
            f"added {len(added)}, removed {len(removed)}, kept {len(kept)}"
        ]
        for goal in new_goals:
            if goal.name in added:
                lines.append(f"  + [{goal.type.upper():<6} P{goal.priority}] {goal.name}")
            else:
                lines.append(f"  = [{goal.type.upper():<6} P{goal.priority}] {goal.name}")
        for name in removed:
            lines.append(f"  - {name}")
        self._write("\n".join(lines))

    def log_game_end(self, turn: int, reason: str, completed_count: int) -> None:
        self._write(
            f"[T{turn:03d}] GAME END: {reason} "
            f"({completed_count} goals completed)"
        )
