"""Goal management for AoE2 LLM Agent."""

from dataclasses import dataclass, field
from typing import Any, Literal

import structlog

from .entity_utils import extract_attrs
from .memory import GameState, AGE_SCORES

log = structlog.get_logger()

# Enemy military classes from YOLO detection that trigger alarm
ALARM_CONFIDENCE_GATE = 0.45
RESOURCE_REWARD_DIVISOR = 1000.0
POPULATION_REWARD_FACTOR = 0.05
MAX_DISPLAY_GOALS = 5

THREAT_CLASSES = frozenset({
    "militia_line", "spearman_line", "eagle_line",
    "archer_line", "skirmisher_line", "cavalry_archer", "hand_cannoneer",
    "scout_line", "knight_line", "camel_line", "battle_elephant",
    "ram", "mangonel_line", "scorpion", "trebuchet", "siege_tower",
    "unique_archer", "unique_cavalry", "unique_infantry", "unique_siege",
    "monk",
})


@dataclass
class Goal:
    """A single goal for the agent to pursue."""

    name: str
    type: Literal["local", "global"]
    metric: str  # "population", "food", "wood", "gold", "stone", "age", "building"
    target: Any  # Numeric value or string (e.g., "Feudal Age")
    priority: int  # 1-10 (10 = most urgent)
    created_turn: int
    deadline_turns: int | None = None
    progress: float = 0.0  # 0.0 to 1.0
    completed: bool = False
    failed: bool = False


class GoalManager:
    """Manages active and completed goals, computes progress and rewards."""

    def __init__(self):
        self.active_goals: list[Goal] = []
        self.completed_goals: list[Goal] = []
        self._prev_state: dict | None = None
        self._resource_readings: dict = {}
        self._alarm_active: bool = False

    def set_goals(self, goals: list[Goal]) -> None:
        """Replace active goals with new ones from strategist.

        Preserves completed goals and merges progress for goals
        that exist in both old and new lists (matched by name).
        """
        old_by_name = {g.name: g for g in self.active_goals}
        for goal in goals:
            old = old_by_name.get(goal.name)
            if old:
                goal.progress = old.progress
                goal.completed = old.completed
        self.active_goals = [g for g in goals if not g.completed]

    def evaluate_progress(self, game_state: GameState, turn: int) -> None:
        """Update progress for all active goals based on current game state."""
        for goal in self.active_goals:
            if goal.completed or goal.failed:
                continue

            progress = self._compute_goal_progress(goal, game_state)
            goal.progress = min(1.0, max(0.0, progress))

            if goal.progress >= 1.0:
                goal.completed = True
                self.completed_goals.append(goal)

            # Check deadline
            if goal.deadline_turns is not None:
                turns_elapsed = turn - goal.created_turn
                if turns_elapsed > goal.deadline_turns and not goal.completed:
                    goal.failed = True

        # Remove completed/failed from active
        self.active_goals = [
            g for g in self.active_goals if not g.completed and not g.failed
        ]

    def _compute_goal_progress(self, goal: Goal, state: GameState) -> float:
        """Compute progress (0-1) for a single goal against game state."""
        metric = goal.metric
        target = goal.target

        # Numeric metrics — target must be convertible to float
        try:
            if metric == "population":
                return state.population / float(target) if float(target) > 0 else 0.0

            if metric in ("food", "wood", "gold", "stone"):
                return state.resources.get(metric, 0) / float(target) if float(target) > 0 else 0.0
        except (ValueError, TypeError):
            return 0.0

        if metric == "age":
            current_score = AGE_SCORES.get(state.current_age, 0.0)
            target_score = AGE_SCORES.get(target, 1.0)
            return current_score / target_score if target_score > 0 else 0.0

        # Unknown metric — can't compute
        return 0.0

    def compute_turn_reward(self, prev_state: GameState, curr_state: GameState) -> dict:
        """Compute reward breakdown from state deltas, weighted by goal priorities."""
        reward = {"total": 0.0}

        # Resource deltas
        for res in ("food", "wood", "gold", "stone"):
            prev = prev_state.resources.get(res, 0)
            curr = curr_state.resources.get(res, 0)
            delta = curr - prev
            reward[res] = round(delta / RESOURCE_REWARD_DIVISOR, 4)
            reward["total"] += reward[res]

        # Population delta
        pop_delta = curr_state.population - prev_state.population
        reward["population"] = round(pop_delta * POPULATION_REWARD_FACTOR, 4)
        reward["total"] += reward["population"]

        # Age progress
        prev_age = AGE_SCORES.get(prev_state.current_age, 0.0)
        curr_age = AGE_SCORES.get(curr_state.current_age, 0.0)
        age_delta = curr_age - prev_age
        reward["age"] = round(age_delta, 4)
        reward["total"] += reward["age"]

        reward["total"] = round(reward["total"], 4)
        return reward

    def get_context_for_llm(self) -> str:
        """Format active goals for injection into executor context."""
        if not self.active_goals:
            return ""

        sorted_goals = sorted(self.active_goals, key=lambda g: g.priority, reverse=True)
        top_goals = sorted_goals[:MAX_DISPLAY_GOALS]

        lines = ["## Active Goals (follow in priority order)"]
        for goal in top_goals:
            priority_label = "HIGH" if goal.priority >= 8 else "MED" if goal.priority >= 5 else "LOW"
            type_label = "LOCAL" if goal.type == "local" else "GLOBAL"
            pct = int(goal.progress * 100)

            if isinstance(goal.target, (int, float)):
                current = int(goal.progress * goal.target)
                progress_str = f"{current}/{int(goal.target)} ({pct}%)"
            else:
                progress_str = f"{pct}%"

            lines.append(f"  [{priority_label}] [{type_label}] {goal.name}: {progress_str}")

        # Recently completed (last 3)
        if self.completed_goals:
            recent_completed = self.completed_goals[-3:]
            lines.append("")
            for goal in recent_completed:
                lines.append(f"  [DONE] {goal.name}")

        return "\n".join(lines)

    def get_state_snapshot(self, game_state: GameState) -> dict:
        """Get a serializable snapshot of game state for the strategist."""
        return {
            "resources": dict(game_state.resources),
            "population": game_state.population,
            "population_cap": game_state.population_cap,
            "age": game_state.current_age,
            "idle_tc": game_state.idle_tc,
            "under_attack": game_state.under_attack,
            "enemy_located": game_state.enemy_located,
        }

    def get_goals_summary(self) -> str:
        """Get a text summary of current goals for the strategist."""
        lines = []
        for goal in self.active_goals:
            pct = int(goal.progress * 100)
            status = "DONE" if goal.completed else f"{pct}%"
            lines.append(f"  {goal.type}/{goal.name} P{goal.priority}: {status}")
        for goal in self.completed_goals[-5:]:
            lines.append(f"  COMPLETED: {goal.name}")
        return "\n".join(lines) if lines else "No goals yet."

    # --- Resource readings cache ---

    def update_resource_readings(self, readings: dict, memory: "AgentMemory | None" = None) -> None:
        """Cache resource readings from strategist and update game state."""
        if not readings:
            return
        self._resource_readings = readings
        log.debug("resource_readings_cached", **readings)

        # Also update the memory's game state if provided
        if memory:
            obs = {}
            if "food" in readings:
                obs["resources"] = {
                    "food": readings.get("food", 0),
                    "wood": readings.get("wood", 0),
                    "gold": readings.get("gold", 0),
                    "stone": readings.get("stone", 0),
                }
            if "population" in readings:
                obs["population"] = readings["population"]
            if obs:
                memory.update_from_observations(obs)
            # Age goes through a dedicated channel — the strategist is the only
            # authoritative source for current_age (executor was hallucinating it).
            if "age" in readings:
                memory.update_age(readings["age"])

    def get_resource_context(self) -> str:
        """Format cached resource readings for executor context."""
        if not self._resource_readings:
            return ""
        r = self._resource_readings
        lines = [
            "## Resource Status (from strategist)",
            f"- Food: {r.get('food', '?')}",
            f"- Wood: {r.get('wood', '?')}",
            f"- Gold: {r.get('gold', '?')}",
            f"- Stone: {r.get('stone', '?')}",
            f"- Population: {r.get('population', '?')}",
            f"- Age: {r.get('age', '?')}",
        ]
        return "\n".join(lines)

    # --- Alarm system ---

    def check_alarm(self, detected_entities: list, screenshot_bytes: bytes | None = None) -> bool:
        """Scan YOLO entities for enemy military threats.

        Uses color-based ownership detection (if screenshot available) to
        distinguish own units (blue) from enemy units. Only triggers alarm
        on confirmed enemy military.

        Returns True if enemy threat detected, also injects emergency goals.
        """
        # Step 1: Find candidate military entities (confidence gate to avoid false alarms)
        candidates = []
        for entity in detected_entities:
            attrs = extract_attrs(entity)
            if attrs.class_name in THREAT_CLASSES and attrs.confidence >= ALARM_CONFIDENCE_GATE:
                candidates.append(entity)

        if not candidates:
            self._alarm_active = False
            return False

        # Step 2: Filter by ownership using color detection
        threats_found = []
        if screenshot_bytes:
            try:
                from detection.inference.ownership import classify_entities, Owner
                results = classify_entities(screenshot_bytes, candidates, THREAT_CLASSES)
                for eid, (owner, ratio) in results.items():
                    if owner == Owner.ENEMY or owner == Owner.UNKNOWN:
                        threats_found.append(eid)
            except Exception as e:
                log.warning("ownership_check_failed", error=str(e))
                # Fallback: treat all candidates as threats
                for entity in candidates:
                    threats_found.append(extract_attrs(entity).class_name)
        else:
            # No screenshot — legacy behavior (all military = threat)
            for entity in candidates:
                threats_found.append(extract_attrs(entity).class_name)

        # Require at least 3 enemy military units before raising the alarm.
        # exp_0013 (turn 14) showed a single spearman triggered alarm reasoning
        # that led the executor to ring the town bell, garrisoning all villagers
        # and collapsing economy. A single scout / spearman is not a threat —
        # the TC's auto-arrows handle one stray unit.
        if len(threats_found) >= 3:
            self._alarm_active = True
            log.warning("alarm_triggered", threats=threats_found[:5], count=len(threats_found))
            self._inject_emergency_goals(threats_found)
            return True

        if threats_found:
            log.debug("alarm_below_threshold", threats=threats_found, count=len(threats_found))

        self._alarm_active = False
        return False

    def _inject_emergency_goals(self, threats: list[str]) -> None:
        """Inject high-priority defensive goals when threats are detected."""
        # Don't duplicate if emergency goals already active
        emergency_names = {g.name for g in self.active_goals if g.priority == 10}
        if "Defend base" in emergency_names:
            return

        threat_types = set(threats)
        emergency_goals = [
            Goal(
                name="Defend base",
                type="local",
                metric="population",
                target=0,
                priority=10,
                created_turn=0,
            ),
        ]

        # Add to front of active goals (highest priority)
        self.active_goals = emergency_goals + self.active_goals
        log.info("emergency_goals_injected", threat_types=list(threat_types)[:5])
