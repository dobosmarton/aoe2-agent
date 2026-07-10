"""Memory management for AoE2 LLM Agent."""

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime

# AoE2 Dark Age starting values
INITIAL_RESOURCES = {"food": 200, "wood": 200, "gold": 100, "stone": 200}
INITIAL_POPULATION = 4
INITIAL_POPULATION_CAP = 5
STUCK_LOOP_THRESHOLD = 3


@dataclass
class Turn:
    """Single decision turn."""

    iteration: int
    timestamp: str
    reasoning: str
    actions: list[dict]
    observed_resources: dict | None = None
    observed_events: list[str] = field(default_factory=list)
    verification: str = ""
    goal_progress: dict = field(default_factory=dict)
    reward: float = 0.0


@dataclass
class GameState:
    """Structured game state extracted from LLM observations."""

    resources: dict = field(default_factory=lambda: dict(INITIAL_RESOURCES))
    population: int = INITIAL_POPULATION
    population_cap: int = INITIAL_POPULATION_CAP
    current_age: str = "Dark Age"
    idle_tc: bool = False
    # Whether any villager is idle, from the HUD badge colour (yellow=idle, grey=none).
    # None = unknown (badge not read yet); True/False = read. Callers must treat None
    # as "skip idle handling", never as False. Presence, not a count — the count digit
    # can't be OCR'd reliably, but the icon's lit state is unambiguous.
    idle_present: bool | None = None
    under_attack: bool = False
    enemy_located: bool = False
    enemy_location: str = ""


AGE_SCORES = {
    "Dark Age": 0.0,
    "Feudal Age": 0.33,
    "Castle Age": 0.66,
    "Imperial Age": 1.0,
}


class AgentMemory:
    """Manages agent memory across turns."""

    def __init__(self, working_memory_size: int = 10) -> None:
        """
        Initialize memory system.

        Args:
            working_memory_size: Number of recent turns to keep in working memory
        """
        self.working_memory: deque[Turn] = deque(maxlen=working_memory_size)
        self.episode_summary: str = ""
        self.game_state = GameState()
        self.turn_count: int = 0

        # Cumulative metrics for autoresearch scoring
        self.total_food_gathered: int = 0
        self.peak_population: int = 0
        self.total_actions: int = 0
        self.successful_actions: int = 0
        self.highest_age: str = "Dark Age"
        self.game_start_time: datetime | None = None
        self.game_end_reason: str = ""  # "victory", "defeat", "timeout", ""
        # Cross-game memory attribution. memories_loaded is the list of titles
        # injected into the system prompt at game start; memories_applied_count
        # accumulates how often the LLM tagged each title via [applied: ...]
        # in its reasoning. See _extract_applied_memories in game_loop.py.
        self.memories_loaded: list[str] = []
        self.memories_applied_count: dict[str, int] = {}

    def add_turn(self, turn: Turn) -> None:
        """Add a turn to working memory."""
        self.working_memory.append(turn)
        self.turn_count += 1

        # Start timer on first turn
        if self.game_start_time is None:
            self.game_start_time = datetime.now(UTC)

        # Track cumulative actions
        self.total_actions += len(turn.actions)

        # Update game state from turn observations
        if turn.observed_resources:
            self.game_state.resources.update(turn.observed_resources)
            # Accumulate food (track delta from previous)
            food = turn.observed_resources.get("food", 0)
            if food > 0:
                self.total_food_gathered = max(self.total_food_gathered, food)

    def update_from_observations(self, observations: dict) -> None:
        """Update game state from LLM observations."""
        if not observations:
            return

        # Update resources
        if "resources" in observations:
            self.game_state.resources.update(observations["resources"])
            # Track peak food across the game — fixes total_food_gathered=0 bug.
            # The strategist's readings flow through this method (via
            # GoalManager.update_resource_readings); the executor's also do via
            # create_turn. max() makes it idempotent across both paths.
            food = observations["resources"].get("food", 0)
            try:
                food_int = int(food)
            except (TypeError, ValueError):
                food_int = 0
            if food_int > 0:
                self.total_food_gathered = max(self.total_food_gathered, food_int)

        # Update population
        if "population" in observations:
            pop_str = observations["population"]
            if "/" in str(pop_str):
                parts = str(pop_str).split("/")
                try:
                    self.game_state.population = int(parts[0])
                    self.game_state.population_cap = int(parts[1])
                    # Track peak population
                    self.peak_population = max(self.peak_population, self.game_state.population)
                except (ValueError, IndexError):
                    pass

        # NOTE: `age` is intentionally NOT read from executor observations.
        # The executor self-reports `observations.age` but was observed hallucinating
        # it (exp_0011: reported "Feudal Age" from turn 2 while game was Dark Age),
        # which misrouted the age-specific prompt. Age is authoritative from the
        # strategist only — see `update_age()` below, called from
        # `GoalManager.update_resource_readings()`.

        # Idle-villager presence read off the HUD badge colour. A missing key leaves
        # the last-known value (never coerced — see GameState.idle_present).
        if "idle_present" in observations:
            self.game_state.idle_present = bool(observations["idle_present"])

        # Update flags
        if "idle_tc" in observations:
            self.game_state.idle_tc = bool(observations["idle_tc"])

        if "under_attack" in observations:
            self.game_state.under_attack = bool(observations["under_attack"])

    def update_age(self, age: str) -> None:
        """Update current age from the strategist's reading (authoritative).

        The strategist reads age directly from the resource bar — this is the
        single source of truth. Do NOT call this from the executor path.
        """
        if not age:
            return
        self.game_state.current_age = age
        if AGE_SCORES.get(age, 0) > AGE_SCORES.get(self.highest_age, 0):
            self.highest_age = age

    def get_context_for_llm(self) -> str:
        """Build context string for LLM prompt.

        NOTE: cross-game memories are NOT loaded here anymore. They live in the
        cached system prompt block (see ClaudeProvider._load_prompts) so they're
        paid once per game instead of every turn. This method only returns
        per-turn state.
        """
        parts = []

        # Current game state
        parts.append(f"## Current Game State\n{self._format_game_state()}")

        # Episode summary (if exists)
        if self.episode_summary:
            parts.append(f"## Previous Events Summary\n{self.episode_summary}")

        # Recent turns (working memory) - last 3 for loop detection
        if self.working_memory:
            recent_turns = list(self.working_memory)[-3:]
            recent_lines = []
            for turn in recent_turns:
                # Summarize actions with target info
                action_summary = ", ".join(
                    f"{a.get('type', '?')}({a.get('key', '')})"
                    if a.get("type") == "press"
                    else f"{a.get('type', '?')}({a.get('target_id') or ''} @ {a.get('x', '?')},{a.get('y', '?')})"
                    for a in turn.actions[:5]
                )
                line = (
                    f"Turn {turn.iteration}: {turn.reasoning[:100]}...\n  Actions: {action_summary}"
                )
                if turn.verification:
                    line += f"\n  Result: {turn.verification[:150]}"
                recent_lines.append(line)

            # Stuck-loop detection: count consecutive failures
            no_change_count = 0
            for turn in reversed(recent_turns):
                if turn.verification and (
                    "no visible change" in turn.verification or "FAILED" in turn.verification
                ):
                    no_change_count += 1
                else:
                    break

            header = "## Recent Decisions\n"
            if no_change_count >= STUCK_LOOP_THRESHOLD:
                header = f"## Recent Decisions\n**WARNING: Last {no_change_count} actions had NO EFFECT. You MUST try a completely different approach — different target, different task, or press H to reset.**\n"

            parts.append(header + "\n".join(recent_lines))

        return "\n\n".join(parts)

    def _format_game_state(self) -> str:
        """Format game state for display."""
        state = self.game_state
        is_housed = state.population >= state.population_cap and state.population_cap > 0
        lines = [
            f"- Resources: Food={state.resources['food']}, Wood={state.resources['wood']}, Gold={state.resources['gold']}, Stone={state.resources['stone']}",
            f"- Population: {state.population}/{state.population_cap}",
            f"- HOUSED (cannot create villagers!): {is_housed}" if is_housed else "- Housed: False",
            f"- Age: {state.current_age}",
            f"- TC Idle: {state.idle_tc}",
            f"- Under Attack: {state.under_attack}",
        ]
        # Idle-villager badge (HUD): None = unknown, so only show a known state.
        if state.idle_present is not None:
            lines.append(f"- Idle Villagers Present: {state.idle_present}")

        if state.enemy_located:
            lines.append(f"- Enemy Located: {state.enemy_location}")

        return "\n".join(lines)

    def set_last_verification(self, verification: str) -> None:
        """Attach verification result to the most recent turn."""
        if self.working_memory:
            self.working_memory[-1].verification = verification

    def record_action_results(self, success_count: int, total: int) -> None:
        """Record action execution results for metrics tracking."""
        self.successful_actions += success_count
        # total_actions already tracked in add_turn, but correct if executor filtered some
        pass

    def record_memories_applied(self, titles: list[str]) -> None:
        """Increment per-title attribution counts.

        Called from game_loop when the executor's reasoning had an
        `[applied: title1, title2]` prefix and the titles matched ones loaded
        into the cached system prompt.
        """
        for t in titles:
            self.memories_applied_count[t] = self.memories_applied_count.get(t, 0) + 1

    def get_game_duration_seconds(self) -> float:
        """Get elapsed game time in seconds."""
        if self.game_start_time is None:
            return 0.0
        return (datetime.now(UTC) - self.game_start_time).total_seconds()

    def get_metrics_snapshot(self) -> dict:
        """Return current cumulative metrics for scoring."""
        return {
            "survival_time": self.get_game_duration_seconds(),
            "peak_population": self.peak_population,
            "highest_age": self.highest_age,
            "age_score": AGE_SCORES.get(self.highest_age, 0.0),
            "total_food_gathered": self.total_food_gathered,
            "total_actions": self.total_actions,
            "successful_actions": self.successful_actions,
            "action_success_rate": (
                self.successful_actions / self.total_actions if self.total_actions > 0 else 0.0
            ),
            "turn_count": self.turn_count,
            "game_end_reason": self.game_end_reason,
            "memories_loaded": list(self.memories_loaded),
            "memories_used": dict(self.memories_applied_count),
        }

    def reset(self) -> None:
        """Reset memory for a new game."""
        self.working_memory.clear()
        self.episode_summary = ""
        self.game_state = GameState()
        self.turn_count = 0
        self.total_food_gathered = 0
        self.peak_population = 0
        self.total_actions = 0
        self.successful_actions = 0
        self.highest_age = "Dark Age"
        self.game_start_time = None
        self.game_end_reason = ""
        self.memories_loaded = []
        self.memories_applied_count = {}

    def create_turn(
        self,
        reasoning: str,
        actions: list[dict],
        observations: dict | None = None,
    ) -> Turn:
        """Create a new turn and add it to memory."""
        turn = Turn(
            iteration=self.turn_count + 1,
            timestamp=datetime.now(UTC).strftime("%Y%m%d_%H%M%S"),
            reasoning=reasoning,
            actions=actions,
            observed_resources=observations.get("resources") if observations else None,
            observed_events=observations.get("events", []) if observations else [],
        )

        # Update state from observations
        if observations:
            self.update_from_observations(observations)

        # Add to working memory
        self.add_turn(turn)

        return turn
