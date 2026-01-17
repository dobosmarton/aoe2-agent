"""Memory management for AoE2 LLM Agent."""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Turn:
    """Single decision turn."""

    iteration: int
    timestamp: str
    reasoning: str
    actions: list[dict]
    observed_resources: dict | None = None
    observed_events: list[str] = field(default_factory=list)


@dataclass
class GameState:
    """Structured game state extracted from LLM observations."""

    resources: dict = field(
        default_factory=lambda: {"food": 0, "wood": 0, "gold": 0, "stone": 0}
    )
    population: int = 0
    population_cap: int = 0
    current_age: str = "Dark Age"
    idle_tc: bool = False
    under_attack: bool = False
    enemy_located: bool = False
    enemy_location: str = ""


class AgentMemory:
    """Manages agent memory across turns."""

    def __init__(self, working_memory_size: int = 10):
        """
        Initialize memory system.

        Args:
            working_memory_size: Number of recent turns to keep in working memory
        """
        self.working_memory: deque[Turn] = deque(maxlen=working_memory_size)
        self.episode_summary: str = ""
        self.game_state = GameState()
        self.turn_count: int = 0

    def add_turn(self, turn: Turn) -> None:
        """Add a turn to working memory."""
        self.working_memory.append(turn)
        self.turn_count += 1

        # Update game state from turn observations
        if turn.observed_resources:
            self.game_state.resources.update(turn.observed_resources)

    def update_from_observations(self, observations: dict) -> None:
        """Update game state from LLM observations."""
        if not observations:
            return

        # Update resources
        if "resources" in observations:
            self.game_state.resources.update(observations["resources"])

        # Update population
        if "population" in observations:
            pop_str = observations["population"]
            if "/" in str(pop_str):
                parts = str(pop_str).split("/")
                try:
                    self.game_state.population = int(parts[0])
                    self.game_state.population_cap = int(parts[1])
                except (ValueError, IndexError):
                    pass

        # Update age
        if "age" in observations:
            self.game_state.current_age = observations["age"]

        # Update flags
        if "idle_tc" in observations:
            self.game_state.idle_tc = bool(observations["idle_tc"])

        if "under_attack" in observations:
            self.game_state.under_attack = bool(observations["under_attack"])

    def get_context_for_llm(self) -> str:
        """Build context string for LLM prompt."""
        parts = []

        # Current game state
        parts.append(f"## Current Game State\n{self._format_game_state()}")

        # Episode summary (if exists)
        if self.episode_summary:
            parts.append(f"## Previous Events Summary\n{self.episode_summary}")

        # Recent turns (working memory) - last 3 for brevity
        if self.working_memory:
            recent_turns = list(self.working_memory)[-3:]
            recent_lines = []
            for t in recent_turns:
                # Summarize actions
                action_summary = ", ".join(
                    f"{a.get('type', '?')}({a.get('key', '')})" if a.get('type') == 'press'
                    else f"{a.get('type', '?')}({a.get('x', '?')},{a.get('y', '?')})"
                    for a in t.actions[:5]  # Limit to first 5 actions
                )
                recent_lines.append(
                    f"Turn {t.iteration}: {t.reasoning[:100]}...\n  Actions: {action_summary}"
                )
            parts.append(f"## Recent Decisions\n" + "\n".join(recent_lines))

        return "\n\n".join(parts)

    def _format_game_state(self) -> str:
        """Format game state for display."""
        gs = self.game_state
        is_housed = gs.population >= gs.population_cap and gs.population_cap > 0
        lines = [
            f"- Resources: Food={gs.resources['food']}, Wood={gs.resources['wood']}, Gold={gs.resources['gold']}, Stone={gs.resources['stone']}",
            f"- Population: {gs.population}/{gs.population_cap}",
            f"- HOUSED (cannot create villagers!): {is_housed}" if is_housed else f"- Housed: False",
            f"- Age: {gs.current_age}",
            f"- TC Idle: {gs.idle_tc}",
            f"- Under Attack: {gs.under_attack}",
        ]

        if gs.enemy_located:
            lines.append(f"- Enemy Located: {gs.enemy_location}")

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset memory for a new game."""
        self.working_memory.clear()
        self.episode_summary = ""
        self.game_state = GameState()
        self.turn_count = 0

    def create_turn(
        self,
        reasoning: str,
        actions: list[dict],
        observations: dict | None = None,
    ) -> Turn:
        """Create a new turn and add it to memory."""
        turn = Turn(
            iteration=self.turn_count + 1,
            timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
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
