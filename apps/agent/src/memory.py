"""Memory management for AoE2 LLM Agent."""

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol, TypedDict, cast

if TYPE_CHECKING:
    from .executor import ActionLedger

from .policy.state import PolicyState
from .turn_timing import ACT_LOOP, PERCEIVE_LOOP, TURN_LOOP, LatencySnapshot


class LatencySource(Protocol):
    """The game loop's latency recorder, kept structural so tests can stub it."""

    def snapshot(self) -> LatencySnapshot: ...


# AoE2 Dark Age starting values
INITIAL_RESOURCES = {"food": 200, "wood": 200, "gold": 100, "stone": 200}
INITIAL_POPULATION = 4
INITIAL_POPULATION_CAP = 5
STUCK_LOOP_THRESHOLD = 3
# Largest believable food gain between two consecutive HUD reads (~30 s of a
# full late-Dark-Age economy). Bigger jumps are OCR glitches, not income.
FOOD_GAIN_SANITY_CAP = 300


@dataclass
class Turn:
    """Single decision turn."""

    iteration: int
    timestamp: str
    reasoning: str
    actions: list[dict[str, object]]
    observed_resources: dict[str, int] | None = None
    observed_events: list[str] = field(default_factory=list)
    verification: str = ""
    goal_progress: dict[str, object] = field(default_factory=dict)
    reward: float = 0.0


@dataclass
class GameState:
    """Structured game state extracted from LLM observations."""

    resources: dict[str, int] = field(default_factory=lambda: dict(INITIAL_RESOURCES))
    population: int = INITIAL_POPULATION
    population_cap: int = INITIAL_POPULATION_CAP
    villagers: int | None = None
    worker_counts: dict[str, int] = field(default_factory=dict)
    current_age: str = "Dark Age"
    idle_tc: bool | None = None
    # Whether any villager is idle, from the HUD badge colour (yellow=idle, grey=none).
    # None = unknown (badge not read yet); True/False = read. Callers must treat None
    # as "skip idle handling", never as False. Presence stays the robust gate.
    idle_present: bool | None = None
    # How many villagers are idle, from the badge's corner digit via template NCC
    # (resource_ocr.read_idle_count). None = digit unreadable this frame — fall back
    # to presence-only behavior, never treat as 0.
    idle_count: int | None = None
    # Consecutive game-loop iterations the badge has stayed lit (idle_present True).
    # Maintained once per iteration by the game loop, NOT by observation updates
    # (those fire more than once per turn). A long streak with a tiny idle_count
    # means the digit is under-reading — the reactive tier then distrusts the count
    # (2026-07-11 run review, F-4: digit pinned at 1 while 8 villagers idled).
    idle_streak: int = 0
    # Building classes known to exist this game — copied once per iteration by
    # the game loop from the executor's build-gate evidence (detection sightings
    # + wood-delta-confirmed purchases). Drives the reactive tier's Feudal prep
    # (build the lumber camp) and gates the age-up press on the two-building
    # requirement (run 6, F-26).
    buildings_seen: frozenset[str] = frozenset()
    # Villagers ordered this game (incl. the starting 4) — copied once per
    # iteration from the executor's order ledger. The reactive queue gates on
    # this, NOT on population: orders lead the delivered HUD population by the
    # TC queue depth, and braking on population over-delivered 40 villagers
    # (run 11, F-38).
    villagers_ordered: int = 0
    under_attack: bool = False
    enemy_located: bool = False
    enemy_location: str = ""


AGE_SCORES = {
    "Dark Age": 0.0,
    "Feudal Age": 0.33,
    "Castle Age": 0.66,
    "Imperial Age": 1.0,
}


class MetricsSnapshot(TypedDict):
    """The cumulative-metrics contract autoresearch scores games by.

    A TypedDict (not a plain dict) so a misspelled key in a consumer is a
    type error, and adding a metric here forces the snapshot literal in
    `get_metrics_snapshot` to supply it.
    """

    survival_time: float
    peak_population: int
    highest_age: str
    age_score: float
    total_food_gathered: int
    total_actions: int
    successful_actions: int
    executed_actions: int
    action_success_rate: float
    accepted_inputs: int
    attempted_inputs: int
    input_acceptance_rate: float
    unverifiable_action_ids: list[int]
    score_valid: bool
    turn_count: int
    game_end_reason: str
    memories_loaded: list[str]
    memories_used: dict[str, int]
    # Executor health (T-533). llm_calls = turns the executor was asked to plan;
    # llm_errors = turns where every LLM path failed (see LLMResult.error).
    # llm_error_rate near 1.0 means the game was played by the reactive tier
    # alone — a dead-executor run that must NOT read as a valid experiment
    # (run 12: 90 errors, still accepted=true).
    llm_calls: int
    llm_errors: int
    llm_error_rate: float
    # Seconds to reach each age; None when the age was never reached (plan 2.1).
    feudal_time_s: float | None
    castle_time_s: float | None
    # Latency (plan 0.3). 0.0 when that loop recorded no tick. The turn_*
    # fields cover the single-tick loop; loop_arch names the architecture.
    turn_latency_p50_ms: float
    turn_latency_p90_ms: float
    turn_latency_max_ms: float
    phase_latency_p50_ms: dict[str, float]
    act_latency_p95_ms: float
    perceive_latency_p50_ms: float
    loop_arch: str


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
        self._last_food_reading: int | None = None
        self.peak_population: int = 0
        self.total_actions: int = 0
        self.successful_actions: int = 0
        # Denominator for action_success_rate: actions actually EXECUTED (the
        # numerator successful_actions counts their successes). total_actions
        # counts PLANNED actions per turn — a different population (fallback and
        # composite executions never enter turn.actions), which is why the old
        # successful/total rate exceeded 1.0 (runs 1 and 3).
        self.executed_actions: int = 0
        self.accepted_inputs: int = 0
        self.attempted_inputs: int = 0
        self.action_ledger: ActionLedger | None = None
        # Executor-outage tracking (T-533). llm_calls/llm_errors feed
        # llm_error_rate; _llm_error_streak is the current run of consecutive
        # failed executor turns, which the game loop alarms on.
        self.llm_calls: int = 0
        self.llm_errors: int = 0
        self._llm_error_streak: int = 0
        self.highest_age: str = "Dark Age"
        # Seconds from game start to the FIRST reading of each age.
        self.age_times: dict[str, float] = {}
        self.game_start_time: datetime | None = None
        self.game_end_reason: str = ""  # "victory", "defeat", "timeout", ""
        # Cross-game memory attribution. memories_loaded is the list of titles
        # injected into the system prompt at game start; memories_applied_count
        # accumulates how often the LLM tagged each title via [applied: ...]
        # in its reasoning. See _extract_applied_memories in game_loop.py.
        self.memories_loaded: list[str] = []
        self.memories_applied_count: dict[str, int] = {}
        # Set by the game loop. None on the scenario and synth paths, which run
        # no timed loop.
        self.latency: LatencySource | None = None

    def start_game(self) -> None:
        """Start the game clock. `survival_time` and every `age_times` entry
        hang off it, so a game with no LLM turn would score 0 on both."""
        if self.game_start_time is None:
            self.game_start_time = datetime.now(UTC)

    def add_turn(self, turn: Turn) -> None:
        """Add a turn to working memory."""
        self.working_memory.append(turn)
        self.turn_count += 1
        self.start_game()  # the `--test` path has no supervisor to start it

        # Track cumulative actions
        self.total_actions += len(turn.actions)

        # A model's observation is useful history, never a HUD measurement.

    def update_from_observations(self, _observations: dict[str, object]) -> None:
        """Keep model hypotheses in turn history, never in observed game state."""

    def apply_hud_readings(self, readings: dict[str, object]) -> None:
        """Perception-only entry point for observed resources and population."""
        for kind in ("food", "wood", "gold", "stone"):
            value = readings.get(kind)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                self.game_state.resources[kind] = value
        population = readings.get("population")
        if isinstance(population, str):
            current, separator, capacity = population.partition("/")
            if separator and current.isdigit() and capacity.isdigit():
                self.game_state.population = int(current)
                self.game_state.population_cap = int(capacity)
                self.peak_population = max(self.peak_population, int(current))
        idle_present = readings.get("idle_present")
        if isinstance(idle_present, bool):
            self.game_state.idle_present = idle_present
        idle_count = readings.get("idle_count")
        self.game_state.idle_count = idle_count if isinstance(idle_count, int) else None
        villagers = readings.get("villagers")
        self.game_state.villagers = villagers if isinstance(villagers, int) else None
        self.game_state.worker_counts = {
            kind: value
            for kind in ("food", "wood", "gold", "stone")
            if isinstance(value := readings.get(f"{kind}_workers"), int)
        }

    def update_age(self, age: str) -> None:
        """Update current age from the strategist's reading (authoritative).

        The strategist reads age directly from the resource bar — this is the
        single source of truth. Do NOT call this from the executor path.
        """
        if not age:
            return
        self.game_state.current_age = age
        # First arrival only — the strategist re-reports the same age every turn.
        if age not in self.age_times and self.game_start_time is not None:
            self.age_times[age] = self.get_game_duration_seconds()
        if AGE_SCORES.get(age, 0) > AGE_SCORES.get(self.highest_age, 0):
            self.highest_age = age

    def get_context_for_llm(self, observed_state: PolicyState | None = None) -> str:
        """Build context string for LLM prompt.

        NOTE: cross-game memories are NOT loaded here anymore. They live in the
        cached system prompt block (see ExecutorProvider._load_prompts) so they're
        paid once per game instead of every turn. This method only returns
        per-turn state.
        """
        parts = []

        # Current game state
        current_state = (
            self._format_game_state()
            if observed_state is None
            else self._format_observed_state(observed_state)
        )
        parts.append(f"## Current Game State\n{current_state}")

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

            no_change_count = self.no_change_streak()
            header = "## Recent Decisions\n"
            if no_change_count >= STUCK_LOOP_THRESHOLD:
                header = f"## Recent Decisions\n**WARNING: Last {no_change_count} actions had NO EFFECT. You MUST try a completely different approach — different target, different task, or press H to reset.**\n"

            parts.append(header + "\n".join(recent_lines))

        return "\n\n".join(parts)

    @staticmethod
    def _format_observed_state(state: PolicyState) -> str:
        resources = ", ".join(
            f"{name}={getattr(state, name) if name in state.known_resources else '?'}"
            for name in ("food", "wood", "gold", "stone")
        )
        return "\n".join(
            (
                f"- Resources: {resources}",
                f"- Population: {state.population}/{state.population_cap}",
                f"- Age: {state.age}",
                f"- Idle Villagers: {state.idle_count if state.idle_count is not None else state.idle_present}",
                f"- Confirmed Buildings: {', '.join(sorted(state.buildings_seen)) or 'none'}",
                f"- Pending Actions: {', '.join(sorted(state.pending_actions)) or 'none'}",
                f"- Reserved Resources: {dict(state.reserved_resources)}",
            )
        )

    def no_change_streak(self) -> int:
        """Trailing turns whose actions had no visible effect.

        The deliberate loop's "stuck" trigger and the prompt warning read the
        same count, so a change to one cannot drift from the other.
        """
        streak = 0
        for turn in reversed(self.working_memory):
            if not turn.verification:
                break
            if "no visible change" not in turn.verification and "FAILED" not in turn.verification:
                break
            streak += 1
        return streak

    def _format_game_state(self) -> str:
        """Format game state for display."""
        state = self.game_state
        is_housed = state.population >= state.population_cap and state.population_cap > 0
        lines = [
            f"- Resources: Food={state.resources['food']}, Wood={state.resources['wood']}, Gold={state.resources['gold']}, Stone={state.resources['stone']}",
            f"- Population: {state.population}/{state.population_cap}",
            f"- HOUSED (cannot create villagers!): {is_housed}" if is_housed else "- Housed: False",
            f"- Age: {state.current_age}",
            f"- TC Idle: {state.idle_tc if state.idle_tc is not None else 'unknown'}",
            f"- Under Attack: {state.under_attack}",
        ]
        # Idle-villager badge (HUD): None = unknown, so only show a known state.
        # The exact count (badge digit) beats the presence boolean when readable.
        if state.idle_count is not None:
            lines.append(f"- Idle Villagers: {state.idle_count}")
        elif state.idle_present is not None:
            lines.append(f"- Idle Villagers Present: {state.idle_present}")

        if state.enemy_located:
            lines.append(f"- Enemy Located: {state.enemy_location}")

        return "\n".join(lines)

    def set_last_verification(self, verification: str) -> None:
        """Attach verification result to the most recent turn."""
        if self.working_memory:
            self.working_memory[-1].verification = verification

    def record_action_results(self, success_count: int, total: int) -> None:
        """Input acceptance is diagnostic; only observed effects score success."""
        self.accepted_inputs += success_count
        self.attempted_inputs += total

    def record_llm_outcome(self, *, errored: bool) -> int:
        """Record one executor turn's success/failure; return the failure streak.

        Every executor call funnels through here (T-533). `errored` is the
        LLMResult.error flag — True only when every LLM path failed and the
        turn is a safe-wait no-op. The returned consecutive-failure streak lets
        the game loop raise a loud outage alarm; a success resets it to 0.
        """
        self.llm_calls += 1
        if errored:
            self.llm_errors += 1
            self._llm_error_streak += 1
        else:
            self._llm_error_streak = 0
        return self._llm_error_streak

    def record_food_reading(self, food: int) -> None:
        """Accumulate gathered food from consecutive HUD (OCR) readings.

        Sum of positive deltas between reads ≈ income; food spent within an
        interval hides some income, so this UNDERCOUNTS, never overcounts.
        Jumps beyond FOOD_GAIN_SANITY_CAP are OCR glitches and are dropped
        (the reading still becomes the new baseline). Call only from the OCR
        path — LLM-echoed resource observations hallucinate.
        """
        prev = self._last_food_reading
        self._last_food_reading = food
        if prev is None:
            return
        delta = food - prev
        if 0 < delta <= FOOD_GAIN_SANITY_CAP:
            self.total_food_gathered += delta

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

    def get_metrics_snapshot(self) -> MetricsSnapshot:
        """Return current cumulative metrics for scoring."""
        ledger = self.action_ledger
        attempted = len(ledger.attempted_economic_ids) if ledger is not None else 0
        confirmed = len(ledger.confirmed_economic_ids) if ledger is not None else 0
        unresolved = sorted(ledger.uncertain_operations) if ledger is not None else []
        latency = self.latency.snapshot() if self.latency is not None else LatencySnapshot()
        turn = latency.of(TURN_LOOP)
        act = latency.of(ACT_LOOP)
        perceive = latency.of(PERCEIVE_LOOP)
        return {
            "survival_time": self.get_game_duration_seconds(),
            "peak_population": self.peak_population,
            "highest_age": self.highest_age,
            "age_score": AGE_SCORES.get(self.highest_age, 0.0),
            "total_food_gathered": ledger.food_gathered
            if ledger is not None
            else self.total_food_gathered,
            "total_actions": self.total_actions,
            "successful_actions": confirmed,
            "executed_actions": attempted,
            "action_success_rate": (confirmed / attempted if attempted > 0 else 0.0),
            "accepted_inputs": self.accepted_inputs,
            "attempted_inputs": self.attempted_inputs,
            "input_acceptance_rate": self.accepted_inputs / self.attempted_inputs
            if self.attempted_inputs
            else 0.0,
            "unverifiable_action_ids": unresolved,
            "score_valid": not unresolved,
            "turn_count": self.turn_count,
            "game_end_reason": self.game_end_reason,
            "memories_loaded": list(self.memories_loaded),
            "memories_used": dict(self.memories_applied_count),
            "llm_calls": self.llm_calls,
            "llm_errors": self.llm_errors,
            "llm_error_rate": (self.llm_errors / self.llm_calls if self.llm_calls > 0 else 0.0),
            "feudal_time_s": self.age_times.get("Feudal Age"),
            "castle_time_s": self.age_times.get("Castle Age"),
            "turn_latency_p50_ms": turn.p50_ms,
            "turn_latency_p90_ms": turn.p90_ms,
            "turn_latency_max_ms": turn.max_ms,
            "phase_latency_p50_ms": turn.phase_p50_ms,
            "act_latency_p95_ms": act.p95_ms,
            "perceive_latency_p50_ms": perceive.p50_ms,
            # Presence, not duration: a fast act tick still rounds to 0.0 ms.
            "loop_arch": "clocks" if ACT_LOOP in latency.loops else TURN_LOOP,
        }

    def reset(self) -> None:
        """Reset memory for a new game."""
        self.working_memory.clear()
        self.episode_summary = ""
        self.game_state = GameState()
        self.turn_count = 0
        self.total_food_gathered = 0
        self._last_food_reading = None
        self.peak_population = 0
        self.total_actions = 0
        self.successful_actions = 0
        self.executed_actions = 0
        self.accepted_inputs = 0
        self.attempted_inputs = 0
        self.action_ledger = None
        self.llm_calls = 0
        self.llm_errors = 0
        self._llm_error_streak = 0
        self.highest_age = "Dark Age"
        self.age_times = {}
        self.game_start_time = None
        self.game_end_reason = ""
        self.memories_loaded = []
        self.memories_applied_count = {}
        # Drop the recorder so a reused AgentMemory cannot report last game's
        # latency; the loop attaches a fresh one.
        self.latency = None

    def create_turn(
        self,
        reasoning: str,
        actions: list[dict[str, object]],
        observations: dict[str, object] | None = None,
    ) -> Turn:
        """Create a new turn and add it to memory."""
        # Boundary casts (LLM-echoed observations) — runtime behavior unchanged.
        turn = Turn(
            iteration=self.turn_count + 1,
            timestamp=datetime.now(UTC).strftime("%Y%m%d_%H%M%S"),
            reasoning=reasoning,
            actions=actions,
            observed_resources=cast(
                "dict[str, int] | None", observations.get("resources") if observations else None
            ),
            observed_events=cast(
                "list[str]", observations.get("events", []) if observations else []
            ),
        )

        # Add to working memory
        self.add_turn(turn)

        return turn
