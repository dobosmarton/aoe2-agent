"""Orchestrator — main autoresearch experiment loop.

Coordinates prompt mutation, game running, scoring, and accept/reject decisions.

Usage:
    python -m autoresearch.orchestrator [--max-experiments 5] [--time-budget 1200]
"""

import argparse
import asyncio
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

import structlog
from autoresearch.experiment_log import (
    get_best_score,
    get_next_experiment_id,
    get_recent_experiments,
    log_experiment,
)
from autoresearch.game_runner import run_game
from autoresearch.metrics import GameScore
from autoresearch.pareto import (
    ParetoEntry,
    dominates,
    load_frontier,
    save_frontier,
    update_frontier,
)
from autoresearch.prompt_mutator import PromptMutator
from autoresearch.trace import load_recent_traces

log = structlog.stdlib.get_logger()

REPO_ROOT = Path(__file__).parent.parent
EPSILON = 0.02  # Accept if score >= best - epsilon

# Successive-halving tournament defaults. Human starts each game, so these are
# kept small: N=3, 2 rounds, keep half ~= 5 supervised games per accepted change.
TOURNAMENT_CANDIDATES = 3
TOURNAMENT_ROUNDS = 2
TOURNAMENT_KEEP_FRACTION = 0.5
TOURNAMENT_GAMES_BUDGET = 6  # hard cap on total games per tournament
_MAX_PARETO_EXTRAS = 1  # extra non-dominated survivors kept per halving round (A2)


@dataclass
class _Candidate:
    """A prompt edit competing in a successive-halving tournament."""

    candidate_id: str
    change: dict
    games: list[GameScore] = field(default_factory=list)

    def mean_composite(self) -> float:
        """Mean composite score across this candidate's games (0.0 if none yet)."""
        return mean(g.composite for g in self.games) if self.games else 0.0

    def vector(self) -> tuple[float, float, float, float, float]:
        """Mean per-component score vector across this candidate's games."""
        if not self.games:
            return (0.0, 0.0, 0.0, 0.0, 0.0)
        return (
            mean(g.age for g in self.games),
            mean(g.age_speed for g in self.games),
            mean(g.economy for g in self.games),
            mean(g.action_success for g in self.games),
            mean(g.survival for g in self.games),
        )


def _latest_breakdown(recent: list[dict]) -> dict[str, float]:
    """Extract the 5 component scores from the most recent experiment row."""
    if not recent:
        return {}
    latest = recent[-1]
    out: dict[str, float] = {}
    for axis in ("survival", "population", "age", "economy", "action_success"):
        try:
            out[axis] = float(latest.get(axis, 0.0))
        except (TypeError, ValueError):
            out[axis] = 0.0
    return out


class Orchestrator:
    """Coordinates the prompt optimization experiment loop."""

    def __init__(self, epsilon: float = EPSILON) -> None:
        self.mutator = PromptMutator()
        self.best_score = get_best_score(loop="prompt")
        self.epsilon = epsilon

    def git_commit(self, message: str) -> str:
        """Commit prompt changes and return short SHA."""
        subprocess.run(
            ["git", "add", "prompts/system.md"],
            cwd=REPO_ROOT,
            capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=REPO_ROOT,
            capture_output=True,
        )
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        return result.stdout.strip()

    def git_revert_prompt(self) -> None:
        """Revert prompts/system.md and commit the revert."""
        self.mutator.revert()
        subprocess.run(
            ["git", "add", "prompts/system.md"],
            cwd=REPO_ROOT,
            capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "[autoresearch] revert: prompt change rejected"],
            cwd=REPO_ROOT,
            capture_output=True,
        )

    @staticmethod
    def _candidate_applies(change: dict, current_prompt: str) -> bool:
        """Whether the change's old_text exists verbatim in the current prompt."""
        old_text = str(change.get("old_text", ""))
        return bool(old_text) and old_text in current_prompt

    @staticmethod
    def _keep_top(candidates: list[_Candidate], keep_fraction: float) -> list[_Candidate]:
        """Keep the top fraction by mean composite, plus up to one non-dominated
        off-axis candidate so a single-axis strength isn't discarded (A2)."""
        ranked = sorted(candidates, key=lambda c: c.mean_composite(), reverse=True)
        keep = max(1, int(len(ranked) * keep_fraction))
        top = ranked[:keep]
        extras = 0
        for c in ranked[keep:]:
            if extras >= _MAX_PARETO_EXTRAS:
                break
            if not any(dominates(o.vector(), c.vector()) for o in candidates if o is not c):
                top.append(c)
                extras += 1
        return top

    @staticmethod
    def _record_frontier(winner: _Candidate) -> None:
        """Persist the accepted winner onto the Pareto frontier (A2)."""
        entry = ParetoEntry(
            candidate_id=winner.candidate_id,
            description=str(winner.change.get("description", "")),
            change={str(k): str(v) for k, v in winner.change.items()},
            vector=winner.vector(),
        )
        save_frontier(update_frontier(load_frontier(), entry))

    async def _play_candidate(self, change: dict, time_budget: float) -> GameScore:
        """Apply a candidate edit, play one game, then revert to baseline.

        Reverting in `finally` guarantees the prompt returns to baseline even if
        the game errors, so the next candidate applies against the same text.
        """
        self.mutator.apply_change(str(change["old_text"]), str(change["new_text"]))
        self.git_commit(f"[autoresearch] trial: {str(change.get('description', ''))[:50]}")
        try:
            result = await run_game(time_budget=time_budget)
        finally:
            self.git_revert_prompt()
        return result["score"]

    async def _run_halving_rounds(
        self,
        survivors: list[_Candidate],
        tournament_id: str,
        time_budget: float,
        halving_rounds: int,
        keep_fraction: float,
        games_budget: int,
    ) -> int:
        """Play successive-halving rounds, returning the total games played.

        Mutates `survivors` in place: after each round it is trimmed to the top
        fraction. Stops early when the games budget is exhausted.
        """
        games_played = 0
        round_num = 0
        while round_num < halving_rounds:
            round_num += 1
            for game_in_round, cand in enumerate(survivors, start=1):
                if games_played >= games_budget:
                    log.info("tournament_games_budget_reached", games=games_played)
                    return games_played
                input(
                    f"\nStart a new game in AoE2 (round {round_num}, {cand.candidate_id}: "
                    f"{str(cand.change.get('description', ''))[:50]}), then press Enter..."
                )
                score = await self._play_candidate(cand.change, time_budget)
                cand.games.append(score)
                games_played += 1
                log_experiment(
                    experiment_id=get_next_experiment_id(),
                    loop="prompt",
                    change_description=str(cand.change.get("description", "?")),
                    score=score,
                    accepted=False,
                    git_sha="(reverted)",
                    tournament_id=tournament_id,
                    candidate_id=cand.candidate_id,
                    round_num=str(round_num),
                    game_in_round=str(game_in_round),
                )
            # Play first, then trim: a lone survivor still gets evaluated.
            if len(survivors) <= 1:
                break
            survivors[:] = self._keep_top(survivors, keep_fraction)
        return games_played

    async def run_tournament(
        self,
        time_budget: float = 1200,
        n_candidates: int = TOURNAMENT_CANDIDATES,
        halving_rounds: int = TOURNAMENT_ROUNDS,
        keep_fraction: float = TOURNAMENT_KEEP_FRACTION,
        games_budget: int = TOURNAMENT_GAMES_BUDGET,
    ) -> dict:
        """Race N candidate prompt edits via successive halving.

        Each round plays the survivors one game each (sequential, human-started),
        keeps the top fraction by mean composite score, then doubles down on the
        rest. The single winner is accepted only if its mean beats the baseline
        by more than epsilon. Trial games always revert; only the winner is kept.
        """
        tournament_id = get_next_experiment_id()
        recent = get_recent_experiments(5)
        failure_modes = self._extract_failure_modes(recent)
        current_prompt = self.mutator.read_current_prompt()
        changes = self.mutator.propose_changes(
            current_prompt,
            load_recent_traces(3),
            _latest_breakdown(recent),
            failure_modes,
            n=n_candidates,
        )
        survivors = [
            _Candidate(candidate_id=f"c{i + 1}", change=c)
            for i, c in enumerate(changes)
            if self._candidate_applies(c, current_prompt)
        ]
        if not survivors:
            log.warning("tournament_no_candidates", tournament_id=tournament_id)
            return {"experiment_id": tournament_id, "error": "no_candidates"}

        games_played = await self._run_halving_rounds(
            survivors, tournament_id, time_budget, halving_rounds, keep_fraction, games_budget
        )

        played = [c for c in survivors if c.games]
        if not played:
            return {"experiment_id": tournament_id, "error": "no_games_played"}

        winner = max(played, key=lambda c: c.mean_composite())
        winner_mean = winner.mean_composite()
        accepted = winner_mean >= self.best_score - self.epsilon
        sha = None
        if accepted:
            self.mutator.apply_change(
                str(winner.change["old_text"]), str(winner.change["new_text"])
            )
            sha = self.git_commit(f"[autoresearch] {tournament_id}: {winner.change['description']}")
            self.best_score = max(self.best_score, winner_mean)
            self._record_frontier(winner)

        log_experiment(
            experiment_id=tournament_id,
            loop="prompt",
            change_description=f"tournament winner: {winner.change.get('description', '?')}",
            score=winner.games[-1],
            accepted=accepted,
            git_sha=sha,
            tournament_id=tournament_id,
            candidate_id=winner.candidate_id,
            round_num="final",
        )
        log.info(
            "tournament_complete",
            tournament_id=tournament_id,
            winner=winner.candidate_id,
            winner_mean=round(winner_mean, 4),
            accepted=accepted,
            games_played=games_played,
        )
        return {
            "experiment_id": tournament_id,
            "score": winner_mean,
            "accepted": accepted,
            "description": winner.change.get("description", "?"),
            "candidates": len(changes),
            "games_played": games_played,
        }

    async def run_tournament_loop(
        self,
        max_experiments: int | None,
        time_budget: float,
        n_candidates: int,
        halving_rounds: int,
        keep_fraction: float,
        games_budget: int,
    ) -> None:
        """Run repeated tournaments, each producing at most one accepted change."""
        count = 0
        while max_experiments is None or count < max_experiments:
            print(f"\n{'=' * 60}")
            print(f"Tournament {count + 1} — Best score: {self.best_score:.4f}")
            print(f"{'=' * 60}")

            result = await self.run_tournament(
                time_budget=time_budget,
                n_candidates=n_candidates,
                halving_rounds=halving_rounds,
                keep_fraction=keep_fraction,
                games_budget=games_budget,
            )

            if "error" in result:
                print(f"\n  Error: {result['error']}")
            else:
                status = "ACCEPTED" if result["accepted"] else "REJECTED"
                print(f"\n  {status}: {result.get('description', '?')}")
                print(f"  Winner mean: {result['score']:.4f} (best: {self.best_score:.4f})")
            count += 1

        print(f"\n{'=' * 60}")
        print(f"Done — {count} tournaments completed")
        print(f"Best score: {self.best_score:.4f}")
        print(f"{'=' * 60}")

    def _extract_failure_modes(self, recent: list[dict]) -> list[str]:
        """Identify failure patterns from recent experiments."""
        modes = []
        if not recent:
            return ["No data yet — this is the first experiment"]

        latest = recent[-1]
        try:
            pop = float(latest.get("population", 0))
            age = float(latest.get("age", 0))
            economy = float(latest.get("economy", 0))
            action_success = float(latest.get("action_success", 0))
            survival = float(latest.get("survival", 0))

            if pop < 0.2:
                modes.append(
                    "Population stayed very low — agent may not be queueing villagers or is getting housed"
                )
            if age == 0:
                modes.append(
                    "Agent never advanced past Dark Age — needs to accumulate 500 food and click age-up"
                )
            if economy < 0.1:
                modes.append(
                    "Very little food gathered — agent may not be assigning villagers to food sources"
                )
            if action_success < 0.3:
                modes.append(
                    "Low action success rate — many actions failing, possibly clicking wrong coordinates"
                )
            if survival < 0.5:
                modes.append("Game ended early — agent may be dying or getting stuck")

            end_reason = latest.get("game_end_reason", "")
            if end_reason == "defeat":
                modes.append("Agent was defeated — needs better military or defensive strategy")
        except (ValueError, TypeError):
            pass

        return (
            modes if modes else ["No clear failure patterns — try optimizing strongest components"]
        )


class _OrchestratorArgs(argparse.Namespace):
    max_experiments: int | None
    time_budget: float
    n_candidates: int
    halving_rounds: int
    keep_fraction: float
    games_budget: int


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run autoresearch prompt optimization (successive-halving tournament)"
    )
    parser.add_argument(
        "--max-experiments",
        type=int,
        default=None,
        help="Maximum number of tournaments (default: run forever)",
    )
    parser.add_argument(
        "--time-budget",
        type=float,
        default=1200,
        help="Maximum game duration in seconds (default: 1200 = 20 min)",
    )
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=TOURNAMENT_CANDIDATES,
        help=f"Candidate edits per tournament round (default: {TOURNAMENT_CANDIDATES})",
    )
    parser.add_argument(
        "--halving-rounds",
        type=int,
        default=TOURNAMENT_ROUNDS,
        help=f"Number of halving rounds (default: {TOURNAMENT_ROUNDS})",
    )
    parser.add_argument(
        "--keep-fraction",
        type=float,
        default=TOURNAMENT_KEEP_FRACTION,
        help=f"Top fraction kept each round (default: {TOURNAMENT_KEEP_FRACTION})",
    )
    parser.add_argument(
        "--games-budget",
        type=int,
        default=TOURNAMENT_GAMES_BUDGET,
        help=f"Hard cap on games per tournament (default: {TOURNAMENT_GAMES_BUDGET})",
    )
    args = parser.parse_args(namespace=_OrchestratorArgs())

    print("AoE2 Autoresearch — Successive-Halving Tournament")
    print(f"Time budget: {args.time_budget}s per game")
    print(f"Max tournaments: {args.max_experiments or 'unlimited'}")
    print()

    asyncio.run(
        Orchestrator().run_tournament_loop(
            max_experiments=args.max_experiments,
            time_budget=args.time_budget,
            n_candidates=args.n_candidates,
            halving_rounds=args.halving_rounds,
            keep_fraction=args.keep_fraction,
            games_budget=args.games_budget,
        )
    )


if __name__ == "__main__":
    main()
