"""Orchestrator — main autoresearch experiment loop.

Coordinates prompt mutation, game running, scoring, and accept/reject decisions.

Usage:
    python -m autoresearch.orchestrator [--max-experiments 5] [--time-budget 1200]
"""

import argparse
import asyncio
import subprocess
import sys
from pathlib import Path

import structlog

sys.path.insert(0, str(Path(__file__).parent.parent))

from autoresearch.experiment_log import (
    get_best_score,
    get_next_experiment_id,
    get_recent_experiments,
    log_experiment,
)
from autoresearch.game_runner import run_game
from autoresearch.prompt_mutator import PromptMutator

log = structlog.get_logger()

REPO_ROOT = Path(__file__).parent.parent
EPSILON = 0.02  # Accept if score >= best - epsilon


class Orchestrator:
    """Coordinates the prompt optimization experiment loop."""

    def __init__(self, epsilon: float = EPSILON):
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

    async def run_experiment(self, time_budget: float = 1200) -> dict:
        """Run one full experiment cycle: mutate -> play -> score -> accept/reject.

        Returns dict with experiment_id, score, accepted, description, or error.
        """
        experiment_id = get_next_experiment_id()
        recent = get_recent_experiments(5)
        failure_modes = self._extract_failure_modes(recent)

        # 1. Propose a prompt change
        current_prompt = self.mutator.read_current_prompt()
        change = self.mutator.propose_change(current_prompt, recent, failure_modes)

        if not change:
            log.warning("mutation_failed", experiment_id=experiment_id)
            return {"experiment_id": experiment_id, "error": "mutation_failed"}

        description = change.get("description", "unknown change")
        old_text = change.get("old_text", "")
        new_text = change.get("new_text", "")

        log.info(
            "mutation_proposed",
            experiment_id=experiment_id,
            description=description,
            rationale=change.get("rationale", ""),
        )

        # 2. Apply the change
        if not self.mutator.apply_change(old_text, new_text):
            return {"experiment_id": experiment_id, "error": "change_not_applicable"}

        # 3. Commit the change
        sha = self.git_commit(f"[autoresearch] {experiment_id}: {description}")

        # 4. Run the game
        print("\n  Playing game with modified prompt...")
        result = await run_game(time_budget=time_budget)
        score = result["score"]

        # 5. Accept or reject
        accepted = score.composite >= self.best_score - self.epsilon

        if accepted:
            self.best_score = max(self.best_score, score.composite)
            log.info("experiment_accepted", score=score.composite, best=self.best_score)
        else:
            self.git_revert_prompt()
            log.info("experiment_rejected", score=score.composite, best=self.best_score)

        # 6. Log result
        log_experiment(
            experiment_id=experiment_id,
            loop="prompt",
            change_description=description,
            score=score,
            accepted=accepted,
            git_sha=sha if accepted else None,
        )

        return {
            "experiment_id": experiment_id,
            "score": score.composite,
            "accepted": accepted,
            "description": description,
        }

    async def run_baseline(self, time_budget: float = 1200) -> dict:
        """Run a baseline game with the current prompt (no mutation).

        Use this for the first experiment to establish a starting score.
        """
        experiment_id = get_next_experiment_id()

        print("\n  Running baseline game (no prompt changes)...")
        result = await run_game(time_budget=time_budget)
        score = result["score"]

        self.best_score = max(self.best_score, score.composite)

        log_experiment(
            experiment_id=experiment_id,
            loop="prompt",
            change_description="baseline — unmodified prompt",
            score=score,
            accepted=True,
        )

        return {
            "experiment_id": experiment_id,
            "score": score.composite,
            "accepted": True,
            "description": "baseline",
        }

    async def run_loop(
        self,
        max_experiments: int | None = None,
        time_budget: float = 1200,
        run_baseline_first: bool = True,
    ):
        """Run the autonomous experiment loop.

        Human starts each game manually (Phase 1).
        Orchestrator mutates prompt between games.

        Args:
            max_experiments: Stop after N experiments (None = run forever)
            time_budget: Seconds per game
            run_baseline_first: Run an unmodified baseline game first
        """
        count = 0

        # Run baseline if no previous experiments
        if run_baseline_first and self.best_score == 0.0:
            print("\n" + "=" * 60)
            print("BASELINE — Playing with unmodified prompt")
            print("=" * 60)
            input("Start a new game in AoE2, then press Enter...")

            result = await self.run_baseline(time_budget=time_budget)
            print(f"\n  Baseline score: {result['score']:.4f}")
            count += 1

            if max_experiments and count >= max_experiments:
                return

        while max_experiments is None or count < max_experiments:
            print(f"\n{'=' * 60}")
            print(f"Experiment {count + 1} — Best score: {self.best_score:.4f}")
            print(f"{'=' * 60}")

            # Propose mutation
            recent = get_recent_experiments(5)
            current_prompt = self.mutator.read_current_prompt()
            change = self.mutator.propose_change(
                current_prompt, recent, self._extract_failure_modes(recent)
            )

            if change:
                print(f"\n  Proposed: {change.get('description', '?')}")
                print(f"  Rationale: {change.get('rationale', '?')}")
            else:
                print("\n  Mutation failed, will retry with current prompt")

            input("\nStart a new game in AoE2, then press Enter...")

            if change:
                result = await self.run_experiment(time_budget=time_budget)
            else:
                # Run with current prompt if mutation failed
                result = await self.run_baseline(time_budget=time_budget)

            if "error" in result:
                print(f"\n  Error: {result['error']}")
            else:
                status = "ACCEPTED" if result["accepted"] else "REJECTED"
                print(f"\n  {status}: {result.get('description', '?')}")
                print(f"  Score: {result['score']:.4f} (best: {self.best_score:.4f})")

            count += 1

        print(f"\n{'=' * 60}")
        print(f"Done — {count} experiments completed")
        print(f"Best score: {self.best_score:.4f}")
        print("Results saved to experiments/results.tsv")
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
                modes.append("Population stayed very low — agent may not be queueing villagers or is getting housed")
            if age == 0:
                modes.append("Agent never advanced past Dark Age — needs to accumulate 500 food and click age-up")
            if economy < 0.1:
                modes.append("Very little food gathered — agent may not be assigning villagers to food sources")
            if action_success < 0.3:
                modes.append("Low action success rate — many actions failing, possibly clicking wrong coordinates")
            if survival < 0.5:
                modes.append("Game ended early — agent may be dying or getting stuck")

            end_reason = latest.get("game_end_reason", "")
            if end_reason == "defeat":
                modes.append("Agent was defeated — needs better military or defensive strategy")
        except (ValueError, TypeError):
            pass

        return modes if modes else ["No clear failure patterns — try optimizing strongest components"]


def main():
    parser = argparse.ArgumentParser(description="Run autoresearch prompt optimization loop")
    parser.add_argument(
        "--max-experiments",
        type=int,
        default=None,
        help="Maximum number of experiments (default: run forever)",
    )
    parser.add_argument(
        "--time-budget",
        type=float,
        default=1200,
        help="Maximum game duration in seconds (default: 1200 = 20 min)",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip baseline game even if no previous experiments",
    )
    args = parser.parse_args()

    print("AoE2 Autoresearch — Prompt Optimization Loop")
    print(f"Time budget: {args.time_budget}s per game")
    print(f"Max experiments: {args.max_experiments or 'unlimited'}")
    print()

    asyncio.run(
        Orchestrator().run_loop(
            max_experiments=args.max_experiments,
            time_budget=args.time_budget,
            run_baseline_first=not args.no_baseline,
        )
    )


if __name__ == "__main__":
    main()
