"""Game runner — wraps game_loop to collect metrics and log results.

Usage:
    python -m autoresearch.game_runner [--time-budget 1200] [--experiment-id exp_0001]
"""

import argparse
import asyncio
import sys
from pathlib import Path

import structlog

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from autoresearch.experiment_log import get_next_experiment_id, log_experiment
from autoresearch.memory_chain import MemoryChain
from autoresearch.metrics import compute_score
from gameplay_agent.config import config
from gameplay_agent.game_loop import game_loop
from gameplay_agent.memory import AgentMemory
from gameplay_agent.providers.claude import ClaudeProvider

log = structlog.get_logger()


async def run_game(
    time_budget: float | None = None,
    max_iterations: int | None = None,
    use_detection: bool = True,
    game_id: str | None = None,
    extract_memories: bool = True,
    use_overlay: bool = False,
) -> dict:
    """Run a single game and return metrics.

    Args:
        time_budget: Maximum game duration in seconds (None = no limit)
        max_iterations: Maximum turns (None = no limit)
        use_detection: Whether to use YOLO detection
        game_id: Experiment ID for memory attribution
        extract_memories: Whether to extract memory fragments after the game

    Returns:
        Dict with metrics snapshot and computed score
    """
    provider = ClaudeProvider()
    memory = AgentMemory()

    log.info(
        "game_start",
        time_budget=time_budget,
        max_iterations=max_iterations,
        model=config.model,
    )

    memory = await game_loop(
        provider=provider,
        max_iterations=max_iterations,
        memory=memory,
        use_detection=use_detection,
        time_budget=time_budget,
        use_overlay=use_overlay,
    )

    metrics = memory.get_metrics_snapshot()
    score = compute_score(metrics)

    log.info(
        "game_complete",
        composite_score=score.composite,
        survival=score.survival,
        population=score.population,
        age=score.age,
        economy=score.economy,
        action_success=score.action_success,
        end_reason=metrics["game_end_reason"],
        turns=metrics["turn_count"],
    )

    # Extract cross-game memory fragments
    memory_files = []
    if extract_memories and metrics["turn_count"] > 0:
        try:
            chain = MemoryChain()
            memory_files = chain.extract_memories(
                memory=memory,
                score=score,
                game_id=game_id or "unknown",
            )
            if memory_files:
                log.info("memories_extracted", count=len(memory_files),
                         files=[f.name for f in memory_files])
        except Exception as e:
            log.warning("memory_extraction_error", error=str(e))

    return {
        "metrics": metrics,
        "score": score,
        "memory_files": memory_files,
    }


async def run_and_log(
    experiment_id: str | None = None,
    loop: str = "manual",
    description: str = "manual game run",
    time_budget: float | None = None,
    max_iterations: int | None = None,
    use_overlay: bool = False,
) -> dict:
    """Run a game and log results to the experiment ledger.

    Args:
        experiment_id: Experiment ID (auto-generated if None)
        loop: Which loop this belongs to (manual, prompt, context, strategy)
        description: What changed for this experiment
        time_budget: Maximum game duration in seconds
        max_iterations: Maximum turns

    Returns:
        Dict with metrics, score, and experiment_id
    """
    if experiment_id is None:
        experiment_id = get_next_experiment_id()

    result = await run_game(
        time_budget=time_budget,
        max_iterations=max_iterations,
        use_overlay=use_overlay,
    )

    # Log to experiment ledger (manual runs are always "accepted")
    log_experiment(
        experiment_id=experiment_id,
        loop=loop,
        change_description=description,
        score=result["score"],
        accepted=True,
    )

    result["experiment_id"] = experiment_id
    return result


def main():
    parser = argparse.ArgumentParser(description="Run AoE2 agent and collect metrics")
    parser.add_argument(
        "--time-budget",
        type=float,
        default=None,
        help="Maximum game duration in seconds (default: no limit)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum number of turns (default: no limit)",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Experiment ID (auto-generated if omitted)",
    )
    parser.add_argument(
        "--description",
        type=str,
        default="manual game run",
        help="Description of this experiment",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Show live YOLO detection overlay on game window",
    )
    args = parser.parse_args()

    result = asyncio.run(
        run_and_log(
            experiment_id=args.experiment_id,
            description=args.description,
            time_budget=args.time_budget,
            max_iterations=args.max_iterations,
            use_overlay=args.overlay,
        )
    )

    # Print summary
    score = result["score"]
    print("\n--- Game Complete ---")
    print(f"Experiment:     {result['experiment_id']}")
    print(f"Composite:      {score.composite:.4f}")
    print(f"Survival:       {score.survival:.4f}")
    print(f"Population:     {score.population:.4f}")
    print(f"Age:            {score.age:.4f}")
    print(f"Economy:        {score.economy:.4f}")
    print(f"Action Success: {score.action_success:.4f}")
    print(f"End Reason:     {result['metrics']['game_end_reason']}")
    print(f"Turns:          {result['metrics']['turn_count']}")


if __name__ == "__main__":
    main()
