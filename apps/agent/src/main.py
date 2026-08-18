"""Entry point for AoE2 LLM Agent."""

import argparse
import asyncio
import sys
from typing import get_args

import structlog

from .config import KEY_ENV, WireName, config
from .game_loop import game_loop, run_single_iteration
from .providers import ExecutorProvider
from .providers.wire_factory import make_wire

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

log = structlog.stdlib.get_logger()


class _AgentArgs(argparse.Namespace):
    wire: WireName | None
    test: bool
    iterations: int | None
    overlay: bool


async def main_async(args: _AgentArgs) -> None:
    """Async main function."""
    if not config.llm_api_key:
        log.error("missing_api_key", message=f"Set {KEY_ENV}")
        sys.exit(1)

    wire_name: WireName = args.wire or config.llm_wire
    provider = ExecutorProvider(
        wire=make_wire(
            wire_name,
            model=config.model,
            api_key=config.llm_api_key,
            base_url=config.llm_base_url,
        )
    )
    log.info("provider_created", wire=wire_name, model=config.model)

    if args.test:
        # Run single iteration for testing
        log.info("running_test_iteration")
        result = await run_single_iteration(provider)
        log.info(
            "test_result",
            screenshot=result["screenshot_path"],
            reasoning=result["reasoning"],
            actions=result["actions"],
        )
    else:
        # Run main game loop
        log.info("starting_game_loop", loop_delay=config.loop_delay)
        await game_loop(provider, max_iterations=args.iterations, use_overlay=args.overlay)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AoE2 LLM Agent - Play Age of Empires 2 using vision LLM"
    )
    parser.add_argument(
        "--wire",
        type=str,
        default=None,
        choices=list(get_args(WireName)),
        help=(
            "Adapter serving the model (default: AOE2_LLM_WIRE, else openai). "
            "Adapter only — the models still follow AOE2_LLM_WIRE."
        ),
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run a single test iteration (capture + analyze)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Maximum number of iterations (default: unlimited)",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Show live YOLO detection overlay on game window",
    )

    args = parser.parse_args(namespace=_AgentArgs())

    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        log.info("agent_stopped")


if __name__ == "__main__":
    main()
