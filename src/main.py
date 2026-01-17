"""Entry point for AoE2 LLM Agent."""

import argparse
import asyncio
import sys

import structlog

from .config import config
from .game_loop import game_loop, run_single_iteration
from .providers import ClaudeProvider

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

log = structlog.get_logger()


def create_provider(provider_name: str):
    """Create an LLM provider by name."""
    providers = {
        "claude": ClaudeProvider,
        # Add more providers here as they're implemented
        # "openai": OpenAIProvider,
        # "gemini": GeminiProvider,
    }

    if provider_name not in providers:
        available = ", ".join(providers.keys())
        raise ValueError(f"Unknown provider: {provider_name}. Available: {available}")

    return providers[provider_name]()


async def main_async(args):
    """Async main function."""
    # Validate API key
    if not config.anthropic_api_key:
        log.error("missing_api_key", message="Set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    # Create provider
    provider = create_provider(args.provider)
    log.info("provider_created", provider=args.provider, model=config.model)

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
        await game_loop(provider, max_iterations=args.iterations)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AoE2 LLM Agent - Play Age of Empires 2 using vision LLM"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="claude",
        choices=["claude"],  # Add more as implemented
        help="LLM provider to use (default: claude)",
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

    args = parser.parse_args()

    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        log.info("agent_stopped")


if __name__ == "__main__":
    main()
