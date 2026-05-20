"""CLI entry point for the synth arena (Phase 6).

Sub-commands:
  python -m arena race [profile.yaml]   — real Claude API (requires ANTHROPIC_API_KEY)
  python -m arena smoke                 — offline mock run, no API key needed
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from arena.config_profile import ConfigProfile, RaceConfig
from arena.metrics import summarise
from arena.race import race, race_with_mock
from evaluation.world_sim import WorldState

_DEFAULT_PROFILE = Path(__file__).parent / "profiles" / "v1.yaml"

_STANDARD_START = WorldState(
    food=200.0,
    wood=150.0,
    gold=0.0,
    stone=0.0,
    population=8,
    pop_cap=25,
    age="Dark Age",
    buildings=[],
    villager_queue=[],
    age_up_ticks_remaining=0,
    turn=0,
)


def _cmd_race(profile_path: Path) -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("error: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    config = RaceConfig.from_yaml(profile_path)
    print(f"Racing {len(config.profiles)} profiles for {config.turns} turns …")
    results = asyncio.run(race(config, api_key, _STANDARD_START))
    print(summarise(results))


def _cmd_smoke() -> None:
    config = RaceConfig(
        turns=10,
        profiles=[
            ConfigProfile(name="mock-a"),
            ConfigProfile(name="mock-b"),
        ],
    )
    print(f"Smoke run: {len(config.profiles)} profiles x {config.turns} turns (mock invoke) ...")
    results = asyncio.run(race_with_mock(config, _STANDARD_START))
    print(summarise(results))


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print("usage: python -m arena <race [profile.yaml] | smoke>", file=sys.stderr)
        sys.exit(1)

    cmd = args[0]
    if cmd == "race":
        path = Path(args[1]) if len(args) > 1 else _DEFAULT_PROFILE
        _cmd_race(path)
    elif cmd == "smoke":
        _cmd_smoke()
    else:
        print(f"error: unknown command '{cmd}'", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
