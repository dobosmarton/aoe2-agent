"""CLI entry point for the synth arena (Phase 6 + Phase 8).

Sub-commands:
  python -m arena race [profile.yaml]   — real model API (key per profile wire)
  python -m arena smoke                 — offline mock run, no API key needed
  python -m arena rank [profile.yaml]   — multi-round BT ranking (real API)

All commands persist the full event log to a DuckDB file under
`logs/arena/<YYYY-MM-DD>/<label>-<timestamp>.duckdb` for post-mortem queries.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import duckdb
from arena.config_profile import ConfigProfile, RaceConfig, RankingConfig
from arena.metrics import summarise
from arena.race import race, race_with_mock
from arena.ranking import RankingResult, rank
from arena.scenarios import DEFAULT_SCENARIOS, get_scenario
from evaluation.broker_factory import make_broker
from evaluation.duckdb_persister import MultiRunBrokerSink
from evaluation.event_log import DuckDBEventSink
from evaluation.world_sim import WorldState
from gameplay_agent.config import KEY_ENV
from gameplay_agent.config import config as agent_config
from gameplay_agent.providers.pricing import price_for

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


_T = TypeVar("_T")

_DEFAULT_PROFILE = Path(__file__).parent / "profiles" / "v1.yaml"
_DEFAULT_RANKING_PROFILE = Path(__file__).parent / "profiles" / "ranking-v1.yaml"
_LOGS_ROOT = Path("logs") / "arena"
# Measured shape of one arena turn: a short WorldState summary in, a small JSON
# array of actions out. Only used for the pre-flight cost projection — the real
# figure comes from each response's usage.
_TURN_INPUT_TOKENS = 215
_TURN_OUTPUT_TOKENS = 15

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


def _new_db_path(label: str) -> Path:
    now = datetime.now(UTC)
    day_dir = _LOGS_ROOT / now.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    return day_dir / f"{label}-{now.strftime('%H%M%S')}.duckdb"


async def _run_through_broker(
    db_path: Path,
    producer: Callable[[MultiRunBrokerSink], Awaitable[_T]],
) -> _T:
    """Phase 2.5 shim: drive a producer through the broker + a shared DuckDB sink.

    Owns the broker, DuckDB connection, and sink lifetimes so each CLI
    entry point stays a one-liner at the call site. The `close_all`
    invariant (close every opened run before exiting the `with`) is what
    guarantees the file is consistent on disk by the time we return.
    """
    broker = make_broker()
    # `<label>-<HHMMSS>.duckdb` per `_new_db_path` — the label the live `/runs`
    # source shows for this run. Deriving it from the path keeps the filename
    # the single source of truth (matches the cold path's `_label_from_filename`).
    label = db_path.stem.split("-", 1)[0]
    with duckdb.connect(str(db_path)) as conn:
        db_sink = DuckDBEventSink(conn)
        sink = MultiRunBrokerSink(broker, db_sink, asyncio.get_running_loop(), label=label)
        try:
            return await producer(sink)
        finally:
            await sink.close_all()


def _estimate_cost(profiles: list[ConfigProfile], turns_per_profile: int) -> float:
    """Project the spend of a ranking run, priced per profile's own model."""
    total = 0.0
    for profile in profiles:
        price = price_for(profile.model)
        per_turn = (
            _TURN_INPUT_TOKENS * price.input + _TURN_OUTPUT_TOKENS * price.output
        ) / 1_000_000
        total += per_turn * turns_per_profile
    return total


def _require_api_key() -> None:
    """Fail fast rather than a third of the way into a paid race."""
    if not agent_config.llm_api_key:
        print(f"error: {KEY_ENV} not set", file=sys.stderr)
        sys.exit(1)


def _cmd_race(profile_path: Path) -> None:
    config = RaceConfig.from_yaml(profile_path)
    _require_api_key()
    db_path = _new_db_path("race")
    print(f"Racing {len(config.profiles)} profiles for {config.turns} turns …")
    print(f"Event log: {db_path}")
    results = asyncio.run(
        _run_through_broker(
            db_path,
            lambda sink: race(config, _STANDARD_START, sink=sink),
        )
    )
    print(summarise(results))


def _cmd_smoke() -> None:
    config = RaceConfig(
        turns=10,
        profiles=[
            ConfigProfile(name="mock-a"),
            ConfigProfile(name="mock-b"),
        ],
    )
    db_path = _new_db_path("smoke")
    print(f"Smoke run: {len(config.profiles)} profiles x {config.turns} turns (mock invoke) ...")
    print(f"Event log: {db_path}")
    results = asyncio.run(
        _run_through_broker(
            db_path,
            lambda sink: race_with_mock(config, _STANDARD_START, sink=sink),
        )
    )
    print(summarise(results))


def _format_ranking(result: RankingResult) -> str:
    lines = [
        f"{'Rank':>4}  {'Profile':<22}  {'Rating':>8}  {'95% CI':<20}  {'Wins/Total':>11}",
        "-" * 73,
    ]
    ranked = sorted(result.ratings.items(), key=lambda kv: kv[1], reverse=True)
    total_per_profile = result.n_races // len(result.ratings) if result.ratings else 0
    for rank_pos, (name, rating) in enumerate(ranked, start=1):
        wins = sum(v for (a, _b), v in result.pairwise_wins.items() if a == name)
        ci = f"[{result.ci_low[name]:>+.2f}, {result.ci_high[name]:>+.2f}]"
        lines.append(
            f"{rank_pos:>4}  {name:<22}  {rating:>+8.3f}  {ci:<20}  {wins:>4}/{total_per_profile:<6}"
        )
    return "\n".join(lines)


def _cmd_rank(profile_path: Path) -> None:
    config = RankingConfig.from_yaml(profile_path)
    _require_api_key()
    scenarios = (
        list(DEFAULT_SCENARIOS)
        if not config.scenarios
        else [get_scenario(n) for n in config.scenarios]
    )
    n_races = config.rounds * len(scenarios)
    est_cost = _estimate_cost(config.profiles, n_races * config.turns)

    print(
        f"Ranking: {len(config.profiles)} profiles x {len(scenarios)} scenarios "
        f"x {config.rounds} rounds x {config.turns} turns = {n_races} race-instances"
    )
    print(f"Estimated cost: ~${est_cost:.2f}")

    if sys.stdin.isatty():
        reply = input("Proceed? [y/N] ").strip().lower()
        if reply != "y":
            print("Aborted.")
            sys.exit(0)

    db_path = _new_db_path("rank")
    print(f"Event log: {db_path}")
    result = asyncio.run(
        _run_through_broker(
            db_path,
            lambda sink: rank(config, sink=sink),
        )
    )
    print(_format_ranking(result))


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(
            "usage: python -m arena <race [profile.yaml] | smoke | rank [profile.yaml]>",
            file=sys.stderr,
        )
        sys.exit(1)

    cmd = args[0]
    if cmd == "race":
        path = Path(args[1]) if len(args) > 1 else _DEFAULT_PROFILE
        _cmd_race(path)
    elif cmd == "smoke":
        _cmd_smoke()
    elif cmd == "rank":
        path = Path(args[1]) if len(args) > 1 else _DEFAULT_RANKING_PROFILE
        _cmd_rank(path)
    else:
        print(f"error: unknown command '{cmd}'", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
