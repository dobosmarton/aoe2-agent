"""Live smoke test of the agent on the running Mac game.

Runs the agent's real perception and action loops against the game in front, with
the in-process detector, and scripts only the policy's choice: each named action
is chosen once, as soon as the agent offers it, and "wait" otherwise. Everything
else (selection checks, freshness limits, ledger settlement, the menu check) is
the agent's own, so failures here are the agent's failures.

It sends real input and spends real resources. Run it from Terminal.app, which
needs Screen Recording and Accessibility, with the game running and no menu open:

    uv run python scripts/mac_smoke.py --actions queue_villager,assign_food,build_house

The log goes to stdout; outcomes are the `action_outcome` lines.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING

from gameplay_agent import executor, game_loop
from gameplay_agent.config import config
from gameplay_agent.game_profile import load_profile
from gameplay_agent.keyboard_macos import use_layout_keys
from gameplay_agent.loops.act import act_loop
from gameplay_agent.loops.perceive import perceive_loop
from gameplay_agent.loops.source import frame_refresh
from gameplay_agent.memory import AgentMemory
from gameplay_agent.policy.advice import PolicyAdvice, readonly_probabilities
from gameplay_agent.screen import close_capture

if TYPE_CHECKING:
    from gameplay_agent.policy.advice import PolicyRequest

PROFILE = Path(__file__).parents[1] / "apps/agent/profiles/magyars_arabia_4v4.macos.example.json"
# Time an action's outcome gets before the next scripted action is chosen.
OUTCOME_SECONDS = 12.0


class ScriptedChoices:
    """Chooses each named action once it is offered, and "wait" otherwise."""

    def __init__(self, actions: list[str], started: float) -> None:
        self.todo = list(actions)
        self.started = started
        self.next_at = 0.0
        self.chosen: list[tuple[str, float]] = []
        self._last_offer: list[str] = []

    async def advise(self, request: PolicyRequest) -> PolicyAdvice:
        offered = {candidate.id for candidate in request.candidates}
        choice = "wait"
        if self.todo and time.monotonic() >= self.next_at:
            ready = [action for action in self.todo if action in offered]
            if ready:
                choice = ready[0]
                self.todo.remove(choice)
                self.next_at = time.monotonic() + OUTCOME_SECONDS
                self.chosen.append((choice, round(time.monotonic() - self.started, 1)))
                print(f"smoke: chose {choice}", flush=True)
            elif sorted(offered) != self._last_offer:
                self._last_offer = sorted(offered)
                print(f"smoke: waiting for {self.todo}; offered {sorted(offered)}", flush=True)
        if choice not in offered:
            choice = "wait" if "wait" in offered else sorted(offered)[0]
        return PolicyAdvice(
            source_tick=request.source_tick,
            source_captured_at=request.source_captured_at,
            model="scripted",
            action_choice=choice,
            action_confidence=1.0,
            action_probabilities=readonly_probabilities({choice: 1.0}),
            allocation_focus="balanced",
            allocation_confidence=1.0,
            allocation_probabilities=readonly_probabilities({"balanced": 1.0}),
        )

    async def aclose(self) -> None:
        return None


async def run(actions: list[str], seconds: float) -> int:
    started = time.monotonic()
    config.detection_host = ""  # The in-process CoreML detector.
    config.save_screenshots = False  # Game frames stay out of the logs.
    layout, missing = use_layout_keys()
    if missing:
        print(f"smoke: the {layout.name} layout can't type {missing}", flush=True)
        return 1
    profile = load_profile(PROFILE)
    executor.set_command_button(profile.command_button)
    ledger = executor.ActionLedger()
    memory = AgentMemory()
    memory.start_game()
    memory.action_ledger = ledger
    token = executor.bind_ledger(ledger)
    try:
        ctx = game_loop._build_context(  # The agent's own wiring.
            memory,
            use_detection=True,
            use_overlay=False,
            time_budget=None,
            max_iterations=None,
            detector_factory=None,
            ledger=ledger,
            profile=profile,
        )
        refresh = frame_refresh(ctx.frames)
        executor.set_rescan_fn(refresh)
        executor.set_rescan_full_fn(refresh)
        executor.set_selection_refresh_fn(frame_refresh(ctx.frames, selection_only=True))
        not_ready = await game_loop._prepare_macos_game(profile)
        if not_ready is not None:
            print(f"smoke: game not ready: {not_ready}", flush=True)
            return 1
        script = ScriptedChoices(actions, started)
        tasks = [
            asyncio.create_task(perceive_loop(ctx)),
            asyncio.create_task(act_loop(ctx, script)),
        ]
        await asyncio.sleep(seconds)
        ctx.request_stop("smoke_done")
        await asyncio.gather(*tasks, return_exceptions=True)
        print(f"smoke: chosen {script.chosen}; not offered {script.todo}", flush=True)
        print(f"smoke: failures {list(ledger.recent_failures)}", flush=True)
        return 0
    finally:
        executor.unbind_ledger(token)
        close_capture()


class _Args(argparse.Namespace):
    actions: str
    seconds: float


def main() -> None:
    parser = argparse.ArgumentParser(description="Live smoke test of the agent on the Mac game")
    parser.add_argument("--actions", default="queue_villager,assign_food,build_house")
    parser.add_argument("--seconds", type=float, default=70.0)
    args = parser.parse_args(namespace=_Args())
    raise SystemExit(asyncio.run(run(args.actions.split(","), args.seconds)))


if __name__ == "__main__":
    main()
