"""A window backend for tests: a fixed game area and a front state the test controls."""

from __future__ import annotations

from dataclasses import dataclass, field

from gameplay_agent.window import GameGeometry, Platform

# The Mac game: full screen on a 3024x1964 Retina panel, below a 66-pixel band.
MAC_GAME = GameGeometry(0, 66, 3024, 1898, 2.0, 42)
# The VMware window the Windows agent was built against.
VM_GAME = GameGeometry(200, 100, 3024, 1672)


@dataclass
class FakeWindowBackend:
    platform: Platform = "macos"
    game: GameGeometry | None = MAC_GAME
    front: bool = True
    # Answers to `in_front`, used in order before falling back to `front`.
    front_answers: list[bool] = field(default_factory=list)
    focus_calls: int = 0

    def geometry(self) -> GameGeometry | None:
        return self.game

    def input_scale(self) -> float:
        return self.game.scale if self.game is not None else 1.0

    def focus(self, retries: int) -> bool:
        del retries
        self.focus_calls += 1
        return self.in_front()

    def in_front(self) -> bool:
        return self.front_answers.pop(0) if self.front_answers else self.front

    def is_running(self) -> bool:
        return self.game is not None
