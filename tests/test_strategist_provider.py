"""StrategistProvider.generate_goals perception source (T-203).

The game loop owns the per-turn HUD OCR; the strategist consumes the passed-in
reading and only self-OCRs on the standalone/eval path (readings=None).
"""

from __future__ import annotations

import asyncio

import pytest
from gameplay_agent.memory import GameState
from gameplay_agent.providers import strategist as strat_mod
from gameplay_agent.providers.strategist import (
    StrategistGoal,
    StrategistProvider,
    StrategistResponse,
    VillagerTargets,
    as_allocation,
)
from pydantic import ValidationError


class _RecordingApi:
    """Stands in for _call_api: records prompts, returns an empty goal set."""

    def __init__(self) -> None:
        self.contents: list[list[dict]] = []

    async def __call__(self, content: list[dict]) -> StrategistResponse:
        self.contents.append(content)
        return StrategistResponse(reasoning="r", goals=[])


@pytest.fixture
def api() -> _RecordingApi:
    return _RecordingApi()


@pytest.fixture
def provider(api: _RecordingApi, monkeypatch: pytest.MonkeyPatch) -> StrategistProvider:
    p = StrategistProvider(model="test-model")
    monkeypatch.setattr(p, "_call_api", api)
    return p


def _forbid_ocr(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fail(_screenshot: bytes) -> tuple[dict, None]:
        raise AssertionError("strategist must not re-OCR the frame")

    monkeypatch.setattr(strat_mod, "read_hud_readings", _fail)


def _prompt_text(api: _RecordingApi) -> str:
    return str(api.contents[0][0]["text"])


def test_passed_readings_skip_ocr(
    provider: StrategistProvider, api: _RecordingApi, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid_ocr(monkeypatch)
    _goals, readings = asyncio.run(
        provider.generate_goals(GameState(), "", "", turn=1, readings={"food": 42})
    )
    assert readings == {"food": 42}
    assert "Food=42" in _prompt_text(api)


def test_empty_readings_mean_bad_frame_not_reocr(
    provider: StrategistProvider, api: _RecordingApi, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid_ocr(monkeypatch)
    asyncio.run(provider.generate_goals(GameState(), "", "", turn=1, readings={}))
    assert "Food=200" in _prompt_text(api)  # game_state fallback fills the gaps


def test_no_readings_self_ocr_eval_path(
    provider: StrategistProvider, api: _RecordingApi, monkeypatch: pytest.MonkeyPatch
) -> None:
    ocr_calls: list[bytes] = []

    async def _ocr(screenshot: bytes) -> tuple[dict, None]:
        ocr_calls.append(screenshot)
        return {"food": 7}, None

    monkeypatch.setattr(strat_mod, "read_hud_readings", _ocr)
    _goals, readings = asyncio.run(
        provider.generate_goals(GameState(), "", "", turn=1, screenshot_bytes=b"png")
    )
    assert ocr_calls == [b"png"]
    assert readings == {"food": 7}


# ---------------------------------------------------------------------------
# as_allocation — the fixed-key model back into the policy tier's Allocation
# ---------------------------------------------------------------------------
#
# `allocation` was a dict[str, int] until 2026-08-20, which 400'd every OpenAI
# call (an open object has no `properties`). A fixed-key model can never be
# empty, so the all-zero answer must still route to the seeded per-age mix.


def test_an_all_zero_target_falls_back_to_the_seeded_mix() -> None:
    """The model says nothing by leaving every field at 0."""
    assert as_allocation(VillagerTargets()) is None


def test_a_declared_target_becomes_an_allocation() -> None:
    targets = VillagerTargets(food=6, wood=4)
    assert as_allocation(targets).targets == {"food": 6, "wood": 4}


def test_a_zero_resource_is_left_out_rather_than_targeted_at_zero() -> None:
    """`share()` divides by the total, so a 0 entry would only add noise."""
    assert "stone" not in as_allocation(VillagerTargets(food=6)).targets


# ---------------------------------------------------------------------------
# Planning inputs and cadence
# ---------------------------------------------------------------------------


def test_the_brief_reaches_the_prompt(provider: StrategistProvider, api: _RecordingApi) -> None:
    brief = "## Next age\nFeudal Age costs 500 food (have 411)."
    asyncio.run(provider.generate_goals(GameState(), "", "", turn=1, readings={}, brief=brief))
    assert brief in _prompt_text(api)


def test_a_goal_metric_the_agent_cannot_measure_is_refused() -> None:
    with pytest.raises(ValidationError):
        StrategistGoal(name="Farms", type="local", metric="farm", target="8", priority=9)


def _planned(provider: StrategistProvider, age: str = "Dark Age") -> StrategistProvider:
    asyncio.run(provider.generate_goals(GameState(), "", "", turn=1, readings={"age": age}))
    provider._has_run = True  # The recording API skips the success path.
    return provider


def test_goals_are_not_replanned_within_the_period(provider: StrategistProvider) -> None:
    planned = _planned(provider)
    assert planned._last_run_at is not None
    assert not planned.should_run(age="Dark Age", now=planned._last_run_at + 5)


def test_goals_are_replanned_after_the_period(provider: StrategistProvider) -> None:
    planned = _planned(provider)
    assert planned._last_run_at is not None
    due = planned._last_run_at + planned.period_seconds
    assert planned.should_run(age="Dark Age", now=due)


def test_a_new_age_replans_at_once(provider: StrategistProvider) -> None:
    planned = _planned(provider)
    assert planned._last_run_at is not None
    assert planned.should_run(age="Feudal Age", now=planned._last_run_at + 1)


def test_a_lasting_alarm_replans_only_after_its_cooldown(provider: StrategistProvider) -> None:
    planned = _planned(provider)
    assert planned._last_run_at is not None
    start = planned._last_run_at + 1
    assert planned.should_run(alarm=True, age="Dark Age", now=start)
    assert not planned.should_run(alarm=True, age="Dark Age", now=start + 2)
