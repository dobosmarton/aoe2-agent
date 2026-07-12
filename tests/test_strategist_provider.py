"""StrategistProvider.generate_goals perception source (T-203).

The game loop owns the per-turn HUD OCR; the strategist consumes the passed-in
reading and only self-OCRs on the standalone/eval path (readings=None).
"""

from __future__ import annotations

import asyncio

import pytest
from gameplay_agent.memory import GameState
from gameplay_agent.providers import strategist as strat_mod
from gameplay_agent.providers.strategist import StrategistProvider, StrategistResponse


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
