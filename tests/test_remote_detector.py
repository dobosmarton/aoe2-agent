"""Remote detection distinguishes resilient play from required experiments."""

from __future__ import annotations

import asyncio
import logging

import httpx
import pytest
from core import DetectedEntity
from detection.inference.remote_detector import (
    DetectionUnavailableError,
    RemoteDetector,
    RequiredRemoteDetector,
)


def _timeout_transport() -> httpx.MockTransport:
    async def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("", request=request)

    return httpx.MockTransport(timeout)


def test_required_remote_detector_propagates_an_outage(caplog: pytest.LogCaptureFixture) -> None:
    detector = RequiredRemoteDetector(
        "http://detector.invalid",
        fallback_detector=None,
        client=httpx.AsyncClient(transport=_timeout_transport()),
    )

    async def detect() -> None:
        try:
            with pytest.raises(DetectionUnavailableError):
                await detector.detect_fast(b"jpeg")
        finally:
            await detector.close()

    with caplog.at_level(logging.WARNING):
        asyncio.run(detect())

    assert "error_type=ReadTimeout" in caplog.text


def test_interactive_remote_detector_uses_its_local_fallback() -> None:
    class _Fallback:
        def detect_fast(self, _screenshot: bytes) -> list[DetectedEntity]:
            return [
                DetectedEntity(
                    id="tree_0",
                    class_name="tree",
                    bbox=(1.0, 2.0, 3.0, 4.0),
                    center=(2.0, 3.0),
                    confidence=0.8,
                )
            ]

    detector = RemoteDetector(
        "http://detector.invalid",
        fallback_detector=_Fallback(),  # pyright: ignore[reportArgumentType]
        client=httpx.AsyncClient(transport=_timeout_transport()),
    )

    async def detect() -> list[DetectedEntity]:
        try:
            return await detector.detect_fast(b"jpeg")
        finally:
            await detector.close()

    entities = asyncio.run(detect())
    assert [entity.class_name for entity in entities] == ["tree"]


def test_interactive_remote_only_detector_degrades_to_an_empty_result() -> None:
    detector = RemoteDetector(
        "http://detector.invalid",
        fallback_detector=None,
        client=httpx.AsyncClient(transport=_timeout_transport()),
    )

    async def detect() -> list[DetectedEntity]:
        try:
            return await detector.detect_fast(b"jpeg")
        finally:
            await detector.close()

    assert asyncio.run(detect()) == []
