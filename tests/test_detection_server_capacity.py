"""A disconnected client cannot release inference capacity prematurely."""

from __future__ import annotations

import asyncio
import io
import threading

import httpx
from detection_server import app as server
from PIL import Image


def _jpeg() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32)).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_busy_response_survives_client_cancellation(monkeypatch) -> None:
    started = threading.Event()
    release = threading.Event()

    def detect(_state: server.ModelState, _image: Image.Image, _imgsz: int):
        started.set()
        release.wait(timeout=3)
        return []

    monkeypatch.setattr(server, "_model_state", server.ModelState("onnx_cpu", object(), (), "test"))
    monkeypatch.setattr(server, "_detect_single", detect)
    application = server.create_app("unused")

    async def run() -> None:
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            first = asyncio.create_task(
                client.post("/detect", files={"file": ("frame.jpg", _jpeg(), "image/jpeg")})
            )
            assert await asyncio.to_thread(started.wait, 1)
            first.cancel()
            await asyncio.gather(first, return_exceptions=True)
            busy = await client.post(
                "/detect", files={"file": ("frame.jpg", _jpeg(), "image/jpeg")}
            )
            assert busy.status_code == 503
            assert busy.json()["reason"] == "detector_busy"
            release.set()
            await asyncio.sleep(0.05)
            ready = await client.post(
                "/detect", files={"file": ("frame.jpg", _jpeg(), "image/jpeg")}
            )
            assert ready.status_code == 200

    try:
        asyncio.run(run())
    finally:
        release.set()
