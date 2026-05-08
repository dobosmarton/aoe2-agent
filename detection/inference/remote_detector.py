"""Remote detector client that offloads inference to a CoreML/ONNX server.

Drop-in async replacement for EntityDetector. Sends JPEG screenshots to a
FastAPI server (coreml_server.py) running on the macOS host, receives JSON
detections, and applies NMS + Kalman tracking locally.

Usage:
    detector = get_remote_detector("http://192.168.64.1:8420")
    entities = await detector.detect(screenshot_bytes)
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from typing import TYPE_CHECKING

from .detector import DetectedEntity, EntityDetector, get_detector

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)

# Re-check server health after this many seconds of being unavailable
_HEALTH_RECHECK_INTERVAL: float = 30.0


class RemoteDetector:
    """Async detector that delegates inference to a remote CoreML/ONNX server.

    NMS and Kalman tracking run locally. On connection failure, falls back
    to a local ONNX EntityDetector if available.
    """

    def __init__(
        self,
        server_url: str,
        confidence_threshold: float = 0.35,
        imgsz: int = 1280,
        fallback_detector: EntityDetector | None = None,
    ) -> None:
        import httpx

        self.server_url = server_url.rstrip("/")
        self.confidence_threshold = confidence_threshold
        self.imgsz = imgsz
        self._fallback = fallback_detector
        self._client = httpx.AsyncClient(timeout=10.0)
        self._server_available = True
        self._last_health_check: float = 0.0

        # Local NMS (reuse from EntityDetector)
        self._nms_helper = EntityDetector(use_mock=True)

        # Local Kalman tracker
        self.tracker = None
        try:
            from .tracker import EntityTracker

            self.tracker = EntityTracker()
        except Exception:
            logger.debug("Tracker not available for RemoteDetector")

        # For adaptive detection compatibility
        self._previous_entities: list[DetectedEntity] = []

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Public detection methods (async)
    # ------------------------------------------------------------------

    async def detect(self, screenshot: bytes | Image.Image) -> list[DetectedEntity]:
        """Full SAHI detection via remote server."""
        t0 = time.monotonic()
        image_bytes = self._to_jpeg(screenshot)

        detections = await self._post_detect("/detect/sahi", image_bytes)
        if detections is None:
            return await self._fallback_detect(screenshot, "detect")

        entities = self._to_entities(detections)
        entities = self._nms_helper._nms(entities, iou_threshold=0.5)

        if self.tracker:
            entities = self.tracker.update(entities)

        self._previous_entities = entities
        elapsed = time.monotonic() - t0
        logger.info("remote_detect elapsed=%.2fs entities=%d", elapsed, len(entities))
        return entities

    async def detect_fast(self, screenshot: bytes | Image.Image) -> list[DetectedEntity]:
        """Single-image detection (no SAHI)."""
        t0 = time.monotonic()
        image_bytes = self._to_jpeg(screenshot)

        detections = await self._post_detect("/detect", image_bytes, imgsz=self.imgsz)
        if detections is None:
            return await self._fallback_detect(screenshot, "detect_fast")

        entities = self._to_entities(detections)
        entities = self._nms_helper._nms(entities, iou_threshold=0.5)

        if self.tracker:
            entities = self.tracker.update(entities)

        self._previous_entities = entities
        elapsed = time.monotonic() - t0
        logger.info("remote_detect_fast elapsed=%.2fs entities=%d", elapsed, len(entities))
        return entities

    async def detect_fast_multi(self, screenshot: bytes | Image.Image) -> list[DetectedEntity]:
        """Two-pass detection: full image + center 50% crop (parallel requests)."""
        from PIL import Image as PILImage

        t0 = time.monotonic()
        if isinstance(screenshot, bytes):
            image = PILImage.open(io.BytesIO(screenshot))
        else:
            image = screenshot

        w, h = image.size
        crop_x1 = w // 4
        crop_y1 = h // 4
        crop_x2 = w - crop_x1
        crop_y2 = h - crop_y1
        center_crop = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

        full_bytes = self._to_jpeg(screenshot)
        crop_bytes = self._to_jpeg(center_crop)

        # Parallel requests
        full_task = self._post_detect("/detect", full_bytes, imgsz=self.imgsz)
        crop_task = self._post_detect("/detect", crop_bytes, imgsz=640)
        full_dets, crop_dets = await asyncio.gather(full_task, crop_task)

        if full_dets is None or crop_dets is None:
            return await self._fallback_detect(screenshot, "detect_fast_multi")

        full_entities = self._to_entities(full_dets)

        # Offset crop detections back to full image coordinates
        crop_entities = self._to_entities(crop_dets)
        for e in crop_entities:
            x1, y1, x2, y2 = e.bbox
            e.bbox = (x1 + crop_x1, y1 + crop_y1, x2 + crop_x1, y2 + crop_y1)
            e.center = ((e.bbox[0] + e.bbox[2]) / 2, (e.bbox[1] + e.bbox[3]) / 2)

        entities = full_entities + crop_entities
        entities = self._nms_helper._nms(entities, iou_threshold=0.5)

        if self.tracker:
            entities = self.tracker.update(entities)

        self._previous_entities = entities
        elapsed = time.monotonic() - t0
        logger.info(
            "remote_detect_fast_multi elapsed=%.2fs entities=%d (full=%d crop=%d)",
            elapsed,
            len(entities),
            len(full_entities),
            len(crop_entities),
        )
        return entities

    async def detect_adaptive(
        self,
        screenshot: bytes | Image.Image,
        force_full: bool = False,
    ) -> list[DetectedEntity]:
        """Adaptive detection. With CoreML's speed, full SAHI is fast enough (~300ms).

        Always delegates to full SAHI detect() since tiling overhead is negligible
        on the Neural Engine.
        """
        return await self.detect(screenshot)

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    async def _post_detect(
        self,
        endpoint: str,
        image_bytes: bytes,
        **params: int,
    ) -> list[dict] | None:
        """POST image to server, return detections list or None on failure."""
        import httpx

        # Check if server was marked unavailable
        if not self._server_available:
            now = time.monotonic()
            if now - self._last_health_check < _HEALTH_RECHECK_INTERVAL:
                return None
            # Re-check health
            self._last_health_check = now
            if not await self._check_health():
                return None
            self._server_available = True
            logger.info("remote_server_reconnected")

        try:
            response = await self._client.post(
                f"{self.server_url}{endpoint}",
                files={"file": ("screenshot.jpg", image_bytes, "image/jpeg")},
                params=params if params else None,
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("detections", [])
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as e:
            logger.warning("remote_detection_failed: %s", e)
            self._server_available = False
            self._last_health_check = time.monotonic()
            return None

    async def _check_health(self) -> bool:
        """Check if server is reachable."""
        import httpx

        try:
            resp = await self._client.get(f"{self.server_url}/health", timeout=3.0)
            return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_jpeg(screenshot: bytes | Image.Image) -> bytes:
        """Convert screenshot to JPEG bytes."""
        if isinstance(screenshot, bytes):
            return screenshot
        buf = io.BytesIO()
        screenshot.save(buf, format="JPEG", quality=85)
        return buf.getvalue()

    @staticmethod
    def _to_entities(detections: list[dict]) -> list[DetectedEntity]:
        """Convert server detection dicts to DetectedEntity list (no IDs yet)."""
        entities: list[DetectedEntity] = []
        counters: dict[str, int] = {}

        for det in detections:
            cls = det["class_name"]
            idx = counters.get(cls, 0)
            counters[cls] = idx + 1

            entities.append(
                DetectedEntity(
                    id=f"{cls}_{idx}",
                    class_name=cls,
                    bbox=tuple(det["bbox"]),
                    center=tuple(det["center"]),
                    confidence=det["confidence"],
                    area=det.get("area", 0.0),
                )
            )

        return entities

    async def _fallback_detect(
        self,
        screenshot: bytes | Image.Image,
        method: str,
    ) -> list[DetectedEntity]:
        """Fall back to local ONNX detector."""
        if self._fallback is None:
            logger.error("remote_detection_failed_no_fallback")
            return []

        logger.info("falling_back_to_local_detector method=%s", method)
        fn = getattr(self._fallback, method)
        return await asyncio.to_thread(fn, screenshot)


def get_remote_detector(
    server_url: str,
    imgsz: int = 1280,
    with_fallback: bool = True,
) -> RemoteDetector:
    """Create a RemoteDetector with optional local ONNX fallback."""
    fallback: EntityDetector | None = None
    if with_fallback:
        try:
            fallback = get_detector(imgsz=imgsz)
        except Exception:
            logger.warning("Could not create local fallback detector")

    return RemoteDetector(
        server_url=server_url,
        imgsz=imgsz,
        fallback_detector=fallback,
    )
