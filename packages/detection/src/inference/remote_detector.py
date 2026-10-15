"""Stateless single-frame HTTP inference on the macOS detector server."""

from __future__ import annotations

import asyncio
import io
import logging
import time
from typing import TYPE_CHECKING, cast

from core import DetectedEntity
from pydantic import BaseModel, ConfigDict, ValidationError

from .detector import EntityDetector, get_detector, resolve_model_path
from .postprocess import nms as _apply_nms

if TYPE_CHECKING:
    from collections.abc import Callable

    import httpx
    from PIL import Image

logger = logging.getLogger(__name__)

# Re-check server health after this many seconds of being unavailable
_HEALTH_RECHECK_INTERVAL: float = 30.0


class DetectionUnavailableError(RuntimeError):
    """The required detector cannot produce observations."""


class DetectionBusyError(RuntimeError):
    """The server is already inferencing; this frame may be discarded."""


class _RemoteDetection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    class_name: str
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    confidence: float
    area: float = 0.0


class _RemoteResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    detections: tuple[_RemoteDetection, ...]
    inference_ms: float
    tile_count: int
    image_size: tuple[int, int]


class _BusyResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    reason: str
    retry_after_ms: int


class RemoteDetector:
    """Async detector that delegates inference to a remote CoreML/ONNX server.

    On connection failure, interactive execution may use a local ONNX fallback.
    """

    def __init__(
        self,
        server_url: str,
        confidence_threshold: float = 0.35,
        imgsz: int = 1280,
        fallback_detector: EntityDetector | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        import httpx

        self.server_url = server_url.rstrip("/")
        self.confidence_threshold = confidence_threshold
        self.imgsz = imgsz
        self._fallback = fallback_detector
        self._client = client if client is not None else httpx.AsyncClient(timeout=10.0)
        self._server_available = True
        self._last_health_check: float = 0.0

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
        entities = _apply_nms(entities, iou_threshold=0.5)

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
        entities = _apply_nms(entities, iou_threshold=0.5)

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

        # Parallel requests. Both legs use self.imgsz: a fixed-shape ONNX graph
        # (static export, required for CoreML/ANE at 1280) only accepts the one
        # size it was exported at, so the crop cannot request a different imgsz.
        # At 1280 the crop is downscaled ~1.2x (vs ~2.4x for the full frame), so
        # it still adds center resolution for small units.
        full_task = self._post_detect("/detect", full_bytes, imgsz=self.imgsz)
        crop_task = self._post_detect("/detect", crop_bytes, imgsz=self.imgsz)
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
        entities = _apply_nms(entities, iou_threshold=0.5)

        elapsed = time.monotonic() - t0
        logger.info(
            "remote_detect_fast_multi elapsed=%.2fs entities=%d (full=%d crop=%d)",
            elapsed,
            len(entities),
            len(full_entities),
            len(crop_entities),
        )
        return entities

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    async def _post_detect(
        self,
        endpoint: str,
        image_bytes: bytes,
        **params: int,
    ) -> tuple[_RemoteDetection, ...] | None:
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

        started_at = time.monotonic()
        try:
            response = await self._client.post(
                f"{self.server_url}{endpoint}",
                files={"file": ("screenshot.jpg", image_bytes, "image/jpeg")},
                params=params if params else None,
                timeout=10.0,
            )
            if response.status_code == 503:
                # An overloaded server is healthy. Never replay this stale frame
                # through the local fallback or mark the server unavailable.
                busy = _BusyResponse.model_validate(cast("object", response.json()))
                if busy.reason == "detector_busy":
                    raise DetectionBusyError(f"{self.server_url}{endpoint} is busy")
            response.raise_for_status()
            # httpx JSON is untyped; validate the external response before use.
            data = _RemoteResponse.model_validate(cast("object", response.json()))
            roundtrip_ms = (time.monotonic() - started_at) * 1000
            logger.info(
                "remote_detection_latency endpoint=%s roundtrip_ms=%.1f "
                "server_ms=%.1f transport_overhead_ms=%.1f",
                endpoint,
                roundtrip_ms,
                data.inference_ms,
                max(0.0, roundtrip_ms - data.inference_ms),
            )
            return data.detections
        except (
            httpx.RequestError,
            httpx.HTTPStatusError,
            ValidationError,
            ValueError,
        ) as e:
            logger.warning(
                "remote_detection_failed server=%s endpoint=%s error_type=%s "
                "error=%r elapsed=%.2fs fallback=%s",
                self.server_url,
                endpoint,
                type(e).__name__,
                e,
                time.monotonic() - started_at,
                self._fallback is not None,
            )
            self._server_available = False
            self._last_health_check = time.monotonic()
            return None

    async def _check_health(self) -> bool:
        """Check if server is reachable."""
        import httpx

        try:
            resp = await self._client.get(f"{self.server_url}/health", timeout=3.0)
            return resp.status_code == 200
        except httpx.RequestError:
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
    def _to_entities(detections: tuple[_RemoteDetection, ...]) -> list[DetectedEntity]:
        """Convert server detection dicts to DetectedEntity list (no IDs yet)."""
        entities: list[DetectedEntity] = []
        counters: dict[str, int] = {}

        for det in detections:
            cls = det.class_name
            idx = counters.get(cls, 0)
            counters[cls] = idx + 1

            entities.append(
                DetectedEntity(
                    id=f"{cls}_{idx}",
                    class_name=cls,
                    bbox=det.bbox,
                    center=det.center,
                    confidence=det.confidence,
                    area=det.area,
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
            logger.error(
                "remote_detection_unavailable_empty_result server=%s method=%s",
                self.server_url,
                method,
            )
            return []

        logger.info("falling_back_to_local_detector method=%s", method)
        fn = cast("Callable[..., list[DetectedEntity]]", getattr(self._fallback, method))
        return await asyncio.to_thread(fn, screenshot)


class RequiredRemoteDetector(RemoteDetector):
    """Remote detector used by experiments where changing observations is invalid."""

    async def _fallback_detect(
        self,
        screenshot: bytes | Image.Image,
        method: str,
    ) -> list[DetectedEntity]:
        del screenshot
        raise DetectionUnavailableError(
            f"required remote detector {self.server_url!r} cannot run {method!r}"
        )


def _local_model_exists() -> bool:
    """True if a real local YOLO model is on disk, so a non-mock fallback is possible.

    On a remote-only deploy (the agent VM) the model files are gitignored and
    absent — there is nothing to fall back to, and a *mock* fallback is worse than
    none (it would silently return fabricated detections on a server outage).
    """
    return resolve_model_path() is not None


def get_remote_detector(
    server_url: str,
    imgsz: int = 1280,
    model_name: str | None = None,
) -> RemoteDetector:
    """Create a RemoteDetector, keeping a local model as fallback only if one exists.

    A *mock* detector is never kept as the fallback: on a server outage it would
    silently return fabricated detections instead of failing loudly. The fallback
    is populated only when a real local model is present (e.g. the serving host);
    remote-only interactive deploys get ``None`` and publish an empty result
    during an outage rather than fabricating detections.

    ``model_name`` is the configured served model (``config.detection_model``) —
    passing it lets ``get_detector`` warn loudly when the local fallback has to
    substitute different bundled weights (e.g. only an old version on disk).
    """
    fallback: EntityDetector | None = None
    if _local_model_exists():
        try:
            candidate = get_detector(imgsz=imgsz, model_name=model_name)
            fallback = candidate if not candidate.use_mock else None
        except Exception:
            logger.warning("Could not create local fallback detector")

    return RemoteDetector(
        server_url=server_url,
        imgsz=imgsz,
        fallback_detector=fallback,
    )


def get_required_remote_detector(
    server_url: str,
    imgsz: int = 1280,
) -> RemoteDetector:
    """Create a detector whose remote-service failure is fatal to the caller."""
    return RequiredRemoteDetector(
        server_url=server_url,
        imgsz=imgsz,
        fallback_detector=None,
    )


__all__ = [
    "DetectionUnavailableError",
    "RemoteDetector",
    "RequiredRemoteDetector",
    "get_remote_detector",
    "get_required_remote_detector",
]
