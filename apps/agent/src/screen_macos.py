"""Capture the native Mac game at full Retina resolution.

Both grabbers name the game's window, so another app's window on top can't leak
into the frame, and both clip it to the game area, without the band under the menu
bar. `StreamGrabber` keeps a ScreenCaptureKit stream running and hands out its
newest frame (about 50 ms per grab on the probe); `WindowGrabber` asks Quartz for
one capture per call (about 95 ms, and Quartz deprecates it).
"""

# pyobjc fills its framework modules at runtime, so a static checker can't see
# Quartz's functions and constants.
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Protocol

import CoreMedia
import numpy as np
import objc
import Quartz
import ScreenCaptureKit
import structlog
from Foundation import NSObject

from .capture import Frame

if TYPE_CHECKING:
    from collections.abc import Callable

    from .window import GameGeometry

log = structlog.stdlib.get_logger()

STREAM_FPS = 30
STREAM_QUEUE_DEPTH = 4
ANSWER_SECONDS = 5.0  # For ScreenCaptureKit's completion handlers and the first frame.

# Best resolution gives pixels, not points; without it the frame comes back half size.
IMAGE_OPTIONS = (
    Quartz.kCGWindowImageBoundsIgnoreFraming
    | Quartz.kCGWindowImageShouldBeOpaque
    | Quartz.kCGWindowImageBestResolution
)


def bgra_to_rgb(raw: bytes, width: int, height: int, row_bytes: int, top: int = 0) -> np.ndarray:
    """RGB pixels from BGRA rows, which may be padded past the image width, and may
    carry spare bytes after the last row. Rows above `top` are dropped."""
    rows = np.frombuffer(raw, dtype=np.uint8, count=row_bytes * height)
    pixels = rows.reshape(height, row_bytes // 4, 4)[top:, :width]
    return np.ascontiguousarray(pixels[..., 2::-1])


def grab_game_area(geometry: GameGeometry) -> np.ndarray:
    """The game area as RGB pixels, exactly `geometry.width` by `geometry.height`."""
    if geometry.window_id is None:
        raise ValueError("a Mac capture needs the game's window number")
    scale = geometry.scale
    area = Quartz.CGRectMake(
        geometry.left / scale,
        geometry.top / scale,
        geometry.width / scale,
        geometry.height / scale,
    )
    image = Quartz.CGWindowListCreateImage(
        area, Quartz.kCGWindowListOptionIncludingWindow, geometry.window_id, IMAGE_OPTIONS
    )
    if image is None:
        raise RuntimeError(
            "the window capture returned nothing; check that this terminal has "
            "Screen Recording permission"
        )
    width, height = Quartz.CGImageGetWidth(image), Quartz.CGImageGetHeight(image)
    if (width, height) != (geometry.width, geometry.height):
        raise RuntimeError(
            f"captured {width}x{height}, but the game area is {geometry.width}x{geometry.height}"
        )
    raw = bytes(Quartz.CGDataProviderCopyData(Quartz.CGImageGetDataProvider(image)))
    return bgra_to_rgb(raw, width, height, Quartz.CGImageGetBytesPerRow(image))


def _game_area(geometry: GameGeometry | None) -> GameGeometry:
    if geometry is None or geometry.window_id is None:
        raise ValueError("a Mac capture needs the game's window number")
    return geometry


class WindowGrabber:
    """One Quartz window capture per call."""

    def grab(self, geometry: GameGeometry | None, monitor: int) -> Frame:
        del monitor  # The window is always known on macOS.
        area = _game_area(geometry)
        stamped = time.monotonic()
        pixels = grab_game_area(area)
        return Frame(pixels.tobytes(), area.width, area.height, stamped)

    def close(self) -> None:
        return None


class LatestFrame:
    """The newest complete frame a stream delivered, and when the screen last matched it.

    ScreenCaptureKit sends no new image while the window is unchanged, only "idle"
    notices. Those keep the last image current, so its time moves forward.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._arrived = threading.Event()
        self._buffer: object = None
        self._at = 0.0

    def offer(self, buffer: object, status: int | None, at: float) -> None:
        with self._lock:
            if buffer is not None and status in (None, ScreenCaptureKit.SCFrameStatusComplete):
                self._buffer, self._at = buffer, at
                self._arrived.set()
            elif status == ScreenCaptureKit.SCFrameStatusIdle and self._buffer is not None:
                self._at = at

    def take(self, wait: float) -> tuple[object, float]:
        if not self._arrived.wait(wait):
            raise RuntimeError("the ScreenCaptureKit stream delivered no frame")
        with self._lock:
            return self._buffer, self._at


def _frame_status(sample: object) -> int | None:
    attachments = CoreMedia.CMSampleBufferGetSampleAttachmentsArray(sample, False)
    if not attachments:
        return None
    status = attachments[0].get(ScreenCaptureKit.SCStreamFrameInfoStatus)
    return None if status is None else int(status)


# pyobjc's metaclass takes `protocols`; the stubs don't know it.
class _Output(NSObject, protocols=[objc.protocolNamed("SCStreamOutput")]):  # pyright: ignore[reportGeneralTypeIssues, reportCallIssue]
    """Hands each delivered sample to a `LatestFrame`."""

    latest: LatestFrame

    def stream_didOutputSampleBuffer_ofType_(
        self, _stream: object, sample: object, kind: int
    ) -> None:
        if kind != ScreenCaptureKit.SCStreamOutputTypeScreen:
            return
        self.latest.offer(
            CoreMedia.CMSampleBufferGetImageBuffer(sample), _frame_status(sample), time.monotonic()
        )


def _answer(ask: Callable[[Callable[..., None]], None]) -> tuple[object, ...]:
    """Call a ScreenCaptureKit method that answers through a completion handler."""
    answered = threading.Event()
    answers: list[tuple[object, ...]] = []

    def handler(*values: object) -> None:
        answers.append(values)
        answered.set()

    ask(handler)
    if not answered.wait(ANSWER_SECONDS):
        raise RuntimeError("ScreenCaptureKit didn't answer")
    return answers[0]


class Stream(Protocol):
    top: int  # Window rows above the game area.

    def take(self) -> tuple[object, float]: ...
    def stop(self) -> None: ...


class _ScreenCaptureKitStream:
    """A running stream of one window at full pixel size."""

    def __init__(self, geometry: GameGeometry) -> None:
        content, error = _answer(
            lambda handler: (
                ScreenCaptureKit.SCShareableContent.getShareableContentExcludingDesktopWindows_onScreenWindowsOnly_completionHandler_(
                    True, False, handler
                )
            )
        )
        if error is not None:
            raise RuntimeError(f"ScreenCaptureKit refused: {error}")
        window = next(
            (item for item in content.windows() if item.windowID() == geometry.window_id), None
        )
        if window is None:
            raise RuntimeError("ScreenCaptureKit can't see the game window")
        frame = window.frame()
        width = round(frame.size.width * geometry.scale)
        height = round(frame.size.height * geometry.scale)
        if width != geometry.width or height < geometry.height:
            raise RuntimeError(
                f"the window is {width}x{height}, but the game area is "
                f"{geometry.width}x{geometry.height}"
            )
        self.top = height - geometry.height  # The band is cut from the window's top.
        config = ScreenCaptureKit.SCStreamConfiguration.alloc().init()
        config.setWidth_(width)
        config.setHeight_(height)
        config.setPixelFormat_(Quartz.kCVPixelFormatType_32BGRA)
        config.setShowsCursor_(False)
        config.setMinimumFrameInterval_(CoreMedia.CMTimeMake(1, STREAM_FPS))
        config.setQueueDepth_(STREAM_QUEUE_DEPTH)
        source = ScreenCaptureKit.SCContentFilter.alloc().initWithDesktopIndependentWindow_(window)
        self._latest = LatestFrame()
        self._output = _Output.alloc().init()
        self._output.latest = self._latest
        self._stream = ScreenCaptureKit.SCStream.alloc().initWithFilter_configuration_delegate_(
            source, config, None
        )
        added, error = self._stream.addStreamOutput_type_sampleHandlerQueue_error_(
            self._output, ScreenCaptureKit.SCStreamOutputTypeScreen, None, None
        )
        if not added:
            raise RuntimeError(f"ScreenCaptureKit wouldn't add the output: {error}")
        (error,) = _answer(self._stream.startCaptureWithCompletionHandler_)
        if error is not None:
            raise RuntimeError(f"ScreenCaptureKit wouldn't start: {error}")
        log.info("capture_stream_started", window_id=geometry.window_id, size=f"{width}x{height}")

    def take(self) -> tuple[object, float]:
        return self._latest.take(ANSWER_SECONDS)

    def stop(self) -> None:
        try:
            _answer(self._stream.stopCaptureWithCompletionHandler_)
        except Exception as error:  # Stopping is cleanup; the process may be exiting.
            log.debug("capture_stream_stop_failed", error=repr(error))


def pixel_buffer_to_rgb(buffer: object, top: int) -> np.ndarray:
    """RGB pixels of a BGRA CVPixelBuffer, without its first `top` rows."""
    Quartz.CVPixelBufferLockBaseAddress(buffer, Quartz.kCVPixelBufferLock_ReadOnly)
    try:
        row_bytes = Quartz.CVPixelBufferGetBytesPerRow(buffer)
        height = Quartz.CVPixelBufferGetHeight(buffer)
        width = Quartz.CVPixelBufferGetWidth(buffer)
        base = Quartz.CVPixelBufferGetBaseAddress(buffer)
        raw = bytes(base.as_buffer(row_bytes * height))
    finally:
        Quartz.CVPixelBufferUnlockBaseAddress(buffer, Quartz.kCVPixelBufferLock_ReadOnly)
    return bgra_to_rgb(raw, width, height, row_bytes, top)


class StreamGrabber:
    """The newest frame of a ScreenCaptureKit stream of the game window.

    The stream starts on the first grab and restarts when the window or the game
    area's size changes.
    """

    def __init__(
        self,
        open_stream: Callable[[GameGeometry], Stream] = _ScreenCaptureKitStream,
        to_rgb: Callable[[object, int], np.ndarray] = pixel_buffer_to_rgb,
    ) -> None:
        self._open_stream = open_stream
        self._to_rgb = to_rgb
        self._lock = threading.Lock()
        self._stream: Stream | None = None
        self._key: tuple[int | None, int, int] | None = None

    def grab(self, geometry: GameGeometry | None, monitor: int) -> Frame:
        del monitor  # The window is always known on macOS.
        area = _game_area(geometry)
        key = (area.window_id, area.width, area.height)
        with self._lock:
            if self._stream is None or self._key != key:
                self._stop()
                self._stream, self._key = self._open_stream(area), key
            stream = self._stream
        buffer, captured_at = stream.take()
        pixels = self._to_rgb(buffer, stream.top)
        if pixels.shape[:2] != (area.height, area.width):
            raise RuntimeError(
                f"the stream gave {pixels.shape[1]}x{pixels.shape[0]}, but the game area "
                f"is {area.width}x{area.height}"
            )
        return Frame(pixels.tobytes(), area.width, area.height, captured_at)

    def _stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream, self._key = None, None

    def close(self) -> None:
        with self._lock:
            self._stop()
