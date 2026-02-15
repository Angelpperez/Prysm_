from __future__ import annotations

import math
import time

import cv2
import numpy as np

from vision_app.vision.vision import BaslerUsbCamera, CameraCfg

from .config import Settings


class FrameSource:
    def read(self, timeout_ms: int = 1000) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class BaslerFrameSource(FrameSource):
    def __init__(self, cam: BaslerUsbCamera):
        self._cam = cam

    def read(self, timeout_ms: int = 1000) -> np.ndarray:
        return self._cam.grab_bgr(timeout_ms=timeout_ms)

    def close(self) -> None:
        self._cam.stop()
        self._cam.close()


class VideoFrameSource(FrameSource):
    def __init__(self, source: str):
        src = int(source) if source.isdigit() else source
        self._cap = cv2.VideoCapture(src)
        if not self._cap.isOpened():
            raise RuntimeError(f"Video source not available: {source}")

    def read(self, timeout_ms: int = 0) -> np.ndarray:
        ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError("Video source read failed")
        return frame

    def close(self) -> None:
        self._cap.release()


class SyntheticFrameSource(FrameSource):
    def __init__(self, width: int, height: int, fps: float = 10.0) -> None:
        self._width = max(320, width)
        self._height = max(240, height)
        self._fps = fps if fps and fps > 0 else 10.0
        self._last = time.time()
        self._t0 = time.time()

    def read(self, timeout_ms: int = 0) -> np.ndarray:
        now = time.time()
        elapsed = now - self._last
        delay = max(0.0, (1.0 / self._fps) - elapsed)
        if delay > 0:
            time.sleep(delay)
        self._last = time.time()

        frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        t = self._last - self._t0
        gap = min(self._width // 2, 900)
        jitter = int(20 * math.sin(t))
        x1 = self._width // 2 - gap // 2 + jitter
        x2 = self._width // 2 + gap // 2 + jitter
        y = self._height // 2
        rect_w = max(20, self._width // 80)
        rect_h = max(120, self._height // 6)

        for x in (x1, x2):
            top_left = (max(0, x - rect_w // 2), max(0, y - rect_h // 2))
            bottom_right = (min(self._width - 1, x + rect_w // 2), min(self._height - 1, y + rect_h // 2))
            cv2.rectangle(frame, top_left, bottom_right, (255, 255, 255), -1)
            cv2.rectangle(frame, (top_left[0], top_left[1] - rect_h // 3), (bottom_right[0], top_left[1]), (255, 255, 255), -1)
            cv2.rectangle(frame, (top_left[0], bottom_right[1]), (bottom_right[0], bottom_right[1] + rect_h // 3), (255, 255, 255), -1)

        return frame

    def close(self) -> None:
        return


def open_frame_source(settings: Settings) -> FrameSource:
    if settings.video_source:
        if settings.video_source.lower() in {"synthetic", "demo", "test"}:
            return SyntheticFrameSource(settings.width, settings.height, settings.fps or 10.0)
        return VideoFrameSource(settings.video_source)

    cfg = CameraCfg(
        width=settings.width,
        height=settings.height,
        offset_x=settings.offset_x,
        offset_y=settings.offset_y,
        pixel_format=settings.pixel_format,
        exposure_us=settings.exposure_us,
        gain=settings.gain,
        fps=settings.fps,
        latest_only=settings.latest_only,
    )

    cam = BaslerUsbCamera(serial=settings.camera_serial).open()
    cam.configure(cfg)
    cam.start(latest_only=cfg.latest_only)
    return BaslerFrameSource(cam)
