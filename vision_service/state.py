from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, Optional

import numpy as np
import time


@dataclass
class Measurement:
    timestamp: float
    status: str
    distance_px: Optional[float]
    distance_mm: Optional[float]
    fps: Optional[float]
    mm_per_pixel: float
    dist_ref_mm: float
    tolerance_mm: float


class VisionState:
    def __init__(self) -> None:
        self._lock = Lock()
        self._frame_bgr: Optional[np.ndarray] = None
        self._frame_jpeg: Optional[bytes] = None
        self._measurement: Optional[Measurement] = None
        self._last_dist_px: Optional[float] = None
        self._mm_per_pixel_est: Optional[float] = None
        self._mm_per_pixel: Optional[float] = None
        self._stream_fps: Optional[float] = None

    def update(
        self,
        frame_bgr: np.ndarray,
        frame_jpeg: bytes,
        measurement: Measurement,
        last_dist_px: Optional[float],
        mm_per_pixel_est: float,
        stream_fps: Optional[float],
    ) -> None:
        with self._lock:
            self._frame_bgr = frame_bgr
            self._frame_jpeg = frame_jpeg
            self._measurement = measurement
            self._last_dist_px = last_dist_px
            self._mm_per_pixel = measurement.mm_per_pixel
            self._mm_per_pixel_est = mm_per_pixel_est
            self._stream_fps = stream_fps

    def set_mm_per_pixel(self, value: float) -> None:
        with self._lock:
            self._mm_per_pixel = value

    def get_mm_per_pixel(self) -> Optional[float]:
        with self._lock:
            return self._mm_per_pixel

    def get_last_dist_px(self) -> Optional[float]:
        with self._lock:
            return self._last_dist_px

    def get_frame_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._frame_jpeg

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            if not self._measurement:
                return {
                    "timestamp": time.time(),
                    "status": "STARTING",
                    "distance_px": None,
                    "distance_mm": None,
                    "fps": None,
                    "mm_per_pixel": self._mm_per_pixel,
                    "mm_per_pixel_est": self._mm_per_pixel_est,
                    "dist_ref_mm": None,
                    "tolerance_mm": None,
                    "stream_fps": self._stream_fps,
                }
            m = self._measurement
            return {
                "timestamp": m.timestamp,
                "status": m.status,
                "distance_px": m.distance_px,
                "distance_mm": m.distance_mm,
                "fps": m.fps,
                "mm_per_pixel": m.mm_per_pixel,
                "mm_per_pixel_est": self._mm_per_pixel_est,
                "dist_ref_mm": m.dist_ref_mm,
                "tolerance_mm": m.tolerance_mm,
                "stream_fps": self._stream_fps,
            }
