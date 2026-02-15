from __future__ import annotations

import time
from typing import Generator

from .state import VisionState


def mjpeg_stream(state: VisionState, fps: float) -> Generator[bytes, None, None]:
    delay = 1.0 / fps if fps and fps > 0 else 0.1
    while True:
        frame = state.get_frame_jpeg()
        if frame:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(delay)
