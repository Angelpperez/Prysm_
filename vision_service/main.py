from __future__ import annotations

import logging
import threading
import time
from contextlib import asynccontextmanager
from typing import Optional

import cv2
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from .calibration import load_scale, save_scale
from .camera import SyntheticFrameSource, open_frame_source
from .config import Settings
from .detection import find_mark_centers, pick_farthest_pair
from .plc import build_plc_client
from .state import Measurement, VisionState
from .streaming import mjpeg_stream

settings = Settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(settings.service_name)

state = VisionState()


class VisionWorker:
    def __init__(self, settings: Settings, state: VisionState) -> None:
        self._settings = settings
        self._state = state
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._mm_lock = threading.Lock()
        self._mm_per_pixel: Optional[float] = None
        self._mm_per_pixel_est: float = 1.0
        self._plc = build_plc_client(settings.plc_mode)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="vision-loop", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def set_mm_per_pixel(self, value: float) -> None:
        with self._mm_lock:
            self._mm_per_pixel = value

    def get_mm_per_pixel(self) -> float:
        with self._mm_lock:
            return self._mm_per_pixel if self._mm_per_pixel is not None else self._mm_per_pixel_est

    def get_mm_per_pixel_est(self) -> float:
        with self._mm_lock:
            return self._mm_per_pixel_est

    def calibrate(self, dist_ref_mm: Optional[float], mm_per_pixel: Optional[float]) -> float:
        if mm_per_pixel is None:
            dist_ref = dist_ref_mm if dist_ref_mm is not None else self._settings.dist_ref_mm
            last_px = self._state.get_last_dist_px()
            if not last_px:
                raise ValueError("No markers available for calibration")
            mm_per_pixel = dist_ref / last_px

        self.set_mm_per_pixel(mm_per_pixel)
        save_scale(self._settings.scale_file, mm_per_pixel)
        return mm_per_pixel

    def _compute_mm_per_pixel_est(self) -> float:
        pixel_pitch_mm = self._settings.pixel_pitch_um / 1000.0
        if self._settings.working_distance_mm > self._settings.focal_mm:
            return (
                pixel_pitch_mm
                * (self._settings.working_distance_mm - self._settings.focal_mm)
                / self._settings.focal_mm
            )
        return 1.0

    def _run(self) -> None:
        source = None
        used_fallback = False
        attempts = 0
        while not self._stop_event.is_set():
            try:
                source = open_frame_source(self._settings)
                break
            except Exception as exc:
                attempts += 1
                if (
                    self._settings.allow_synthetic_fallback
                    and not self._settings.video_source
                    and not used_fallback
                ):
                    logger.error("Camera open failed: %s", exc)
                    logger.warning("Falling back to synthetic video source.")
                    source = SyntheticFrameSource(
                        self._settings.width,
                        self._settings.height,
                        self._settings.fps or 10.0,
                    )
                    used_fallback = True
                    break
                if not self._settings.allow_synthetic_fallback and attempts >= self._settings.camera_open_retries:
                    logger.critical(
                        "Camera open failed after %d attempts. Exiting vision loop.", attempts
                    )
                    return
                logger.exception("Camera open failed: %s", exc)
                time.sleep(self._settings.camera_open_retry_s)

        if source is None:
            return

        self._mm_per_pixel_est = self._compute_mm_per_pixel_est()
        mm_per_pixel = load_scale(self._settings.scale_file, self._mm_per_pixel_est)
        self.set_mm_per_pixel(mm_per_pixel)

        t_last = time.time()
        fps_smoothed: Optional[float] = None
        alpha = 0.2
        last_plc_color: Optional[str] = None
        last_log = time.time()

        try:
            while not self._stop_event.is_set():
                try:
                    frame = source.read(timeout_ms=1000)
                except Exception as exc:
                    logger.warning("Frame grab failed: %s", exc)
                    time.sleep(0.2)
                    continue

                now = time.time()
                dt = now - t_last
                t_last = now
                fps_inst = (1.0 / dt) if dt > 0 else 0.0
                fps_smoothed = fps_inst if fps_smoothed is None else (alpha * fps_inst + (1 - alpha) * fps_smoothed)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                centers, _ = find_mark_centers(gray, self._settings.threshold_val, self._settings.min_area)

                status = "LIVE"
                distance_px = None
                distance_mm = None
                last_dist_px = None

                mm_per_pixel = self.get_mm_per_pixel()

                if len(centers) >= 2:
                    pair = pick_farthest_pair(centers)
                    if pair:
                        c1, c2, distance_px = pair
                        last_dist_px = distance_px
                        distance_mm = distance_px * mm_per_pixel

                        if abs(distance_mm - self._settings.dist_ref_mm) <= self._settings.tolerance_mm:
                            status = "NORMAL"
                            color = (0, 255, 0)
                        else:
                            status = "ALERTA"
                            color = (0, 0, 255)

                        cv2.line(frame, c1, c2, color, 2)
                        mid = ((c1[0] + c2[0]) // 2, (c1[1] + c2[1]) // 2)
                        cv2.putText(
                            frame,
                            f"{distance_mm:.1f} mm",
                            (mid[0] + 10, mid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            color,
                            2,
                            cv2.LINE_AA,
                        )

                header = f"Status: {status}"
                if distance_mm is not None:
                    header += f"  Dist: {distance_mm:.1f} mm"
                cv2.putText(
                    frame,
                    header,
                    (15, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                color_name = "YELLOW" if status == "LIVE" else ("GREEN" if status == "NORMAL" else "RED")
                if color_name != last_plc_color:
                    try:
                        self._plc.set_light(color_name)
                        last_plc_color = color_name
                    except Exception as exc:
                        logger.warning("PLC update failed: %s", exc)

                ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self._settings.jpeg_quality])
                if not ok:
                    continue

                measurement = Measurement(
                    timestamp=now,
                    status=status,
                    distance_px=distance_px,
                    distance_mm=distance_mm,
                    fps=fps_smoothed,
                    mm_per_pixel=mm_per_pixel,
                    dist_ref_mm=self._settings.dist_ref_mm,
                    tolerance_mm=self._settings.tolerance_mm,
                )

                self._state.update(
                    frame_bgr=frame,
                    frame_jpeg=buf.tobytes(),
                    measurement=measurement,
                    last_dist_px=last_dist_px,
                    mm_per_pixel_est=self._mm_per_pixel_est,
                    stream_fps=self._settings.stream_fps,
                )

                if now - last_log >= 5.0:
                    logger.info(
                        "vision status=%s dist_mm=%s fps=%.1f mm_per_px=%.6f",
                        status,
                        f"{distance_mm:.1f}" if distance_mm is not None else "None",
                        fps_smoothed or 0.0,
                        mm_per_pixel,
                    )
                    last_log = now
        finally:
            try:
                source.close()
            except Exception:
                pass


worker = VisionWorker(settings, state)


@asynccontextmanager
async def lifespan(app: FastAPI):
    worker.start()
    yield
    worker.stop()


app = FastAPI(title="Prysm Vision Service", lifespan=lifespan)


class CalibrateRequest(BaseModel):
    dist_ref_mm: Optional[float] = None
    mm_per_pixel: Optional[float] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status")
def status():
    return state.snapshot()


@app.get("/frame")
def frame():
    data = state.get_frame_jpeg()
    if not data:
        raise HTTPException(status_code=503, detail="No frame available")
    return Response(content=data, media_type="image/jpeg")


@app.get("/stream")
def stream(request: Request):
    client_host = request.client.host if request.client else "unknown"
    logger.info("stream: client connected from %s", client_host)
    return StreamingResponse(
        mjpeg_stream(state, settings.stream_fps),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/calibrate")
def calibrate(req: CalibrateRequest):
    try:
        value = worker.calibrate(req.dist_ref_mm, req.mm_per_pixel)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"mm_per_pixel": value, "dist_ref_mm": req.dist_ref_mm or settings.dist_ref_mm}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("vision_service.main:app", host=settings.host, port=settings.port, log_level=settings.log_level)
