from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PRYSM_", env_file=".env", extra="ignore")

    service_name: str = "prysm-vision"
    host: str = "0.0.0.0"
    port: int = 8001
    log_level: str = "info"

    camera_serial: Optional[str] = None
    width: int = 3840
    height: int = 2160
    offset_x: int = 0
    offset_y: int = 0
    pixel_format: Optional[str] = None
    exposure_us: Optional[float] = None
    gain: Optional[float] = None
    fps: Optional[float] = 43.0
    latest_only: bool = True
    video_source: Optional[str] = None

    threshold_val: int = 110
    min_area: int = 100
    dist_ref_mm: float = 1000.0
    tolerance_mm: float = 10.0

    pixel_pitch_um: float = 2.0
    focal_mm: float = 8.0
    working_distance_mm: float = 1600.0

    stream_fps: float = 10.0
    jpeg_quality: int = 80

    plc_mode: str = "noop"
    allow_synthetic_fallback: bool = False
    camera_open_retries: int = 5
    camera_open_retry_s: float = 2.0

    data_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent / "data")
    scale_file: Optional[Path] = None

    def model_post_init(self, __context) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if self.scale_file is None:
            self.scale_file = self.data_dir / "mm_per_pixel.txt"
