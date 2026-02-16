from __future__ import annotations

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PRYSM_", env_file=".env", extra="ignore")

    host: str = "0.0.0.0"
    port: int = 8000
    ui_title: str = "Prysmian Monitor"
    status_refresh_ms: int = 1000
    vision_service_url: str = "http://localhost:8001"
    vision_public_url: Optional[str] = None
    use_stream_proxy: bool = True
