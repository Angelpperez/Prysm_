from __future__ import annotations

from pathlib import Path


def load_scale(path: Path, default: float) -> float:
    try:
        if path.exists():
            value = float(path.read_text().strip())
            return value
    except Exception:
        pass
    return default


def save_scale(path: Path, mm_per_pixel: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{mm_per_pixel:.12f}")
