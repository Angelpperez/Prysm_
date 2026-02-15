from __future__ import annotations

from typing import Any, Dict, Optional

import httpx


DEFAULT_TIMEOUT = 2.0


def _timeout() -> float:
    return DEFAULT_TIMEOUT


async def fetch_status(base_url: str) -> Optional[Dict[str, Any]]:
    try:
        async with httpx.AsyncClient(timeout=_timeout()) as client:
            resp = await client.get(f"{base_url}/status")
            resp.raise_for_status()
            return resp.json()
    except Exception:
        return None


async def post_calibrate(base_url: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(f"{base_url}/calibrate", json=payload)
            resp.raise_for_status()
            return resp.json()
    except Exception:
        return None
