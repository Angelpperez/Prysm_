from __future__ import annotations

import uvicorn
from pathlib import Path
import logging

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fasthtml.common import Link, NotStr, Script, Style, Title, fast_app
import httpx

from .client import fetch_status, post_calibrate
from .config import Settings
from .ui import render_index, render_status

settings = Settings()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("prysm-api")

app, rt = fast_app(
    pico=False,
    hdrs=[
        Title(settings.ui_title),
        NotStr('<meta name="viewport" content="width=device-width, initial-scale=1">'),
        Script(src="https://unpkg.com/htmx.org@1.9.12"),
        Link(rel="icon", href="/favicon.ico", type="image/x-icon"),
        Link(rel="shortcut icon", href="/favicon.ico", type="image/x-icon"),
        Link(
            rel="stylesheet",
            href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600&display=swap",
            type="text/css",
        ),
        Link(
            rel="stylesheet",
            href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap",
            type="text/css",
        ),
        Link(
            rel="stylesheet",
            href="https://cdn.jsdelivr.net/npm/@picocss/pico@latest/css/pico.min.css",
            type="text/css",
        ),
        Style(
            "@font-face {"
            "  font-family: 'Digitek';"
            "  src: local('Digitek'), local('Digitek Regular'),"
            "       url('/static/fonts/Digitek.woff2') format('woff2'),"
            "       url('/static/fonts/Digitek.woff') format('woff');"
            "  font-weight: 400;"
            "  font-style: normal;"
            "  font-display: swap;"
            "}"
            "html, body { height: 100%; }"
            "body { margin: 0; font-family: 'Space Grotesk', sans-serif;"
            "background: radial-gradient(1200px 600px at 10% 10%, #1b2a3a 0%, #0b0f14 45%, #05070a 100%);"
            "color: #f2f2f2; overflow: hidden; }"
            "#app { height: 100vh; display: flex; flex-direction: column; padding: 18px 24px; gap: 12px; }"
            "#navbar { display: grid; grid-template-columns: auto 1fr auto; align-items: center; gap: 16px; }"
            "#brand { display: flex; align-items: center; gap: 10px; min-width: 0; }"
            "#title-center { text-align: center; font-size: clamp(20px, 2.4vw, 34px); font-weight: 700; letter-spacing: 0.02em; line-height: 1; }"
            "#repo-icon { justify-self: end; width: 28px; height: 28px; opacity: 0.8;"
            "background-image: url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='white'><path d='M12 0.5C5.73 0.5.5 5.73.5 12a11.5 11.5 0 0 0 7.86 10.94c.58.1.79-.25.79-.56v-2.02c-3.2.7-3.87-1.54-3.87-1.54-.53-1.35-1.3-1.7-1.3-1.7-1.06-.73.08-.72.08-.72 1.17.08 1.78 1.2 1.78 1.2 1.04 1.78 2.73 1.27 3.4.97.1-.75.4-1.27.72-1.56-2.56-.29-5.26-1.28-5.26-5.7 0-1.26.45-2.29 1.2-3.1-.12-.3-.52-1.52.12-3.16 0 0 .97-.31 3.18 1.18.92-.26 1.9-.39 2.88-.39s1.96.13 2.88.39c2.2-1.49 3.18-1.18 3.18-1.18.64 1.64.24 2.86.12 3.16.75.81 1.2 1.84 1.2 3.1 0 4.43-2.7 5.4-5.27 5.69.41.36.78 1.07.78 2.16v3.2c0 .31.21.66.8.55A11.5 11.5 0 0 0 23.5 12C23.5 5.73 18.27.5 12 .5Z'/></svg>\");"
            "background-size: cover; background-repeat: no-repeat; }"
            "#content { flex: 1; display: flex; flex-direction: column; gap: 10px; min-height: 0; }"
            "#status { backdrop-filter: blur(6px); padding: 10px 14px; border-radius: 10px;"
            "background: rgba(10,10,10,0.65); font-size: 0.9rem; line-height: 1.2; letter-spacing: 0.01em;"
            "display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));"
            "column-gap: 18px; row-gap: 6px; }"
            "#status p { margin: 0; }"
            "#video-wrap { flex: 1; border-radius: 10px; border: 1px solid #2a2f3a; overflow: hidden; }"
            "#video { width: 100%; height: 100%; object-fit: contain; display: block; }"
            "img { display: block; }"
            "@media (max-width: 900px) {"
            "  #app { padding: 12px 14px; }"
            "  #navbar { grid-template-columns: auto 1fr auto; gap: 10px; }"
            "  #title-center { font-size: clamp(18px, 4vw, 22px); }"
            "  #repo-icon { width: 24px; height: 24px; }"
            "  #status { font-size: 0.85rem; grid-template-columns: 1fr; }"
            "}"
        ),
    ]
)

ROOT_DIR = Path(__file__).resolve().parents[1]
ASSET_DIR = Path(__file__).resolve().parent / "static"
ICON_ICO = ASSET_DIR / "favicon.ico"
ICON_WEBP = ASSET_DIR / "favicon.webp"
ICON_WEBP_LEGACY = ASSET_DIR / "favicon.ico.webp"
ROOT_ICON_ICO = ROOT_DIR / "favicon.ico"
ROOT_ICON_WEBP = ROOT_DIR / "favicon.webp"
if ASSET_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(ASSET_DIR)), name="static")


@app.get("/favicon.ico")
def favicon():
    if ICON_ICO.exists():
        return FileResponse(ICON_ICO, media_type="image/x-icon")
    if ROOT_ICON_ICO.exists():
        return FileResponse(ROOT_ICON_ICO, media_type="image/x-icon")
    if ICON_WEBP.exists():
        return FileResponse(ICON_WEBP, media_type="image/webp")
    if ROOT_ICON_WEBP.exists():
        return FileResponse(ROOT_ICON_WEBP, media_type="image/webp")
    return FileResponse(ICON_WEBP_LEGACY, media_type="image/webp")


@app.get("/favicon.ico.webp")
def favicon_webp():
    if ICON_WEBP.exists():
        return FileResponse(ICON_WEBP, media_type="image/webp")
    return FileResponse(ICON_WEBP_LEGACY, media_type="image/webp")


@app.get("/stream")
async def stream_proxy(request: Request):
    if not settings.use_stream_proxy:
        raise HTTPException(status_code=404, detail="stream proxy disabled")
    client_host = request.client.host if request.client else "unknown"
    logger.info("stream proxy: client connected from %s", client_host)

    async def gen():
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                url = f"{settings.vision_service_url}/stream"
                logger.info("stream proxy: connecting to %s", url)
                async with client.stream("GET", url) as resp:
                    logger.info("stream proxy: upstream status %s", resp.status_code)
                    resp.raise_for_status()
                    async for chunk in resp.aiter_raw():
                        if await request.is_disconnected():
                            logger.info("stream proxy: client disconnected")
                            break
                        yield chunk
        except Exception as exc:
            logger.exception("stream proxy error: %s", exc)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@rt("/")
def index():
    if settings.use_stream_proxy:
        stream_url = "/stream"
    else:
        public_url = settings.vision_public_url or settings.vision_service_url
        stream_url = f"{public_url}/stream"
    return render_index(settings.ui_title, stream_url, settings.status_refresh_ms)


@rt("/status")
async def status_fragment():
    status = await fetch_status(settings.vision_service_url)
    if status is None:
        logger.warning("status fragment: vision_service unavailable")
    return render_status(status)


@app.get("/api/status")
async def api_status():
    status = await fetch_status(settings.vision_service_url)
    if status is None:
        logger.warning("api status: vision_service unavailable")
        raise HTTPException(status_code=503, detail="vision_service unavailable")
    return status


@app.post("/api/calibrate")
async def api_calibrate(payload: dict):
    result = await post_calibrate(settings.vision_service_url, payload)
    if result is None:
        raise HTTPException(status_code=503, detail="vision_service unavailable")
    return result


if __name__ == "__main__":
    uvicorn.run("api_service.main:app", host=settings.host, port=settings.port, log_level="info")
