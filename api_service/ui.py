from __future__ import annotations

from typing import Any, Dict, Optional

from fasthtml.common import A, Div, Img, P, Span


def render_index(title: str, stream_url: str, refresh_ms: int):
    citec_style = (
        "margin: 0;"
        "font-family: \"Roboto\", \"Helvetica\", \"Arial\", sans-serif;"
        "font-weight: 400;"
        "font-size: 1rem;"
        "line-height: 1.5;"
        "letter-spacing: 0.00938em;"
        "color: #EDEDED;"
        "margin-right: 16px;"
        "letter-spacing: .2rem;"
        "font-size: 24px;"
        "-webkit-text-decoration: none;"
        "text-decoration: none;"
        "padding-left: 14.4px;"
        "font-family: Digitek;"
        "width: 100%;"
    )
    brand = Div(
        Img(
            src="/favicon.ico",
            alt="CITEC",
            style="height:32px;width:32px;object-fit:contain;",
        ),
        Span("CITEC", style=citec_style),
        id="brand",
    )
    title_center = Div(title, id="title-center")
    repo_icon = A(
        Div(id="repo-icon", title="GitHub"),
        href="https://github.com/Angelpperez/Prysm_",
        target="_blank",
        rel="noopener noreferrer",
        style="display:inline-block;",
    )
    return Div(
        Div(
            brand,
            title_center,
            repo_icon,
            id="navbar",
            role="navigation",
        ),
        Div(
            Div(
                id="status",
                hx_get="/status",
                hx_trigger=f"load, every {refresh_ms}ms",
                hx_swap="innerHTML",
            ),
            Div(
                Img(src=stream_url, id="video"),
                id="video-wrap",
            ),
            id="content",
        ),
        id="app",
    )


def render_status(status: Optional[Dict[str, Any]]):
    if not status:
        return Div(P("Sin conexion con vision_service"))

    estado = status.get("status", "UNKNOWN")
    color = status_color(estado)
    dist_mm = status.get("distance_mm")
    dist_px = status.get("distance_px")
    fps = status.get("fps")
    stream_fps = status.get("stream_fps")
    mm_per_pixel = status.get("mm_per_pixel")
    tol = status.get("tolerance_mm")
    ref = status.get("dist_ref_mm")

    lines = [
        P(Span("Estado: "), Span(estado, style=f"color:{color};font-weight:700;")),
    ]

    if dist_mm is not None and dist_px is not None:
        lines.append(P(f"Distancia: {dist_mm:.1f} mm ({dist_px:.1f} px)"))
    if ref is not None and tol is not None:
        lines.append(P(f"Referencia: {ref:.1f} mm  Tolerancia: {tol:.1f} mm"))
    if mm_per_pixel is not None:
        lines.append(P(f"Escala: {mm_per_pixel:.6f} mm/px"))
    if fps is not None:
        lines.append(P(f"FPS: {fps:.1f}"))
    if stream_fps is not None:
        lines.append(P(f"Stream FPS: {stream_fps:.1f}"))

    return Div(*lines, style="padding:10px;background:#111;color:#f2f2f2;border-radius:6px;")


def status_color(estado: str) -> str:
    if estado == "NORMAL":
        return "#2e8b57"
    if estado == "ALERTA":
        return "#b00020"
    if estado == "LIVE":
        return "#f0a500"
    return "#888888"
