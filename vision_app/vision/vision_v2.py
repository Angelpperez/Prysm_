from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    from vision_app.vision.vision import (
        BaslerUsbCamera,
        CameraCfg,
        draw_hud,
        load_scale,
        pick_farthest_pair,
        save_scale,
    )
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from vision_app.vision.vision import (  # type: ignore
        BaslerUsbCamera,
        CameraCfg,
        draw_hud,
        load_scale,
        pick_farthest_pair,
        save_scale,
    )

try:
    from pypylon import pylon
except Exception:  # pragma: no cover - runtime dependency
    pylon = None


@dataclass
class IDetectParams:
    thresh_val: int = 120
    min_area: int = 300
    blur_ksize: int = 5
    use_otsu: bool = False
    morph_kernel: int = 3
    morph_open: int = 1
    morph_close: int = 2
    aspect_min: float = 1.6
    aspect_max: float = 10.0
    cap_ratio_min: float = 1.2
    fill_min: float = 0.10
    fill_max: float = 0.65
    vertical_coverage_min: float = 0.75


def _threshold_mask(gray: np.ndarray, params: IDetectParams) -> np.ndarray:
    if params.blur_ksize and params.blur_ksize > 1:
        k = params.blur_ksize if params.blur_ksize % 2 == 1 else params.blur_ksize + 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)
    if params.use_otsu:
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(gray, params.thresh_val, 255, cv2.THRESH_BINARY)
    return mask


def _clean_mask(mask: np.ndarray, params: IDetectParams) -> np.ndarray:
    k = max(1, params.morph_kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    if params.morph_open > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=params.morph_open)
    if params.morph_close > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=params.morph_close)
    return mask


def _width_profile(mask_roi: np.ndarray) -> np.ndarray:
    h, _ = mask_roi.shape
    widths = np.zeros(h, dtype=np.float32)
    for y in range(h):
        xs = np.flatnonzero(mask_roi[y] > 0)
        if xs.size:
            widths[y] = float(xs[-1] - xs[0] + 1)
    return widths


def _i_shape_score(mask_roi: np.ndarray, params: IDetectParams) -> Tuple[float, float, float]:
    h, w = mask_roi.shape
    if h < 8 or w < 3:
        return 0.0, 0.0, 0.0

    widths = _width_profile(mask_roi)
    if widths.max() <= 0:
        return 0.0, 0.0, 0.0

    top_end = max(1, int(0.25 * h))
    mid_start = int(0.35 * h)
    mid_end = int(0.65 * h)
    bot_start = int(0.75 * h)

    top = widths[:top_end]
    mid = widths[mid_start:mid_end] if mid_end > mid_start else widths
    bot = widths[bot_start:] if bot_start < h else widths

    def _median_nonzero(arr: np.ndarray) -> float:
        nz = arr[arr > 0]
        return float(np.median(nz)) if nz.size else 0.0

    top_w = _median_nonzero(top)
    mid_w = _median_nonzero(mid)
    bot_w = _median_nonzero(bot)

    if mid_w <= 0:
        return 0.0, 0.0, 0.0

    cap_ratio = (top_w + bot_w) / (2.0 * mid_w)
    fill = float(mask_roi.sum() / 255.0) / float(h * w)

    covered_rows = float(np.count_nonzero(widths)) / float(h)

    score = cap_ratio * (0.5 + fill)
    if covered_rows < params.vertical_coverage_min:
        score *= 0.5

    return score, cap_ratio, fill


def detect_i_markers(gray: np.ndarray, params: IDetectParams) -> List[Tuple[Tuple[int, int], Tuple[int, int, int, int], float]]:
    mask = _threshold_mask(gray, params)
    mask = _clean_mask(mask, params)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    markers: List[Tuple[Tuple[int, int], Tuple[int, int, int, int], float]] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < params.min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w <= 0 or h <= 0:
            continue

        aspect = float(h) / float(w)
        if aspect < params.aspect_min or aspect > params.aspect_max:
            continue

        roi = mask[y : y + h, x : x + w]
        score, cap_ratio, fill = _i_shape_score(roi, params)
        if cap_ratio < params.cap_ratio_min:
            continue
        if not (params.fill_min <= fill <= params.fill_max):
            continue

        if score < 0.8:
            continue

        cx = x + w // 2
        cy = y + h // 2
        markers.append(((cx, cy), (x, y, w, h), score))

    return markers


def _open_video_source(source: str) -> cv2.VideoCapture:
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Video source not available: {source}")
    return cap


def _select_pixel_format(cam: BaslerUsbCamera, candidates: List[str]) -> Optional[str]:
    if not cam.cam:
        return None
    supported = list(cam.cam.PixelFormat.Symbolics)
    for cand in candidates:
        if cand in supported:
            return cand
    return None


def _get_max_resolution(cam: BaslerUsbCamera) -> Tuple[int, int]:
    if not cam.cam:
        return 0, 0
    return int(cam.cam.Width.Max), int(cam.cam.Height.Max)


def _grab_with_converter(cam: BaslerUsbCamera, output_pixel: Optional[int]) -> np.ndarray:
    if not cam.cam:
        raise RuntimeError("Camera not initialized")
    if pylon is None:
        return cam.grab_bgr(timeout_ms=1000)

    converter = pylon.ImageFormatConverter()
    if output_pixel is not None:
        converter.OutputPixelFormat = output_pixel
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    res = cam.cam.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
    try:
        if not res.GrabSucceeded():
            raise RuntimeError(f"Grab failed: {res.ErrorCode} - {res.ErrorDescription}")
        img = converter.Convert(res)
        return img.GetArray()
    finally:
        res.Release()


def _resolve_output_pixel(name: str) -> Optional[int]:
    if pylon is None:
        return None
    return getattr(pylon, name, None)


def capture_dataset(
    cam: BaslerUsbCamera,
    base_cfg: CameraCfg,
    output_dir: str,
    count: int = 10,
) -> None:
    from pathlib import Path
    import time

    root = Path(__file__).resolve().parents[2] / "Dataset" / output_dir
    default_dir = root / "default_resolution"
    full_dir = root / "full_resolution"
    mono_dir = root / "full_resolution_mono"
    default_dir.mkdir(parents=True, exist_ok=True)
    full_dir.mkdir(parents=True, exist_ok=True)
    mono_dir.mkdir(parents=True, exist_ok=True)

    full_w, full_h = _get_max_resolution(cam)
    if full_w <= 0 or full_h <= 0:
        full_w, full_h = base_cfg.width, base_cfg.height

    capture_sets = [
        {
            "name": "default_resolution",
            "dir": default_dir,
            "width": base_cfg.width,
            "height": base_cfg.height,
            "pixel_candidates": ["RGB8", "BayerRG8", "BayerBG8", "BayerGR8", "BayerGB8"],
            "output_pixel": _resolve_output_pixel("PixelType_BGR8packed"),
            "to_gray": False,
        },
        {
            "name": "full_resolution",
            "dir": full_dir,
            "width": full_w,
            "height": full_h,
            "pixel_candidates": ["RGB12", "BayerRG12", "BayerBG12", "BayerGR12", "BayerGB12", "RGB8"],
            "output_pixel": _resolve_output_pixel("PixelType_BGR16"),
            "to_gray": False,
        },
        {
            "name": "full_resolution_mono",
            "dir": mono_dir,
            "width": full_w,
            "height": full_h,
            "pixel_candidates": ["Mono8", "Mono12", "Mono16"],
            "output_pixel": _resolve_output_pixel("PixelType_Mono8"),
            "to_gray": True,
        },
    ]

    print(f"[DATASET] Saving {count} frames per set into {root}")

    for spec in capture_sets:
        pixel_format = _select_pixel_format(cam, spec["pixel_candidates"])
        if pixel_format is None:
            print(f"[DATASET] Skip {spec['name']}: no supported pixel format from {spec['pixel_candidates']}")
            continue

        cam.stop()
        cfg = CameraCfg(
            width=spec["width"],
            height=spec["height"],
            offset_x=base_cfg.offset_x,
            offset_y=base_cfg.offset_y,
            pixel_format=pixel_format,
            exposure_us=base_cfg.exposure_us,
            gain=base_cfg.gain,
            fps=base_cfg.fps,
            latest_only=base_cfg.latest_only,
        )
        cam.configure(cfg)
        cam.start(latest_only=cfg.latest_only)

        time.sleep(0.4)

        for i in range(count):
            frame = _grab_with_converter(cam, spec["output_pixel"])
            if spec["to_gray"] and frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            out_path = spec["dir"] / f"frame_{i:03d}.png"
            cv2.imwrite(str(out_path), frame)

        print(f"[DATASET] {spec['name']} saved to {spec['dir']} (pixel_format={pixel_format})")


def main():
    # ====== CONFIG ======
    USE_BASLER = True
    VIDEO_SOURCE = "0"

    PIXEL_PITCH_UM = 2.0
    FOCAL_MM = 8.0
    WORKING_DISTANCE_MM = 1600.0
    DIST_REF_MM = 1000.0
    TOL_MM = 10.0

    params = IDetectParams(
        thresh_val=120,
        min_area=300,
        blur_ksize=5,
        use_otsu=False,
        morph_kernel=3,
        morph_open=1,
        morph_close=2,
        aspect_min=1.8,
        aspect_max=8.0,
        cap_ratio_min=1.2,
        fill_min=0.12,
        fill_max=0.65,
        vertical_coverage_min=0.75,
    )

    if USE_BASLER:
        cfg = CameraCfg(width=3840, height=2160, fps=35, latest_only=True)
        cam = BaslerUsbCamera().open()
        cam.configure(cfg)
        cam.start(latest_only=cfg.latest_only)
        cap = None
    else:
        cap = _open_video_source(VIDEO_SOURCE)
        cam = None

    win = "Basler RT (I Detect v2)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    pixel_pitch_mm = PIXEL_PITCH_UM / 1000.0
    if WORKING_DISTANCE_MM > FOCAL_MM:
        mm_per_pixel_est = pixel_pitch_mm * (WORKING_DISTANCE_MM - FOCAL_MM) / FOCAL_MM
    else:
        mm_per_pixel_est = 1.0

    MM_PER_PIXEL = load_scale(default=0.0)
    if MM_PER_PIXEL <= 0:
        MM_PER_PIXEL = mm_per_pixel_est
        save_scale(MM_PER_PIXEL)

    t_last = cv2.getTickCount()
    fps_smoothed: Optional[float] = None
    alpha = 0.2
    last_dist_px: Optional[float] = None

    try:
        while True:
            if cam is not None:
                frame = np.ascontiguousarray(cam.grab_bgr(timeout_ms=1000))
            else:
                ok, frame = cap.read()
                if not ok:
                    break

            now = cv2.getTickCount()
            dt = (now - t_last) / cv2.getTickFrequency()
            t_last = now
            fps_inst = (1.0 / dt) if dt > 0 else 0.0
            fps_smoothed = fps_inst if fps_smoothed is None else (alpha * fps_inst + (1 - alpha) * fps_smoothed)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            markers = detect_i_markers(gray, params)

            centers = []
            for (cx, cy), (x, y, w, h), score in markers:
                centers.append((cx, cy))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 180, 255), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                cv2.putText(
                    frame,
                    f"I {score:.2f}",
                    (x, max(0, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            distancia_px = None
            distancia_mm = None
            estado = "LIVE"
            color = (0, 255, 255)

            if len(centers) >= 2:
                c1, c2, distancia_px = pick_farthest_pair(centers)
                last_dist_px = distancia_px
                distancia_mm = distancia_px * MM_PER_PIXEL

                if abs(distancia_mm - DIST_REF_MM) <= TOL_MM:
                    estado = "NORMAL"
                    color = (0, 255, 0)
                else:
                    estado = "ALERTA"
                    color = (0, 0, 255)

                cv2.line(frame, c1, c2, color, 2)
                mid = ((c1[0] + c2[0]) // 2, (c1[1] + c2[1]) // 2)
                cv2.putText(
                    frame,
                    f"{distancia_mm:.1f} mm",
                    (mid[0] + 10, mid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            extra = f"I-detect v2 | mm/px:{MM_PER_PIXEL:.6f} (est:{mm_per_pixel_est:.6f})"
            draw_hud(
                frame,
                fps=fps_smoothed,
                estado=estado,
                distancia_mm=distancia_mm,
                distancia_px=distancia_px,
                nominal_mm=DIST_REF_MM,
                tol_mm=TOL_MM,
                extra=extra,
            )

            cv2.imshow(win, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                if last_dist_px and last_dist_px > 0:
                    MM_PER_PIXEL = DIST_REF_MM / last_dist_px
                    save_scale(MM_PER_PIXEL)
                    print(f"[CAL] mm/px = {MM_PER_PIXEL:.12f}")
                else:
                    print("[CAL] Need 2 I markers visible.")
            if key == ord("o"):
                params.use_otsu = not params.use_otsu
                print(f"[CFG] use_otsu = {params.use_otsu}")
            if key == ord("s") and cam is not None:
                capture_dataset(cam, cfg, "13-02-2026", count=10)

    finally:
        if cam is not None:
            cam.stop()
            cam.close()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
