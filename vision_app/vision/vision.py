from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from pypylon import pylon, genicam
from PIL import Image
import cv2
import time

@dataclass
class CameraCfg:
    width: int = 3840
    height: int = 2160
    offset_x: int = 0
    offset_y: int = 0
    pixel_format: Optional[str] = None   # e.g. "BayerRG8", "RGB8", etc. (depende de tu cámara)
    exposure_us: Optional[float] = None  # microsegundos (si tu nodo es ExposureTime en us)
    gain: Optional[float] = None         # puede ser Gain (float) o GainRaw (int) según SFNC
    fps: Optional[float] = None
    latest_only: bool = True             # baja latencia


class BaslerUsbCamera:
    def __init__(self, serial: Optional[str] = None):
        self.serial = serial
        self.cam: Optional[pylon.InstantCamera] = None
        self.converter = pylon.ImageFormatConverter()
        # Entrega BGR8 para OpenCV (si tu PixelFormat es Bayer, pylon debayeriza aquí)
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    def open(self) -> "BaslerUsbCamera":
        tl = pylon.TlFactory.GetInstance()
        if self.serial:
            di = pylon.DeviceInfo()
            di.SetSerialNumber(self.serial)
            self.cam = pylon.InstantCamera(tl.CreateFirstDevice(di))
        else:
            self.cam = pylon.InstantCamera(tl.CreateFirstDevice())

        self.cam.Open()
        return self

    def close(self) -> None:
        if not self.cam:
            return
        if self.cam.IsGrabbing():
            self.cam.StopGrabbing()
        self.cam.Close()
        self.cam = None

    # --------- helpers ---------
    @staticmethod
    def _clamp_and_align(value: int, vmin: int, vmax: int, inc: int) -> int:
        value = max(vmin, min(vmax, value))
        if inc > 0:
            value = vmin + ((value - vmin) // inc) * inc
        return value

    def _safe_set_enum(self, node, value: str) -> None:
        if not genicam.IsWritable(node):
            raise RuntimeError(f"Node '{node.GetName()}' no es escribible ahora.")
        node.SetValue(value)

    # --------- config ---------
    def configure(self, cfg: CameraCfg) -> None:
        if not self.cam:
            raise RuntimeError("Cámara no abierta. Llama open().")

        cam = self.cam
        if cam.IsGrabbing():
            cam.StopGrabbing()

        # PixelFormat: lista soportada rápida
        if cfg.pixel_format:
            supported = list(cam.PixelFormat.Symbolics)  # útil para no adivinar
            if cfg.pixel_format not in supported:
                raise ValueError(f"PixelFormat '{cfg.pixel_format}' no soportado. Opciones: {supported}")
            self._safe_set_enum(cam.PixelFormat, cfg.pixel_format)

        # ROI (ojo con orden recomendado: Offset -> Width/Height o viceversa según modelo;
        # aquí lo hacemos con alineación y clamps)
        ox = self._clamp_and_align(cfg.offset_x, cam.OffsetX.Min, cam.OffsetX.Max, cam.OffsetX.GetInc())
        oy = self._clamp_and_align(cfg.offset_y, cam.OffsetY.Min, cam.OffsetY.Max, cam.OffsetY.GetInc())
        cam.OffsetX.Value = ox
        cam.OffsetY.Value = oy

        w = self._clamp_and_align(cfg.width, cam.Width.Min, cam.Width.Max, cam.Width.GetInc())
        h = self._clamp_and_align(cfg.height, cam.Height.Min, cam.Height.Max, cam.Height.GetInc())
        cam.Width.Value = w
        cam.Height.Value = h

        # Exposure / Gain (los nombres pueden variar por SFNC/cámara)
        if cfg.exposure_us is not None and hasattr(cam, "ExposureTime"):
            if hasattr(cam, "ExposureAuto"):
                cam.ExposureAuto.SetValue("Off")
            cam.ExposureTime.Value = float(cfg.exposure_us)

        if cfg.gain is not None:
            if hasattr(cam, "Gain"):
                if hasattr(cam, "GainAuto"):
                    cam.GainAuto.SetValue("Off")
                cam.Gain.Value = float(cfg.gain)
            elif hasattr(cam, "GainRaw"):
                cam.GainRaw.Value = int(cfg.gain)

        # FPS (si está disponible)
        if cfg.fps is not None and hasattr(cam, "AcquisitionFrameRateEnable"):
            cam.AcquisitionFrameRateEnable.SetValue(True)
            cam.AcquisitionFrameRate.Value = float(cfg.fps)

    # --------- grabbing ---------
    def start(self, latest_only: bool = True) -> None:
        if not self.cam:
            raise RuntimeError("Cámara no abierta.")
        strategy = pylon.GrabStrategy_LatestImageOnly if latest_only else pylon.GrabStrategy_OneByOne
        self.cam.StartGrabbing(strategy)


    def grab_bgr(self, timeout_ms: int = 1000) -> np.ndarray:
        if not self.cam or not self.cam.IsGrabbing():
            raise RuntimeError("No está grabando. Llama start().")

        res = self.cam.RetrieveResult(timeout_ms, pylon.TimeoutHandling_ThrowException)
        try:
            if not res.GrabSucceeded():
                raise RuntimeError(f"Grab falló: {res.ErrorCode} - {res.ErrorDescription}")
            img = self.converter.Convert(res)
            return img.GetArray()  # numpy array (H,W,3) BGR8
        finally:
            res.Release()

    def stop(self) -> None:
        if self.cam and self.cam.IsGrabbing():
            self.cam.StopGrabbing()

    # --------- “leer registros” (GenICam nodemap) ---------
    def get_feature_as_string(self, feature_name: str) -> str:
        """
        Lee cualquier feature desde el nodemap como string (útil para debug/config).
        La idea viene de GenApi: cualquier parámetro se puede leer/setear como string. :contentReference[oaicite:8]{index=8}
        """
        if not self.cam:
            raise RuntimeError("Cámara no abierta.")
        node = self.cam.GetNodeMap().GetNode(feature_name)
        if node is None:
            raise KeyError(f"No existe feature '{feature_name}' en el nodemap.")
        return node.ToString()

    def set_feature_from_string(self, feature_name: str, value: str) -> None:
        if not self.cam:
            raise RuntimeError("Cámara no abierta.")
        node = self.cam.GetNodeMap().GetNode(feature_name)
        if node is None:
            raise KeyError(f"No existe feature '{feature_name}' en el nodemap.")
        if not genicam.IsWritable(node):
            raise RuntimeError(f"Feature '{feature_name}' no es escribible ahora.")
        node.FromString(value)


import cv2
import numpy as np
import time
import math

# ---------- Overlay HUD ----------
def draw_hud(frame, fps=None, estado="LIVE", distancia_mm=None, distancia_px=None,
             nominal_mm=None, tol_mm=None, extra=""):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    box_h = 90
    cv2.rectangle(overlay, (0, 0), (w, box_h), (0, 0, 0), -1)
    alpha = 0.55
    frame[:box_h, :] = cv2.addWeighted(overlay[:box_h, :], alpha, frame[:box_h, :], 1 - alpha, 0)

    if estado == "NORMAL":
        color = (0, 255, 0)
    elif estado == "ALERTA":
        color = (0, 0, 255)
    else:
        color = (0, 255, 255)

    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line1 = f"TS: {ts} | Estado: {estado} | Res: {w}x{h}"
    line2 = f"FPS: {fps:.1f}" if fps is not None else "FPS: ..."

    if distancia_mm is not None and distancia_px is not None:
        line2 += f" | Dist: {distancia_mm:.1f} mm ({distancia_px:.1f} px)"
    if nominal_mm is not None and tol_mm is not None:
        line2 += f" | Nom: {nominal_mm}±{tol_mm} mm"
    if extra:
        line2 += f" | {extra}"

    cv2.putText(frame, line1, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    cv2.putText(frame, line2, (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)


# ---------- Detección de marcas ----------
def calcular_centroide(contorno):
    M = cv2.moments(contorno)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)
    return None


def main():
    """
    43.3 fps
    45.7 fps (Device Link Throughput Limit mode set to Off)
    24.3 fps (with triggering via Frame Start trigger)
    """

    # ----- Parámetros de medición -----
    MM_PER_PIXEL = 1.0
    DISTANCIA_NOMINAL_MM = 1000.0
    TOL_MM = 10.0

    cfg = CameraCfg(width=3840, height=2160, fps=43, latest_only=True)

    cam = BaslerUsbCamera().open()
    cam.configure(cfg)
    cam.start(latest_only=cfg.latest_only)

    win = "Basler RT"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # FPS suavizado (más estable)
    t_last = time.time()
    fps_smoothed = None
    alpha = 0.2
    extra = "LatestOnly" if cfg.latest_only else "OneByOne"

    # Umbral (si no detecta, este es el primer dial a ajustar)
    THRESH_VAL = 110
    MIN_AREA = 100

    try:
        while True:
            frame = cam.grab_bgr(timeout_ms=1000)
            frame = np.ascontiguousarray(frame)

            # --- FPS ---
            now = time.time()
            dt = now - t_last
            t_last = now
            fps_inst = (1.0 / dt) if dt > 0 else 0.0
            fps_smoothed = fps_inst if fps_smoothed is None else (alpha * fps_inst + (1 - alpha) * fps_smoothed)

            # --- Pre-proceso ---
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Opción A: umbral fijo (rápido)
            _, thresh = cv2.threshold(gray, THRESH_VAL, 255, cv2.THRESH_BINARY)

            # Opción B (mejor si la iluminación cambia): Otsu
            # _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            centros = []
            # Dibuja boxes + centroides
            for cnt in contornos:
                area = cv2.contourArea(cnt)
                if area > MIN_AREA:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                    centro = calcular_centroide(cnt)
                    if centro:
                        centros.append(centro)
                        cv2.circle(frame, centro, 6, (0, 255, 0), -1)

            # Para consistencia, ordena por X (izq->der)
            centros.sort(key=lambda c: c[0])

            distancia_px = None
            distancia_mm = None
            estado = "LIVE"
            color = (0, 255, 255)

            if len(centros) >= 2:
                c1, c2 = centros[0], centros[1]
                distancia_px = math.dist(c1, c2)
                distancia_mm = distancia_px * MM_PER_PIXEL

                if abs(distancia_mm - DISTANCIA_NOMINAL_MM) <= TOL_MM:
                    estado = "NORMAL"
                    color = (0, 255, 0)
                else:
                    estado = "ALERTA"
                    color = (0, 0, 255)

                # Línea y etiqueta (overlay “encima”)
                cv2.line(frame, c1, c2, color, 3)
                mid = ((c1[0] + c2[0]) // 2, (c1[1] + c2[1]) // 2)
                cv2.putText(frame, f"{distancia_mm:.1f} mm", (mid[0] + 10, mid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

            # HUD primero o después:
            # - Si lo dibujas al final, puede “oscurecer” overlays en la franja superior.
            # - Aquí lo dibujamos al final, pero SOLO afecta a la franja superior.
            draw_hud(frame, fps=fps_smoothed, estado=estado,
                     distancia_mm=distancia_mm, distancia_px=distancia_px,
                     nominal_mm=DISTANCIA_NOMINAL_MM, tol_mm=TOL_MM,
                     extra=extra)

            cv2.imshow(win, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            # Debug rápido: presiona 'd' para ver el binario y validar detección
            if key == ord('d'):
                cv2.imshow("thresh_debug", thresh)

    finally:
        cam.stop()
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
