from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from pypylon import pylon, genicam
from PIL import Image
import cv2
import time
import math
import os

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




SCALE_FILE = "mm_per_pixel.txt"

def load_scale(default=0.0):
    if os.path.exists(SCALE_FILE):
        try:
            with open(SCALE_FILE, "r") as f:
                return float(f.read().strip())
        except Exception:
            pass
    return default

def save_scale(mm_per_pixel: float):
    with open(SCALE_FILE, "w") as f:
        f.write(f"{mm_per_pixel:.12f}")

def pick_farthest_pair(points):
    best = None
    best_d = -1.0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            d = math.dist(points[i], points[j])
            if d > best_d:
                best_d = d
                best = (points[i], points[j], d)
    return best  # (p1, p2, dist_px)

def main():
    """
    Teclas:
      q / ESC : salir
      d       : ver thresh_debug
      c       : calibrar usando DIST_REF_MM (ej. 1000 mm) con las marcas visibles
      r       : reset a estimación óptica (o a 1.0 si WD inválido)
    """

    # ========= CONFIG MANUAL (EDITA AQUÍ) =========
    PIXEL_PITCH_UM = 2.0     # Pixel Size (H x V) = 2.0 µm
    FOCAL_MM = 8.0           # Computar M0814-MP2: 8 mm
    WORKING_DISTANCE_MM = 1600.0  # <-- Distancia lente->plano del cable (mm) MEDIDA A MANO
    DIST_REF_MM = 1000.0     # referencia física para calibrar (1 metro)
    TOL_MM = 10.0

    # ========= CÁMARA =========
    cfg = CameraCfg(width=3840, height=2160, fps=43, latest_only=True)

    cam = BaslerUsbCamera().open()
    cam.configure(cfg)
    cam.start(latest_only=cfg.latest_only)

    win = "Basler RT"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # ========= ESTIMACIÓN ÓPTICA (mm/px) =========
    pixel_pitch_mm = PIXEL_PITCH_UM / 1000.0  # µm -> mm
    if WORKING_DISTANCE_MM > FOCAL_MM:
        mm_per_pixel_est = pixel_pitch_mm * (WORKING_DISTANCE_MM - FOCAL_MM) / FOCAL_MM
    else:
        mm_per_pixel_est = 1.0  # fallback si WD inválido

    # Si hay una escala guardada, úsala; si no, usa la estimación óptica
    MM_PER_PIXEL = load_scale(default=0.0)
    if MM_PER_PIXEL <= 0:
        MM_PER_PIXEL = mm_per_pixel_est
        save_scale(MM_PER_PIXEL)

    # ========= DEBUG / DETECCIÓN =========
    THRESH_VAL = 110
    MIN_AREA = 100

    # FPS suavizado
    t_last = time.time()
    fps_smoothed = None
    alpha = 0.2
    extra = "LatestOnly" if cfg.latest_only else "OneByOne"

    last_dist_px = None  # para calibración con tecla 'c'

    try:
        while True:
            frame = np.ascontiguousarray(cam.grab_bgr(timeout_ms=1000))

            # FPS
            now = time.time()
            dt = now - t_last
            t_last = now
            fps_inst = (1.0 / dt) if dt > 0 else 0.0
            fps_smoothed = fps_inst if fps_smoothed is None else (alpha * fps_inst + (1 - alpha) * fps_smoothed)

            # Threshold
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, THRESH_VAL, 255, cv2.THRESH_BINARY)

            contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            centros = []
            for cnt in contornos:
                area = cv2.contourArea(cnt)
                if area > MIN_AREA:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                    centro = calcular_centroide(cnt)
                    if centro:
                        centros.append(centro)
                        cv2.circle(frame, centro, 6, (0, 255, 0), -1)

            distancia_px = None
            distancia_mm = None
            estado = "LIVE"
            color = (0, 255, 255)

            if len(centros) >= 2:
                # Usa el par más separado para evitar agarrar dos falsos positivos
                c1, c2, distancia_px = pick_farthest_pair(centros)
                last_dist_px = distancia_px

                distancia_mm = distancia_px * MM_PER_PIXEL

                if abs(distancia_mm - DIST_REF_MM) <= TOL_MM:
                    estado = "NORMAL"
                    color = (0, 255, 0)
                else:
                    estado = "ALERTA"
                    color = (0, 0, 255)

                cv2.line(frame, c1, c2, color, 3)
                mid = ((c1[0] + c2[0]) // 2, (c1[1] + c2[1]) // 2)
                cv2.putText(frame, f"{distancia_mm:.1f} mm", (mid[0] + 10, mid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

            # HUD (reutiliza tu draw_hud)
            extra2 = (f"{extra} | mm/px:{MM_PER_PIXEL:.6f} (est:{mm_per_pixel_est:.6f}) "
                      f"| WD:{WORKING_DISTANCE_MM:.0f}mm f:{FOCAL_MM:.1f}mm p:{PIXEL_PITCH_UM:.1f}um | c=cal")
            draw_hud(frame, fps=fps_smoothed, estado=estado,
                     distancia_mm=distancia_mm, distancia_px=distancia_px,
                     nominal_mm=DIST_REF_MM, tol_mm=TOL_MM,
                     extra=extra2)

            cv2.imshow(win, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('d'):
                cv2.imshow("thresh_debug", thresh)

            # Calibrar con referencia física (1 m) cuando tengas marcas bien detectadas
            if key == ord('c'):
                if last_dist_px and last_dist_px > 0:
                    MM_PER_PIXEL = DIST_REF_MM / last_dist_px
                    save_scale(MM_PER_PIXEL)
                    print(f"[CALIBRADO] mm/px = {MM_PER_PIXEL:.12f} (guardado en {SCALE_FILE})")
                else:
                    print("[CALIBRADO] No hay 2 marcas confiables en pantalla.")

            # Reset a estimación óptica (o fallback)
            if key == ord('r'):
                MM_PER_PIXEL = mm_per_pixel_est if mm_per_pixel_est > 0 else 1.0
                save_scale(MM_PER_PIXEL)
                print(f"[RESET] mm/px = {MM_PER_PIXEL:.12f} (est óptica)")

    finally:
        cam.stop()
        cam.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

