from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from pypylon import pylon, genicam


@dataclass
class CameraCfg:
  
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
    def start(self) -> None:
        if not self.cam:
            raise RuntimeError("Cámara no abierta.")
        strategy = pylon.GrabStrategy_LatestImageOnly if True else pylon.GrabStrategy_OneByOne
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

