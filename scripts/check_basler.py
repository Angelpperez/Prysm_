from __future__ import annotations

import os
import sys

from pypylon import pylon


def main() -> int:
    serial = os.getenv("PRYSM_CAMERA_SERIAL")
    tl = pylon.TlFactory.GetInstance()
    devices = tl.EnumerateDevices()
    print(f"Found devices: {len(devices)}")
    for dev in devices:
        print(f"- {dev.GetModelName()} SN={dev.GetSerialNumber()}")

    if not devices:
        return 2

    if serial:
        for dev in devices:
            if dev.GetSerialNumber() == serial:
                di = pylon.DeviceInfo()
                di.SetSerialNumber(serial)
                cam = pylon.InstantCamera(tl.CreateFirstDevice(di))
                break
        else:
            print(f"Serial {serial} not found")
            return 3
    else:
        cam = pylon.InstantCamera(tl.CreateFirstDevice())

    cam.Open()
    print("Camera opened ok")
    print("Model:", cam.GetDeviceInfo().GetModelName())
    print("Serial:", cam.GetDeviceInfo().GetSerialNumber())
    cam.Close()
    print("Camera closed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
