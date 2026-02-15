from __future__ import annotations

import logging
from dataclasses import dataclass


@dataclass
class PlcCommand:
    color: str


class PlcClient:
    def set_light(self, color: str) -> None:
        raise NotImplementedError


class NoopPlcClient(PlcClient):
    def set_light(self, color: str) -> None:
        return


class LoggingPlcClient(PlcClient):
    def __init__(self) -> None:
        self._log = logging.getLogger("plc")

    def set_light(self, color: str) -> None:
        self._log.info("PLC light -> %s", color)


def build_plc_client(mode: str) -> PlcClient:
    if mode == "log":
        return LoggingPlcClient()
    return NoopPlcClient()
