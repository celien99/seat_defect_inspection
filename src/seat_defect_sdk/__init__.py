"""外部系统集成用 SDK 包。"""

from .client import (
    CameraFrame,
    ConfigSource,
    InspectionSdkResponse,
    SeatDefectInspector,
    inspect_once,
)

__all__ = [
    "CameraFrame",
    "ConfigSource",
    "InspectionSdkResponse",
    "SeatDefectInspector",
    "inspect_once",
]
