"""Core inspect runtime for seat defect inspection."""

from .config import InspectionConfig
from .runtime_config import load_config
from .types import (
    CameraInspectionResult,
    FramePacket,
    InspectionFrame,
    InspectionResponse,
    InspectionResult,
)

__all__ = [
    "CameraInspectionResult",
    "ConfigSource",
    "FramePacket",
    "InspectionFrame",
    "InspectionConfig",
    "InspectionResponse",
    "InspectionResult",
    "SeatDefectInspector",
    "inspect_once",
    "load_config",
    "resolve_config",
]

_LAZY_EXPORTS = {
    "ConfigSource": (".api", "ConfigSource"),
    "SeatDefectInspector": (".api", "SeatDefectInspector"),
    "inspect_once": (".api", "inspect_once"),
    "resolve_config": (".api", "resolve_config"),
}


def __getattr__(name: str):
    """Lazily load heavy inspect runtime exports."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
