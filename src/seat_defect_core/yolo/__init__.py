"""YOLO detection entrypoint for core runtime."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .detection import DetectionService

__all__ = [
    "DetectionService",
]

_LAZY_EXPORTS = {
    "DetectionService": (".detection", "DetectionService"),
}


def __getattr__(name: str):
    """Import YOLO detection lazily."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
