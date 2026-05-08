"""Core inspect runtime service internals."""

__all__ = [
    "InspectionService",
    "PreparedCameraSample",
    "inspect_frames",
    "inspect_one_camera",
]

_LAZY_EXPORTS = {
    "InspectionService": (".core", "InspectionService"),
    "PreparedCameraSample": (".core", "PreparedCameraSample"),
    "inspect_frames": (".inspection", "inspect_frames"),
    "inspect_one_camera": (".inspection_camera", "inspect_one_camera"),
}


def __getattr__(name: str):
    """Lazily load heavy service internals."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
