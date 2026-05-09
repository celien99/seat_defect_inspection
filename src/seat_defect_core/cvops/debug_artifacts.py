"""Backward-compatible imports for debug artifact helpers."""

from ..artifacts.debug import (
    DEFAULT_DEBUG_ARTIFACT_NAMES,
    _overlay_heatmap,
    _render_detections,
    _render_heatmap,
    save_debug_artifacts,
)

__all__ = [
    "DEFAULT_DEBUG_ARTIFACT_NAMES",
    "_overlay_heatmap",
    "_render_detections",
    "_render_heatmap",
    "save_debug_artifacts",
]
