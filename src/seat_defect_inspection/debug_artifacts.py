"""Unified inspection artifact definitions."""

from __future__ import annotations

DEFAULT_DEBUG_ARTIFACT_NAMES: tuple[str, ...] = (
    "raw",
    "detections",
    "heatmap",
    "overlay",
)


def get_debug_artifact_names() -> set[str]:
    """Return the concise artifact set exported by every inspection run."""
    return set(DEFAULT_DEBUG_ARTIFACT_NAMES)
