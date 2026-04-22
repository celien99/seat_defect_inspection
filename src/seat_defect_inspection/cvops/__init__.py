"""OpenCV 中间层能力入口。"""

from .debug_artifacts import (
    DEFAULT_DEBUG_ARTIFACT_MODE,
    resolve_debug_artifact_names,
    save_debug_artifacts,
)
from .quality import ImageQualityGuard
from .roi import RoiRefineEngine

__all__ = [
    "DEFAULT_DEBUG_ARTIFACT_MODE",
    "ImageQualityGuard",
    "RoiRefineEngine",
    "resolve_debug_artifact_names",
    "save_debug_artifacts",
]
