"""检测运行时类型聚合导出。"""

from __future__ import annotations

from .geometry import BoundingBox
from .input import FramePacket, InspectionFrame
from .pipeline import (
    DetectionObject,
    DetectionResult,
    ImageQualityDecision,
    ImageQualityMetrics,
    RoiRefineResult,
)
from .results import (
    CameraInspectionResult,
    ColorAnomalyResult,
    InspectionError,
    InspectionResponse,
    InspectionResult,
    RegionPatchCoreResult,
    TextureAnomalyResult,
)

__all__ = [
    "BoundingBox",
    "CameraInspectionResult",
    "ColorAnomalyResult",
    "DetectionObject",
    "DetectionResult",
    "FramePacket",
    "ImageQualityDecision",
    "ImageQualityMetrics",
    "InspectionError",
    "InspectionFrame",
    "InspectionResponse",
    "InspectionResult",
    "RegionPatchCoreResult",
    "RoiRefineResult",
    "TextureAnomalyResult",
]
