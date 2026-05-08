"""OpenCV 中间层能力入口。"""

from .debug_artifacts import (
    save_debug_artifacts,
)
from .quality import ImageQualityGuard
from .regions import RegionRoiSample, build_region_roi_sample, split_roi_regions
from .roi import RoiRefineEngine

__all__ = [
    "ImageQualityGuard",
    "RegionRoiSample",
    "RoiRefineEngine",
    "build_region_roi_sample",
    "save_debug_artifacts",
    "split_roi_regions",
]
