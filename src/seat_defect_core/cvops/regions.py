"""标准 ROI 内的局部 PatchCore 区域切分。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import RegionConfig
from ..schemas import BoundingBox, RoiRefineResult
from ..util import select_patchcore_input


@dataclass(slots=True)
class RegionRoiSample:
    """单个局部区域送入 PatchCore 的图像和掩膜。"""

    region_id: str
    box: BoundingBox
    image: np.ndarray
    aligned_roi_image: np.ndarray
    target_mask: np.ndarray
    valid_mask: np.ndarray
    ignore_mask: np.ndarray


def split_roi_regions(
    roi: RoiRefineResult,
    regions: list[RegionConfig],
) -> list[RegionRoiSample]:
    """按配置从标准 ROI 中切出局部区域。"""
    samples: list[RegionRoiSample] = []
    for region in regions:
        if not region.enabled:
            continue
        sample = build_region_roi_sample(roi, region)
        if sample is not None:
            samples.append(sample)
    return samples


def build_region_roi_sample(
    roi: RoiRefineResult,
    region: RegionConfig,
) -> RegionRoiSample | None:
    """从一个标准 ROI 中切出一个局部 PatchCore 样本。"""
    patchcore_input = select_patchcore_input(roi)
    height, width = patchcore_input.shape[:2]
    x1, y1, x2, y2 = _normalized_box_to_pixels(region.box, width, height)
    if x2 <= x1 or y2 <= y1:
        return None

    image = patchcore_input[y1:y2, x1:x2].copy()
    aligned_roi_image = roi.aligned_roi_image[y1:y2, x1:x2].copy()
    target_mask = (roi.target_mask[y1:y2, x1:x2] > 0).astype(np.uint8)
    valid_mask = (roi.valid_mask[y1:y2, x1:x2] > 0).astype(np.uint8)
    ignore_mask = (roi.ignore_mask[y1:y2, x1:x2] > 0).astype(np.uint8)
    if image.size == 0 or int(valid_mask.sum()) <= 0:
        return None

    return RegionRoiSample(
        region_id=region.region_id,
        box=BoundingBox(
            x1=float(x1),
            y1=float(y1),
            x2=float(x2),
            y2=float(y2),
        ),
        image=image,
        aligned_roi_image=aligned_roi_image,
        target_mask=target_mask,
        valid_mask=valid_mask,
        ignore_mask=ignore_mask,
    )


def _normalized_box_to_pixels(
    box: list[float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1 = int(round(float(box[0]) * width))
    y1 = int(round(float(box[1]) * height))
    x2 = int(round(float(box[2]) * width))
    y2 = int(round(float(box[3]) * height))
    return (
        min(max(x1, 0), width),
        min(max(y1, 0), height),
        min(max(x2, 0), width),
        min(max(y2, 0), height),
    )
