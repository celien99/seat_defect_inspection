"""ROI 几何、掩膜与基础裁剪辅助。"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import cv2
import numpy as np

from ..types import BoundingBox, DetectionObject


def _expand_box(
    box: BoundingBox,
    image_shape: Tuple[int, int],
    *,
    expand_ratio: float,
    shrink_ratio: float,
) -> BoundingBox:
    """按比例扩缩检测框，并限制在图像范围内。"""
    height, width = image_shape
    box_width = max(1.0, box.width)
    box_height = max(1.0, box.height)
    shrink_x = box_width * max(0.0, float(shrink_ratio))
    shrink_y = box_height * max(0.0, float(shrink_ratio))
    expand_x = box_width * max(0.0, float(expand_ratio))
    expand_y = box_height * max(0.0, float(expand_ratio))

    x1 = max(0.0, box.x1 + shrink_x - expand_x)
    y1 = max(0.0, box.y1 + shrink_y - expand_y)
    x2 = min(float(width), box.x2 - shrink_x + expand_x)
    y2 = min(float(height), box.y2 - shrink_y + expand_y)
    return BoundingBox(x1=x1, y1=y1, x2=max(x1 + 1.0, x2), y2=max(y1 + 1.0, y2))


def _resolve_crop_source_box(
    target: DetectionObject,
    image_shape: Tuple[int, int],
) -> BoundingBox:
    """优先使用分割掩膜外接框，否则回退到检测框。"""
    if target.segmentation_mask is not None:
        mask_box = _mask_to_box(target.segmentation_mask, image_shape)
        if mask_box is not None:
            return mask_box
    return target.bounding_box


def _mask_to_box(mask: Any, image_shape: Tuple[int, int]) -> Optional[BoundingBox]:
    """把掩膜转换成最小外接矩形。"""
    if mask is None:
        return None
    normalized = np.asarray(mask)
    if normalized.ndim != 2:
        return None

    height, width = image_shape
    if normalized.shape[:2] != (height, width):
        normalized = cv2.resize(
            normalized.astype(np.float32),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
    ys, xs = np.nonzero(normalized > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x1 = float(xs.min())
    y1 = float(ys.min())
    x2 = float(xs.max() + 1)
    y2 = float(ys.max() + 1)
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def _box_to_ints(box: BoundingBox) -> Tuple[int, int, int, int]:
    """把浮点框稳定转换成整数裁剪坐标。"""
    return (
        max(0, int(round(box.x1))),
        max(0, int(round(box.y1))),
        max(1, int(round(box.x2))),
        max(1, int(round(box.y2))),
    )


def _crop_shape(box: BoundingBox) -> Tuple[int, int]:
    """返回裁剪框对应的高宽。"""
    x1, y1, x2, y2 = _box_to_ints(box)
    return max(1, y2 - y1), max(1, x2 - x1)


def _crop_mask(mask: Any, crop_box: BoundingBox) -> np.ndarray:
    """按裁剪框裁出局部掩膜。"""
    x1, y1, x2, y2 = _box_to_ints(crop_box)
    cropped = mask[y1:y2, x1:x2]
    if cropped.size == 0:
        height, width = _crop_shape(crop_box)
        return np.zeros((height, width), dtype=np.uint8)
    return (cropped > 0).astype(np.uint8)
