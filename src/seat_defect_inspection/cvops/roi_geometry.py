"""ROI 几何、掩膜与基础裁剪辅助。"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ..schemas import BoundingBox, DetectionObject


def _expand_box(
    box: BoundingBox,
    image_shape: tuple[int, int],
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
    image_shape: tuple[int, int],
) -> BoundingBox:
    """优先使用分割掩膜外接框，否则回退到检测框。"""
    if target.segmentation_mask is not None:
        mask_box = _mask_to_box(target.segmentation_mask, image_shape)
        if mask_box is not None:
            return mask_box
    return target.bounding_box


def _mask_to_box(mask: Any, image_shape: tuple[int, int]) -> BoundingBox | None:
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


def _box_to_ints(box: BoundingBox) -> tuple[int, int, int, int]:
    """把浮点框稳定转换成整型裁剪坐标。"""
    return (
        max(0, int(round(box.x1))),
        max(0, int(round(box.y1))),
        max(1, int(round(box.x2))),
        max(1, int(round(box.y2))),
    )


def _crop_shape(box: BoundingBox) -> tuple[int, int]:
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


def _grabcut_foreground(roi_image: Any) -> np.ndarray:
    """在没有分割掩膜时，用 GrabCut 估计前景。"""
    height, width = roi_image.shape[:2]
    margin_x = max(2, width // 20)
    margin_y = max(2, height // 20)
    rect = (
        margin_x,
        margin_y,
        max(1, width - 2 * margin_x),
        max(1, height - 2 * margin_y),
    )
    mask = np.zeros((height, width), dtype=np.uint8)
    bg_model = np.zeros((1, 65), dtype=np.float64)
    fg_model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(roi_image, mask, rect, bg_model, fg_model, 3, cv2.GC_INIT_WITH_RECT)
        return np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
            1,
            0,
        ).astype(np.uint8)
    except cv2.error:
        return np.ones((height, width), dtype=np.uint8)


def _clean_mask(mask: np.ndarray, kernel_size: int, dilate: bool = False) -> np.ndarray:
    """对目标或忽略掩膜做基础形态学清理。"""
    normalized = (mask > 0).astype(np.uint8)
    if kernel_size <= 1:
        return normalized
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    if dilate:
        return cv2.dilate(normalized, kernel, iterations=1)
    cleaned = cv2.morphologyEx(normalized, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
