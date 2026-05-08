"""ROI 裁剪与 mask 构造。"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ..config import RoiRefineConfig
from ..types import BoundingBox, DetectionResult, RoiRefineResult
from .roi_geometry import (
    _box_to_ints,
    _crop_mask,
    _expand_box,
    _resolve_crop_source_box,
)


class RoiRefineEngine:
    """把 YOLO 分割结果整理成 PatchCore 可直接消费的 ROI。"""

    def __init__(self, config: RoiRefineConfig) -> None:
        self.config = config

    def refine(self, image: Any, detection_result: DetectionResult) -> RoiRefineResult:
        """只保留裁剪、缩放和 mask 处理，不再做 ROI 二次增强。"""
        if detection_result.target is None:
            raise ValueError("ROI 精修必须提供目标检测框")

        base_box = _resolve_crop_source_box(
            detection_result.target,
            image.shape[:2],
        )
        crop_box = _expand_box(
            base_box,
            image.shape[:2],
            expand_ratio=self.config.crop_expand_ratio,
            shrink_ratio=self.config.crop_shrink_ratio,
        )
        x1, y1, x2, y2 = _box_to_ints(crop_box)
        original_roi_image = image[y1:y2, x1:x2].copy()
        if original_roi_image.size == 0:
            raise ValueError("ROI 裁剪结果为空")

        target_mask = self._build_target_mask(
            original_roi_image,
            detection_result,
            crop_box,
        )
        aligned_roi_image, target_mask, alignment_applied = self._resize_bundle(
            original_roi_image,
            target_mask,
        )
        target_mask = (target_mask > 0).astype(np.uint8)
        valid_mask = self._build_valid_mask(target_mask)
        ignore_mask = self._build_ignore_mask(target_mask, valid_mask)

        return RoiRefineResult(
            crop_box=crop_box,
            roi_image=original_roi_image,
            aligned_roi_image=aligned_roi_image,
            texture_ready_image=_apply_mask(aligned_roi_image, target_mask),
            target_mask=target_mask,
            valid_mask=valid_mask,
            ignore_mask=ignore_mask,
            foreground_weight=None,
            alignment_applied=alignment_applied,
        )

    def _build_target_mask(
        self,
        roi_image: Any,
        detection_result: DetectionResult,
        crop_box: BoundingBox,
    ) -> np.ndarray:
        target = detection_result.target
        if target is None:
            raise ValueError("target_missing")
        if target.segmentation_mask is not None:
            cropped = _crop_mask(target.segmentation_mask, crop_box)
            if int(cropped.sum()) <= 0:
                raise ValueError("target_mask_empty")
            return cropped
        if not detection_result.used_fallback:
            raise ValueError("target_mask_missing")
        # 只有显式 fallback_box 路径才允许退回矩形 ROI，避免静默污染 PatchCore 输入。
        return np.ones(roi_image.shape[:2], dtype=np.uint8)

    def _resize_bundle(
        self,
        roi_image: np.ndarray,
        target_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        output_width = max(1, int(self.config.alignment.output_width or roi_image.shape[1]))
        output_height = max(1, int(self.config.alignment.output_height or roi_image.shape[0]))
        return _letterbox_bundle(
            roi_image,
            target_mask,
            output_width=output_width,
            output_height=output_height,
        )

    def _build_valid_mask(
        self,
        target_mask: np.ndarray,
    ) -> np.ndarray:
        valid_mask = (target_mask > 0).astype(np.uint8)

        edge_ignore = int(max(0, self.config.edge_ignore_pixels))
        if edge_ignore > 0:
            valid_mask[:edge_ignore, :] = 0
            valid_mask[-edge_ignore:, :] = 0
            valid_mask[:, :edge_ignore] = 0
            valid_mask[:, -edge_ignore:] = 0

        if valid_mask.sum() > 0:
            return valid_mask
        if target_mask.sum() > 0:
            return (target_mask > 0).astype(np.uint8)
        return np.ones_like(target_mask, dtype=np.uint8)

    def _build_ignore_mask(
        self,
        target_mask: np.ndarray,
        valid_mask: np.ndarray,
    ) -> np.ndarray:
        ignore_mask = np.zeros_like(target_mask, dtype=np.uint8)
        ignore_mask[np.logical_and(target_mask > 0, valid_mask == 0)] = 1
        return ignore_mask


def _apply_mask(image: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Build a BGRA PatchCore input whose mask background is transparent."""
    if image.ndim == 2:
        base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        base = image[:, :, :3].copy()
    else:
        base = image.copy()
    alpha = np.where(valid_mask > 0, 255, 0).astype(np.uint8)
    return np.dstack([base, alpha])


def _letterbox_bundle(
    roi_image: np.ndarray,
    target_mask: np.ndarray,
    *,
    output_width: int,
    output_height: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Preserve ROI aspect ratio when mapping to the canonical PatchCore canvas."""
    src_height, src_width = roi_image.shape[:2]
    scale = min(float(output_width) / float(src_width), float(output_height) / float(src_height))
    resized_width = max(1, int(round(src_width * scale)))
    resized_height = max(1, int(round(src_height * scale)))
    offset_x = max(0, (output_width - resized_width) // 2)
    offset_y = max(0, (output_height - resized_height) // 2)

    roi_interpolation = cv2.INTER_AREA if scale <= 1.0 else cv2.INTER_LINEAR
    resized_roi = cv2.resize(
        roi_image,
        (resized_width, resized_height),
        interpolation=roi_interpolation,
    )
    resized_target = cv2.resize(
        target_mask,
        (resized_width, resized_height),
        interpolation=cv2.INTER_NEAREST,
    )

    canvas = np.zeros((output_height, output_width, roi_image.shape[2]), dtype=roi_image.dtype)
    canvas_mask = np.zeros((output_height, output_width), dtype=np.uint8)
    canvas[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width] = resized_roi
    canvas_mask[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width] = (
        resized_target > 0
    ).astype(np.uint8)

    alignment_applied = (
        src_width != output_width
        or src_height != output_height
        or offset_x != 0
        or offset_y != 0
    )
    return canvas, canvas_mask, alignment_applied
