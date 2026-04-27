"""ROI 裁剪与 mask 构造。"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ..config import RoiRefineConfig
from ..schemas import BoundingBox, DetectionObject, DetectionResult, RoiRefineResult
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
            detection_result.target,
            crop_box,
        )
        aligned_roi_image, target_mask = self._resize_bundle(
            original_roi_image,
            target_mask,
        )
        target_mask = (target_mask > 0).astype(np.uint8)
        valid_mask = self._build_valid_mask(target_mask)

        return RoiRefineResult(
            crop_box=crop_box,
            roi_image=original_roi_image,
            aligned_roi_image=aligned_roi_image,
            texture_ready_image=_apply_mask(aligned_roi_image, valid_mask),
            target_mask=target_mask,
            valid_mask=valid_mask,
            foreground_weight=None,
            alignment_applied=False,
        )

    def _build_target_mask(
        self,
        roi_image: Any,
        target: DetectionObject,
        crop_box: BoundingBox,
    ) -> np.ndarray:
        if target.segmentation_mask is not None:
            return _crop_mask(target.segmentation_mask, crop_box)
        # 没有分割 mask 时直接退回矩形 ROI，不再做 GrabCut 推断。
        return np.ones(roi_image.shape[:2], dtype=np.uint8)

    def _resize_bundle(
        self,
        roi_image: np.ndarray,
        target_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        output_width = max(1, int(self.config.alignment.output_width or roi_image.shape[1]))
        output_height = max(1, int(self.config.alignment.output_height or roi_image.shape[0]))
        output_size = (output_width, output_height)
        resized_roi = cv2.resize(roi_image, output_size, interpolation=cv2.INTER_AREA)
        resized_target = cv2.resize(target_mask, output_size, interpolation=cv2.INTER_NEAREST)
        return resized_roi, resized_target

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


def _apply_mask(image: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """把 mask 外区域清零，避免背景继续干扰 PatchCore。"""
    masked = image.copy()
    masked[valid_mask == 0] = 0
    return masked
