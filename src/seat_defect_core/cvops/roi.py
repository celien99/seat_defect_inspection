"""ROI 裁剪与 mask 构造。"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import cv2
import numpy as np

from ..config import RoiRefineConfig
from ..types import BoundingBox, DetectionResult, RoiRefineResult
from .roi_geometry import (
    _box_to_ints,
    _crop_mask,
    _expand_box,
    _mask_to_box,
    _resolve_crop_source_box,
)


KEEP_LARGEST_COMPONENT_ONLY = True
MIN_COMPONENT_AREA_RATIO = 0.001
MIN_COMPONENT_AREA_PIXELS = 200


class RoiRefineEngine:
    """把 YOLO 分割结果整理成 PatchCore 可直接消费的 ROI。"""

    def __init__(self, config: RoiRefineConfig) -> None:
        self.config = config

    def refine(self, image: Any, detection_result: DetectionResult) -> RoiRefineResult:
        """只保留裁剪、缩放和 mask 处理，不再做 ROI 二次增强。"""
        if detection_result.target is None:
            raise ValueError("ROI 精修必须提供目标检测框")

        cleaned_target_mask = self._clean_target_segmentation_mask(
            detection_result.target.segmentation_mask,
            image.shape[:2],
        )
        base_box = _resolve_crop_source_box(
            detection_result.target,
            image.shape[:2],
        )
        mask_box = _mask_to_box(cleaned_target_mask, image.shape[:2])
        if mask_box is not None:
            base_box = mask_box
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
            cleaned_target_mask,
        )
        aligned_roi_image, target_mask, alignment_applied = self._resize_bundle(
            original_roi_image,
            target_mask,
        )
        target_mask = (target_mask > 0).astype(np.uint8)
        target_mask = self._erode_target_mask(target_mask)
        if int(target_mask.sum()) <= 0:
            raise ValueError("target_mask_empty_after_erode")
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
        cleaned_target_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        target = detection_result.target
        if target is None:
            raise ValueError("target_missing")
        source_mask = cleaned_target_mask if cleaned_target_mask is not None else target.segmentation_mask
        if source_mask is not None:
            cropped = _crop_mask(source_mask, crop_box)
            if int(cropped.sum()) <= 0:
                raise ValueError("target_mask_empty")
            return cropped
        raise ValueError("target_mask_missing")

    def _clean_target_segmentation_mask(
        self,
        mask: Any,
        image_shape: Tuple[int, int],
    ) -> np.ndarray:
        if mask is None:
            raise ValueError("target_mask_missing")
        normalized = np.asarray(mask)
        if normalized.ndim != 2:
            raise ValueError("target_mask_missing")

        height, width = image_shape
        if normalized.shape[:2] != (height, width):
            normalized = cv2.resize(
                normalized.astype(np.float32),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        binary = (normalized > 0).astype(np.uint8)
        if int(binary.sum()) <= 0:
            raise ValueError("target_mask_empty")

        cleaned = _filter_mask_components(
            binary,
            keep_largest_only=KEEP_LARGEST_COMPONENT_ONLY,
            min_area_ratio=MIN_COMPONENT_AREA_RATIO,
            min_area_pixels=MIN_COMPONENT_AREA_PIXELS,
        )
        if int(cleaned.sum()) <= 0:
            raise ValueError("target_mask_empty_after_component_filter")
        return cleaned

    def _erode_target_mask(self, target_mask: np.ndarray) -> np.ndarray:
        """按配置把 YOLO 前景 mask 向内收缩，避免边缘噪声进入 PatchCore。"""
        erode_pixels = int(max(0, self.config.mask_erode_pixels))
        if erode_pixels <= 0:
            return target_mask
        kernel_size = erode_pixels * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        return cv2.erode(target_mask, kernel, iterations=1).astype(np.uint8)

    def _resize_bundle(
        self,
        roi_image: np.ndarray,
        target_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
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


def _filter_mask_components(
    mask: np.ndarray,
    *,
    keep_largest_only: bool,
    min_area_ratio: float,
    min_area_pixels: int,
) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
    if component_count <= 1:
        return binary

    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int64)
    if areas.size == 0:
        return np.zeros_like(binary, dtype=np.uint8)

    min_area = max(
        int(max(0, min_area_pixels)),
        int(round(max(0.0, min_area_ratio) * float(binary.shape[0] * binary.shape[1]))),
    )
    keep_labels = [
        label
        for label, area in enumerate(areas, start=1)
        if int(area) >= min_area
    ]
    if not keep_labels:
        return np.zeros_like(binary, dtype=np.uint8)
    if keep_largest_only:
        keep_labels = [max(keep_labels, key=lambda label: int(stats[label, cv2.CC_STAT_AREA]))]

    return np.isin(labels, keep_labels).astype(np.uint8)


def _letterbox_bundle(
    roi_image: np.ndarray,
    target_mask: np.ndarray,
    *,
    output_width: int,
    output_height: int,
) -> Tuple[np.ndarray, np.ndarray, bool]:
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
