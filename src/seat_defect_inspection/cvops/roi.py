"""ROI 精修、对齐与掩膜生成。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..config import RoiRefineConfig
from ..schemas import BoundingBox, DetectionObject, DetectionResult, RoiRefineResult
from .roi_geometry import (
    _box_to_ints,
    _clean_mask,
    _crop_mask,
    _crop_shape,
    _expand_box,
    _grabcut_foreground,
    _resolve_crop_source_box,
)
from .roi_texture import (
    _apply_masked_clahe,
    _apply_masked_illumination_correction,
    _build_foreground_weight,
    _denoise_texture_image,
    _enhance_texture_edges,
    _safe_texture_mask,
    _suppress_background,
)


class RoiRefineEngine:
    """把粗检测框变成可直接送入 PatchCore 的 ROI。"""

    def __init__(self, config: RoiRefineConfig) -> None:
        self.config = config
        self._template = self._load_template(config.alignment.template_image_path)

    def refine(self, image: Any, detection_result: DetectionResult) -> RoiRefineResult:
        """裁剪 ROI,并生成目标、忽略和有效区域掩膜。"""
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
        roi_image = image[y1:y2, x1:x2].copy()
        if roi_image.size == 0:
            raise ValueError("ROI 裁剪结果为空")
        original_roi_image = roi_image.copy()

        target_mask = self._build_target_mask(roi_image, detection_result.target, crop_box)
        ignore_mask = self._build_ignore_mask(image.shape[:2], detection_result.ignores, crop_box)

        roi_image, target_mask, ignore_mask, aligned = self._align(roi_image, target_mask, ignore_mask)
        target_mask = _clean_mask(target_mask, self.config.morphology_kernel_size)
        ignore_mask = _clean_mask(ignore_mask, self.config.ignore_dilate_kernel_size, dilate=True)
        valid_mask = np.logical_and(target_mask > 0, ignore_mask == 0).astype(np.uint8)

        edge_ignore = int(max(0, self.config.edge_ignore_pixels))
        if edge_ignore > 0:
            valid_mask[:edge_ignore, :] = 0
            valid_mask[-edge_ignore:, :] = 0
            valid_mask[:, :edge_ignore] = 0
            valid_mask[:, -edge_ignore:] = 0

        if valid_mask.sum() == 0:
            valid_mask = (target_mask > 0).astype(np.uint8)

        texture_ready_image, texture_valid_mask, foreground_weight = self._prepare_texture_image(
            roi_image,
            valid_mask.astype(np.uint8),
        )

        return RoiRefineResult(
            crop_box=crop_box,
            roi_image=original_roi_image,
            aligned_roi_image=roi_image,
            texture_ready_image=texture_ready_image,
            target_mask=(target_mask > 0).astype(np.uint8),
            ignore_mask=(ignore_mask > 0).astype(np.uint8),
            valid_mask=texture_valid_mask.astype(np.uint8),
            foreground_weight=foreground_weight,
            alignment_applied=aligned,
        )

    def _build_target_mask(
        self,
        roi_image: Any,
        target: DetectionObject,
        crop_box: BoundingBox,
    ) -> np.ndarray:
        if target.segmentation_mask is not None:
            return _crop_mask(target.segmentation_mask, crop_box)

        mode = self.config.mask_mode.strip().lower()
        if mode == "full":
            return np.ones(roi_image.shape[:2], dtype=np.uint8)

        return _grabcut_foreground(roi_image)

    def _build_ignore_mask(
        self,
        image_shape: tuple[int, int],
        ignores: list[DetectionObject],
        crop_box: BoundingBox,
    ) -> np.ndarray:
        height, width = _crop_shape(crop_box)
        mask = np.zeros((height, width), dtype=np.uint8)
        crop_x1, crop_y1, _, _ = _box_to_ints(crop_box)

        for detection in ignores:
            if detection.segmentation_mask is not None:
                cropped = _crop_mask(detection.segmentation_mask, crop_box)
                mask = np.maximum(mask, cropped)
                continue

            x1 = max(0, int(round(detection.bounding_box.x1)) - crop_x1)
            y1 = max(0, int(round(detection.bounding_box.y1)) - crop_y1)
            x2 = min(width, int(round(detection.bounding_box.x2)) - crop_x1)
            y2 = min(height, int(round(detection.bounding_box.y2)) - crop_y1)
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 1
        return mask

    def _align(
        self,
        roi_image: Any,
        target_mask: np.ndarray,
        ignore_mask: np.ndarray,
    ) -> tuple[Any, np.ndarray, np.ndarray, bool]:
        output_size = (
            int(self.config.alignment.output_width),
            int(self.config.alignment.output_height),
        )
        resized_roi = cv2.resize(roi_image, output_size, interpolation=cv2.INTER_AREA)
        resized_target = cv2.resize(target_mask, output_size, interpolation=cv2.INTER_NEAREST)
        resized_ignore = cv2.resize(ignore_mask, output_size, interpolation=cv2.INTER_NEAREST)

        if not self.config.alignment.enabled or self.config.alignment.method.lower() != "ecc" or self._template is None:
            return resized_roi, resized_target, resized_ignore, False

        template = cv2.resize(self._template, output_size, interpolation=cv2.INTER_AREA)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        roi_gray = cv2.cvtColor(resized_roi, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            max(1, int(self.config.alignment.ecc_iterations)),
            1e-5,
        )
        try:
            cv2.findTransformECC(
                template_gray,
                roi_gray,
                warp_matrix,
                cv2.MOTION_EUCLIDEAN,
                criteria,
            )
        except cv2.error:
            return resized_roi, resized_target, resized_ignore, False

        aligned_roi = cv2.warpAffine(
            resized_roi,
            warp_matrix,
            output_size,
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )
        aligned_target = cv2.warpAffine(
            resized_target,
            warp_matrix,
            output_size,
            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        aligned_ignore = cv2.warpAffine(
            resized_ignore,
            warp_matrix,
            output_size,
            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return aligned_roi, aligned_target, aligned_ignore, True

    def _load_template(self, template_image_path: str | None) -> Any | None:
        if not template_image_path:
            return None
        template = cv2.imread(str(Path(template_image_path)))
        return template

    def _prepare_texture_image(
        self,
        roi_image: Any,
        valid_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        prepared = roi_image.copy()
        safe_mask = _safe_texture_mask(
            valid_mask,
            self.config.safe_margin_erode_kernel_size,
        )
        foreground_weight = _build_foreground_weight(
            safe_mask,
            self.config.mask_feather_kernel_size,
        )
        prepared = _apply_masked_clahe(
            prepared,
            safe_mask,
            enabled=self.config.apply_texture_clahe,
            clip_limit=self.config.texture_clahe_clip_limit,
            tile_grid_size=self.config.texture_clahe_tile_grid_size,
        )
        prepared = _apply_masked_illumination_correction(
            prepared,
            safe_mask,
            enabled=self.config.texture_illumination_correction,
            kernel_size=self.config.texture_illumination_blur_kernel_size,
            strength=self.config.texture_illumination_strength,
        )
        prepared = _denoise_texture_image(
            prepared,
            method=self.config.texture_denoise_method,
            gaussian_kernel_size=self.config.texture_gaussian_kernel_size,
            bilateral_diameter=self.config.texture_bilateral_diameter,
            bilateral_sigma_color=self.config.texture_bilateral_sigma_color,
            bilateral_sigma_space=self.config.texture_bilateral_sigma_space,
        )
        prepared = _enhance_texture_edges(
            prepared,
            safe_mask,
            foreground_weight=foreground_weight,
            method=self.config.edge_enhance_method,
            weight=self.config.edge_enhance_weight,
        )
        if self.config.suppress_background:
            prepared = _suppress_background(
                prepared,
                safe_mask,
                foreground_weight=foreground_weight,
                fill_mode=self.config.background_fill_mode,
                blur_kernel_size=self.config.background_blur_kernel_size,
            )
        return prepared, safe_mask, foreground_weight
