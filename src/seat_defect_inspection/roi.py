"""ROI 精修、对齐与掩膜生成。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import RoiRefineConfig
from .schemas import BoundingBox, DetectionObject, DetectionResult, RoiRefineResult


class RoiRefineEngine:
    """把粗检测框变成可直接送入 PatchCore 的 ROI。"""

    def __init__(self, config: RoiRefineConfig) -> None:
        self.config = config
        self._template = self._load_template(config.alignment.template_image_path)

    def refine(self, image: Any, detection_result: DetectionResult) -> RoiRefineResult:
        """裁剪 ROI，并生成目标、忽略和有效区域掩膜。"""
        if detection_result.target is None:
            raise ValueError("ROI 精修必须提供目标检测框")

        crop_box = _expand_box(
            detection_result.target.bounding_box,
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

        return RoiRefineResult(
            crop_box=crop_box,
            roi_image=original_roi_image,
            aligned_roi_image=roi_image,
            target_mask=(target_mask > 0).astype(np.uint8),
            ignore_mask=(ignore_mask > 0).astype(np.uint8),
            valid_mask=valid_mask.astype(np.uint8),
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


def _expand_box(
    box: BoundingBox,
    image_shape: tuple[int, int],
    *,
    expand_ratio: float,
    shrink_ratio: float,
) -> BoundingBox:
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


def _box_to_ints(box: BoundingBox) -> tuple[int, int, int, int]:
    return (
        max(0, int(round(box.x1))),
        max(0, int(round(box.y1))),
        max(1, int(round(box.x2))),
        max(1, int(round(box.y2))),
    )


def _crop_shape(box: BoundingBox) -> tuple[int, int]:
    x1, y1, x2, y2 = _box_to_ints(box)
    return max(1, y2 - y1), max(1, x2 - x1)


def _crop_mask(mask: Any, crop_box: BoundingBox) -> np.ndarray:
    x1, y1, x2, y2 = _box_to_ints(crop_box)
    cropped = mask[y1:y2, x1:x2]
    if cropped.size == 0:
        height, width = _crop_shape(crop_box)
        return np.zeros((height, width), dtype=np.uint8)
    return (cropped > 0).astype(np.uint8)


def _grabcut_foreground(roi_image: Any) -> np.ndarray:
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
    normalized = (mask > 0).astype(np.uint8)
    if kernel_size <= 1:
        return normalized
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    if dilate:
        return cv2.dilate(normalized, kernel, iterations=1)
    cleaned = cv2.morphologyEx(normalized, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
