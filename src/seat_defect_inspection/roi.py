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


def _resolve_crop_source_box(
    target: DetectionObject,
    image_shape: tuple[int, int],
) -> BoundingBox:
    if target.segmentation_mask is not None:
        mask_box = _mask_to_box(target.segmentation_mask, image_shape)
        if mask_box is not None:
            return mask_box
    return target.bounding_box


def _mask_to_box(mask: Any, image_shape: tuple[int, int]) -> BoundingBox | None:
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


def _safe_texture_mask(mask: np.ndarray, erode_kernel_size: int) -> np.ndarray:
    normalized = (mask > 0).astype(np.uint8)
    if erode_kernel_size <= 1:
        return normalized
    kernel = np.ones((erode_kernel_size, erode_kernel_size), dtype=np.uint8)
    eroded = cv2.erode(normalized, kernel, iterations=1)
    if eroded.sum() == 0:
        return normalized
    return eroded


def _build_foreground_weight(mask: np.ndarray, feather_kernel_size: int) -> np.ndarray:
    normalized = (mask > 0).astype(np.float32)
    if normalized.sum() == 0:
        return normalized
    if feather_kernel_size <= 1:
        return normalized
    kernel = _odd_kernel(feather_kernel_size)
    weighted = cv2.GaussianBlur(normalized, (kernel, kernel), 0)
    weighted = np.maximum(weighted, normalized)
    return np.clip(weighted, 0.0, 1.0).astype(np.float32)


def _apply_masked_clahe(
    image: np.ndarray,
    valid_mask: np.ndarray,
    *,
    enabled: bool,
    clip_limit: float,
    tile_grid_size: int,
) -> np.ndarray:
    if not enabled or valid_mask.sum() == 0:
        return image

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(
            max(1, int(tile_grid_size)),
            max(1, int(tile_grid_size)),
        ),
    )
    enhanced_l = clahe.apply(l_channel)
    mask = valid_mask.astype(bool)
    l_channel = l_channel.copy()
    l_channel[mask] = enhanced_l[mask]
    return cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)


def _apply_masked_illumination_correction(
    image: np.ndarray,
    valid_mask: np.ndarray,
    *,
    enabled: bool,
    kernel_size: int,
    strength: float,
) -> np.ndarray:
    if not enabled or valid_mask.sum() == 0:
        return image

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    corrected_l = _flatten_illumination_channel(
        l_channel,
        kernel_size=kernel_size,
        strength=strength,
    )
    mask = valid_mask.astype(bool)
    l_channel = l_channel.copy()
    l_channel[mask] = corrected_l[mask]
    return cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)


def _denoise_texture_image(
    image: np.ndarray,
    *,
    method: str,
    gaussian_kernel_size: int,
    bilateral_diameter: int,
    bilateral_sigma_color: float,
    bilateral_sigma_space: float,
) -> np.ndarray:
    normalized = method.strip().lower()
    if normalized == "none":
        return image
    if normalized == "gaussian":
        kernel = _odd_kernel(gaussian_kernel_size)
        return cv2.GaussianBlur(image, (kernel, kernel), 0)
    return cv2.bilateralFilter(
        image,
        d=max(1, int(bilateral_diameter)),
        sigmaColor=float(bilateral_sigma_color),
        sigmaSpace=float(bilateral_sigma_space),
    )


def _enhance_texture_edges(
    image: np.ndarray,
    valid_mask: np.ndarray,
    *,
    foreground_weight: np.ndarray,
    method: str,
    weight: float,
) -> np.ndarray:
    normalized_weight = max(0.0, float(weight))
    if normalized_weight <= 0.0 or valid_mask.sum() == 0:
        return image

    normalized_method = method.strip().lower()
    if normalized_method == "none":
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if normalized_method == "laplacian":
        detail = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    else:
        grad_x = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
        grad_y = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
        detail = cv2.magnitude(grad_x, grad_y)

    mask = valid_mask.astype(bool)
    detail_scale = float(np.percentile(detail[mask], 95))
    if detail_scale <= 1e-6:
        return image

    detail = np.clip(detail / detail_scale, 0.0, 1.0)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    enhanced_l = l_channel.astype(np.float32) + (255.0 * normalized_weight * detail * foreground_weight)
    enhanced_l = np.clip(enhanced_l, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(cv2.merge((enhanced_l, a_channel, b_channel)), cv2.COLOR_LAB2BGR)


def _suppress_background(
    image: np.ndarray,
    valid_mask: np.ndarray,
    *,
    foreground_weight: np.ndarray,
    fill_mode: str,
    blur_kernel_size: int,
) -> np.ndarray:
    mask = valid_mask.astype(bool)
    if not np.any(mask):
        return image

    normalized_mode = fill_mode.strip().lower()
    if normalized_mode == "none":
        return image

    background = _build_background_reference(
        image,
        mask,
        fill_mode=normalized_mode,
        blur_kernel_size=blur_kernel_size,
    )
    alpha = np.clip(foreground_weight, 0.0, 1.0)[..., None]
    blended = image.astype(np.float32) * alpha + background.astype(np.float32) * (1.0 - alpha)
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)


def _build_background_reference(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    fill_mode: str,
    blur_kernel_size: int,
) -> np.ndarray:
    output = image.copy()
    if fill_mode == "blur":
        kernel = _odd_kernel(blur_kernel_size)
        return cv2.GaussianBlur(output, (kernel, kernel), 0)
    if fill_mode == "neutral_gray":
        neutral = np.empty_like(output)
        neutral[:] = np.array([127, 127, 127], dtype=np.uint8)
        return neutral

    foreground_pixels = output[mask]
    median_color = np.median(foreground_pixels, axis=0).astype(np.uint8)
    filled = np.empty_like(output)
    filled[:] = median_color
    return filled


def _flatten_illumination_channel(
    channel: np.ndarray,
    *,
    kernel_size: int,
    strength: float,
) -> np.ndarray:
    normalized_strength = float(np.clip(strength, 0.0, 1.0))
    if normalized_strength <= 0.0:
        return channel

    kernel = _odd_kernel(kernel_size)
    float_channel = channel.astype(np.float32)
    estimated_background = cv2.GaussianBlur(float_channel, (kernel, kernel), 0)
    estimated_background = np.maximum(estimated_background, 1.0)
    corrected = cv2.divide(
        float_channel,
        estimated_background,
        scale=float(estimated_background.mean()),
    )
    corrected = np.clip(corrected, 0.0, 255.0)
    blended = cv2.addWeighted(
        float_channel,
        1.0 - normalized_strength,
        corrected,
        normalized_strength,
        0.0,
    )
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)


def _odd_kernel(value: int) -> int:
    normalized = max(1, int(value))
    return normalized if normalized % 2 == 1 else normalized + 1
