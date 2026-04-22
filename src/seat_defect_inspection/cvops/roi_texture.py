"""ROI 纹理增强、背景抑制与前景权重辅助。"""

from __future__ import annotations

import cv2
import numpy as np


def _safe_texture_mask(mask: np.ndarray, erode_kernel_size: int) -> np.ndarray:
    """给纹理分支收缩一圈安全边界，减少边缘污染。"""
    normalized = (mask > 0).astype(np.uint8)
    if erode_kernel_size <= 1:
        return normalized
    kernel = np.ones((erode_kernel_size, erode_kernel_size), dtype=np.uint8)
    eroded = cv2.erode(normalized, kernel, iterations=1)
    if eroded.sum() == 0:
        return normalized
    return eroded


def _build_foreground_weight(mask: np.ndarray, feather_kernel_size: int) -> np.ndarray:
    """把二值前景掩膜平滑成软权重图。"""
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
    """只在有效前景区域内做 CLAHE。"""
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
    """只在有效前景区域内做光照校正。"""
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
    """按配置对纹理图做去噪。"""
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
    """在前景区域内轻量增强纹理边缘。"""
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
    """把背景压平，减少对纹理分支的干扰。"""
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
    """构造背景填充参考图。"""
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
    """拉平亮度通道中的慢变化光照。"""
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
    """把核大小标准化为正奇数。"""
    normalized = max(1, int(value))
    return normalized if normalized % 2 == 1 else normalized + 1
