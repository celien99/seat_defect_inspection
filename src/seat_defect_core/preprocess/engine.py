"""OpenCV preprocessing before YOLO detection."""

from __future__ import annotations

import cv2
import numpy as np

from ..config import PreprocessConfig


class PreprocessEngine:
    """Apply distortion correction, denoising, normalization, and sharpening."""

    def __init__(self, config: PreprocessConfig) -> None:
        self.config = config
        self._camera_matrix = (
            np.asarray(config.camera_matrix, dtype=np.float32)
            if config.camera_matrix
            else None
        )
        self._distortion_coeffs = (
            np.asarray(config.distortion_coeffs, dtype=np.float32)
            if config.distortion_coeffs
            else None
        )

    def process(self, image):
        """Return a stabilized BGR image for the downstream pipeline."""
        processed = image.copy()
        # 畸变校正
        if self._camera_matrix is not None and self._distortion_coeffs is not None:
            processed = cv2.undistort(processed, self._camera_matrix, self._distortion_coeffs)

        # 缩放
        if self.config.resize_width and self.config.resize_height:
            processed = cv2.resize(
                processed,
                (int(self.config.resize_width), int(self.config.resize_height)),
                interpolation=cv2.INTER_AREA,
            )

        # 去噪
        processed = self._denoise(processed)
        # 白平衡
        processed = self._white_balance(processed)
        # 光照归一化
        processed = self._normalize_lighting(processed)
        # 锐化
        processed = self._sharpen(processed)
        return processed

    def _sharpen(self, image):
        if not self.config.sharpen:
            return image
        return _apply_unsharp_mask(
            image,
            sigma=float(self.config.sharpen_sigma),
            amount=float(self.config.sharpen_amount),
        )
    def _denoise(self, image):
        method = self.config.denoise_method.strip().lower()
        if method == "none":
            return image
        if method == "bilateral":
            d = int(self.config.bilateral_diameter)
            return cv2.bilateralFilter(
                image,
                d=d,
                sigmaColor=float(self.config.bilateral_sigma_color),
                sigmaSpace=float(self.config.bilateral_sigma_space),
            )
        kernel_size = _odd_kernel(self.config.gaussian_kernel_size)
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def _white_balance(self, image):
        method = self.config.white_balance_method.strip().lower()
        if method == "none":
            return image
        if method != "gray_world":
            return image

        float_image = image.astype(np.float32)
        channel_means = float_image.reshape(-1, 3).mean(axis=0)
        gray_mean = float(channel_means.mean())
        max_gain = max(1.0, float(self.config.max_white_balance_gain))
        gains = gray_mean / np.maximum(channel_means, 1e-6)
        gains = np.clip(gains, 1.0 / max_gain, max_gain)
        balanced = float_image * gains.reshape(1, 1, 3)
        return np.clip(balanced, 0.0, 255.0).astype(np.uint8)

    def _normalize_lighting(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        if self.config.apply_illumination_correction:
            l_channel = _flatten_illumination_channel(
                l_channel,
                kernel_size=int(self.config.illumination_blur_kernel_size),
                strength=float(self.config.illumination_strength),
            )

        if self.config.apply_clahe:
            clahe = cv2.createCLAHE(
                clipLimit=float(self.config.clahe_clip_limit),
                tileGridSize=(
                    max(1, int(self.config.clahe_tile_grid_size)),
                    max(1, int(self.config.clahe_tile_grid_size)),
                ),
            )
            l_channel = clahe.apply(l_channel)

        normalized = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
        if self.config.gamma is None or self.config.gamma <= 0 or abs(self.config.gamma - 1.0) < 1e-6:
            return normalized

        gamma = float(self.config.gamma)
        table = np.asarray(
            [((index / 255.0) ** (1.0 / gamma)) * 255.0 for index in range(256)],
            dtype=np.uint8,
        )
        return cv2.LUT(normalized, table)


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


def _apply_unsharp_mask(
    image: np.ndarray,
    *,
    sigma: float,
    amount: float,
) -> np.ndarray:
    """反锐化掩模算法 及: 结果 = 原图 + amount x (原图 - 模糊图)"""
    normalized_amount = max(0.0, float(amount))
    if normalized_amount <= 0.0:
        return image

    blurred = cv2.GaussianBlur(image, (0, 0), max(0.1, float(sigma)))
    # dst=(1+amount)×image−amount×blurred
    sharpened = cv2.addWeighted(
        image.astype(np.float32),
        1.0 + normalized_amount,
        blurred.astype(np.float32),
        -normalized_amount,
        0.0,
    )
    return np.clip(sharpened, 0.0, 255.0).astype(np.uint8)


def _odd_kernel(value: int) -> int:
    normalized = max(1, int(value))
    return normalized if normalized % 2 == 1 else normalized + 1
