"""OpenCV preprocessing before YOLO detection."""

from __future__ import annotations

import cv2
import numpy as np

from .config import PreprocessConfig


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

        if self._camera_matrix is not None and self._distortion_coeffs is not None:
            processed = cv2.undistort(processed, self._camera_matrix, self._distortion_coeffs)

        if self.config.resize_width and self.config.resize_height:
            processed = cv2.resize(
                processed,
                (int(self.config.resize_width), int(self.config.resize_height)),
                interpolation=cv2.INTER_AREA,
            )

        processed = self._denoise(processed)
        processed = self._normalize_lighting(processed)

        if self.config.sharpen:
            processed = cv2.filter2D(
                processed,
                -1,
                np.asarray([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
            )
        return processed

    def _denoise(self, image):
        method = self.config.denoise_method.strip().lower()
        if method == "none":
            return image
        if method == "bilateral":
            return cv2.bilateralFilter(
                image,
                d=max(1, int(self.config.bilateral_diameter)),
                sigmaColor=float(self.config.bilateral_sigma_color),
                sigmaSpace=float(self.config.bilateral_sigma_space),
            )
        return cv2.GaussianBlur(
            image,
            (
                _odd_kernel(self.config.gaussian_kernel_size),
                _odd_kernel(self.config.gaussian_kernel_size),
            ),
            0,
        )

    def _normalize_lighting(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

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


def _odd_kernel(value: int) -> int:
    normalized = max(1, int(value))
    return normalized if normalized % 2 == 1 else normalized + 1
