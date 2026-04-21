"""基础图像质量与预处理配置。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class QualityGuardConfig:
    """检测前的图像质量阈值。"""

    min_laplacian_variance: float = 80.0
    min_brightness_mean: float = 30.0
    max_brightness_mean: float = 225.0
    max_overexposed_ratio: float = 0.25
    max_underexposed_ratio: float = 0.35


@dataclass(slots=True)
class PreprocessConfig:
    """YOLO 前的 OpenCV 预处理参数。"""

    resize_width: int | None = None
    resize_height: int | None = None
    denoise_method: str = "gaussian"
    gaussian_kernel_size: int = 5
    bilateral_diameter: int = 5
    bilateral_sigma_color: float = 30.0
    bilateral_sigma_space: float = 30.0
    white_balance_method: str = "none"
    max_white_balance_gain: float = 1.25
    apply_illumination_correction: bool = False
    illumination_blur_kernel_size: int = 51
    illumination_strength: float = 0.7
    apply_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    gamma: float | None = None
    sharpen: bool = False
    sharpen_sigma: float = 1.2
    sharpen_amount: float = 1.0
    camera_matrix: list[list[float]] | None = None
    distortion_coeffs: list[float] | None = None
