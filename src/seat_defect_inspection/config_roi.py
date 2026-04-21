"""ROI 裁剪、掩膜与对齐配置。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class AlignmentConfig:
    """ROI 裁剪后的对齐参数。"""

    enabled: bool = False
    method: str = "resize"
    template_image_path: str | None = None
    output_width: int = 256
    output_height: int = 256
    ecc_iterations: int = 50


@dataclass(slots=True)
class RoiRefineConfig:
    """ROI 精修与掩膜生成配置。"""

    crop_expand_ratio: float = 0.05
    crop_shrink_ratio: float = 0.0
    mask_mode: str = "grabcut"
    morphology_kernel_size: int = 5
    ignore_dilate_kernel_size: int = 9
    edge_ignore_pixels: int = 6
    texture_denoise_method: str = "bilateral"
    texture_gaussian_kernel_size: int = 5
    texture_bilateral_diameter: int = 7
    texture_bilateral_sigma_color: float = 40.0
    texture_bilateral_sigma_space: float = 40.0
    apply_texture_clahe: bool = True
    texture_clahe_clip_limit: float = 2.0
    texture_clahe_tile_grid_size: int = 8
    texture_illumination_correction: bool = True
    texture_illumination_blur_kernel_size: int = 41
    texture_illumination_strength: float = 0.85
    mask_feather_kernel_size: int = 15
    edge_enhance_method: str = "scharr"
    edge_enhance_weight: float = 0.18
    suppress_background: bool = True
    background_fill_mode: str = "median"
    background_blur_kernel_size: int = 31
    safe_margin_erode_kernel_size: int = 3
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
