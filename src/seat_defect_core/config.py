"""SDK runtime configuration models."""

from __future__ import annotations

from dataclasses import dataclass, field
from .schemas import BoundingBox


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
    """预处理参数。"""

    # 尺寸统一：为空时保持原图大小。
    resize_width: int | None = None
    resize_height: int | None = None

    # 去噪。
    denoise_method: str = "gaussian"
    gaussian_kernel_size: int = 5
    bilateral_diameter: int = 5
    bilateral_sigma_color: float = 30.0
    bilateral_sigma_space: float = 30.0

    # 白平衡和光照校正。
    white_balance_method: str = "none"
    max_white_balance_gain: float = 1.25
    apply_illumination_correction: bool = False
    illumination_blur_kernel_size: int = 51
    illumination_strength: float = 0.7

    # 对比度和锐化。
    apply_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    gamma: float | None = None
    sharpen: bool = False
    sharpen_sigma: float = 1.2
    sharpen_amount: float = 1.0

    # 畸变校正：只在确实有标定参数时启用。
    camera_matrix: list[list[float]] | None = None
    distortion_coeffs: list[float] | None = None


@dataclass(slots=True)
class AlignmentConfig:
    """ROI 裁剪后的输出尺寸。"""

    output_width: int = 256
    output_height: int = 256


@dataclass(slots=True)
class RoiRefineConfig:
    """ROI 裁剪与有效区域配置。"""

    # 基于 YOLO 分割外接框做轻量扩缩，避免裁得过紧或过松。
    crop_expand_ratio: float = 0.05
    crop_shrink_ratio: float = 0.0

    # 屏蔽边缘像素，减少座椅边界和背景混入 PatchCore。
    edge_ignore_pixels: int = 6
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)


@dataclass(slots=True)
class DetectionConfig:
    """单机位 YOLO 检测配置。"""

    model_path: str | None = None
    target_class: str = "seat"
    # 保持与历史版本一致，避免结构重构时悄悄改变 YOLO 检测策略。
    confidence: float = 0.25
    iou: float = 0.45
    device: str = "cpu"
    fallback_box: BoundingBox | None = None


@dataclass(slots=True)
class PatchCoreConfig:
    """PatchCore model and decision parameters."""

    # patch 提取和 memory bank。
    backend: str = "full"
    image_size: int = 256
    patch_size: int = 32
    stride: int = 16
    max_memory: int = 512
    threshold_quantile: float = 0.99
    texture_input: str = "lab_l"

    # 有效 patch 过滤。
    min_target_coverage: float = 0.8
    max_ignore_overlap: float = 0.1
    min_valid_patch_ratio: float = 0.65

    # 图像级与连通域判定。
    decision_score_margin: float = 1.08
    strong_patch_score_ratio: float = 0.9
    min_strong_patch_count: int = 3
    min_strong_component_count: int = 2
    min_strong_patch_ratio: float = 0.015
    min_strong_component_ratio: float = 0.01

    # 小面积高峰值缺陷的快速放行规则。
    critical_score_margin: float = 1.35
    critical_peak_score_margin: float = 1.45
    critical_min_component_patch_count: int = 2

    # full 后端的骨干网络参数。
    backbone_name: str = "wide_resnet50_2"
    feature_layers: list[str] = field(default_factory=lambda: ["layer2", "layer3"])
    backbone_pretrained: bool = False
    backbone_weights_path: str | None = None
    backbone_device: str = "cpu"
    feature_pool_kernel_size: int = 3
    coreset_sampling_ratio: float = 0.1


@dataclass(slots=True)
class ColorBranchConfig:
    """颜色一致性分支配置。"""

    enabled: bool = False
    threshold_quantile: float = 0.99
    threshold: float | None = None
    min_valid_pixel_ratio: float = 0.4


@dataclass(slots=True)
class RegionConfig:
    """单机位标准 ROI 内的局部 PatchCore 区域。"""

    region_id: str
    # 标准 ROI 内的归一化矩形：[x1, y1, x2, y2]，取值范围 0-1。
    box: list[float]
    patchcore_model_path: str
    enabled: bool = True
    patchcore: PatchCoreConfig | None = None


@dataclass(slots=True)
class CameraConfig:
    """单机位 runtime 配置。"""

    camera_id: str
    source: str
    patchcore_model_path: str
    enabled: bool = True
    color_insensitive_mode: bool = False
    quality: QualityGuardConfig = field(default_factory=QualityGuardConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    roi: RoiRefineConfig = field(default_factory=RoiRefineConfig)
    patchcore: PatchCoreConfig = field(default_factory=PatchCoreConfig)
    color_branch: ColorBranchConfig = field(default_factory=ColorBranchConfig)
    regions: list[RegionConfig] = field(default_factory=list)


@dataclass(slots=True)
class FusionConfig:
    """多机位融合判定策略。"""

    reject_on_any_reject: bool = True
    ng_strategy: str = "any"
    # 默认输出所有机位结果，避免首个 NG 提前截断整件复盘信息。
    early_stop_on_ng: bool = False
    defect_overrides_reject: bool = True


@dataclass(slots=True)
class SeatModelConfig:
    """按座椅型号组织的多机位配置。"""

    seat_model_id: str
    cameras: list[CameraConfig] = field(default_factory=list)
    display_name: str | None = None


@dataclass(slots=True)
class InspectionConfig:
    """SDK 顶层 runtime 配置。"""

    cameras: list[CameraConfig] = field(default_factory=list)
    seat_models: list[SeatModelConfig] = field(default_factory=list)
    default_seat_model_id: str | None = None
    output_json_path: str = "outputs/seat_defect_inspection/results.json"
    debug_dir: str = "outputs/seat_defect_inspection/debug"
    part_id: str = "seat_demo"
    fusion: FusionConfig = field(default_factory=FusionConfig)


__all__ = [
    "AlignmentConfig",
    "CameraConfig",
    "ColorBranchConfig",
    "DetectionConfig",
    "FusionConfig",
    "InspectionConfig",
    "PatchCoreConfig",
    "PreprocessConfig",
    "QualityGuardConfig",
    "RegionConfig",
    "RoiRefineConfig",
    "SeatModelConfig",
]
