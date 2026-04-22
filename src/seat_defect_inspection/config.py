"""项目配置模型。"""

from __future__ import annotations

from dataclasses import dataclass, field

# 这里只依赖调试图默认档位常量，避免导入 cvops 包形成循环依赖。
from .debug_artifacts import DEFAULT_DEBUG_ARTIFACT_MODE
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
    """YOLO 前的 OpenCV 预处理参数。
    resize_width: 缩放宽度。
    resize_height: 缩放高度。
    denoise_method: 去噪方法。
    gaussian_kernel_size: 高斯核大小。
    bilateral_diameter: 双边滤波直径。
    bilateral_sigma_color: 双边滤波颜色标准差。
    bilateral_sigma_space: 双边滤波空间标准差。
    white_balance_method: 白平衡方法。
    max_white_balance_gain: 最大白平衡增益。
    apply_illumination_correction: 是否应用光照校正。
    illumination_blur_kernel_size: 光照模糊核大小。
    illumination_strength: 光照强度。
    apply_clahe: 是否应用 CLAHE。
    clahe_clip_limit: CLAHE 截断限制。
    clahe_tile_grid_size: CLAHE 网格大小。
    gamma: 伽马值。
    sharpen: 是否应用锐化。
    sharpen_sigma: 锐化核大小。
    sharpen_amount: 锐化强度。
    camera_matrix: 相机矩阵。
    distortion_coeffs: 畸变系数。
    """

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
    """ROI 精修与掩膜生成配置。
    crop_expand_ratio: 扩框比例。
    crop_shrink_ratio: 缩框比例。
    mask_mode: 掩膜模式。
    morphology_kernel_size: 形态学核大小。
    ignore_dilate_kernel_size: 忽略区域膨胀核大小。
    edge_ignore_pixels: 边缘忽略像素数。
    texture_denoise_method: 纹理去噪方法。
    texture_gaussian_kernel_size: 纹理高斯核大小。
    texture_bilateral_diameter: 纹理双边滤波直径。
    texture_bilateral_sigma_color: 纹理双边滤波颜色标准差。
    texture_bilateral_sigma_space: 纹理双边滤波空间标准差。
    apply_texture_clahe: 是否应用纹理 CLAHE。
    texture_clahe_clip_limit: 纹理 CLAHE 截断限制。
    texture_clahe_tile_grid_size: 纹理 CLAHE 网格大小。
    texture_illumination_correction: 是否应用纹理光照校正。
    texture_illumination_blur_kernel_size: 纹理光照模糊核大小。
    texture_illumination_strength: 纹理光照强度。
    mask_feather_kernel_size: 掩膜羽化核大小。
    edge_enhance_method: 边缘增强方法。
    edge_enhance_weight: 边缘增强权重。
    suppress_background: 是否抑制背景。
    background_fill_mode: 背景填充模式。
    background_blur_kernel_size: 背景模糊核大小。
    safe_margin_erode_kernel_size: 安全边距腐蚀核大小。
    alignment: 对齐配置。
    """

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


@dataclass(slots=True)
class DetectionConfig:
    """单机位 YOLO 检测配置。"""

    model_path: str | None = None
    target_class: str = "seat"
    ignore_classes: list[str] = field(default_factory=list)
    # 保持与历史版本一致，避免结构重构时悄悄改变 YOLO 检测策略。
    confidence: float = 0.25
    iou: float = 0.45
    device: str = "cpu"
    fallback_box: BoundingBox | None = None


@dataclass(slots=True)
class PatchCoreConfig:
    """PatchCore 风格模型与 patch 过滤配置。
    backend: 模型后端，可选值为 "handcrafted" 或 "patchcore"。
    image_size: 输入图像大小。
    patch_size: 补丁大小。
    stride: 步长。
    max_memory: 最大内存。
    threshold_quantile: 阈值量化。
    texture_input: 纹理输入。
    min_target_coverage: 最小目标覆盖率。
    max_ignore_overlap: 最大忽略重叠。
    min_valid_patch_ratio: 最小有效补丁比例。
    decision_score_margin: 决策得分边际。
    strong_patch_score_ratio: 强补丁得分比例。
    min_strong_patch_count: 最小强补丁数量。
    min_strong_component_count: 最小强组件数量。
    min_strong_patch_ratio: 最小强补丁比例。
    min_strong_component_ratio: 最小强组件比例。
    critical_score_margin: 关键得分边际。
    critical_peak_score_margin: 关键峰值得分边际。
    critical_min_component_patch_count: 关键最小组件补丁数量。
    backbone_name: 骨干网络名称。
    feature_layers: 特征层。
    backbone_pretrained: 骨干网络预训练。
    backbone_weights_path: 骨干网络权重路径。
    backbone_device: 骨干网络设备。
    feature_pool_kernel_size: 特征池化核大小。
    coreset_sampling_ratio: 核心集采样比例。
    texture_input: 纹理输入。
    min_target_coverage: 最小目标覆盖率。
    max_ignore_overlap: 最大忽略重叠。
    min_valid_patch_ratio: 最小有效补丁比例。
    decision_score_margin: 决策得分边际。
    strong_patch_score_ratio: 强补丁得分比例。
    min_strong_patch_count: 最小强补丁数量。
    min_strong_component_count: 最小强组件数量。
    min_strong_patch_ratio: 最小强补丁比例。
    min_strong_component_ratio: 最小强组件比例。
    critical_score_margin: 关键得分边际。
    critical_peak_score_margin: 关键峰值得分边际。
    critical_min_component_patch_count: 关键最小组件补丁数量。
    backbone_name: 骨干网络名称。
    feature_layers: 特征层。
    backbone_pretrained: 骨干网络预训练。
    backbone_weights_path: 骨干网络权重路径。
    backbone_device: 骨干网络设备。
    feature_pool_kernel_size: 特征池化核大小。
    coreset_sampling_ratio: 核心集采样比例。
    """

    backend: str = "handcrafted"
    image_size: int = 256
    patch_size: int = 32
    stride: int = 16
    max_memory: int = 512
    threshold_quantile: float = 0.99
    texture_input: str = "lab_l"
    min_target_coverage: float = 0.8
    max_ignore_overlap: float = 0.1
    min_valid_patch_ratio: float = 0.65
    decision_score_margin: float = 1.08
    strong_patch_score_ratio: float = 0.9
    min_strong_patch_count: int = 3
    min_strong_component_count: int = 2
    min_strong_patch_ratio: float = 0.015
    min_strong_component_ratio: float = 0.01
    critical_score_margin: float = 1.35
    critical_peak_score_margin: float = 1.45
    critical_min_component_patch_count: int = 2
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
class YoloTrainingConfig:
    """YOLO 训练配置。"""

    model_path: str = "yolo11m-seg.pt"
    data_config_path: str = "configs/seat_defect_yolo.dataset.example.yaml"
    epochs: int = 100
    imgsz: int = 1280
    batch: int = 8
    device: str = "cpu"
    project: str = "outputs/yolo_training"
    name: str = "seat_defect"
    workers: int = 4
    patience: int = 20
    cache: bool = False
    pretrained: bool = True
    seat_model_id: str | None = None


@dataclass(slots=True)
class CameraConfig:
    """单机位完整配置。"""

    camera_id: str
    source: str
    patchcore_model_path: str
    train_good_dir: str | None = None
    enabled: bool = True
    color_insensitive_mode: bool = False
    quality: QualityGuardConfig = field(default_factory=QualityGuardConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    roi: RoiRefineConfig = field(default_factory=RoiRefineConfig)
    patchcore: PatchCoreConfig = field(default_factory=PatchCoreConfig)
    color_branch: ColorBranchConfig = field(default_factory=ColorBranchConfig)


@dataclass(slots=True)
class FusionConfig:
    """多机位融合判定策略。"""

    reject_on_any_reject: bool = True
    ng_strategy: str = "any"
    early_stop_on_ng: bool = True
    defect_overrides_reject: bool = True


@dataclass(slots=True)
class SeatModelConfig:
    """按座椅型号组织的多机位配置。"""

    seat_model_id: str
    cameras: list[CameraConfig] = field(default_factory=list)
    display_name: str | None = None
    yolo_training: YoloTrainingConfig | None = None


@dataclass(slots=True)
class InspectionConfig:
    """项目顶层配置。"""

    cameras: list[CameraConfig] = field(default_factory=list)
    seat_models: list[SeatModelConfig] = field(default_factory=list)
    default_seat_model_id: str | None = None
    output_json_path: str = "outputs/seat_defect_inspection/results.json"
    debug_dir: str = "outputs/seat_defect_inspection/debug"
    capture_dir: str = "outputs/seat_defect_inspection/capture"
    save_debug_artifacts: bool = True
    debug_artifact_mode: str = DEFAULT_DEBUG_ARTIFACT_MODE
    capture_retries: int = 3
    part_id: str = "seat_demo"
    fusion: FusionConfig = field(default_factory=FusionConfig)
    yolo_training: YoloTrainingConfig | None = None


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
    "RoiRefineConfig",
    "SeatModelConfig",
    "YoloTrainingConfig",
]
