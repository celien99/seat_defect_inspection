"""Core inspect runtime configuration models."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class QualityGuardConfig:
    """检测前的图像质量阈值。"""

    min_laplacian_variance: float = 80.0
    min_brightness_mean: float = 30.0
    max_brightness_mean: float = 225.0
    max_overexposed_ratio: float = 0.25
    max_underexposed_ratio: float = 0.35


@dataclass
class AlignmentConfig:
    """ROI 裁剪后的输出尺寸。"""

    output_width: int = 256
    output_height: int = 256


@dataclass
class RoiRefineConfig:
    """ROI 裁剪与有效区域配置。"""

    # 基于 YOLO 分割外接框做轻量扩缩，避免裁得过紧或过松。
    crop_expand_ratio: float = 0.05
    crop_shrink_ratio: float = 0.0

    # 对 YOLO 前景 mask 做保守内缩，剔除座椅轮廓边缘的无效像素。
    mask_erode_pixels: int = 1

    # 屏蔽边缘像素，减少座椅边界和背景混入 PatchCore。
    edge_ignore_pixels: int = 6
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)


@dataclass
class DetectionConfig:
    """单机位 YOLO 检测配置。"""

    model_path: str | None = None
    target_class: str = "seat"
    # 保持与历史版本一致，避免结构重构时悄悄改变 YOLO 检测策略。
    confidence: float = 0.25
    iou: float = 0.45
    device: str = "cpu"
    imgsz: int = 960
    # YOLO 实例 mask 可能在目标内部留下低置信空洞；PatchCore 需要完整前景区域。
    fill_segmentation_holes: bool = True
    segmentation_hole_fill_max_area_ratio: float = 0.08


@dataclass
class ClassificationConfig:
    """缺陷分类配置。"""

    enabled: bool = False
    model_path: str | None = None
    confidence_threshold: float = 0.5
    inference_timeout_ms: float = 200.0
    sam_refinement_enabled: bool = False
    enable_zero_shot_fallback: bool = False
    zero_shot_prompts: dict[str, str] = field(default_factory=dict)


@dataclass
class FalsePositiveVetoConfig:
    """基于启发式规则的误报过滤。"""

    enabled: bool = False
    min_defect_area_ratio: float = 0.0002
    max_defect_aspect_ratio: float = 0.05
    edge_proximity_ratio: float = 0.02


@dataclass
class PatchCoreConfig:
    """PatchCore model and decision parameters."""

    # patch 提取和 memory bank。
    backend: str = "full"
    image_size: int = 256
    patch_size: int = 32
    stride: int = 16
    max_memory: int = 1024
    threshold_quantile: float = 0.99
    texture_input: str = "lab_l"

    # 有效 patch 过滤。
    min_target_coverage: float = 0.8
    max_ignore_overlap: float = 0.1
    min_valid_patch_ratio: float = 0.65

    # 训练时阈值上限分位数（替代 max*1.1 的统计鲁棒上界）。
    training_threshold_upper_quantile: float = 0.995

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

    # 峰值规则（peak_rule）最小连通 patch 数，防止单 patch 噪声误触发。
    min_peak_component_patch_count: int = 1

    # full 后端的骨干网络参数。
    backbone_name: str = "wide_resnet50_2"
    feature_layers: list[str] = field(default_factory=lambda: ["layer2", "layer3"])
    backbone_pretrained: bool = False
    backbone_weights_path: str | None = None
    backbone_device: str = "cpu"
    feature_pool_kernel_size: int = 3
    coreset_sampling_ratio: float = 0.1


@dataclass
class ColorBranchConfig:
    """颜色一致性分支配置。"""

    enabled: bool = False
    threshold_quantile: float = 0.99
    threshold: float | None = None
    min_valid_pixel_ratio: float = 0.4
    training_threshold_upper_quantile: float = 0.995


@dataclass
class RegionConfig:
    """单机位标准 ROI 内的局部 PatchCore 区域。"""

    region_id: str
    # 标准 ROI 内的归一化矩形：[x1, y1, x2, y2]，取值范围 0-1。
    box: list[float]
    patchcore_model_path: str
    enabled: bool = True
    patchcore: PatchCoreConfig | None = None


@dataclass
class CameraConfig:
    """单机位 runtime 配置。"""

    camera_id: str
    patchcore_model_path: str
    source: str = ""
    enabled: bool = True
    color_insensitive_mode: bool = False
    quality: QualityGuardConfig = field(default_factory=QualityGuardConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    roi: RoiRefineConfig = field(default_factory=RoiRefineConfig)
    patchcore: PatchCoreConfig = field(default_factory=PatchCoreConfig)
    color_branch: ColorBranchConfig = field(default_factory=ColorBranchConfig)
    regions: list[RegionConfig] = field(default_factory=list)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    veto: FalsePositiveVetoConfig = field(default_factory=FalsePositiveVetoConfig)


@dataclass
class FusionConfig:
    """多机位融合判定策略。"""

    reject_on_any_reject: bool = True
    ng_strategy: str = "any"
    defect_overrides_reject: bool = True


@dataclass
class FlywheelConfig:
    """自学习数据闭环配置。"""

    enabled: bool = False
    buffer_dir: str = "flywheel_data/"
    auto_label_threshold: float = 0.92
    human_validation_threshold: float = 0.60
    min_samples_before_retrain: int = 200
    retrain_cooldown_hours: int = 72
    sampling_rate_ok: float = 0.01
    incremental_patchcore_enabled: bool = True
    max_samples_per_class: int = 5000
    retrain_trigger_mode: str = "any"


@dataclass
class SeatModelConfig:
    """按座椅型号组织的多机位配置。"""

    seat_model_id: str
    cameras: list[CameraConfig] = field(default_factory=list)
    display_name: str | None = None


@dataclass
class InspectionConfig:
    """core 顶层 inspect runtime 配置。"""

    cameras: list[CameraConfig] = field(default_factory=list)
    seat_models: list[SeatModelConfig] = field(default_factory=list)
    default_seat_model_id: str | None = None
    output_json_path: str = "outputs/seat_defect_inspection/results.json"
    debug_dir: str = "outputs/seat_defect_inspection/debug"
    debug_artifacts_enabled: bool = True
    debug_artifact_names: list[str] = field(
        default_factory=lambda: ["overlay"],
    )
    part_id: str = "seat_demo"
    fusion: FusionConfig = field(default_factory=FusionConfig)
    flywheel: FlywheelConfig = field(default_factory=FlywheelConfig)
    model_registry_dir: str | None = None


__all__ = [
    "AlignmentConfig",
    "CameraConfig",
    "ClassificationConfig",
    "ColorBranchConfig",
    "DetectionConfig",
    "FalsePositiveVetoConfig",
    "FlywheelConfig",
    "FusionConfig",
    "InspectionConfig",
    "PatchCoreConfig",
    "QualityGuardConfig",
    "RegionConfig",
    "RoiRefineConfig",
    "SeatModelConfig",
]
