"""座椅缺陷检测项目的运行配置。"""

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
    """YOLO 前的 OpenCV 预处理参数。"""

    resize_width: int | None = None
    resize_height: int | None = None
    denoise_method: str = "gaussian"
    gaussian_kernel_size: int = 5
    bilateral_diameter: int = 5
    bilateral_sigma_color: float = 30.0
    bilateral_sigma_space: float = 30.0
    apply_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    gamma: float | None = None
    sharpen: bool = False
    camera_matrix: list[list[float]] | None = None
    distortion_coeffs: list[float] | None = None


@dataclass(slots=True)
class DetectionConfig:
    """单机位 YOLO 检测配置。"""

    model_path: str | None = None
    target_class: str = "seat_main"
    ignore_classes: list[str] = field(default_factory=list)
    confidence: float = 0.25
    iou: float = 0.45
    device: str = "cpu"
    prefer_segmentation_mask: bool = True
    fallback_box: BoundingBox | None = None


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
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)


@dataclass(slots=True)
class PatchCoreConfig:
    """PatchCore 风格模型与 patch 过滤配置。"""

    image_size: int = 256
    patch_size: int = 32
    stride: int = 16
    max_memory: int = 512
    threshold_quantile: float = 0.99
    texture_input: str = "lab_l"
    min_target_coverage: float = 0.8
    max_ignore_overlap: float = 0.1
    min_valid_patch_ratio: float = 0.65


@dataclass(slots=True)
class ColorBranchConfig:
    """颜色一致性分支配置。"""

    enabled: bool = False
    threshold_quantile: float = 0.99
    threshold: float | None = None
    min_valid_pixel_ratio: float = 0.4


@dataclass(slots=True)
class CameraConfig:
    """单机位完整配置。"""

    camera_id: str
    source: str
    patchcore_model_path: str
    train_good_dir: str | None = None
    enabled: bool = True
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


@dataclass(slots=True)
class InspectionConfig:
    """项目顶层配置。"""

    cameras: list[CameraConfig]
    output_json_path: str = "outputs/seat_defect_inspection/results.json"
    debug_dir: str = "outputs/seat_defect_inspection/debug"
    capture_dir: str = "outputs/seat_defect_inspection/capture"
    save_debug_artifacts: bool = True
    capture_retries: int = 3
    part_id: str = "seat_demo"
    fusion: FusionConfig = field(default_factory=FusionConfig)


@dataclass(slots=True)
class YoloTrainingConfig:
    """YOLO 训练配置。"""

    model_path: str = "yolo11n.pt"
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
