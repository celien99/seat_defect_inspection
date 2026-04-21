"""检测、异常分析与训练配置。"""

from __future__ import annotations

from dataclasses import dataclass, field

from .schemas import BoundingBox


@dataclass(slots=True)
class DetectionConfig:
    """单机位 YOLO 检测配置。"""

    model_path: str | None = None
    target_class: str = "seat"
    ignore_classes: list[str] = field(default_factory=list)
    confidence: float = 0.25
    iou: float = 0.45
    device: str = "cpu"
    fallback_box: BoundingBox | None = None


@dataclass(slots=True)
class PatchCoreConfig:
    """PatchCore 风格模型与 patch 过滤配置。"""

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
