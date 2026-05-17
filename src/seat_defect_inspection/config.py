"""工程层配置模型。

检测 runtime 配置统一从 ``seat_defect_core`` 导出；本模块只扩展 CLI/采图/训练字段。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from seat_defect_core.config import (
    CameraConfig as CoreCameraConfig,
    InspectionConfig as CoreInspectionConfig,
    SeatModelConfig as CoreSeatModelConfig,
)

@dataclass(slots=True)
class CameraConfig(CoreCameraConfig):
    """工程层单机位配置，额外携带 PatchCore 正常样本目录。"""

    train_good_dir: str | None = None


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
    cache: bool | str = False
    pretrained: bool = True
    amp: bool = True
    optimizer: str = "auto"
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    mixup: float = 0.0
    copy_paste: float = 0.0
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    rect: bool = False
    seat_model_id: str | None = None


@dataclass(slots=True)
class SeatModelConfig(CoreSeatModelConfig):
    """工程层座椅型号配置，额外携带 YOLO 训练块。"""

    yolo_training: YoloTrainingConfig | None = None


@dataclass(slots=True)
class InspectionConfig(CoreInspectionConfig):
    """工程层顶层配置，额外携带采图和训练字段。"""

    capture_dir: str = "outputs/seat_defect_inspection/capture"
    capture_retries: int = 3
    yolo_training: YoloTrainingConfig | None = None
    seat_models: list[SeatModelConfig] = field(default_factory=list)


__all__ = [
    "CameraConfig",
    "InspectionConfig",
    "SeatModelConfig",
    "YoloTrainingConfig",
]
