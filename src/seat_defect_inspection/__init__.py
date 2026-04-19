"""座椅缺陷检测独立项目。"""

from .config import (
    AlignmentConfig,
    CameraConfig,
    ColorBranchConfig,
    DetectionConfig,
    FusionConfig,
    InspectionConfig,
    PatchCoreConfig,
    PreprocessConfig,
    QualityGuardConfig,
    RoiRefineConfig,
    SeatModelConfig,
    YoloTrainingConfig,
)
from .runtime_config import load_config, load_yolo_training_config
from .schemas import BoundingBox, CaptureRecord, CaptureSummary, InspectionResult
from .service import (
    InspectionService,
    capture_samples,
    run_inspection,
    train_patchcore_models,
)
from .yolo_training import train_yolo_model

__all__ = [
    "AlignmentConfig",
    "BoundingBox",
    "CameraConfig",
    "ColorBranchConfig",
    "CaptureRecord",
    "CaptureSummary",
    "DetectionConfig",
    "FusionConfig",
    "InspectionConfig",
    "InspectionResult",
    "InspectionService",
    "PatchCoreConfig",
    "PreprocessConfig",
    "QualityGuardConfig",
    "RoiRefineConfig",
    "SeatModelConfig",
    "YoloTrainingConfig",
    "capture_samples",
    "load_config",
    "load_yolo_training_config",
    "run_inspection",
    "train_patchcore_models",
    "train_yolo_model",
]
