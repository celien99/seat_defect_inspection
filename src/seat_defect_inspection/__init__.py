"""座椅缺陷检测独立项目。"""

from importlib import import_module

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

_LAZY_EXPORTS = {
    "InspectionService": (".service", "InspectionService"),
    "capture_samples": (".service", "capture_samples"),
    "run_inspection": (".service", "run_inspection"),
    "train_patchcore_models": (".service", "train_patchcore_models"),
    "train_yolo_model": (".yolo", "train_yolo_model"),
}


def _load_lazy_export(name: str):
    """按需加载重依赖导出，避免顶层导入时把整条主流程一起拉起。"""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


def __getattr__(name: str):
    """延迟暴露主流程入口，减少顶层包导入成本。"""
    return _load_lazy_export(name)
